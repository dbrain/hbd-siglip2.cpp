# siglip2.cpp megakernel-v0 — handoff

> ## Mood of the room (read this first)
>
> User on 2026-05-09 after the QKV-fusion + graph-cache-dead-end pass:
> *"i like that megakernel was a 'meme' over in qwen3-tts.cpp land and
> now its the saviour. i mean its working over there, so why not."*
>
> Translation: greenlit. The user has lived through this exact arc on
> qwen3-tts.cpp (skeptical → "go nuts" → shipped → wins). They're
> calibrated to the multi-day commitment. Iterate, measure, ship.
>
> **Standing rules from the parent project (`HANDOFF.md` "user's
> standing instructions") still apply:**
> - **VRAM**: full-tower resident is the metric. Currently 1504 MiB
>   post-load (vs Python 2322). Don't regress this. Megakernel done
>   right *reduces* VRAM (smaller scheduler arena from 486-launch graph
>   collapsing to ~30 launches).
> - **Cosine ≥ 0.999** on parity scripts is the floor. Float-accumulator
>   reordering inside a fused kernel can cost you ~10-50 µcos per layer;
>   verify with `parity_check_image.py` (heavy-downsample, the strictest)
>   not just `parity_check_vision.py` (synthetic, easier). Today's
>   real-image cosine is **0.999545** — there's only ~545 µcos of
>   headroom before you blow the floor across all 27 fused layers.
> - **Q8_0 weights are the cap**, F32 activations.
> - **Don't push commits.** Bank locally, the user pushes themselves.
> - **Bench discipline.** GPU is shared with `kobbler-vision-1`,
>   `kobbler-tts-qwen3-1`, sometimes `tts-qwen3-iter`. Quote ratios over
>   absolute ms; do 30-run p50 minimum.

## Cold-pickup orientation

**Phase A (A0-A4) and Phase B all landed.** Megakernel collapsed text
launches ~745 → ~30. Phase B added per-shape graph caching (TextEncoder
+ VisionEncoder) and bypassed `ggml_backend_sched`, which unlocked
ggml-cuda's CUDA-graph warmup path → ~30 → 1 submit per encode.

Current state on `dbrain/siglip2-v0`:

- All 4 endpoints beat python clean (0.76-0.98×, see Phase B receipts below).
- Internal text n=5 compute = **19.1 ms ±0.05** (rock-steady; CUDA graph
  replay killed jitter).
- All 4 parity scripts PASS, bit-identical to A4.
- VRAM unchanged ~1.5 GiB full-tower post-warmup.

**Phase C is the kernel/quant lap** — see "Phase C — handoff for the
kernel/quant lap" below. Graph-layer is exhausted; further gains need
custom MMQ for siglip2's M-shapes or Q4_K_M quant. Target: cfe ≤ 0.85×
python ("real embarrassment").

## ✅ Phase A0 landed (2026-05-09)

**Fused LayerNorm-with-affine** is in. First custom kernel under
`src/cuda/`. Plan-builder finds **55 LN groups for text** and **56 for
vision**; op-hook fires `fused_layernorm_affine_kernel` per anchor and
short-circuits the upstream norm + mul as followers. Per text encode
drops 110 launches; per vision encode drops 112; `/v1/classify` drops
662. Wins are **3-4 % across all four endpoints** on a clean GPU
(kobbler-vision-1 stopped):

  - `/v1/embeddings`:               58.2 → 55.8 ms (**−4.1 %**)
  - `/v1/text_embeddings`:          38.3 → 37.1 ms (**−3.1 %**)
  - `/v1/classify`:                 94.1 → 90.2 ms (**−4.1 %**)
  - `/v1/classify_from_embeddings`: 38.5 → 37.1 ms (**−3.6 %**)

HF parity holds: vision 0.999944, text 0.999756, image-1920×1080→729
0.999523, score 0.999986. Self-parity (ON vs OFF same binary) is
0.99992-0.99997 across the board — drift is reduction-tree shuffle
noise, not a numerical bug.

Smaller than expected because per-launch cost on these tiny ops
(`norm` + 1D-bcast `mul` + 1D-bcast `add`) is **~10-12 µs**, not the
12-15 µs estimated. The bigger wins live in the **QKV-prep chain**
(see Phase A1 below).

The scaffolding is now load-bearing — `siglip2_megakernel::install()`
in `src/siglip2_server.cpp::main()` (and both CLIs), the graph-begin
hook, the per-op anchor/follower pattern. Phase A1 is the same
machinery, bigger fusion.

## ✅ Phase A1 landed (2026-05-09) — QKV-prep chain fusion

**2-kernel design: copy mm+bias to persistent device scratch at the qkv_add
anchor; split-permute-pad-cast from scratch into Q_pad / K_cast / V_cast at
the topo-last of those three.** 9 launches per encoder block (1 add + 3 cont
+ 3 pad + 2 cast) → 2 kernels, saves 7 per block × 27 = 189 per encode.

Plan-builder finds 27 chains for both text and vision; dispatcher fires the
copy at qkv_add and the split at V_cast (or whichever of {q_pad, k_cast,
v_cast} happens to be last). Followers: 7 per block (3 cont + 3 pad +
non-anchor cast/pad). Scratch buffer is a single `cudaMalloc`'d slab sized
for `3 * H * max(text_n_pos, vision_n_pos) * 4 B` ≈ 9.6 MiB, lazy-allocated
on the first qkv_add anchor fire and reused across all 27 blocks per
encode (CUDA stream is sequential, so block N's split kernel reads scratch
before block N+1's copy kernel writes).

**Self-parity (A1 ON vs A1 OFF, same C++ binary):** vision **1.0000000**,
text **1.0000000** across all five prompts. Numerically bit-identical to
ggml's split path — scratch holds exactly what the bias-add would produce,
and the split kernel writes exactly what cont/pad/cast would. **HF parity
unchanged from A0-only:** vision 0.999944, text 0.999756, image-1920×1080→
729 0.999523, score 0.999986 / 5.6e-7.

**Bench A/B (clean GPU, kobbler-vision-1 stopped):**

  - `/v1/embeddings`:               55.9 → 49.6 ms (**−11.3 %**)
  - `/v1/text_embeddings`:          36.9 → 33.5 ms (**−9.2 %**)
  - `/v1/classify`:                 90.5 → 80.8 ms (**−10.7 %**)
  - `/v1/classify_from_embeddings`: 37.3 → 33.6 ms (**−9.9 %**)

Kill switch: `SIGLIP2_DISABLE_QKV_PREP=1`. Default ON in the install hook.

## ✅ Phase A2 landed (2026-05-09) — pointwise tail fusion (bias+residual, bias+gelu)

**Per-block tail fusion: collapse `add(add(mm, 1D-bias), 2D-residual)` and
`gelu(add(mm, 1D-bias))` into one kernel each.** Anchor on the OUTER op
(residual_add or GELU); the inner bias-add is a follower. Two new pointwise
kernels (`fused_bias_residual_kernel`, `fused_bias_gelu_kernel`) replace the
3-op pattern with 1 launch (mm itself stays put — no MMQ). Per encoder
block: 2 P1 (o-proj + down-proj) + 1 P2 (up-proj) → **3 launches/block × 27
= 81 per encode** (text and vision).

GELU formula matches `ggml_cuda_op_gelu_single` bit-for-bit (tanh-approx,
GELU_COEF_A=0.044715, SQRT_2_OVER_PI=0.79788...). Inner-add uses `__fadd_rn`
to suppress nvcc FMA fusion across the bias+residual chain — preserves the
per-store F32 rounding identical to the split (bias_add → residual_add)
path, so cosine drift is reduction-tree noise only.

**Plan-builder gates (BOTH required):**

1. **`is_blk_bias`** — bias name must contain `"blk"` (per-encoder-block
   tensors only). Probe head (`v.head.*`) and patch_embedding (`v.patch_embd.*`)
   matches are filtered. Hard-required: see "gallocr trap take 2" below.
2. **Consecutive-in-topo** — `inner.idx + 1 == outer.idx`. Defense in depth;
   per-block patterns always satisfy this, non-block patterns often don't.

**Self-parity (A2 ON vs A2 OFF, same C++ binary):** all four parity scripts
have **identical cosine + identical MSE bit-for-bit** between the two runs
(text 0.999756 / 2.41e-4, vision 0.999944 / 1.94e-5, image-1920×1080→729
0.999523 / 1.36e-4, score 0.999986). Per-block kernels are numerically
indistinguishable from the split path — bit-clean.

**HF parity unchanged from A1:** vision 0.999944, text 0.999756, image-1920×
1080→729 0.999523, score 0.999986. No drift on the 0.999 floor.

**Bench A/B (clean GPU, kobbler-vision-1 stopped, no dev containers):**

  - `/v1/embeddings`:               53.2 → 49.9 ms (**−6.2 %**)
  - `/v1/text_embeddings`:          39.8 → 36.5 ms (**−8.3 %**)
  - `/v1/classify`:                 86.8 → 82.2 ms (**−5.3 %**)
  - `/v1/classify_from_embeddings`: 38.0 → 36.7 ms (**−3.4 %**)

Absolute ms drift between this session and the A1 receipts is GPU contention
noise (this session's clean state had a different mix than A1's; the A/B is
the receipts you should trust). Computed against post-restart python (sharing
GPU with the cpp server during the python pass), three of four endpoints now
beat python:

  - `/v1/embeddings`:               49.9 cpp / 67.6 python = **0.74×**
  - `/v1/text_embeddings`:          36.5 cpp / 37.2 python = **0.98×**
  - `/v1/classify`:                 82.2 cpp / 86.5 python = **0.95×**
  - `/v1/classify_from_embeddings`: 36.7 cpp / 31.2 python = **1.18×**

Down from pre-A0's 1.07× / 1.43× / 1.34× / 1.79× — text-embeddings hit the
target ≤1.0× ratio called out in the original A scoping.

Kill switch: `SIGLIP2_DISABLE_TAIL_FUSION=1`. Default ON in the install hook.

## ✅ Phase A3 landed (2026-05-09) — post-FA cont specialization

**One specialized kernel replaces ggml's generic `cpy_f32_f32` for the post-FA
cont in `fa_attn_pad_slice` when `d_pad != d_head`.** Same launch count (1 per
block, 27 text / 28 vision incl. probe head), but the kernel is shape-specialised
(input strided over `d_pad`, output contiguous `(d_head, n_head, n_pos)`) so the
launch path skips ggml-cuda's generic-cpy dispatch overhead. Anchor on the cont
node; the upstream view is metadata-only (no follower needed).

Plan-builder gates: source must be `FLASH_ATTN_EXT`, view must be a 3D F32
zero-offset slice off the d-axis only (other axes preserved), and cont must be
contiguous F32 with matching `(d_head, n_head, n_pos)` shape. Defensive
`is_already_claimed` check guards against overlap with LN/QKV/tail anchors —
in practice they don't overlap, but cheap to keep honest.

**Self-parity (A3 ON vs OFF, same C++ binary):** all four parity scripts have
**bit-identical** first-8 outputs and HF cosine unchanged: vision **0.999944**,
text **0.999756**, image-1920×1080→729 **0.999523**, score **0.999986** /
probs MAE **5.6e-7**. The kernel is functionally identical to ggml's cpy
(same memory pattern, same per-store rounding); the only difference is dispatch
path, so any drift is reduction-tree noise — and we measure none.

**Bench A/B (clean GPU, only `tts-qwen3-iter` co-resident, 30 runs each, 2 reps):**

  - `/v1/embeddings`:               46.45 → 45.75 ms (**−1.5 %**)
  - `/v1/text_embeddings`:          32.20 → 31.75 ms (**−1.4 %**)
  - `/v1/classify`:                 76.15 → 75.05 ms (**−1.4 %**)
  - `/v1/classify_from_embeddings`: 32.40 → 32.05 ms (**−1.1 %**)

Modest, but free given the scaffolding. The win is purely launch-path overhead
(generic ggml-cuda cpy dispatch is heavier than a single-shape specialised
kernel), not launch count — counts are unchanged. ROI math: 27-28 launches per
encode × ~150-200 ns saved per launch ≈ 4-6 µs / encode total. The actual ms
delta is larger than that math predicts; either the generic-cpy dispatch is
heavier than estimated, or downstream ops benefit from slightly tighter CUDA
stream packing. Either way the receipts are real and reproducible.

Kill switch: `SIGLIP2_DISABLE_POST_FA_CONT=1`. Default ON in the install hook.

**Cumulative since pre-megakernel (A0+A1+A2+A3):** text 38.3 → 31.75 ms (**−17 %**),
embeddings 58.2 → 45.75 ms (**−21 %**), classify 94.1 → 75.05 ms (**−20 %**),
classify_from_embeddings 38.5 → 32.05 ms (**−17 %**).

**The natural extension** (not landed): swallow the o_proj's mul_mat read into
the post-FA cont kernel, eliminating the post-FA cont launch entirely. That
needs a custom Q8_0 × F32 MMQ for o_proj (M=64 text, M=729 vision) — Phase B
territory.

## ✅ Phase A4 landed (2026-05-09) — batched text encode + hooks-in-batched-mode

**The win:** `/v1/text_embeddings` and `/v1/classify_from_embeddings` were the
last endpoints behind python (1.18× and 1.49× per the pre-A4 baseline). Root
cause: cpp ran 5 sequential `encode_text` graph computes for 5 prompts, while
python batched all 5 into a single forward pass. Even after A3 the 5× sequential
graph build + launch chain dwarfed python's batched cuDNN path.

**What landed:**

1. **`TextEncoder::encode_batch`** (`src/siglip2_text.cpp`). Single graph
   compute over (n_pos, n_batch). Activations 3D `(H, n_pos, n_batch)`; Q/K/V
   views become 4D with the prompt-batch on `ne[3]`. ggml's
   `ggml_flash_attn_ext` natively supports 4D batches (per `ggml.h:2452`:
   `q: [n_embd_k, n_batch, n_head, ne3]`). For `n_batch=1` the call dispatches
   to the existing `encode()` so single-prompt CLI / parity scripts are
   untouched.

2. **Server wires through `encode_text_batch`** for all three text endpoints
   (`/v1/text_embeddings`, `/v1/classify`, `/v1/classify_from_embeddings`).
   Tokenizes all prompts in one batch, runs one graph compute, fans the
   batch-major output back into per-prompt response rows.

3. **A0 + A2 hooks extended for 3D-contig** via a new
   `is_2d_or_3d_f32_contiguous` predicate. Both fusions are pointwise across
   positions, so flattening `flat_n_pos = ne[1]*ne[2]` is sufficient — no
   kernel changes, just dispatcher param. The A0 anchor's shape gates were
   tightened to also compare `ne[2]` across `x/norm/mul/add` to keep the
   chain self-consistent. A2's `match_inner` got the same `ne[2]` consistency
   gate, and the residual/outer checks compare flat positions.

4. **A1 QKV-prep kernels learned the batch dim.** Both
   `qkv_copy_to_scratch_kernel` and `fused_qkv_prep_kernel` now take
   `blockIdx.z = batch index`, with per-batch source/dest pointer offsets.
   Plan-builder: `qkv_add` accepts `ne[2] > 1` (now strictly contiguous-3D);
   `q_pad / k_cast / v_cast` must have `ne[3] == n_batch`. Scratch buffer
   sized to `3*H*n_pos*n_batch*4` — for the 5-prompt text path that's 4.4 MiB,
   well below vision's 10 MiB ceiling, so no allocator regression.

5. **A3 post-FA cont kernel** got the same batch-dim treatment. Plan-builder
   gates ne[3] consistency between view/fa_out/cont and records `n_batch`.

**Self-parity (A4 batched ON, n=5 prompts vs n=1 sequential, same C++ binary):**
five test prompts cosine 0.99994–0.99996 across all of them. Drift is reduction-tree
shuffle from the 5× wider matmul; well above the 0.999 floor.

**HF parity (CLI scripts, single-prompt path) unchanged from A3:**
- text 0.999779
- vision 0.999907
- image 0.999545
- score 0.999987 / probs MAE 4.97e-7

CLI runs through the unmodified `encode()`, so this confirms A4 didn't disturb
the single-prompt path. **Server batched parity vs python**: 0.99927–0.99994
across the 5 bench prompts, well above the 0.999 floor.

**Bench A/B (clean GPU, only `kobbler-tts-qwen3-1` + `tts-qwen3-iter` co-resident,
30 runs each, 2 reps; ratios are cpp/python p50):**

|                                  | A3 head (sequential) | Phase A4 batched | python  |
|----------------------------------|---------------------:|-----------------:|--------:|
| `/v1/embeddings`                 | 45.5  (0.84×)        | 45.6   (**0.84×**) | 54.3  |
| `/v1/text_embeddings`            | 31.5  (1.18×)        | 21.7   (**0.81×**) | 26.9  |
| `/v1/classify`                   | 74.7  (1.07×)        | 65.2   (**0.93×**) | 70.4  |
| `/v1/classify_from_embeddings`   | 31.7  (1.49×)        | 21.9   (**1.02×**) | 21.6  |

Three of four endpoints now beat python clean. Cfe is measurement-noise tied
(within 0.3-0.5 ms; python wins one rep, cpp the other).

**Verbose plan-build counts confirm full hook coverage in batched mode:**
```
LN groups=55 (followers=110)
QKV groups=27 (copy_anchors=27 split_anchors=27 followers=189)
tail bias-residual=54 bias-gelu=27 (followers=81)
post-FA cont=27
total_nodes=958
```
Same as single-prompt, just one graph for all 5 prompts instead of five.

**Kill switches:** existing per-fusion kill switches still work
(`SIGLIP2_DISABLE_QKV_PREP`, `SIGLIP2_DISABLE_TAIL_FUSION`,
`SIGLIP2_DISABLE_POST_FA_CONT`, `SIGLIP2_DISABLE_MEGAKERNEL`). Plus per-call
timing under `SIGLIP2_TIME_ENCODE=1` and `SIGLIP2_TIME_HANDLER=1`.

## Phase A4 — investigation: where is the remaining 22 ms?

Per-call breakdown for steady-state cfe (`SIGLIP2_TIME_ENCODE=1`,
`SIGLIP2_TIME_HANDLER=1`):

```
[cfe] parse=0.10 imgs=0.00 encode=22.0 score=0.01 respond=0.01 total=22.1 ms
  └─ encode_batch: init=0.65 build=0.15 alloc=0.15 inputs=0.00 compute=21.6 get=0.04 reset=0.00
```

**99 % of cfe is `sched_graph_compute`.** All "overhead" — JSON parse, image-emb
parse, tokenize, ggml setup, tensor sets, response — totals <1 ms.

Cross-check: vision encode at n_pos=729 takes 49 ms; text batched at effective
n_pos=320 takes 22 ms. Ratio 2.23× ≈ 729/320 = 2.28×. **Compute scales linearly
with n_pos — we're at the GPU compute floor for ggml's matmul implementation
on Q8_0 × F32 at these shapes.**

The megakernel work eliminated CUDA driver/launch overhead. Python uses
cuDNN/cuBLAS GEMMs that are hand-tuned for these shapes; ggml dispatches
Q8_0×F32 via dequant+cuBLAS for M=320. **Both backends are now hitting
roughly the same TensorCores and bandwidth ceiling.** That's why we tied,
not embarrassed, on cfe.

## ❌ Phase A4 dead end — n_pos trimming

**Tested and proven dead.** SigLIP2 is hard-coded to expect padding to
`max_position_embeddings=64`. Tokenized 5 typical prompts (≤6 tokens each)
into `padding="longest"` (n_pos=7) and various smaller `max_length`s, then ran
HF reference and compared to `max_length=64` on a real image:

```
config            top-1 prediction              correct-prompt prob
max_length=64     "an abstract gradient"        0.401  ← REFERENCE
max_length=32     "a photo of a cat"            0.000000
max_length=16     "a colorful landscape"        0.000000
max_length=12     "white noise on a TV"         0.000004
max_length=8      "a TV showing static"         0.000011
longest(=7)       "a TV showing static"         0.000006
```

**Confidence in the correct answer collapses by ≥5 orders of magnitude.** Not
just rank scramble — the model's signal vanishes. SigLIP2 is trained with the
pooler at position 63 (a pad-token's hidden state after 64-position self-attention),
and that mechanism is load-bearing. Trimming pools at the EOS token after a
shorter attention context, which is fundamentally a different computation.

**Don't try this lever.** The optimization-vs-correctness boundary is at the
graph layer (megakernel + CUDA graphs) and the kernel layer (custom MMQ),
not at the input length.

## Where the next big win lives

The remaining theoretical ceiling on cpp text encode at n_pos=64×n_batch=5:

| Lever                                  | Est. win on cfe | Status |
|----------------------------------------|----------------:|--------|
| Phase A4 batched + hooks (this lap)    | 31.7 → 21.9 ms (-9.8) | landed |
| **CUDA graph capture** (Phase B)       | ~3-5 ms more    | gallocr path 3 (handoff parent doc) |
| Custom MMQ for M=320 shapes            | ~5-7 ms more    | multi-week, parity risk per rewrite |
| Q4_K_M weights + rank-parity bench     | bandwidth half + VRAM 1504→~750 MiB | needs rank-parity test infra first |

**The honest bottom line:** A4 is the realistic ceiling at the graph-fusion
layer. Going below ~17 ms cfe needs a different layer of attack (either kernel-
level MMQ work or CUDA graph capture, both multi-day projects).

## ✅ Phase B landed (2026-05-08) — graph cache + CUDA graphs

**Headline: every endpoint now beats python clean.** Internal text n=5 compute
collapsed **30 → 19.1 ms** (-36%) once the warmup-graph path could fire, and
p95 sits within 0.5 ms of p50 across all four endpoints — CUDA graphs killed
the jitter. cfe crossed below 1× python for the first time.

### Bench A/B (clean GPU, kobbler-vision-1 active, 50 runs each, p50 ms)

| Endpoint                              | A4 cpp | A4 ratio | **Phase B cpp** | **Phase B ratio** |
|---------------------------------------|-------:|---------:|----------------:|------------------:|
| `/v1/embeddings`                      | 45.6   | 0.84×    | **43.3**        | **0.81×**         |
| `/v1/text_embeddings` (n=5)           | 21.7   | 0.81×    | **20.1**        | **0.76×**         |
| `/v1/classify`                        | 65.2   | 0.93×    | **62.6**        | **0.91×**         |
| `/v1/classify_from_embeddings`        | 21.9   | 1.02×    | **19.9**        | **0.98×**         |

Internal `encode_batch n=5` compute (`SIGLIP2_TIME_ENCODE=1`, post-warmup):
**19.14 ms ±0.05** rock-steady across 30+ samples. p95 within 0.5 ms of p50
on every endpoint — CUDA-graph replay is deterministic.

### Parity unchanged from A4

- text 0.999756 / vision 0.999944 / image 0.999523 / score 0.999986 / probs MAE 5.6e-7
- 200/200 alternating-cache stress test passes (single n=1 ⟷ batched n=5
  with both shapes hot in the LRU).

### What landed in code

1. **`src/siglip2_text.cpp`** — `GraphCacheEntry` (n_pos, n_batch, has_mask
   key) holding `(arena, ctx, gf, gallocr, named tensors)`. Both `encode()`
   and `encode_batch()` look up / build / reuse. Cap = 8 entries (LRU evicts
   front). On miss: `ggml_gallocr_new` + `reserve` + `alloc_graph`. On hit:
   skip rebuild, no alloc_graph, just `tensor_set` inputs → `ggml_backend_graph_compute`
   → `tensor_get`. **No scheduler in the path** (see "Path 2 unlock" below).
2. **`src/siglip2_vision.cpp`** — `VisionGraphCacheEntry` (n_patches_h,
   n_patches_w, pooling key). LRU cap = 4 (NaFlex hits a small finite set
   per max_num_patches). Same gallocr/direct-compute pattern.
3. **`CMakeLists.txt`** — `set(GGML_CUDA_GRAPHS ON CACHE BOOL ...)` before
   `add_subdirectory(${GGML_DIR})`. Default-on; override on cmake line if
   needed. Drives `GGML_CUDA_USE_GRAPHS` → `USE_CUDA_GRAPH` define in
   `ggml-cuda.cu`. Without stable tensor pointers across calls (provided
   by the cache) this flag was a no-op-with-overhead in A4.

### Path 3 was a dead end (Path 2 — bypass scheduler — won)

**Path 3 (the recommended starting experiment from the prior handoff)
crashed on call 2 with the same gallocr-aliasing CUDA error as Variants
1 and 2.** Caching `(ctx, gf)` and re-running `ggml_graph_clear` +
`ggml_build_forward_expand` doesn't help because **`ggml_backend_sched_split_graph`
creates fresh split-copy tensors per call regardless of how we manage our
own ctx.** The gallocr's `node_allocs[]` is keyed by position in
`sched->graph`, and on call 2 position N points to a fresh split-copy
tensor whose `data == NULL`, so the gallocr re-allocates at the saved
offset and overlaps a still-live cached tensor.

**Pivot: skip the scheduler entirely (Path 2 from the parent doc).**
`GGML_SCHED_DEBUG=1` confirms the text/vision encoders run as ONE split on
ONE backend (CUDA) — sched is pure overhead for us. Per-entry
`ggml_gallocr_t` + `ggml_backend_graph_compute(backend, gf)` directly:

- gallocr's offset table is per-entry, derived once at `reserve()` time,
  reused on each `alloc_graph()` (we only call `alloc_graph` once at miss
  time; the binding sticks across `graph_compute`s for our private
  buffer).
- Tensor pointers stay stable across calls (we own the ctx; nothing
  recreates them) → `ggml_cuda_graph_update_required`'s `memcmp` matches
  → warmup completes after 2 calls → CUDA graph replay takes over.

### The trap: std::vector<CacheEntry> emplace_back UAF

First Path-2 cut crashed silently (SEGV in `set_tensor`, no CUDA error)
on the third call after the cache had two entries. **Cause: vector
growth on `emplace_back` triggers a move of existing entries to the new
underlying array. `GraphCacheEntry`'s default move = memberwise copy of
raw pointers WITHOUT nulling the source. The destructor on the moved-from
slot then frees the `ggml_gallocr_t` + `ggml_context *` that the live
entry in the new array still owns.**

Fix: explicit move ctor / assign that transfers and nulls source pointers.
Same pattern used in `VisionGraphCacheEntry`. **Any RAII-pointer member
in a `std::vector<T>` element type needs this** — common pitfall, easy
to miss because `=default` looks correct.

### Files touched (commits ahead of A4 head)

- `src/siglip2_text.cpp` (+158, -67) — GraphCacheEntry + encode/encode_batch refactor
- `src/siglip2_vision.cpp` (+128, -54) — VisionGraphCacheEntry + encode refactor
- `CMakeLists.txt` (+8, -0) — flip GGML_CUDA_GRAPHS default
- `HANDOFF-megakernel-v0.md` (this file)

VRAM (full-tower post-warmup): unchanged at ~1.5 GiB. Each per-shape
activation buffer is ~30-50 MiB; with two text shapes + 1 vision shape
typically resident the activation budget is ~150 MiB total — well under
the previous A4 sched arena.

## Phase C — handoff for the kernel/quant lap

**Mood from the user 2026-05-08:** "we've hit our target, lets blow it
away." Phase B beat python on every endpoint, but cfe is at 0.98× — tied,
not embarrassed. Going below 0.9× ratio needs a different layer of attack;
graph-layer is exhausted.

### Why graph-layer is done

The 19.1 ms cfe compute is GPU-compute-bound on Q8_0×F32 matmuls at M=320
(text n_pos=64 × n_batch=5). Both backends hit the same RTX 3060
TensorCores (or scalar dp4a path) at roughly the same throughput.
Megakernel collapsed launch overhead from ~745 → ~30 launches; CUDA
graphs collapsed ~30 → 1 submit. **There is no launch-overhead left
to recover.**

Quick math: 27 layers × 4 matmuls/layer = 108 Q8_0×F32 matmuls per
encode, plus FA. At M=320 each Q8_0×F32 is M×K×N flops with M=320,
K=1152, N∈{1152, 3456, 4304}. Total ≈ 27 GFLOPS per encode. RTX 3060 at
~10 TFLOPS Q8 peak ≈ 2.7 ms compute floor. We're at 19 ms — 7× the
peak. The gap is in matmul efficiency at small M (cuBLAS GEMM at M=320
gets ~30% peak), not launch overhead.

### Levers ranked by ROI

| Lever                                  | Est. cfe gain | Difficulty | Parity risk |
|----------------------------------------|--------------:|-----------:|------------:|
| **Custom MMQ for siglip2 shapes**      | -5 to -7 ms   | Multi-week | High (FP rounding per kernel) |
| **Q4_K_M weights + rank-parity bench** | -3 to -5 ms   | 1-2 weeks  | Medium (need rank-parity infra first) |
| Pre-pad weights at conversion time     | <0.5 ms       | 1 day      | None |
| Probe-head fusion (vision only)        | ~0.5 ms       | 2 days     | None |
| FA d_head=72 fast path on GA106        | -1 to -2 ms   | 1 week     | High |

The first two are where the next agent should plant their flag. The rest
are "while you're here" cleanup that don't move the needle on the headline
ratios.

### Lever 1: custom MMQ for M=320, K=1152, N∈{1152, 3456, 4304}

ggml's MMQ dispatcher specializes on small M (vector path, `mmvq.cu`)
and falls back to dequantize+cuBLAS for M > some threshold. siglip2
text at M=320 (or vision at M=729) lands in cuBLAS-GEMM territory, which
isn't tuned for these specific shapes.

The qwen3-tts megakernel's `mmvq_q8_0_q8_1_kernel` is the reference for
small-M MMQ on Ampere. **Don't lift that kernel directly** — it's M=1
specialized. siglip2 needs M=64, M=320, M=729. The right move is a fresh
kernel templated on `(M_tile, K_tile, N_tile)`, picked per shape, with
mma.m16n8k32 tensor cores.

**Critical:** the auto-memory entry `project_qwen3tts_int8_mma_kill.md`
says mma m16n8k32 is *0.31-0.70× dp4a* on hot wide-N shapes on RTX 3060
(consumer GA106 is 1:1 mma:dp4a peak, not 16:1 like A100). Verify mma
beats dp4a *for this specific (M, K, N)* before committing — bench the
shape at M=320 with both paths. dp4a may still be the winner here, in
which case the lever is "specialize the dp4a path for these shapes,"
not "use mma."

Parity discipline: every kernel rewrite is a fresh source of FP-reduction
order. Use the **5-prompt batched vs 1-prompt sequential cosine ≥ 0.99994**
self-parity check as your fast smoke (it ran 0.99994-0.99996 in A4 and
hasn't drifted). HF parity (text/vision/image/score) must stay ≥ 0.999.

### Lever 2: Q4_K_M weights + rank-parity bench

Weight bandwidth halves (Q8_0 → Q4_K_M = 8 → 4.5 bits/weight). On a
bandwidth-bound shape (which SigLIP2 text at M=320 is), that's a direct
proportional speedup on matmul time. Estimate: ~3-5 ms off cfe.

VRAM also halves: 1504 MiB → ~750 MiB post-load. Big win for the multi-
service deploy where qwen3-tts is co-resident.

**Blocker for shipping:** rank-parity test infrastructure. The user's
position is documented in `feedback_no_hf_publish.md` and adjacent: the
0.999 cosine floor is the bar, AND we need to verify that classification
RANKS don't shuffle on real workloads (top-1 image-prompt match must
stay correct). The recipe is in
`/home/dbrain/.claude/projects/-home-dbrain-dev-kobbler/memory/project_siglip2_n_pos_trim_dead.md`
— that file documents the n_pos-trim dead end but the rank-test framework
is the same one Q4_K_M needs. Rebuild that test bed first, THEN convert.

### Lever 3 (small): pre-pad weights at conversion

`scripts/convert_siglip2_to_gguf.py` writes qkv_w / o_w at native d_head=72.
The runtime pads to d_head=80 for tensor-core MMA-FA via `ggml_pad`. Pad
at conversion → eliminate 27 × 3 = 81 launches per encode. Already
collapsed by megakernel A1, but the actual padding compute happens
inside the megakernel kernel — pre-padding the weights would let A1's
copy-to-scratch read directly without runtime padding logic.

ROI is small (<0.5 ms) but the change is mechanical and reduces complexity.

### What's NOT a lever (don't relitigate)

- **n_pos trim** — `project_siglip2_n_pos_trim_dead.md`. Pad-to-64 is
  load-bearing for SigLIP2's pooler. Confidence collapses 5+ orders of
  magnitude with shorter padding.
- **CUDA graphs alone** — already on. Won't help further.
- **Sched re-introduction** — landed Path 2 to skip it. Don't reverse.
- **Lever J (ffn_down padding)** — parked per the standing rule (126 MiB
  doesn't justify sub-0.999 cosine on 12 GB GPU).

### What "done" looks like for Phase C

- cfe (`/v1/classify_from_embeddings`) **≤ 0.85× python p50** on a clean
  GPU with kobbler-vision-1 co-resident. (Equivalent to **≤ ~17 ms**
  absolute at current python ratios.) That's "real embarrassment."
- All four parity scripts PASS — vision/text/image/score cosines ≥ 0.999,
  probs MAE ≤ 5e-3.
- 5-prompt batched vs 1-prompt sequential self-parity ≥ 0.99994.
- VRAM regression is OK if the lever justifies it (Q4_K_M halves; MMQ
  may grow scratch by ~50 MiB) — full-tower post-warmup VRAM target
  is the metric. Today: ~1.5 GiB.
- Add a Phase C receipts section here mirroring Phase B's table.

### Files the next agent will likely touch

- `src/cuda/siglip2_megakernel.cu` — add a custom MMQ kernel under a new
  function (`launch_siglip2_mmq_q8_0_f32_kernel`?). Plumb through the
  op_hook for `MUL_MAT` ops on attn/o/up/down weights.
- `src/cuda/siglip2_megakernel.cuh` — kernel forward decls.
- `scripts/convert_siglip2_to_gguf.py` — Q4_K_M conversion option +
  optional pre-padded qkv_w/o_w.
- `scripts/parity_check_*.py` — rank-parity test bed (top-1 stability
  on a labeled image+prompt corpus). Probably new
  `scripts/rank_parity_check.py` script.
- `models/` — emit `siglip2-so400m-naflex-q4_k_m.gguf` alongside Q8_0.

## Phase A2 — gallocr trap take 2 (read before extending)

The first cut of A2 had no `is_blk_bias` gate — it matched every
`add(add(mm, 1D-bias), 2D-residual)` and `gelu(add(mm, 1D-bias))` chain. Text
came back **bit-clean** (cosine 0.999756 ON, identical to OFF). Vision came
back at cosine **0.694** — broken.

### Bisect

- `SIGLIP2_DISABLE_TAIL_BG=1` (P1 only)  → vision 0.635 (still broken)
- `SIGLIP2_DISABLE_TAIL_BR=1` (P2 only)  → vision 0.647 (still broken)
- `SIGLIP2_TAIL_NAME_SKIP=v.head` (skip probe head only) → vision 0.999944
  (PASS); skipping just `patch_embd` left it broken at 0.694
- After landing the consecutive-in-topo gate alone (which skips patch_embd
  + probe head's outer P1 since the chain has many ops between inner and
  outer), vision STILL failed at 0.647 — because the probe head's `gelu(up_b
  + up_mm)` IS a consecutive 2-op chain, so the safety gate alone passes it
  through, and that single match is enough to break vision parity.
- `SIGLIP2_TAIL_BLK_ONLY=1` (only `*.blk.*` biases) → vision 0.999944 PASS,
  image 0.999523 PASS, text 0.999756 PASS, score 0.999986 PASS — all clean.

### Why per-block is safe and probe-head/patch_embd are not

Same root family as the A1 single-kernel trap: late-anchor design depends on
gallocr letting `mm.dst`, `inner_add.dst`, and `outer.dst` share one physical
slot (their lifetimes are disjoint, shapes match). When the slot aliases,
follower-skipping inner_add leaves mm's value intact in the slot through to
outer.idx — our late anchor reads the right value.

Per-block chains satisfy this aggressively: 27 layers × identical (H, n_pos)
shapes × every block has the same fan-out structure. gallocr packs them
tightly with shared slots — works in practice for both text n_pos=64 and
vision n_pos=729, even at 12.5 MiB up_intermediate slots.

Probe-head and patch_embd chains do NOT satisfy this:

- **Probe head** has Q/K/V mm-bias-add nodes whose outers are reshapes (not
  ADD/GELU) — those don't match anyway. The matches that DO fire are
  `gelu(up_b + up_mm)` (consecutive 2-op chain, gate allows) and `out =
  ggml_add(attn, normed)` where `attn` is the inner of the o_b add but
  separated from `out` by the entire LN+up_mm+gelu+down_mm+down_b sub-chain
  (gate skips this one). The `gelu(up_b + ...)` match is the surviving
  failure mode: gallocr packs probe head's mixed-shape slots differently
  enough that mm.dst's slot doesn't alias with up_bias.dst → late anchor
  reads garbage. Confirmed by single-pattern bisect.
- **Patch embedding** has `add(mm, patch_embd_b)` followed by ~10 pos_embd
  compute ops then `add(x, pos_embd)`. The pos_embd ops materialize tensors
  that gallocr packs into mm.dst's freed slot. The consecutive-in-topo gate
  catches this case correctly — non-consecutive, skipped.

### What this rules in/out for future fusions

- **Generic op_hook fusions can't safely follower-skip an upstream op** in a
  one-off graph location — gallocr's slot packing is shape-and-graph-shape-
  dependent and drifts between regions. The A1 scratch-buffer pattern is the
  only path that's robust across all locations.
- **Long, repetitive sub-graphs** (encoder blocks) DO permit follower-skip
  because gallocr's packing is uniform across layers and we can verify
  alias survival empirically with a single self-parity pass.
- The `is_blk_bias` gate is brittle (couples to `convert_siglip2_to_gguf.py`
  naming), but empirically necessary. Future agents extending A2 to other
  patterns must either (a) prove the alias survives across all 4 parity
  scripts via self-parity, or (b) use the A1 2-kernel scratch pattern.

## Phase A1 — design history & gallocr aliasing trap (read before extending)

The first attempt at A1 was a single-kernel late-anchor design — anchored
on V_cast (last in topo of {Q_pad, K_cast, V_cast}), reading directly from
`qkv_add->data` and writing the three FA inputs in one launch. Plan-builder
found 27 chains correctly and the kernel produced bit-clean vision output
(cosine 1.0 vs A1-OFF), but **every text encode produced NaN**. Hours of
delight follow.

### What the late-anchor approach tried

One fused kernel anchored on `V_cast` (the highest-topo of {Q_pad, K_cast, V_cast} per block), reading from `qkv_add->data` and writing the three FA inputs in one launch. Plan-builder finds 27 chains for both text and vision (verified via `SIGLIP2_MEGAKERNEL_VERBOSE=1`). With A1 ON:
- vision self-parity vs A1-OFF: cosine **1.0000000** (looks bit-perfect)
- text self-parity vs A1-OFF: every output is **NaN** (silent NaN propagation through subsequent attention)

### Why it breaks

`gallocr`'s static analysis for `qkv_add`:
- last consumer = the three `cont` ops (which read it via the strided permuted views)
- so `qkv_add.data`'s lifetime = [qkv_add.idx, last_cont.idx]
- after `last_cont.idx`, the slot is freeable → reusable for any other tensor

Our late-anchor fires at `V_cast.idx > last_cont.idx`. By that point gallocr has already let the slot host a different tensor whose lifetime intersects this window. We read garbage. For vision (n_pos=729), the colliding tensor happens to land somewhere that doesn't matter for cosine; for text (n_pos=64) the smaller tensor sizes change gallocr's packing and the collision corrupts a live attention output → NaN.

### Why we can't just move the anchor earlier

Any anchor index `i` must satisfy:
- `i ≤ qkv_add.idx`  to read `qkv_add.data` (or `mm.data + bias.data`)
- `i ≥ max(Q_pad.idx, K_cast.idx, V_cast.idx)` to write to those dsts (their slots may be aliased with active tensors before their lifetime starts)

These constraints are **mutually exclusive** because `Q_pad.idx > qkv_add.idx` by graph topology. No single anchor works.

### The 2-kernel fix

Decouple read and write into two separately-anchored kernels:

1. **Anchor on `qkv_add`** (replaces the bias-add): launch a kernel that reads `mm.data + bias.data`, sums them, writes to a **persistent scratch buffer** allocated in `install()` once. Followers: nothing — qkv_add itself is the anchor.
2. **Anchor on `V_cast`** (last in topo): launch the existing fused split-permute-pad-cast kernel reading from scratch, writing to Q_pad / K_cast / V_cast. Followers: 3 conts + 3 pads + 2 casts (skip qkv_add since it's the other anchor).

Scratch-buffer sizing: `3 * H * max(text_n_pos, vision_n_pos) * sizeof(float)` = `3 * 1152 * 729 * 4` ≈ **10 MiB** for the hottest shape. Allocate once in `install()` (or lazy-on-first-fire). Reuse across all 27 blocks per encode — block N's scratch is consumed before block N+1's write because the CUDA stream is sequenced.

Total: 9 ops/block (1 add + 3 cont + 3 pad + 2 cast) → 2 kernels. **Saves 7 launches per block × 27 = 189/encode** — same as the abandoned single-kernel design. Roughly 1.5-2 ms per text encode at ~10 µs/launch.

### Where to anchor in code

Plan-builder needs two entries per block — one for each anchor. The followers list shrinks accordingly. `g_qkv_anchors` already maps `dst → QkvPrepEntry` so a second map (or a discriminated `enum AnchorRole { COPY_TO_SCRATCH, SPLIT_FROM_SCRATCH }`) handles both. The scratch pointer is per-block (since blocks are sequential, one block's scratch is fine — but if you ever go pipelined / multi-stream, you'd need n_blocks scratches).

### Don't fall into these adjacent traps

- **Don't try to anchor on the FIRST cont and read qkv_add inline.** Same lifetime problem: the cont's compute (returning true via op_hook) is at last_cont.idx for the LAST cont but earlier for the others; gallocr still expires qkv_add.data at last_cont.idx and any earlier read sees the data alive only until its OWN cont fires.
- **Don't try to redirect FA's input pointers via `tensor->data = scratch_ptr`** post-allocation. ggml's gallocr keeps internal bookkeeping; mutating `data` after `sched_alloc_graph` corrupts the allocator's state on the next graph compute.
- **Don't insert a no-op "consumer" node into the graph to extend qkv_add's lifetime.** That changes the graph structure (build_block in two TUs); doable but more invasive than scratch + 2 kernels, and you'd still need to walk the graph twice anyway.
- The auto-memory entry on the qwen3-tts.cpp A2 sub-megakernel (`project_qwen3tts_a2_autofusion_block.md`) describes a related but distinct trap: ggml's `try_fuse({RMS_NORM, MUL})` eats the anchor before op_hook fires. SigLIP2 doesn't use RMSNorm so that specific pre-fusion doesn't bite us, but the take-away is the same — **ggml-cuda's static behaviour limits where you can hook from, design around it.**

### Original Phase A1 framing (kept for context)


## Where things stand

- Branch: `dbrain/siglip2-v0` @ A2 head + ggml submodule fast-
  forwarded to `a3fc6843` (the qwen3-tts megakernel hook commits:
  graph-compute-begin hook, cgraph passed to it, generic per-op hook).
  All three needed for the dispatcher pattern; no more ggml plumbing
  required to start phase A.
- Builds: `build-cuda/` clean with `-DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES=86`.
- Tower state: full both towers (vision + text), 1504 MiB post-load,
  1566 MiB post-warmup. **This is the deploy mode** — do not optimize
  for `--vision-only` or `--text-only` at the cost of full-tower wins.
- Parity (post-QKV-fusion):
  - vision (synthetic): 0.999907
  - text:               0.999779
  - score logits:       0.999987 / probs MAE 5.0e-7
  - image (1920×1080→729): **0.999545** ← strictest, watch this one
- Bench (RTX 3060, GPU shared):
  - /v1/embeddings:               58 ms  (cpp 1.07× python)
  - /v1/text_embeddings:          38 ms  (cpp 1.43× python — **target**)
  - /v1/classify:                 94 ms  (cpp 1.34× python — text-bound)
  - /v1/classify_from_embeddings: 38 ms  (cpp 1.79× python — **target**)

After QKV mul_mat + bias_add (2 launches), `build_block` fans out 11
ops just to prep Q/K/V for `flash_attn_ext`:

```
qkv = mul_mat(qkv_w, cur) + qkv_b               (already done)
Q = view_3d(qkv, off=0)        K = view(off=1H) V = view(off=2H)   (free)
Q = permute(0,2,1,3)           K = permute(...)  V = permute(...)  (free metadata)
Q = pad(cont(Q), pad=8)        K = pad(cont(K))  V = pad(cont(V))  (6 launches)
                               K_f16 = cast(K)   V_f16 = cast(V)   (2 launches)
KQV = flash_attn_ext(Q, K_f16, V_f16, mask, kq_scale)              (1 launch)
KQV = view_3d(slice off pad)
KQV = cont                                                          (1 launch — for o_proj input)
```

That's 9 launches per block × 27 blocks = **243 launches** that one
custom CUDA kernel can replace with **3** (one writer per Q/K/V), or
plausibly **1** if you write all three outputs from one grid.

**The fused kernel:** read the strided F32 qkv buffer, write three
contiguous outputs:
- Q F32 of shape `(d_pad=80, n_pos, n_head)`, with the padded slot
  (d=72..79) zero-filled.
- K F16 of the same shape, zero-padded, fp32→fp16 cast inline.
- V F16 likewise.

Each output goes directly into the existing dst tensor that ggml
allocated for the post-cast (K) / post-pad (V) / post-pad (Q) node.
The 9 follower ops short-circuit via the per-op hook. Same pattern
as the LN fusion landed in A0.

**ROI:** ~3.5 ms per text encode, ~3.5 ms per vision encode (assuming
~10-12 µs per skipped launch). Combined with A0, this should put text
at ~33 ms (vs 26 ms python = 1.27×) and embeddings at ~52 ms (already
sub-Python on a clean GPU; ratio stays favourable). /v1/classify drops
the most because it walks both towers + 5 text iterations, all
benefitting.

**Plan-builder shape:** anchor on each block's QKV `add` (the bias
add), walk forward in topo order recording `(view, view, view,
permute*3, cont*3, pad*3, cast*2)` until you hit the FA node. If the
shape matches, fill `QkvPrepEntry { qkv_dst, q_pad_dst, k_cast_dst,
v_cast_dst, n_pos, d_head, n_head, d_pad, kv_scale_offset }`. Anchor
the fused kernel on the LAST cast (V_cast) — by then ggml has
allocated the dsts for Q_pad, K_cast, V_cast, and the qkv data is
ready. Followers: every other intermediate node in the chain.

**Risk:** strided source memory layout (qkv is non-contig along
positions), F16 quant must round-to-nearest matching ggml's
`ggml_compute_forward_cpy_f16` semantics, the d_head=72→80 zero-pad
needs to be exactly zero-bit (not denormal). All achievable; just
careful kernel work.

**Don't fuse the post-FA `cont` into this** — that one runs on FA's
output and a different shape. Phase A2.

## What "phase A" originally looked like (kept for context)

**Goal:** one fused CUDA kernel per text encoder block at fixed shape
n_pos=64, H=1152, d_head=72, n_head=16. Inputs: input activations
(H, n_pos), 8 weight tensors (qkv, o, ln1, ln2, up, down — quantized
or not depending on which); outputs: (H, n_pos) for the next block.
Internal kernel does:

```
LN1 → fused QKV mul_mat → reshape to (d_head=80 padded, n_head, n_pos)
    → strided FA on padded d ← already in-place
    → slice back to d_head=72
    → O mul_mat → +residual
    → LN2 → up mul_mat → GELU → down mul_mat → +residual
```

Currently each block fires ~18 launches; megakernel = 1.

**Reference template:** `qwen3-tts.cpp/src/cuda/qwen3_megakernel.cu`
(1533 lines). Key sub-patterns to lift:

1. **F32 → Q8_1 quantize-x kernel** (`quantize_x_q8_1_kernel`,
   templated on K). Used for the activation side of mul_mats. siglip2
   needs the same — input activations are F32, weights are Q8_0,
   so we Q8_1 the input once per block.
2. **Q8_0 × Q8_1 specialized MMVQ** (`mmvq_q8_0_q8_1_kernel`). qwen3
   specializes on (K, N) pairs for M=1. **siglip2 is M=64 (text) or
   M=729 (vision), not M=1, so MMVQ isn't directly applicable.** You
   want MMQ-style matmul (multi-row) at fixed (M, K, N). Either lift
   ggml's existing MMQ kernel and specialize, or write a fresh one
   tuned for siglip2's shapes (probably what you'll do — siglip2's M
   range is small enough that template instantiation is cheap and
   per-shape tuning beats the generic dispatcher).
3. **Graph-begin hook** (`qwen3_graph_begin_hook` in qwen3-tts) — the
   pattern for scanning `cgraph->nodes` at graph-compute-begin and
   building a per-graph fusion plan. ggml hooks are already in place
   (`ggml_cuda_set_graph_begin_hook`, `ggml_cuda_set_op_hook`). For
   siglip2 you scan for the 27-block transformer pattern (look for
   the LN1 → mul_mat(qkv) → ... → ggml_add(residual) chain), record
   per-block tensor pointers, install the per-op hook that anchors
   on the LN1 node and runs the megakernel.
4. **Per-op hook** (`g_ggml_cuda_op_hook`) returns true when it
   handled the op (so ggml-cuda's normal dispatch is skipped). Anchor
   on the first node of the fused pattern; subsequent nodes in the
   chain look themselves up in the fusion plan and return true (no-op).

**Where to put the new code:**

```
siglip2.cpp/
  src/cuda/                         (new dir)
    siglip2_megakernel.cu           (block-fused kernel + dispatcher)
    siglip2_megakernel.cuh          (templated kernel definitions)
    siglip2_megakernel.h            (C API: hook installer, init/deinit)
  CMakeLists.txt                    (add cuda lib + link to siglip2-server)
```

Wire installer in `siglip2_server.cpp::main()` once at startup, before
the encoders' `load()` calls, so the hooks are registered when the
first graph compute fires. The hook checks `cuda_ctx->device == 0`
and the per-graph plan for siglip2 patterns; if no match, returns
false and ggml's normal dispatch handles it (this is your safety
fallback while developing).

## Phase A scoping — what's in / out

**In:**
- Text encoder block megakernel at fixed n_pos=64, H=1152, d_head=72,
  n_head=16, intermediate=4304. Single shape — no template
  instantiation across n_pos values.
- F32 activations, Q8_0 weights (current GGUF format).
- Drop-in compat: keep `SIGLIP2_DISABLE_FA=1` working, keep the
  non-megakernel path intact. Add `SIGLIP2_DISABLE_MEGAKERNEL=1` env
  knob (default off) so the user can A/B with two seeds during ear
  testing — although for an encoder there's no audio, "ear testing"
  here means re-running parity scripts with both modes and verifying
  cosine + score MAE stay above 0.999 / under 5e-3.
- Padded MMA-FA (d_head=72→80) inside the megakernel. The pad is
  trivial when you control the kernel; just allocate the padded slice
  inline.
- Bias-add fused into the mul_mat output (free with shared-memory
  accumulation).

**Out (defer to phase B/C):**
- Vision encoder megakernel (different shape, mostly compute-bound,
  smaller ROI). After phase A measures clean, decide whether vision
  warrants a separate per-shape kernel or shares the text one with a
  shape param.
- Multi-shape vision (variable n_pos via NaFlex). The user's prod
  setting is `max_num_patches=729`, which hits one or two specific
  shapes after the binary search (`(27, 27)` for square, plus the
  AR-clamped variants for tall/wide images). Phase B can specialize
  on those if it's worth it.
- `--text-only` / `--vision-only` selective fusion. Both towers
  benefit from the same megakernel (probe head is a separate pass);
  no need for mode-specific paths.

## Bench / parity discipline

Before any megakernel work touches CUDA code: capture the **exact**
baseline numbers for the head you're at, with the exact bench script.

```bash
# Baseline bench script lives at /tmp/bench_cpp.py (used during
# 2026-05-09 pass) — 30 runs, p50/p95/mean per endpoint. Re-create it
# from siglip2.cpp/scripts/bench_vs_python.py if /tmp lost it.

# 1. Boot the server in a clean container
docker rm -f siglip2-bench-cuda 2>/dev/null
docker run -d --rm --name siglip2-bench-cuda --gpus all -p 18890:18890 \
  -v $PWD:/work:ro -w /work tts-qwen3-dev:builder \
  bash -lc '/work/build-cuda/siglip2-server \
    --model /work/models/siglip2-so400m-naflex-q8_0.gguf \
    --tokenizer /work/reference/hf/siglip2-so400m-patch16-naflex/tokenizer.model \
    --port 18890 --host 0.0.0.0'

# 2. Sleep 4 (warmup + load)
# 3. Run the parity scripts inside kobbler-vision (need libgomp1):
docker run --rm -v $PWD:/work -w /work kobbler-vision bash -c '
  apt-get update -qq >/dev/null && apt-get install -y -qq libgomp1 >/dev/null
  HF=reference/hf/siglip2-so400m-patch16-naflex
  GGUF=models/siglip2-so400m-naflex-q8_0.gguf
  python3 scripts/parity_check_vision.py --hf-model $HF --gguf $GGUF --cli build/siglip2-cli
  python3 scripts/parity_check_text.py   --hf-model $HF --gguf $GGUF --cli build/siglip2-text-cli
  python3 scripts/parity_check_image.py  --hf-model $HF --gguf $GGUF --cli build/siglip2-cli
  python3 scripts/parity_check_score.py  --hf-model $HF --gguf $GGUF \
       --cli build/siglip2-cli --text-cli build/siglip2-text-cli'

# 4. Bench:
python3 /tmp/bench_cpp.py http://localhost:18890   # cpp
python3 /tmp/bench_cpp.py http://localhost:8890    # python (kobbler-vision-1 already running)
```

**Per-iteration discipline (cribbing from qwen3-tts handoff):**
- Don't ship from a 3-prompt smoke test. Bench is 30 runs, both
  servers warmed, GPU shared.
- Quote ratios (cpp/python) not absolute ms — GPU contention with
  `kobbler-vision-1` / `kobbler-tts-qwen3-1` / `tts-qwen3-iter` makes
  absolutes wobble.
- Watch parity cosines after every kernel rewrite. Image-parity
  (heavy-downsample) is the strictest of the four.
- Watch `nvidia-smi --query-compute-apps=...,used_memory` post-load
  — VRAM regression is a non-starter.

## Wild ideas (start here if Phase A goes smoothly)

- **CUDA graphs on top of the megakernel**: 27 megakernel launches
  per text encode × 12-15 µs = 0.3-0.4 ms launch overhead. Graph
  capture replays the whole 27-launch sequence as one submit →
  ~12 µs total. Combined with phase A this puts text encode below
  1 ms per prompt. ggml's CUDA graph capture path needs stable
  tensor pointers across calls (currently broken; see HANDOFF.md
  "Dead ends" → graph caching path 3 is the unlock). With megakernel
  in place, the launch sequence is *deterministic per shape* by
  construction, so the warmup path should fire after 2 calls.
- **Pre-pad weights at conversion time**: instead of running
  `ggml_pad` to round d_head 72→80 at runtime (3 launches per layer
  in the non-fused path), pad qkv_w / o_w in the GGUF so the
  weight already has the padded shape. Removes the runtime pad ops
  entirely. Marginal once megakernel lands, but trivial to implement
  and reduces complexity.
- **Probe head fusion**: the vision probe head fires once per image
  (separate from the 27 encoder blocks). It's 6 mul_mats + an FA. Not
  on the 486-launch hot path so low-priority, but a "while you're
  here" cleanup.
- **Skip-connection accumulator**: in the megakernel, fold the
  residual into the second mul_mat's accumulator instead of doing a
  separate ggml_add. Saves a launch per block. Free when you control
  the kernel.

## What you won't fix this lap (don't get distracted)

- **Vision encoder is mostly compute-bound.** Don't burn cycles on a
  vision-specific megakernel until phase A text is solid; the ROI is
  ~1-2 ms on a 58 ms encode (~3%) vs phase A's 5-6 ms on a 7.6 ms
  encode (~70%).
- **Q8_K_M quants**: prior agents flagged this as "next big lever";
  irrelevant here, parity floor is the gate, not bandwidth.
- **Graph caching variant 1/2**: both blocked at the gallocr layer
  (HANDOFF.md "Dead ends"). Megakernel + CUDA graphs is the path
  *around* that, not a fix for it.
- **Lever J (ffn_down padding)**: parked. 126 MiB doesn't justify
  the cosine cost on a 12 GB GPU with comfortable headroom.

## Phase B / C — what comes next

After phase A lands and is benched (target: text 1.43× → ≤0.9× python):

- **Phase B**: vision encoder megakernel at n_pos=729. Same kernel
  structure, bigger M. Reuse the per-block pattern but specialize
  the K/N tile params for the larger-batch matmuls. Probably ROI is
  ~3-5% on /v1/embeddings.
- **Phase C**: CUDA graphs on top of phase A + B. Need graph caching
  path 3 (cache build, refresh gf each call) to stabilize tensor
  pointers across calls. ROI: ~0.3-0.5 ms / encode (eliminates the
  ~30 megakernel launch overhead).
- **Phase D**: probe-head + final-LN + projection-head fusion. The
  rest of the encoder graph that isn't in the 27-block loop.
  Marginal.

## Filesystem snapshot at handoff

```
siglip2.cpp/
├── HANDOFF.md                       ← parent doc; read for full state
├── HANDOFF-megakernel-v0.md         ← this file
├── src/                             ← unchanged from QKV-fusion pass
├── scripts/                         ← unchanged
├── ggml/                            ← submodule @ a3fc6843 (megakernel
│                                       hooks ready: graph-begin + per-op +
│                                       cgraph-passed-to-begin)
└── build-cuda/                      ← rebuild after editing
```

Now go.
