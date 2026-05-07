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

The win you're chasing: **text path is launch-overhead-bound**, ~80%
of the 38 ms / 5-prompt p50 is CUDA kernel launch overhead (~745
launches per encode × ~10-12 µs each — a hair lower than the original
12-15 µs estimate; A0 measurements re-grounded the per-launch cost).
PyTorch beats us at ~26 ms because it issues maybe 100-200 launches
via cuDNN/cuBLAS fusion.

Megakernel collapses that ~745 → ~30, putting text encode at
**~2-3 ms per prompt** (well below Python). /v1/text_embeddings and
/v1/classify_from_embeddings drop from 1.43× and 1.79× python to
sub-1×. /v1/classify (vision + text loop) closes too because the
text-loop cost dominated.

Vision (n_pos=729) is **already 1.07× python** because it's
compute-bound at that batch size — megakernel saves ~1-2 ms there
(launches are <10% of the encode). Phase A targets text. Phase B
targets vision but the ROI is much smaller; consider that phase
optional.

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

## Phase A1 — QKV-prep chain fusion *(SCAFFOLDED, PARKED — gallocr aliasing trap)*

> **Status:** kernel + plan-builder + dispatcher are all in `src/cuda/siglip2_megakernel.{cu,cuh}` already; default-off (`SIGLIP2_ENABLE_QKV_PREP=1` to opt in). When enabled, vision output is bit-clean vs A1-OFF (cosine 1.0000000) but **every text encode produces NaN**. The bug is a single-kernel late-anchor design colliding with ggml's gallocr; the fix is a 2-kernel design with a persistent scratch buffer. Receipts below — read before re-attempting.

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

- Branch: `dbrain/siglip2-v0` @ `0ff5f03` + ggml submodule fast-
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
