# siglip2.cpp — performance next-targets handoff (post-Phase-C)

> ## ⛔ CHAPTER CLOSED — 2026-05-10
>
> All actionable perf levers exhausted on RTX 3060 / Ampere. See **"Final
> outcomes"** at the end of this doc for receipts. The next focus is
> productionization — see `HANDOFF-prod.md`.
>
> **Current state (RTX 3060, clean GPU, per-process VRAM):**
> - `/v1/classify` p50 **57.6 ms** (0.81× python), p95 **57.8 ms**
> - `/v1/embeddings` p50 **44.3 ms** (0.82× python)
> - `/v1/text_embeddings` p50 **20.1 ms** (0.76× python)
> - `/v1/classify_from_embeddings` p50 **19.9 ms** (0.92× python)
> - VRAM resident **1542 MiB** post-load / **1668 MiB** post-warmup (CUDA
>   graph captures), unchanged under burst
> - Cold start **1.33 s**, reload after `/unload` **0.79 s**
> - 400/400 concurrent classify pass clean
> - All 4 parity scripts pass at the 0.999 floor
>
> ## Original mood-of-the-room (kept for context)
>
> User on 2026-05-09 after the Plan-A Q4_K_M attempt and the v5b/v6 custom
> kernel exploration:
> *"lets just stick with Q8 - keep the code supporting Q4_K - but this
> service was previously 2400 peak VRAM Q8 has already dropped it a lot. i
> guess where do we sit performance wise on Q8 vs python? what are the
> next steps to get more out?"* and *"lets have the next guy target 1, 3
> and 5."*
>
> Translation: the megakernel chapter (Phase A0-A4 + Phase B graph cache)
> is closed and shipped on `dbrain/siglip2-v0`. The custom-kernel and
> Q4_K_M side experiments rode the same branch in via fast-forward
> (commits below) but their runtime paths are off-by-default for Q8_0
> GGUFs. **You're on `dbrain/siglip2-v0`. Stay there.** The next agent's
> punch list is **(1) async parallel streams in /v1/classify, (3) cuBLASLt
> int8 GEMM swap on the hot mul_mats, and (5) FA d_head=72 native fast
> path.**
>
> **Standing rules from `HANDOFF.md` still apply:**
> - **VRAM**: full-tower resident is the metric. Q8_0 today is **1558 MiB**
>   (vs python's ~2400 MiB peak — already a -842 MiB win). Don't regress.
>   Q4_K_M plumbing on the wip branch can drop another 670 MiB if quality
>   constraints ever relax.
> - **Cosine ≥ 0.999** on parity scripts is the floor. Q8_0 holds:
>   text 0.999756 / vision 0.999944 / image 0.999523 / score 0.999986.
> - **Q8_0 is the cap.** Q4_K_M was tried (Plan A, this session) and fails
>   parity AND is +12% slower than Q8_0 on the cfe shape. The plumbing
>   stays on the wip branch for opt-in; production stays Q8_0.
> - **Don't push commits.** Bank locally; the user pushes.
> - **Bench discipline.** GPU is shared with `kobbler-vision-1`,
>   `kobbler-tts-qwen3-1`, sometimes `tts-qwen3-iter`. Quote ratios over
>   absolute ms; do 30-run p50 minimum.

## Where Q8_0 sits today

Phase B head (clean GPU, RTX 3060, 50 runs each, p50; vs python on the
same GPU). All four endpoints beat python clean:

| endpoint                              | cpp Q8_0 | python | ratio  |
|---------------------------------------|---------:|-------:|-------:|
| `/v1/embeddings`                      |  43.3 ms | 54.3 ms| **0.81×** |
| `/v1/text_embeddings` (n=5)           |  20.1 ms | 26.9 ms| **0.76×** |
| `/v1/classify`                        |  62.6 ms | 70.4 ms| **0.91×** |
| `/v1/classify_from_embeddings` (cfe)  |  19.9 ms | 21.6 ms| **0.98×** |

VRAM: **1558 MiB** resident.

Re-confirmed mid-session (server endpoint /v1/text_embeddings n=5 form-encoded,
30 reqs): 20.2 ms p50 / min 20.1 / mean 20.3. Holds.

## What landed in this session (commits now on `dbrain/siglip2-v0`)

These rode in via fast-forward from the now-archived `dbrain/phase-c-cpasync-wip`
branch. **All Q8_0 paths run identically to Phase B head** — the new
runtime helpers (`pad_x_to_w()`, custom-mmq kernel installation) gate
themselves off when W's K matches activation's K (Q8_0) or are only
invoked by the microbench. None of this dispatches at runtime under
default config when running the production Q8_0 GGUF.

```
2f8a1c8 chore: track phaseC handoff trail + ignore build-cuda-* variants
4f1ccdd phase C plan-A: Q4_K_M / Q5_K_M conversion + runtime K-padding (WIP, parity below 0.999)
bf42b34 phase C profile: clock64() cycle breakdown across load/sync/compute regions
9c86d81 phase C v6: cp.async + double-buffered shmem (WIP — edge-tile cosine drift)
cda641f phase C v5b: custom Q8_0×F32 MMQ kernel — bit-clean baseline
```

**Custom Q8_0×F32 mma m16n8k32 kernel (cda641f, bf42b34, 9c86d81):**
- v5b is bit-clean (cosine = 1.0 across all 9 microbench Q8_0 shapes) but
  ~2.7× slower than ggml's mma at the cfe shape on a clean GPU.
- Profile (clock64 instrumentation): 80% of v5b's per-block cycles are
  global→shmem load latency, only 19% mma compute. Structural cause is
  redundant memory traffic — v5b launches 135 small blocks each loading
  their own 128-row W slab; ggml's stream-K runs 28 fixed blocks (=NSM)
  each loading 1/28th of unique data. ~7× more memory traffic in ours.
- v6 added cp.async + double-buffered shmem. Cosine drops to 0.99997 on
  edge-tile shapes (M=729 / N=4304); diagnosis + plausible fixes in the
  v6 commit message (cp.async with zero-fill source-bytes parameter).
- Path forward (multi-week, parity risk per phase): stream-K + cp.async
  fix + custom MMQ for siglip2's finite (M, N) set. Lands at ggml-mma
  parity, not 1.5× past it. The Phase C "≤ 0.85× python on cfe" target
  is structurally hard to hit on this GPU at the kernel layer alone —
  see `HANDOFF-megakernel-v0-phaseC-microbench.md` for the dp4a vs mma
  receipts that frame the ceiling.

**Plan A: Q4_K_M / Q5_K_M conversion + runtime K-padding (4f1ccdd):**
- `tools/siglip2-quantize/main.cpp`: C++ tool that re-quantizes a F16
  GGUF to Q4_K_M / Q5_K_M / Q6_K / Q8_0 / Q4_0. Python's `gguf` library
  raises NotImplementedError on K-quants — needed C++ via
  `ggml_quantize_chunk` (parakeet's pattern). Per-tensor recipe matches
  llama.cpp's "M": ffn_down → Q6_K (residual-stream-sensitive),
  token_embd → Q4_K when base is Q5_K (no Q5_K get_rows kernel in
  ggml-cuda yet), everything else → base quant.
- ggml submodule bumped to `4eec5550` (dbrain/ggml@master) — adds the
  Q4_K get_rows kernel needed for K-quant token_embd lookups on GPU.
  GGML_CUDA_NO_MMA debug knob preserved on top (uncommitted dirty in
  submodule, intentionally — debug-only).
- `scripts/convert_siglip2_to_gguf.py`: --type q4_k_m branch falls back
  to F16 (Python lib limit). Use the C++ tool.
- `src/siglip2_{text,vision}.cpp`: pad_x_to_w() helper inserts ggml_pad
  on every mul_mat activation when W's innermost K exceeds the
  activation's. Threaded through all build_block paths + token_embd
  get_rows slice + cont. Q8_0 baseline parity unchanged (regression
  clean — re-verified mid-session).

Headline: Plan A fails on both axes against the user's "fast + quality":
- **Perf**: Q4_K_M 22.7 ms / Q5_K_M 23.0 ms vs Q8_0 20.2 ms (+12-14%
  slower). ggml's K-quant MMQ at M=320 has per-super-block dequant cost
  that swamps the bandwidth halving. The qwen3-tts memory note "Q5_K_M
  slower than Q4_K_M on Ampere via MMQ" generalizes to siglip2's batched
  shapes too (not just M=1 decode).
- **Quality**: text fail / vision 0.993 / image 0.980 / score 0.99939.
  Score parity holds (the actual classification metric); individual
  embedding cosines drop below 0.999 floor. Recoverable with imatrix
  calibration (~1-2 day plumbing) but not attempted this session.
- VRAM win is real: -670 MiB on Q4_K_M (888 MiB resident), -594 MiB on
  Q5_K_M.

User decision: stay Q8_0 in production. Keep the Plan A code on the
branch for emergency opt-in.

## The next agent's three targets — in priority order

### Target 1: async parallel streams in /v1/classify  (1-2 days, low risk)

**The win:** /v1/classify currently runs vision encode (~49 ms) → text
encode (~20 ms) → score (CPU ~0 ms) in series, totaling 62.6 ms. The
two encodes are independent (vision processes the uploaded image, text
processes the prompt list — no data dependency between them until the
score step). Putting them on parallel CUDA streams reduces wall time to
≈ max(vision, text) + score ≈ 49 ms.

**Expected receipts:** /v1/classify drops from 62.6 → ~49 ms = **0.91× →
~0.65× python**. No effect on the other three endpoints.

**Where to wire it (`src/siglip2_server.cpp` around line 462, the
`/v1/classify` handler):**
- The handler today holds `st.encode_mutex` while running both encodes
  back-to-back. With parallel streams the lock can wrap submission, and
  we wait for both via a shared `cudaEventRecord/cudaStreamWaitEvent`
  pattern.
- ggml-cuda's `ggml_backend_cuda` already supports per-context streams
  (the Phase B graph-cache entries each have their own gallocr binding).
  The text and vision encoders own separate ggml_backend_t — currently
  they share the global default stream because we use `cudaStream_t s = 0`
  in the megakernel install.
- Cleanest implementation: extend `siglip2_megakernel::install()` to
  accept a stream-set, OR have the server allocate two `ggml_backend_cuda`
  contexts (one per encoder), each with its own dedicated stream.
- The `siglip2_score` step is CPU; trivial.

**Pitfalls:**
- The megakernel hooks (A0/A1/A2/A3) currently use the per-op stream
  passed by ggml-cuda. They should already work on different streams.
- Each `GraphCacheEntry` (`siglip2_text.cpp`, `siglip2_vision.cpp`) has
  its own gallocr buffer — disjoint memory, safe to run concurrently.
- The `g_qkv_scratch` slab in `siglip2_megakernel.cu` is single-instance
  per process. With concurrent encodes both touching it, you'll race.
  Either give it per-stream slabs OR serialize the QKV-prep section
  (still wins because most of the encode is OUTSIDE QKV-prep).

**Verify:**
- `/v1/classify` p50 drops from ~62 to ~50 ms.
- Other endpoints unchanged (especially cfe at 19.9 ms).
- Parity scripts unchanged.

### Target 3: cuBLASLt int8 GEMM swap on the hot mul_mats  (1-2 weeks)

**The win:** v5b's profile in this session quantified the matmul ceiling
on ggml's mma at our shapes — ~25 TFLOPS achieved, ~50% of RTX 3060's
~50 TOPS INT8 peak. cuBLASLt's int8 tensor-core GEMM on Ampere routinely
hits 70-80% peak — that's a ~1.4-1.6× win on the matmul subtotal alone.

**Expected receipts:** matmul subtotal at the cfe shape is ~11.6 ms / 19
ms = ~60% of compute. A 1.4× win on matmuls alone = -3.3 ms. cfe goes
from 19.9 → ~16.6 ms = **0.98× → 0.77× python**. Hits the Phase C
"real embarrassment" target (≤ 0.85×).

**Plumbing required:**
1. Pull cuBLASLt into the build (header `cublasLt.h`, library
   `-lcublasLt`). It's in the CUDA toolkit; no new external dep.
2. Q8_0 → cuBLASLt-int8 packing layer at load time. Q8_0 stores 32
   elements per block with one fp16 scale; cuBLASLt's int8 IMMA wants
   flat int8 + per-row OR per-tile scaling. Per-row scaling loses
   per-32-element block fidelity (siglip2 will not survive this — vision
   transformer weights have block-level outliers). Per-tile scaling
   (16- or 32-element groups) preserves it. Use `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`
   with per-K-tile scales.
3. Activation quantize-x: today ggml's MMQ does the F32 → Q8_1 cast on
   the fly. Replace with a custom F32 → flat-int8 + per-row-scale kernel
   that matches cuBLASLt's expected layout.
4. Dispatch hook: in `src/cuda/siglip2_megakernel.cu`, add an op_hook
   on MUL_MAT for the per-block Q8_0 weight names (`*.attn_qkv.weight`,
   `*.attn_o.weight`, `*.ffn_up.weight`, `*.ffn_down.weight` — the four
   per-block weights × 27 layers). Same gating pattern as A0-A3, kill
   switch `SIGLIP2_DISABLE_CUBLASLT=1`.

**Pitfalls:**
- Weight packing is one-time at load — store packed weights in a
  separate device buffer alongside the original Q8_0 (or replace if you
  control the GGUF format). Either way the loader needs an extension.
- Per-tile scale precision: 16-element vs 32-element groups. siglip2's
  K=1152 % 32 == 0 ✓ but K=4304 % 32 ≠ 0 (ffn_down). The Plan A K-padding
  plumbing on the wip branch already handles this — could lift parts of
  it.
- Score parity is the gate. Run all 4 parity scripts after each step.
- ffn_down today is F16 — leave it on the existing F16 cuBLAS HGEMM
  path; only swap the Q8_0 mul_mats.

**Where to start:**
- Read `ggml/src/ggml-cuda/mmq.cu` `ggml_cuda_mul_mat_q` to understand
  the existing dispatch. The cuBLASLt path replaces the
  `ggml_cuda_mul_mat_q_switch_type` call for our specific shape set.
- `bench/microbench_mmq.cpp` (this session) is the right harness — add
  a cuBLASLt column alongside ggml-mma. If cuBLASLt at our exact (M, K, N)
  doesn't beat ggml-mma in the microbench, abort early.
- The `microbench_mmq` already includes the custom-mmq column (v5b),
  so a 4-way comparison is a one-file change.

### Target 5: FA d_head=72 native fast path  (1 week, parity risk)

**The win:** SigLIP2's d_head=72 doesn't divide by 16, so the CUDA MMA-FA
kernels can't accept it natively (their tile templates require
`d_head % 16 == 0`). Today we zero-pad d to 80 before FA and slice back
after — 6 launches per encoder block (3 Q/K/V pads + post-FA slice) are
overhead on top of the FA itself.

**Expected receipts:** ~1 ms savings on cfe (~5% improvement). cfe goes
from 19.9 → ~18.9 ms = **0.98× → ~0.88× python**. Below "real
embarrassment" threshold.

**Plumbing required:**
1. Read `ggml/src/ggml-cuda/fattn-mma-f16.cuh` — the MMA-FA kernel. Find
   the tile-D constant.
2. Either:
   - **(a)** Specialize a d_head=72 variant of the MMA-FA kernel via
     ldmatrix+mma plumbing that handles 72-wide tiles natively (3 sub-tiles
     of 24, or 4.5 sub-tiles of 16 with the last being half-zero). Write a
     siglip2-specific FA kernel under `src/cuda/siglip2_fa72.cu`.
   - **(b)** Use the `fattn-vec-instance-*` (vector path) variants which
     are more flexible on d_head but slower per element.
3. Hook via op_hook on FLASH_ATTN_EXT for our d_head=72 shape; gate
   `SIGLIP2_DISABLE_FA72=1`. Same pattern as A3 post-FA cont.

**Pitfalls:**
- FA's numerical sensitivity is high — softmax + accumulator interaction.
  Any custom kernel needs to match ggml's `flash_attn_ext` cosine on the
  image-1920×1080 path (the strictest of the four parity scripts).
- The post-FA cont (A3) currently slices the d_pad=80 → d_head=72 tail.
  If FA outputs native d=72, A3 becomes a no-op — but A3's anchor
  detection currently requires a strided-cont source, which won't fire
  on a contiguous output. Wire the slice removal into the same op_hook.
- d_head=72 is unusual; vendor benchmarks may not exist for the exact
  shape. Plan to spend a day microbenching the kernel before integrating.

**Reference for ggml's existing FA:**
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh` — MMA path (d_head must be
  multiple of 16).
- `ggml/src/ggml-cuda/fattn-vec-instance-*.cu` — vector path (any d).

## What's NOT a target (don't relitigate)

- **n_pos trim** — `project_siglip2_n_pos_trim_dead.md`. SigLIP2's
  pooler is hard-coded to position 63 of a 64-padded sequence.
- **Q4_K_M / K-quants in production** — Plan A receipts above. Code on
  branch for emergency opt-in only.
- **Custom Q8_0 MMQ matching ggml-mma** — v5b receipts above. Multi-week
  to land at ggml-mma parity, not past it. cuBLASLt (target 3) is the
  better swing for matmul perf.
- **Lever J (ffn_down padding)** — parked. 126 MiB doesn't justify the
  cosine cost on a 12 GB GPU.

## What "done" looks like for the next agent

After targets 1, 3, 5:
- `/v1/classify` ≤ 0.7× python (from 0.91×).
- `/v1/classify_from_embeddings` ≤ 0.85× python (from 0.98×).
- All four parity scripts PASS — vision/text/image/score cosines ≥
  0.999, probs MAE ≤ 5e-3.
- VRAM resident still ≤ 1600 MiB.
- Add a "Phase D receipts" section here mirroring Phase B's table.

## Files the next agent will likely touch

- `src/siglip2_server.cpp` — handler `/v1/classify` (target 1).
- `src/cuda/siglip2_megakernel.{cu,cuh,h}` — add cuBLASLt dispatch hook
  (target 3); add d_head=72 FA dispatch hook (target 5).
- `src/cuda/siglip2_cublaslt_mm.{cu,cuh}` (new) — Q8_0 → int8 packing,
  per-K-tile scales, cuBLASLt matmul wrapper (target 3).
- `src/cuda/siglip2_fa72.{cu,cuh}` (new) — d_head=72 native FA kernel
  (target 5).
- `bench/microbench_mmq.cpp` — extend with cuBLASLt column (target 3).
- `CMakeLists.txt` — link cuBLASLt; add new CUDA TUs.
- `models/` — no new artifacts.

## Files / facts unchanged

- Production branch: `dbrain/siglip2-v0`. The Phase C exploration commits
  (v5b/v6/profile/Plan A/handoff updates) fast-forwarded onto this branch
  after the user's "lets just stick with Q8" call — code is here, but
  runtime paths are off-by-default for Q8_0 GGUFs. No separate wip branch
  to track.
- ggml submodule at `4eec5550` (dbrain/ggml@master). Adds the Q4_K
  get_rows kernel — only relevant if you ever opt into Plan A. The
  GGML_CUDA_NO_MMA debug knob is preserved as uncommitted local content
  inside the submodule (intentional; debug-only).
- VRAM 1558 MiB resident.
- Models: `siglip2-so400m-naflex-q8_0.gguf` (1.46 GB) is the production
  artifact. Q4_K_M / Q5_K_M / new F16 GGUFs from this session are in
  `models/` for reference only.
- Build: `cmake --build build-cuda -j$(nproc)` inside
  `tts-qwen3-dev:builder`. Mount paths: `-v $PWD:/src -w /src`.
- Parity test runtime: `kobbler-vision` image with `/tmp/siglip-libs`
  bind-mounted as `/cuda-libs` and `LD_LIBRARY_PATH=/cuda-libs`. The
  builder image has CUDA libs natively but lacks Python deps; kobbler-vision
  has Python deps but lacks CUDA libs. The bind-mount stitches them.

Now go.

---

# Final outcomes — 2026-05-10

## Receipts table

| Lever | Status | Δ on `/v1/classify` | Notes |
|---|---|---:|---|
| **Target 1: parallel CUDA streams in `/v1/classify`** | ✅ shipped | 62.6 → **57.6 ms** (-5.0 ms, -8 %; 0.91× → 0.81× python) | Private `ggml_backend_t` per encoder + persistent worker thread + `text_worker_mutex` for concurrent-handler serialization. Per-encoder QKV-prep scratch (vision 9.6 MiB, text 27 MiB) allocated once at `load()`, captured CUDA graphs reference encoder-owned slabs. Less than the handoff's projected 49 ms ceiling because RTX 3060's 28 SMs saturate on vision matmuls — under concurrent execution vision goes 43→56 ms (+30 %) and text goes 20→48 ms (+140 %). Wall = max(under_contention) ≈ 56 ms is the floor on this GPU. |
| **Target 4: `/unload` + `/v1/classify` SIGSEGV** | ✅ fixed | n/a (correctness) | Two root causes: (1) concurrent `graph_begin_hook` on parallel host threads racing on global `g_*_anchors` `unordered_map`s (clear+rebuild was UB) → moved to `thread_local`. (2) QKV-prep `g_qkv_scratch` was a process-global slab with grow-on-resize semantics, where realloc invalidated the pointer baked into already-captured CUDA graphs of OTHER shapes → moved to per-encoder pre-sized slab, registered via `siglip2_megakernel::set_active_qkv_scratch` thread-local. 3× (30 classifies + unload) cycle survives clean. |
| **PersistentWorker race fix** | ✅ fixed | n/a (correctness) | `wait_done()` was waiting on `task == nullptr`, but the worker sets task to null at the START of the lambda execution → `wait_done()` could return mid-encode, handler reads uninitialized `text_ok`/`text_err`, returns 500 with empty error string under concurrent load (~3 % under 4-thread burst). Two fixes: separate `pending` flag set false only after `f()` returns, and handler holds `text_worker_mutex` across submit + wait_done so concurrent classify handlers don't clobber each other's submitted lambdas. **400/400 concurrent classify clean post-fix.** |
| **Target 5b: vision stream priority HIGH** | ✅ shipped (default-on) | 57.9 → 57.6 ms (-0.3 ms p50, -0.8 ms p95) | `ggml_backend_cuda_init_with_priority(0, -1)` for the vision backend in dual-tower mode (text stays DEFAULT). Vision is the long pole; HIGH lets it hold SMs under text contention. Sub-noise on p50 but real on p95. Free win, no parity drift. Override via `SIGLIP2_VISION_STREAM_PRIORITY` env (-1/0/1 = HIGH/DEFAULT/LOW). |
| **Target 3: cuBLASLt int8 GEMM swap** | ❌ aborted at gating | — | `bench/microbench_mmq` now has a `cuBLLt` column (per-row int8 + fp32 row-scale + cuBLASLt int32 IMMA + custom rescale). Receipts on RTX 3060: cuBLLt only wins at M=64 (text n=1: 0.75-0.93× across attn_qkv/attn_o/ffn_up); at M=320 (cfe shape, the primary target) it's tied or slower (1.00-1.25×); at M=729 (vision) it's tied or much slower (up to 1.43× at attn_o). And this is the SPEED upper bound — per-K-tile scales (the only form preserving Q8_0 per-32-element fidelity) require K=32 sub-GEMMs which would be slower. ggml-mma's stream-K already saturates Ampere SMs at our shapes. Code stays in tree as a future-reference receipt + re-check harness for sm89+ where per-tile scales are supported. |
| **Target 5: native d=72 FA fast path** | ❌ aborted | — | Discovered ggml's tile FA kernel has a NATIVE `case 72` (`fattn-tile.cu:17-20`). Wired via `SIGLIP2_DISABLE_FA_PAD=1` env to drop the d=72→80 pad and route to the tile path. All 4 parity scripts pass on the env-on path. But A/B clean-GPU bench shows tile@72 SLOWER than mma@80+pad: vision 43.4 → **55.1 ms (+27 %)**, classify 56.8 → 70.5 ms (+24 %). Text barely moves (20.2 → 20.4 ms) — the regression concentrates on n_pos=729 vision where tile@72's `// TODO optimize kernel parameters for head sizes 40, 72, 80, 96, 112` (`fattn-tile.cuh:9`) bites hard. The custom MMA@72 kernel the original handoff sketched as the alternative would need to BEAT mma@80+pad cleanly to be worth shipping (best-case ~1 ms savings on cfe), plus 1 week of MMA surgery + parity risk. Not a viable swing on Ampere. Env-gated, default-off; useful as a re-test knob on sm89+. |
| **Q4_K_M / K-quant production path** | ❌ failed (Plan A receipts above) | — | Code on branch for emergency opt-in; -670 MiB VRAM but fails 0.999 floor + 12 % slower. |
| **Lever J: ffn_down padding** | ❌ parked (user decision) | — | -126 MiB available but cosine drops below 0.999. Floor is the working bar. |

## What's NOT a target anymore (don't relitigate)

- **Custom MMQ for M=320** — the cuBLASLt microbench above just disproved the
  premise that the matmul layer has slack to find on RTX 3060 / Ampere. ggml-
  mma's stream-K is at the SM-saturation ceiling at our shapes. Anything else
  custom would need to BEAT ggml-mma without benefit of cuBLASLt's internal
  algos — multi-week with parity risk and no headroom.
- **Anything else FA-related** — d=72 is the only siglip2 head dim that doesn't
  divide by 16, and the tile/MMA dispatch is already as good as it gets.
- **VRAM at 0.999 floor** — Q4_K_M failed, Lever J parked, ffn_down already
  Q8_0 in prod GGUF. If you ever lower the floor, Q4_K_M with imatrix
  calibration is the next move (~1-2 day plumbing).

## API gaps vs python kobbler-vision

1:1 endpoint surface match (`/health`, `/unload`, `/v1/embeddings`,
`/v1/text_embeddings`, `/v1/classify`, `/v1/classify_from_embeddings`).
**One feature gap**: `return_last_hidden=true` returns 501 (python returns
the full hidden state). **No kobbler caller exercises this flag** — checked
across `~/dev/kobbler/api/src/skip_detect/vision.rs` (the only kobbler
client). Functionally siglip2.cpp can replace kobbler-vision-1 today.

## New env knobs introduced this session

| Env | Default | Purpose |
|---|---|---|
| `SIGLIP2_VISION_STREAM_PRIORITY` | `-1` (HIGH) | Vision backend stream priority. `-1`/`0`/`1` = HIGH/DEFAULT/LOW. On RTX 3060 only HIGH actually changes anything (priority range [-5,0]). |
| `SIGLIP2_DISABLE_FA_PAD` | unset (=off) | Skip d=72→80 zero-pad → route to tile FA. Slower on Ampere; useful as a re-test knob on sm89+. |
| `SIGLIP2_VISION_MAX_PATCHES` | `729` | Worst-case vision QKV-prep scratch sizing. Larger requests fall back to ggml's split path (no fusion). |
| `SIGLIP2_TEXT_MAX_BATCH` | `32` | Worst-case text n_batch for QKV-prep scratch. Larger fall back. |
| `SIGLIP2_TIME_HANDLER` | unset | Per-stage timing for `/v1/classify` and `/unload`. |

## Files touched this session

- `src/siglip2_server.cpp` — split `encode_mutex` → `vision_mutex` + `text_mutex`; private backends in `load_model` (dual-tower); persistent text worker; `text_worker_mutex` serializing handler access; per-stage timing on `/v1/classify` + `/unload`.
- `src/siglip2_vision.cpp` — `private_backend` flag wired through `load()`; per-encoder `qkv_scratch_buf`; HIGH stream priority default; env-gated FA pad disable.
- `src/siglip2_text.cpp` — same private-backend + scratch-buf shape as vision.
- `src/siglip2_vision.h` / `siglip2_text.h` — `load(path, private_backend)` signature.
- `src/cuda/siglip2_megakernel.cu` — anchor maps + counters → `thread_local`; QKV scratch is now a thread-local `(dptr, cap)` pair set by `set_active_qkv_scratch`.
- `src/cuda/siglip2_megakernel.h` / `_stub.cpp` — `set_active_qkv_scratch` / `clear_active_qkv_scratch` API.
- `src/gguf_loader.{h,cpp}` — `init_separate_backend(name, err, stream_priority=0)` bypasses the singleton; passes priority to `ggml_backend_cuda_init_with_priority` when non-zero.
- `bench/microbench_mmq.cpp` + `bench/microbench_cublaslt_helpers.{cu,cuh}` — cuBLASLt int8 column for the gating bench.
- `CMakeLists.txt` — link `CUDA::cublasLt` to microbench.

ggml submodule unchanged this session (still `4eec5550` on `dbrain/ggml@master`).

Now go: `HANDOFF-prod.md`.

