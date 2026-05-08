# siglip2.cpp — performance next-targets handoff (post-Phase-C)

> ## Mood of the room (read this first)
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
> Q4_K_M side experiments are committed on `dbrain/phase-c-cpasync-wip`
> for future reference but the user wants the production line to stay on
> Q8_0. The next agent's punch list is **(1) async parallel streams in
> /v1/classify, (3) cuBLASLt int8 GEMM swap on the hot mul_mats, and
> (5) FA d_head=72 native fast path.**
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

## What landed in this session (commits on `dbrain/phase-c-cpasync-wip`)

**The wip branch is divergent from `dbrain/siglip2-v0`** — none of this
work shipped to prod. It's all preserved for future opt-in or reuse.

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

- Production branch: `dbrain/siglip2-v0` at Phase B head. Phase C work
  is on `dbrain/phase-c-cpasync-wip` — not merged.
- ggml submodule on `siglip2-v0` is still at `a3fc6843`. The wip branch
  bumped to `4eec5550` for the Q4_K get_rows kernel — only relevant if
  you ever opt into the Plan A code.
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
