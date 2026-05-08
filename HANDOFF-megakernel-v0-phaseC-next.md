# siglip2.cpp Phase C — handoff for the kernel lap (custom mma-MMQ)

> ## Mood of the room (read this first)
>
> User on 2026-05-09 after the dp4a-force microbench:
> *"sorry i guess assume ill never say 'stop' until there's nothing
> left without affecting quality - what is the next best bet? Q4_K_M
> ive mentioned lets skip for now - anything else left? quick wins
> lap? custom mma-MMQ is the kind of crazy i think we should do? but
> probably 'clean brained agent work' - i.e. handoff with that task
> in mind not 'start doing it yourself'."*
>
> Translation: greenlit for a multi-week kernel rewrite, but **do it
> with fresh context**, not as the tail of a long session. Standing
> rules from `HANDOFF.md` and `HANDOFF-megakernel-v0.md` still apply
> (VRAM resident is the metric, cosine ≥ 0.999, Q8_0 weights F32
> activations, don't push commits, bench discipline).

## Cold-pickup orientation

Phase A0-A4 + Phase B all landed and ship in `dbrain/siglip2-v0`.
3 of 4 endpoints beat python clean (0.76-0.91× ratio). cfe is at
**0.98× python = tied, not embarrassed.** Phase C target was 0.85×,
which needs ~3 ms shaved off the 19.1 ms steady-state cfe compute.

**The graph-fusion layer is exhausted.** Phase B receipts in the
parent doc; Phase C microbench receipts in
`HANDOFF-megakernel-v0-phaseC-microbench.md`. Custom mma-MMQ for
siglip2's specific shapes is the only remaining lever that doesn't
sacrifice quality.

## What the previous lap (2026-05-09) ruled out

### ❌ dp4a-MMQ — proven slower than ggml's mma at our shapes

Hypothesis from the original handoff: hand-tuned dp4a-MMQ would beat
ggml's path by 1.2-1.7×, citing `project_qwen3tts_int8_mma_kill`
("mma m16n8k32 at 0.31-0.70× dp4a on hot wide-N shapes, consumer
GA106 1:1 mma:dp4a peak").

Reality: **ggml's mma path is 1.22-2.03× faster than its dp4a path
at every Q8_0 shape siglip2 hits.** Microbench in
`bench/microbench_mmq.cpp` measured both paths back-to-back:

| shape (Q8_0)              | mma p50 | dp4a p50 | dp4a/mma |
|---------------------------|--------:|---------:|---------:|
| attn_qkv  M=320 (cfe)     | 0.105   | 0.157    | **1.50×** mma faster |
| attn_qkv  M=729 (vision)  | 0.218   | 0.281    | **1.29×** mma faster |
| ffn_up    M=320           | 0.134   | 0.196    | **1.46×** mma faster |
| ffn_up    M=729           | 0.263   | 0.345    | **1.31×** mma faster |
| attn_o    M=320           | 0.046   | 0.064    | **1.39×** mma faster |
| ... all others            | ...     | ...      | mma 1.22-2.03× faster |

The qwen3-tts memory entry is **shape-specific to M=1 talker decode**
(launch-overhead and bandwidth dominated; mma's setup cost overwhelms
its compute density). At siglip2's M ≥ 64 wide-N batch matmuls, mma's
16× compute density per warp wins decisively even on consumer GA106.

**Note for the auto-memory:** the
`project_qwen3tts_int8_mma_kill.md` entry needs a "shape-specific to
M=1; at M ≥ 64 wide-N mma wins on RTX 3060" caveat, or a future agent
will repeat this experiment.

### ❌ Quick-wins lap (Levers 3+4 from original handoff) — too small

- **Lever 3 (pre-pad qkv_w/o_w at conversion):** the original handoff
  estimated <0.5 ms by removing runtime `ggml_pad` ops. After A1
  landed, those pads are already absorbed into the qkv-prep
  megakernel. Net new gain ≈ 0 ms.
- **Lever 4 (probe-head fusion):** vision-only, 1× per image. Fixed
  overhead today is ~0.4-0.5 ms; the megakernel-friendly portion
  (k/v projections at M=729) already shares the encoder hot path
  (A0-A3 fusions fire on it). Realistic launch-only saving: 0.1-0.2 ms
  on /v1/embeddings = ~0.4 % of endpoint time. **Not worth 1-2 hours
  of work for that ROI when there's a real lever waiting.**

### Tooling left in tree from the dp4a experiment

- **`bench/microbench_mmq.cpp` + CMake target `microbench_mmq`** —
  standalone Q8_0/F16 × F32 mul_mat harness for the 12 shapes
  siglip2 hits per encode. Use this as the first measurement on any
  kernel rewrite — cleaner signal than the full server bench.
  Build: `cmake --build build-cuda --target microbench_mmq`. Run:
  `./build-cuda/microbench_mmq` (CUDA 12.6 builder image: `tts-qwen3-dev:builder`,
  mount `-v $PWD:/src -w /src`).
- **`-DGGML_CUDA_NO_MMA=ON` cmake option in `ggml/`** — gates
  `TURING_MMA_AVAILABLE`/`AMPERE_MMA_AVAILABLE`/`BLACKWELL_MMA_AVAILABLE`
  and the host helpers. OFF by default. Keep it as a comparison
  knob for any future mma-vs-dp4a question. Don't remove without
  cause.

## Where the time is

Steady-state cfe (`/v1/classify_from_embeddings`, n=5 batched):

```
[cfe] parse=0.10 imgs=0.00 encode=21.6 score=0.01 respond=0.01 total=21.9 ms
  └─ encode_batch: init=0.65 build=0.15 alloc=0.15 inputs=0.00 compute=21.6 ...
```

Of the 19.1 ms steady-state graph compute:

| component                          |   ms |   % |
|------------------------------------|-----:|----:|
| Q8_0 mul_mat × 81 (qkv + o + up)   |  6.5 | 34% |
| F16 ffn_down × 27 (cuBLAS HGEMM)   |  4.1 | 21% |
| **matmul subtotal**                | **10.6** | **55%** |
| FA + post-FA cont × 27             |  ~3.5 | 18% |
| LN + tail fusion + residual + glue |  ~5.0 | 26% |
| total                              | 19.1 | 100% |

(matmul totals interpolated from the microbench p50s × 27 layers; the
6.5+4.1 ≠ 11.6 quoted in the receipts file because that one used a
slightly different M=320 sample. Either is within noise.)

## The actual lever — custom mma-MMQ tuned for siglip2 shapes

**Premise:** ggml's MMQ achieves 24-28 TFLOPS at our M=320/729
shapes ≈ **50% of RTX 3060 INT8 peak** (~50 TOPS). A kernel
specialized for our exact (M, K, N) set could plausibly reach
60-75% of peak = 30-37 TFLOPS = matmul time × 0.7-0.8× current.

**Estimated win** (revised from the original handoff which got the
mma-vs-dp4a direction wrong):

| component                | current | best-case | Δ on cfe |
|--------------------------|--------:|----------:|---------:|
| Q8_0 matmul × 81 @ M=320 |   6.5 ms |   ~4.6 ms | **-1.9 ms** |
| Q4_K_M (Lever 2 detour)  |   6.5 ms |   ~3.3 ms | -3.2 ms (but parked) |

So custom mma-MMQ targeting siglip2's three Q8_0 weight shapes at
M=320 puts cfe at ~17.2 ms = **0.80× python**, hitting the ≤0.85×
target. ffn_down (F16, K=4304 not Q8_0-aligned) is unaffected — it
stays on the cuBLAS HGEMM path.

The *theoretical* ceiling without quant changes is roughly 14-15 ms
cfe (60-75% of peak utilization) = 0.65-0.70× python. Whether that's
reachable is the real research question for this lap.

### The shape table (also in the microbench receipts)

```
Q8_0 weights, K=1152 (innermost), output rows N. F32 activations, F32 out.

  attn_qkv: K=1152, N=3456,  M ∈ {64, 320, 729}
  attn_o  : K=1152, N=1152,  M ∈ {64, 320, 729}
  ffn_up  : K=1152, N=4304,  M ∈ {64, 320, 729}

Plus F16 ffn_down (K=4304, N=1152) which goes through cuBLAS HGEMM —
out of scope unless you also re-quantize that weight to Q8_0 with
K-padded to 4320 (16 zero pad). Possible but separate effort, +0.4 %
weight storage. Estimated additional gain ~1 ms cfe if MMQ beats
HGEMM at this shape.
```

### Reference points

- **`ggml/src/ggml-cuda/mmq.cuh`** — the kernel to beat. ~4000 lines,
  highly templated. The mma path lives behind
  `#if defined(TURING_MMA_AVAILABLE)`. Per-shape constants like
  `mmq_x_max=128`, `mmq_y=128`, `nwarps=8` are tuned for breadth, not
  for siglip2's specific (K=1152, N ∈ {1152, 3456, 4304}) trio.
- **`qwen3-tts.cpp/src/cuda/qwen3_megakernel.cu`** — already-shipped
  custom MMVQ for M=1, but not directly applicable (we need M ∈
  {64, 320, 729}).
- **CUTLASS / cuBLASLt int8 GEMM** — could be the easy lift if the
  build-system pain is acceptable. cuBLASLt's int8 tensor-core GEMM
  on Ampere routinely reaches 70-80% peak. Cost: pulling cuBLASLt
  into siglip2's runtime (heavy dep), and writing the Q8_0 →
  cuBLASLt-int8 packing layer (Q8_0 stores 32 elements per block
  with one fp16 scale; cuBLASLt expects flat int8 with separate
  per-row/per-tile scales).

### Plan-of-attack sketch (not prescriptive — the next agent should
own this)

1. **Microbench-first.** Add a custom-kernel column to
   `bench/microbench_mmq.cpp`. Goal of week 1: a hand-rolled
   mma m16n8k32 kernel for one shape (suggest `attn_qkv M=320,
   K=1152, N=3456`) that *matches* ggml's mma at that shape. If you
   can't match, you can't beat — abort early.

2. **Specialize.** Once matched, exploit siglip2-specific facts that
   ggml's generic kernel can't:
   - K=1152 is fixed at every Q8_0 weight (innermost dim). Compile-
     time loop unroll the K-reduction.
   - M ∈ {64, 320, 729} is finite. Three template instantiations,
     not the 8..128-step loop ggml does at line `mmq.cuh:4069`.
   - N ∈ {1152, 3456, 4304} is finite. Same.
   - All three Q8_0 weights are stored with the same Q8_0 layout.
     One load_tiles instantiation, not the 20-quant table-driven
     dispatch.

3. **Activation-share.** In the encoder block, attn_qkv and the
   subsequent o_proj read from different sources (LN1 output vs FA
   output). But within a single residual stream, the LN1 output
   feeds *only* attn_qkv. The LN2 output feeds *only* ffn_up. So
   activation-quantize sharing across matmuls is **NOT a lever** —
   each Q8_0 matmul has a unique source. Don't waste time here.

4. **Megakernel hook integration.** Add the custom kernel as a new
   anchor under `siglip2_megakernel.cu`. Op-hook on `MUL_MAT` whose
   `src0` matches one of the per-block Q8_0 weights by name. Same
   gating pattern as A0-A3 (kill switch
   `SIGLIP2_DISABLE_CUSTOM_MMQ=1`).

5. **Parity discipline.**
   - 5-prompt batched vs 1-prompt sequential cosine ≥ **0.99994**
     (was 0.99994-0.99996 in A4). This is the fast smoke.
   - All 4 HF parity scripts ≥ **0.999** (image-1920×1080→729 is
     strictest, currently 0.999523 — there's only ~545 µcos of
     headroom across 27 layers × 3 matmul rewrites).
   - Re-run after every kernel rewrite, not just at the end.

### Pitfalls inherited from prior phases

- **gallocr aliasing** — the A1 single-kernel design crashed because
  gallocr lets `qkv_add.dst` reuse the slot after `last_cont.idx`.
  Custom MMQ writes to `dst` (the matmul output), not to a re-used
  intermediate, so this trap doesn't directly apply — but if you add
  a "fuse mm + bias-add" version, watch the same lifetime issue.
- **Reduction-tree determinism** — every kernel rewrite is a fresh
  source of FP-reduction order. Drift up to ~50 µcos per layer is
  noise; drift > 100 µcos per layer signals a numerical bug.
- **CUDA graph cache stability** — Phase B's per-shape graph cache
  needs stable tensor pointers. Custom MMQ allocates output via
  ggml's gallocr (same as ggml-mma), so no impact. But if you add a
  scratch buffer (like A1's persistent qkv-prep slab), allocate it
  once in `install()` and never `cudaFree` until shutdown.

## Smaller knobs that might be combinable with MMQ

These are tucked here so they don't get forgotten — none alone moves
the headline, but several together could trim 0.5-1 ms.

- **Stream-K already on.** Verified at `mmq.cu:121`. Not a knob.
- **`mmq_x_max=128`** is fine for M=320 (3 tiles) and M=729 (6
  tiles) but might be over-tiled for M=64 (1 tile). The M=64 path
  goes through MMVQ anyway (single-prompt CLI only), not MMQ.
- **MMVQ_MAX_BATCH_SIZE=8.** Single-prompt text path (CLI scripts)
  hits MMVQ at M=1; batched (server) hits MMQ at M=320. No knob to
  raise/lower at runtime.
- **Pre-quantized Q8_1 activation cache.** ggml re-runs
  `quantize_mmq_q8_1_cuda` on the activation for every matmul. At 4
  matmuls/layer × 27 layers = 108 quantize launches. If the upstream
  fusion produced Q8_1 activations directly (i.e., the LN-with-
  affine kernel writes Q8_1 instead of F32 to scratch), MMQ could
  skip its quantize. ROI: 108 × ~5 µs = ~0.5 ms cfe. Big design lift
  (changes A0's contract); only worth attempting if the custom MMQ
  also lands.

## What "done" looks like for this lap

- cfe (`/v1/classify_from_embeddings`) **≤ 0.85× python p50** clean
  GPU, 50-run p50. Currently at 0.98×.
- All 4 HF parity scripts PASS (≥ 0.999 cosine, ≤ 5e-3 probs MAE).
- 5-prompt batched vs 1-prompt sequential cosine ≥ 0.99994.
- No VRAM regression past the standing 1504 MiB resident target
  (custom kernel scratch ~50 MiB is acceptable).
- Add a "Phase C kernel-lap receipts" section to
  `HANDOFF-megakernel-v0.md` with bench numbers + parity confirmation.
- Update `bench/microbench_mmq.cpp` to include custom-kernel column
  alongside ggml-mma so future regressions are caught.

## Files this lap will touch

- `src/cuda/siglip2_megakernel.cu` — add custom MMQ kernel + op_hook
  on the 3 (Q8_0) per-block weights.
- `src/cuda/siglip2_megakernel.cuh` / `.h` — new kernel decls + env
  knob.
- `bench/microbench_mmq.cpp` — extend with custom-kernel column.
- (Maybe) `ggml/...` — only if you find a real bug or need an
  upstream hook. Don't fork ggml lightly.

## Files / facts unchanged from parent docs

- `dbrain/siglip2-v0` is still the working branch. ggml submodule at
  `a3fc6843`. No upstream fast-forward needed.
- VRAM 1504 MiB resident, 1566 MiB warmup.
- Models: Q8_0 (`models/siglip2-so400m-naflex-q8_0.gguf`).
- Build: `cmake --build build-cuda -j$(nproc)` inside
  `tts-qwen3-dev:builder`. Mount paths: `-v $PWD:/src -w /src`.

Now go. (After you've drunk a coffee and re-loaded context.)
