# Phase C microbench — receipts (2026-05-09)

Standalone harness in `bench/microbench_mmq.cpp`, 1-op `mul_mat` graphs,
20 warmup + 200 measure iterations, std::chrono after `ggml_backend_synchronize`.
Built into `build-cuda/microbench_mmq` via the new CMake target.

## GPU state during run

- Device: NVIDIA GeForce RTX 3060
- Co-resident at run time: `kobbler-api` (104 MiB, idle), `qwen3-tts-server`
  (148 MiB, idle), `crispasr` parakeet (1712 MiB, idle). 0% GPU util at
  start.

## Numbers

```
shape (M K N)                         min ms      p50 ms      p99 ms   GFLOPS @p50   GB/s W @p50
--------------------------------  ----------  ----------  ----------  ------------  ------------
attn_qkv  M=64  (text n=1)             0.037       0.038       0.039       13403           111.3
attn_qkv  M=320 (text n=5)             0.104       0.105       0.106       24184            40.1
attn_qkv  M=729 (vision)               0.215       0.217       0.219       26698            19.5
attn_o    M=64  (text n=1)             0.026       0.026       0.027        6452            53.6
attn_o    M=320 (text n=5)             0.045       0.046       0.047       18445            30.6
attn_o    M=729 (vision)               0.085       0.090       0.091       21504            15.7
ffn_up    M=64  (text n=1)             0.053       0.054       0.055       11724            97.3
ffn_up    M=320 (text n=5)             0.123       0.126       0.128       25271            42.0
ffn_up    M=729 (vision)               0.254       0.261       0.270       27711            20.2
ffn_down  M=64  (text n=1)  (F16)      0.050       0.050       0.051       12601           197 *
ffn_down  M=320 (text n=5)  (F16)      0.147       0.152       0.159       20850            65 *
ffn_down  M=729 (vision)    (F16)      0.278       0.280       0.292       25791            35 *
```

`*` ffn_down GB/s is at 16 bits/weight (F16); the Q8_0 shapes are at 8.5
bits/weight (Q8_0 storage incl. fp16 scale).

## Per-encode matmul totals (×27 layers)

| path                  | sum/block | ×27       | as % of bench compute |
|-----------------------|----------:|----------:|----------------------:|
| text M=64  (n=1 CLI)  | 0.168 ms  | 4.54 ms   | (CLI not benched)     |
| **text M=320 (cfe)**  | 0.429 ms  | **11.58 ms** | **60.6 %** of 19.1 ms |
| vision M=729          | 0.848 ms  | 22.90 ms  | ~60 % of vision encode |

**60-61 % of encode time is in mul_mat.** The other ~40 % is FA, LN, fusion
kernels, residuals, copies, kernel launch glue.

## What ggml is dispatching to right now

Read of `ggml/src/ggml-cuda/ggml-cuda.cu:2448-2535` + `mmq.cu:267` +
`mmq.cuh:110-276`:

- `ggml_cuda_should_use_mmq` returns **true** at all our shapes
  (cc=86 on RTX 3060 → `turing_mma_available=true` → returns true at
  line 308 unconditionally).
- Inside MMQ, `mmq.cuh:276` picks the **INT8 mma m16n8k32** path when
  `turing_mma_available && mmq_x >= 48`. M=320 and M=729 both clear that
  bar. M=64 (single-prompt CLI) hits the dp4a sub-path (`mmq_x < 48`).
- ffn_down is **F16, not Q8_0** (K=4304 isn't 32-aligned for Q8_0
  blocks). It dispatches to `ggml_cuda_op_mul_mat_f` / cuBLAS HGEMM.

So at the cfe path (M=320), every Q8_0 mul_mat is going through the mma
m16n8k32 kernel — the exact path the `project_qwen3tts_int8_mma_kill`
memory says runs **0.31-0.70× dp4a on consumer GA106 hot wide-N shapes**.

Achieved throughput at M=320:
- Q8_0 matmuls: 18-25 TFLOPS
- F16 ffn_down: 21 TFLOPS

RTX 3060 (GA106) ceilings:
- FP32 peak: ~12.7 TFLOPS
- INT8 dp4a peak: ~50 TOPS
- INT8 mma m16n8k32 peak: ~50 TOPS (consumer GA106 is 1:1 mma:dp4a, not
  16:1 like A100)

Realistic dp4a kernel ceiling at our shapes (60-80% of peak): **30-40
TOPS**. Current ggml-mma achieves 25 TOPS at the best shape (ffn_up
M=320, M=729). So **dp4a has 20-60% theoretical headroom over current
ggml-mma at our shapes**, plus the qwen3-tts evidence that on this GPU
dp4a actually outpaces mma despite identical peaks.

## Estimated cfe win if dp4a beats current ggml-mma path

| Assumption                    | matmul ratio | matmul Δ on cfe | new cfe |
|-------------------------------|-------------:|----------------:|--------:|
| dp4a 1.2× current (low end)   |        0.83× |          -2.0 ms | 17.1 ms (0.79× py) |
| dp4a 1.4× current (qwen3 low) |        0.71× |          -3.4 ms | 15.7 ms (0.73× py) |
| dp4a 1.7× current (qwen3 mid) |        0.59× |          -4.7 ms | 14.4 ms (0.67× py) |

Phase C target is **cfe ≤ 0.85× python ≈ 17 ms**. Even the low end
clears it. The qwen3-tts mid-case puts cfe at 0.67× python — well past
"embarrassment."

## Recommended next experiment (cheap, decisive)

**Force ggml's MMQ to use dp4a instead of mma at our shapes**, re-run
this microbench and the full bench, compare. ~1 hour of work, no custom
kernel, decides whether to commit to a hand-tuned MMQ.

Two ways to force it:

1. **Patch `ggml/src/ggml-cuda/mmq.cuh:276`** — change the condition so
   the mma branch never fires (or only fires above a much higher
   `mmq_x` threshold). Rebuild siglip2-server only — submodule edit.
2. **Recompile ggml without `-DGGML_CUDA_FORCE_CUBLAS` but with a new
   `SIGLIP2_FORCE_DP4A_MMQ` define** that gates the same condition.

Option 1 is the simplest first stab. If dp4a wins → option 2 lands as
the durable shape, gated behind an env knob like the existing megakernel
phases.

If dp4a does NOT meaningfully beat current path here → the lever isn't
where we thought, and the next move is either:
- Accept the 0.98× cfe and call Phase B "good enough"
- Move to Lever 2 (Q4_K_M with rank-parity) when the user wants the
  VRAM win

## ❌ Force-dp4a experiment — dead lever (2026-05-09)

Patched `ggml/src/ggml-cuda/common.cuh` to gate `TURING_MMA_AVAILABLE`,
`AMPERE_MMA_AVAILABLE`, `BLACKWELL_MMA_AVAILABLE` and the host-side
`turing_mma_available()` / `ampere_mma_available()` behind a new
`-DGGML_CUDA_NO_MMA=ON` cmake option (added to `ggml/CMakeLists.txt` and
`ggml/src/ggml-cuda/CMakeLists.txt`). Built `build-cuda-dp4a/` from that
config, ran the microbench back-to-back with the default mma build under
matched (idle) GPU conditions:

```
shape (Q8_0)              mma p50 ms   dp4a p50 ms   dp4a/mma   winner
-----------------------  -----------  ------------  ---------  -------
attn_qkv  M=64              0.039         0.079       2.03×    mma 2.0× faster
attn_qkv  M=320 (cfe)       0.105         0.157       1.50×    mma 1.5× faster
attn_qkv  M=729             0.218         0.281       1.29×    mma faster
attn_o    M=64              0.027         0.033       1.22×    mma faster
attn_o    M=320             0.046         0.064       1.39×    mma faster
attn_o    M=729             0.091         0.112       1.23×    mma faster
ffn_up    M=64              0.056         0.108       1.93×    mma 1.9× faster
ffn_up    M=320             0.134         0.196       1.46×    mma faster
ffn_up    M=729             0.263         0.345       1.31×    mma faster
ffn_down  M=*  (F16)        unchanged   unchanged    1.00×    F16 path bypasses MMQ (sanity)
```

**ggml's mma path beats dp4a at every Q8_0 shape we hit, by 1.22–2.03×.**
Opposite of what `project_qwen3tts_int8_mma_kill` predicted — that
memory's shapes are M=1 talker decode, where mma's launch-and-fragment
overhead dominates and dp4a wins. At siglip2's M ≥ 64 wide-N batch
matmuls, mma's 16× compute density per warp pays off even on consumer
GA106. The qwen3-tts memory is **shape-specific**, not a global
GPU-level fact.

**Implication for Phase C:**

- The hand-tuned dp4a-MMQ lever (Lever 1 as scoped) is **dead**.
  ggml-mma is already winning by 1.5×+ at the M=320 cfe shapes; a
  hand-tuned dp4a couldn't even match it.
- A hand-tuned **mma**-MMQ would have to beat ggml's already-good mma.
  Currently achieving 24-28 TFLOPS ≈ 50 % of RTX 3060 INT8 peak. A
  custom kernel might claw 10-20 %, not the -5 to -7 ms cfe estimated
  in the original handoff.
- **The 11.6 ms matmul total at M=320 (cfe) is essentially the floor**
  for ggml + Q8_0×F32 + RTX 3060 + this dispatch.

The `GGML_CUDA_NO_MMA` cmake knob is left in tree as a future debug
tool; not turned on anywhere. If a future agent wants to compare again
on a different GPU (sm_75, A100, etc.), the build flag is wired and
reusable.

## Things this microbench did NOT measure

- **Per-call quantize-x cost**: `quantize_mmq_q8_1_cuda` runs once per
  matmul on the activation. At M=320 × K=1152 that's ~370k floats →
  small but nonzero. Not isolated here.
- **Cross-matmul pipeline benefits**: in the real encoder, several
  matmuls share quantized inputs (e.g., the same LN-output feeds qkv
  and o). MMQ could pre-quantize once per residual stream. Not a graph
  feature today.
- **FA on padded d_head=80**: not a mul_mat, separate analysis.
