# Agent: start here

You're picking up siglip2.cpp **post-perf**. The kernel work is done; what's
left is structural: get this thing released to prod.

## What this project is

A pure C++/GGML port of `google/siglip2-so400m-patch16-naflex` — vision
encoder + text encoder + sentencepiece tokenizer + sigmoid image-text
scoring. Drop-in HTTP replacement for `kobbler-vision-1` (the python
service in `~/dev/kobbler/docker/kobbler-vision/`). Endpoints, request
shapes, response shapes match 1:1.

## State of play

| Axis | Status |
|---|---|
| API parity vs python | ✅ all 4 endpoints, only gap is `return_last_hidden=true` (no kobbler caller uses it) |
| Quality | ✅ all 4 parity scripts ≥ 0.999 cosine on Q8_0 prod GGUF |
| Performance | ✅ /v1/classify 0.81× python on RTX 3060, **at the GPU compute ceiling per profiling** — see `HANDOFF-perf-next.md` "Final outcomes" |
| Lifecycle | ✅ cold-start 1.3 s, unload 36 ms, reload 0.79 s, 162 MiB residual after unload (CUDA driver context, unavoidable) |
| Concurrency | ✅ 400/400 concurrent classify pass clean |
| **Where the source lives** | ⚠️ **on a feature branch of `dbrain/qwen3-tts.cpp` (yes, weird)** |
| **Production Docker image** | ❌ doesn't exist yet — kobbler still runs the python service |
| **README** | ⚠️ stale — says "early scaffolding" |

## The weird thing about this project (read first)

**This directory is a clone of `github.com/dbrain/qwen3-tts.cpp` with the
`dbrain/siglip2-v0` branch checked out.** `main` is the original Qwen3-TTS
implementation. siglip2 source coexists with TTS source on the same repo,
sharing `qwen3_tts::GGUFLoader`, the `ggml` submodule (`dbrain/ggml@master`,
shared with qwen3-tts.cpp), and the build helpers. The directory name
`~/dev/siglip2.cpp` is just what dbrain named the local checkout.

This is the **first thing to fix**. Do NOT productionize from this branch
— it ties siglip2 releases to qwen3-tts churn and namespaces siglip2 code
under `qwen3_tts::*`. Recommended fork plan + namespace cull is in
`HANDOFF-prod.md` §1.

## Your punch list

`HANDOFF-prod.md` is the comprehensive doc. Numbered punch list at §5.
Highlights in order:

1. **Fork to `github.com/dbrain/siglip2.cpp`** as its own repo. Cull the
   qwen3-tts source per §1's checklist. Rename `qwen3_tts::` → `siglip2::`
   in `gguf_loader.{h,cpp}`. ggml submodule keeps pointing at
   `dbrain/ggml.git@master` (shared with qwen3-tts is fine — see §2).

2. **README rewrite** (§3). Current is 38 lines of "early scaffolding"
   and "Milestone 1 in progress" — both stale.

3. **Production Docker image** at `kobbler/docker/siglip2-cpp/Dockerfile`.
   Pattern: copy `kobbler/docker/tts-qwen3/Dockerfile`. Multi-stage
   builder (nvidia/cuda devel) → runtime (nvidia/cuda runtime) with the
   binary + tokenizer baked in. §4 has the full template + the
   `vision-cpp:` compose service block ready to drop into
   `kobbler/docker-compose.yml`.

4. **Dev-vs-prod parity bench** (§4 last paragraph). Build the prod
   image, run the perf bench against it, confirm numbers match
   `HANDOFF-perf-next.md` "Final outcomes" within ~5 %. This is the
   gate before flipping the kobbler API client.

5. **Flip `vision` → `vision-cpp`** in compose, watch logs for 24-48 h,
   then retire the python image.

## Where NOT to spend time

- **Don't relitigate perf.** The handoffs document every lever tried,
  every receipt, every aborted target. Specifically: cuBLASLt int8 GEMM
  was gated and aborted (microbench column in `bench/microbench_mmq.cpp`
  is the receipt; saved in tree for sm89+ re-test). Native d=72 FA via
  tile kernel was tried via `SIGLIP2_DISABLE_FA_PAD=1` and was 27%
  SLOWER on Ampere. Q4_K_M failed the 0.999 floor. Custom MMQ for M=320
  rejected based on ggml-mma SM-saturation. Per-anchor profiling
  decisively shows every kernel is at or near the GPU compute ceiling.

- **Don't push without the user's explicit OK.** Standing rule. Bank
  commits locally.

## How to verify nothing regresses during productionization

```bash
# Build
docker run --rm --gpus all -v $PWD:/src -w /src tts-qwen3-dev:builder \
  cmake --build build-cuda -j$(nproc)

# All 4 parity scripts must PASS (cosine ≥ 0.999)
docker run --rm --gpus all -v $PWD:/work -v /tmp/siglip-libs:/cuda-libs \
  -v /tmp/parity_runner.sh:/parity_runner.sh -w /work \
  kobbler-vision bash /parity_runner.sh

# Lifecycle smoke test — see HANDOFF-perf-next.md final outcomes for
# expected numbers (cold start, unload time, post-unload VRAM, etc.)
```

## Doc index

- `HANDOFF.md` — original M1/M2 handoff (mostly historical)
- `HANDOFF-megakernel-v0.md` — Phase A0-A3 megakernel landing
- `HANDOFF-megakernel-v0-phaseC-microbench.md` — microbench harness origin
- `HANDOFF-megakernel-v0-phaseC-next.md` — Phase C exploration receipts
- **`HANDOFF-perf-next.md`** — perf-next chapter, **closed** — final
  outcomes, every lever tried, every env knob introduced
- **`HANDOFF-prod.md`** — comprehensive prod plan; your reference doc

Now go.
