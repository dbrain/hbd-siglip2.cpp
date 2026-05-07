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
of the 38 ms / 5-prompt p50 is CUDA kernel launch overhead (486
launches per encode × ~12-15 µs each). PyTorch beats us at ~26 ms
because it issues maybe 100-200 launches via cuDNN/cuBLAS fusion.

Megakernel collapses that 486 → ~30, putting text encode at
**~2-3 ms per prompt** (well below Python). /v1/text_embeddings and
/v1/classify_from_embeddings drop from 1.43× and 1.79× python to
sub-1×. /v1/classify (vision + text loop) closes too because the
text-loop cost dominated.

Vision (n_pos=729) is **already 1.07× python** because it's
compute-bound at that batch size — megakernel saves ~1-2 ms there
(launches are <10% of the encode). Phase A targets text. Phase B
targets vision but the ROI is much smaller; consider that phase
optional.

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

## What "phase A" looks like

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
