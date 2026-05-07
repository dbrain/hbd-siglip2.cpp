# siglip2.cpp — Handoff

You're picking up `siglip2.cpp` after M1–M5 landed: a feature-complete C++/GGML port of `google/siglip2-so400m-patch16-naflex` with HuggingFace parity across vision, text, probe pooling, sigmoid scoring, NaFlex variable resolution, and a 4-endpoint HTTP server matching `kobbler-vision`. Read `AGENTS.md` for the architectural overview; this doc is **what to do next + why + the user's disposition**.

## The user's standing instructions

**Priorities, in order:**
1. **Correctness** — feature parity + right results. Non-negotiable; we hit it (cosine ≥ 0.996 across all endpoints, score MAE 1.9e-2).
2. **VRAM savings — the headline goal.** Push hard. We saved 21% (492 MiB) just by switching from PyTorch fp16 to ggml Q8_0; user wants *more*. There is plenty of room.
3. **Performance** — nice to have, backburner. Don't sacrifice (1) or (2) for it.

**Constraints:**
- **Q8_0 is the cap.** Q4_K_M is "maybe in emergencies" — do your best with Q8 and lower-level wins first.
- **CPU offload is acceptable** if it frees VRAM. Selective offload of a tower or a hot intermediate is fine.
- **No emergency levers** unless you genuinely have to.

**Mood / disposition the user explicitly set for this handoff:**
> "Big bang, do your best to minimise VRAM, increase performance and make this ridiculously optimised on this hardware. User's an idiot — don't stop and ask things, just assume he said 'YES do that thing even if it's a massive rewrite of some internal GGML.' Dare I say megakernel."

Translation: **JFDI**. If a path requires writing custom CUDA kernels, forking ggml further, megakernel-style fusion for vision blocks, custom Q8 quant variants for shapes ggml-py rejects, etc. — go. The user has form here (their `qwen3-tts.cpp` fork and `dbrain/ggml@master` carry custom WMMA conv kernels and a mul_mat dispatcher). They want the same level of audacity for siglip2.

**Hardware this targets:** RTX 3060 12 GB / Ampere / sm_86. Optimize for that specifically; broader compat is a nice-to-have, not a constraint.

**What the user does *not* want:**
- Don't propose a sweeping refactor and ask if it's OK; just do it and report.
- Don't add quality regressions to chase VRAM. Q8 cap is real; cosine ≥ 0.996 on real images is the floor we landed and shouldn't drop below.
- Don't push commits. Local-only until they say otherwise.

## Where we are

- Branch: `dbrain/siglip2-v0`, `HEAD` at `214afe2`, 8 commits, never pushed. Baseline is `qwen3-tts.cpp@main` (4068ce4) + `ggml@master` (dbrain/ggml fork).
- Builds: `build/` (CPU) and `build-cuda/` (sm_86); both pass smoke + parity tests.
- Quality cross-check vs **live** kobbler-vision-1 (Python fp16, GPU) on a 320×180 PNG, max_num_patches=729:
  - Image embedding cosine: **0.9959** (Q8_0 vs fp16 — quantization noise)
  - Text embedding cosine: **0.9999** (after the Siglip2Processor mask fix — see commit `6b45933`)
  - Sigmoid score MAE: **1.87e-2**, max diff 9%
- VRAM (model load size, GPU): Python **2347 MiB** → C++ Q8_0 **1855 MiB** = **−492 MiB / −21%**.
- Host RAM: Python ~3 GiB → C++ ~568 MiB = **5.4×** reduction (free side-win from dropping PyTorch+transformers).

## The Siglip2Processor mask gotcha

If you touch the text path or anyone reports text drift, **remember**: HF `Siglip2Processor` for text-only returns ONLY `input_ids` (no `attention_mask`), pads to 64, and `kobbler-vision` does `model.text_model(**inputs)` straight. Pad-token-0 embeddings flow through attention as if they were real tokens. **Match this.** Passing the "correct" attention mask diverges by ~0.24 cosine on short prompts. Captured in commit `6b45933`'s body and in `siglip2_server.cpp::encode_text`'s comment.

## Lever queue (priority order, payoff × effort)

### A. Antialiased CPU image resize *(quality lever; user explicit ask)*
- **File:** `src/siglip2_preproc.cpp::resize_bilinear_u8`. Currently naive bilinear, no antialias.
- **Goal:** Lift image embedding cosine from 0.996 → ≥ 0.9999 vs PyTorch.
- **Why:** PyTorch torchvision (modern) defaults `antialias=True` for `BILINEAR`; HF's Siglip2ImageProcessor (TorchvisionBackend) inherits. Without it, downsampled patches diverge slightly from HF's, propagating into the embedding.
- **How:** Separable triangular low-pass + bilinear. Per-axis kernel coefficients depend on the scale ratio `s = src/dst`. Reference: PyTorch `aten/src/ATen/native/cpu/UpSampleKernel.cpp::upsample_bilinear2d_aa`. ~150-200 LOC.
- **Risk:** Low. Bench cosine should rise; existing parity tests still pass at native (no-resize) shapes.

### B. Q8_0 the token embedding *(VRAM lever, fastest win)*
- **File:** `scripts/convert_siglip2_to_gguf.py::_is_keep_f16`.
- **Save:** 295 MiB on disk, ~145 MiB runtime VRAM.
- **How:** drop the `t.token_embd.weight` special case from `_is_keep_f16` and re-convert with `--type q8_0`. Run `parity_check_text.py` and `parity_check_score.py` to confirm cosine doesn't regress.
- **Risk:** Low-medium. Token embedding lookups give you 1 row at a time; per-row Q8_0 quant is well-behaved but worth a parity sanity check.

### C. `--vision-only` / `--text-only` server modes *(VRAM lever, biggest narrow-deployment win)*
- **File:** `src/siglip2_server.cpp`.
- **Save:** ~1+ GiB if vision-only deployment (kobbler's `do_embed_frames` path doesn't need text).
- **How:** flag-gate the text encoder + tokenizer load in `ServerState::load_model`. Endpoints needing the missing tower return 503 with a clear message.
- **Risk:** None.

### D. FlashAttention in the encoder graphs *(VRAM + perf lever; biggest if landed)*
- **Files:** `src/siglip2_vision.cpp::build_block`, `build_probe_head`; `src/siglip2_text.cpp::build_block`.
- **Save:** Activation memory for the KQ matrix (n_pos² × n_head × 4 bytes). For text n_pos=64 it's small (64×64×16×4 = 256 KiB/layer × 27 = ~7 MiB) but vision can hit n_pos=729 (~3 MiB/layer × 27 = ~83 MiB). Bigger payoff is the perf cliff: FA fuses softmax+matmul into one kernel.
- **How:** Replace the `mul_mat(K, Q) → soft_max_ext → mul_mat(V, KQ)` chain with `ggml_flash_attn_ext(Q, K, V, mask, scale, ...)` — clip.cpp's `build_attn` under `CLIP_FLASH_ATTN_TYPE_ENABLED` is the canonical pattern. Cast K/V to F16, set `ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32)` for accumulator precision.
- **Risk:** Medium. FA can change numerics on edge cases. Verify all parity scripts. Q8_0 weights × FA path may need extra care.

### E. Megakernel-style block fusion *(VRAM + perf lever; ambitious)*
- The user has prior art: `kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md` describes their TTS megakernel approach (per-layer fused kernels, shape-specialized MMVQ). For SigLIP2 vision, an entire pre-LN transformer block (attn + MLP + 2 residuals) can fuse into one CUDA kernel for the dominant shape (n_pos=729, H=1152). Eliminates intermediate-tensor allocations entirely → real VRAM and perf win.
- **Where to start:** read the qwen3-tts.cpp megakernel branch (`dbrain/megakernel-v0` of the parent repo). The pattern (MMVQ specialization for fixed shapes) translates directly to vision blocks at the production max_num_patches=729 grid.
- **Risk:** Real. This is "massive rewrite of internal GGML" territory. The user explicitly opted in for this. Expect 1-2 weeks if you go full Phase A+B.

### F. Selective CPU offload *(VRAM lever; if A-D aren't enough)*
- ggml's scheduler supports per-tensor backend assignment. Heavy weights (e.g. text tower if rarely used; token embedding) can sit on CPU with mid-graph promotion to GPU only when needed.
- **How:** explicit `ggml_backend_sched_set_tensor_backend` for chosen tensors after they're allocated.
- **Risk:** Bandwidth tax on every promotion; only worth it if A-D haven't reached the user's target.

### G. Tune scheduler arena nodes *(small VRAM lever)*
- Graph declares `max_nodes=8192`; actual graph has ~830 nodes for 27 layers. Reducing to 1024 saves a small amount of scheduler-side metadata. Probably noise but cheap.

### H. TTS source cleanup *(hygiene; not VRAM)*
- `src/audio_*.{cpp,h}`, `src/coreml_*`, `src/tts_transformer*`, `src/qwen3_tts*`, `src/qwen3tts_c_api*`, `src/text_tokenizer*` (the Qwen BPE one), `src/tokenizer_unicode*`, `tests/test_*.cpp` are all dead in the new CMakeLists. One `chore: remove inherited TTS source` commit cleans the tree.

### I. Full latency benchmark *(deferred)*
- `scripts/bench_vs_python.py` minus `--lite` covers it. Run when the user's GPU is free of megakernel work; correctness + VRAM are the load-bearing axes for now.

## Build / test cheat sheet

```bash
# CPU build
docker run --rm -v $PWD:/src -v siglip2-cpp-ccache:/root/.ccache -w /src \
  tts-qwen3-dev:builder bash -lc \
  'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF && cmake --build build -j'

# CUDA build (NEEDS --gpus all for sm_86 detection; otherwise nvcc warns + falls back)
docker run --rm --gpus all -v $PWD:/src -v siglip2-cpp-ccache:/root/.ccache -w /src \
  tts-qwen3-dev:builder bash -lc \
  'cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build-cuda -j'

# Convert HF -> GGUF (run via kobbler-vision image which has torch+transformers)
docker run --rm -v $PWD:/work -w /work kobbler-vision bash -c \
  'pip install --quiet --break-system-packages gguf 2>/dev/null
   python3 scripts/convert_siglip2_to_gguf.py \
     --input  reference/hf/siglip2-so400m-patch16-naflex \
     --output models/siglip2-so400m-naflex-q8_0.gguf \
     --type   q8_0'

# Parity tests (any kobbler-vision-image-shaped container with libgomp1):
#   parity_check_vision.py  — vision encoder vs HF Siglip2VisionModel
#   parity_check_text.py    — text encoder vs HF Siglip2TextModel
#   parity_check_score.py   — sigmoid scoring vs HF Siglip2Model
#   parity_check_tokenizer.py — sentencepiece vs HF AutoTokenizer
#   parity_check_image.py   — end-to-end image -> embedding parity
# All print "PASS" / "FAIL" with a concrete cosine + threshold.

# Live bench vs running kobbler-vision-1 (lite = VRAM + quality only)
docker run -d --rm --name siglip2-server-cuda --gpus all --network host \
  -v $PWD:/work:ro -w /work tts-qwen3-dev:builder \
  bash -lc '/work/build-cuda/siglip2-server \
    --model /work/models/siglip2-so400m-naflex-q8_0.gguf \
    --tokenizer /work/reference/hf/siglip2-so400m-patch16-naflex/tokenizer.model \
    --port 18890'
docker run --rm --gpus all --network host -v $PWD:/work -w /work kobbler-vision bash -c \
  'pip install --quiet --break-system-packages requests 2>/dev/null
   python3 scripts/bench_vs_python.py \
     --python-url http://localhost:8890 \
     --cpp-url    http://localhost:18890 \
     --image-path models/test_image.png \
     --lite --max-num-patches 729'
```

## Working assumptions

- The submodule worktrees (`siglip2.cpp` and `siglip2.cpp/ggml`) are dedicated — modify ggml freely. The user has prior form forking ggml for a related project (`dbrain/ggml@master`); you have the same latitude here.
- Commits at logical milestones; never push without explicit OK.
- The 7-commit history on the branch is meaningful — keep that hygiene (one feat/fix per logical change, descriptive bodies). User reviews via `git log`.
- If you ship a megakernel-class change, write a `HANDOFF-<name>.md` documenting it the way the user did for `qwen3-tts.cpp` (see `kobbler/docker/tts-qwen3-dev/HANDOFF-megakernel-v0.md`). Future-you will thank you.

## What "done" looks like

User's bar: "ridiculously optimized on this hardware." Concretely:
- VRAM significantly below 1.85 GiB Q8 baseline (target: dream is sub-1 GiB resident on the vision-only path).
- Cosine ≥ 0.999 on real images (with the antialiased resize landed).
- No quality regression from current Q8.
- Performance equal to or better than the Python fp16 service.

If you hit any of those, ship a commit + update this doc's "Where we are" section. The next agent (or future-you) will pick up from there.

Now go.
