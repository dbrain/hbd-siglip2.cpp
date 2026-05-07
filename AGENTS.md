# AGENTS.md

Coding conventions and project guide for AI agents working on `siglip2.cpp`.

## Project Overview

`siglip2.cpp` is a pure C++17 / GGML port of [`google/siglip2-so400m-patch16-naflex`](https://huggingface.co/google/siglip2-so400m-patch16-naflex). Goal: feature parity with the HuggingFace `transformers` reference — vision encoder + text encoder + multilingual tokenizer + sigmoid image-text scoring — packaged for lean GPU residency (~650 MiB Q8_0) and clean swap composition with other GPU consumers.

Initial driver: drop-in replacement for kobbler's `docker/kobbler-vision/server.py` (4 endpoints — `/v1/embeddings`, `/v1/classify`, `/v1/classify_from_embeddings`, `/v1/text_embeddings`).

## Origin & Worktree Setup

This worktree is branched off `dbrain/qwen3-tts.cpp@main` as `dbrain/siglip2-v0`. The `ggml/` submodule is itself a worktree off `dbrain/ggml@master` as `dbrain/siglip2-ggml-v0`. Many TTS-specific source files inherited from the parent are still present as reference but **not built** by this CMakeLists; cleanup pass to delete them lands once siglip2 has its own stable surface.

## Repository Layout (in flight)

```
siglip2.cpp/
  src/
    siglip2_vision.{h,cpp}     # vision encoder (M1 — stub)
    siglip2_cli.cpp            # CLI: model + image -> embedding
    gguf_loader.{h,cpp}        # GGUF backend init (inherited; namespace qwen3_tts)
    ...                        # legacy TTS files unbuilt; will be removed
  scripts/
    convert_siglip2_to_gguf.py # M1 (TODO): HF safetensors -> GGUF
    parity_check.py            # M1 (TODO): HF reference vs C++ CLI cosine compare
  reference/
    hf/                        # local HF snapshot (gitignored)
  ggml/                        # submodule worktree on dbrain/siglip2-ggml-v0
  CMakeLists.txt
  README.md
  AGENTS.md
```

## Milestones

- **M1 (current):** vision-only fp16 parity. One cosine ≥ 0.999 between HF and C++ on a fixed image.
- **M2:** probe pooling head (`MultiheadAttentionPoolingHead`) + Q8_0 quant + NaFlex preprocessing.
- **M3:** text encoder + multilingual tokenizer + sigmoid scoring (`logit_scale` + `logit_bias`).
- **M4:** HTTP server reproducing kobbler-vision's 4 endpoints.

## Build

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/siglip2-cli --model models/siglip2-so400m-naflex-f16.gguf --image test.jpg
```

GGML backend flags pass through to the submodule (`-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON`, etc.).

## Coding Conventions

- C++17, no exceptions, no RTTI
- `fprintf(stderr, ...)` for logging (not `std::cerr`)
- Methods return `bool`; error details stored in `error_msg_` member
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for macros
- Header guards: `#pragma once`
- Public types in `siglip2` namespace (legacy `qwen3_tts::GGUFLoader` retained until cleanup)

### GGML Forward-Pass Pattern

```cpp
struct ggml_cgraph * gf = build_graph(...);
ggml_backend_sched_alloc_graph(state_->sched, gf);
ggml_backend_tensor_set(input, data, 0, size);
ggml_backend_sched_graph_compute(state_->sched, gf);
ggml_backend_tensor_get(output, out_buf, 0, size);
ggml_backend_sched_reset(state_->sched);
```

Backend selection via `qwen3_tts::init_preferred_backend()` (`gguf_loader.cpp`): `IGPU -> GPU -> ACCEL -> CPU`. CPU backend is added as scheduler fallback when the runtime backend isn't CPU.

## Reference Material

- `reference/hf/siglip2-so400m-patch16-naflex/` — local HF snapshot (config, weights, tokenizer, preprocessor)
- Upstream graph template: `tools/mtmd/models/siglip.cpp` in llama.cpp (`clip_graph_siglip::build`)
- Tensor name map: `gguf-py/gguf/tensor_mapping.py:1414+` in llama.cpp (`siglip2.vision_model.*`)
- Reference converter: `convert_hf_to_gguf.py:13469` in llama.cpp (`YoutuVLVisionModel`)
- HF transformers reference: `src/transformers/models/siglip2/modeling_siglip2.py`
- HF preprocessor reference: `src/transformers/models/siglip2/image_processing_siglip2.py`

## Parity Strategy

Every C++ component lands a `scripts/` parity check that runs the HF reference and the C++ binary on identical input and compares cosine + per-element MSE. Don't ship a component without a green parity test against PyTorch.

## Known Wart

Bilinear pos-embed interpolation in ggml is **not antialiased**; HF uses `F.interpolate(antialias=True)`. Expect ~1-2% cosine drift relative to PyTorch reference; acceptable for ranking, may need a custom op for absolute parity later.

## Git Conventions

- Conventional commits: `feat(scope):`, `fix(scope):`, `docs:`
- Scopes: `vision`, `text`, `tokenizer`, `convert`, `cli`, `server`, `parity`, `build`
- Never push these branches without explicit user OK (project is local-only until usable)
- One logical change per commit
- Do not commit model files (`*.gguf`) or HF snapshots (`reference/hf/*`)
