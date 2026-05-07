# siglip2.cpp

Pure C++/GGML implementation of [`google/siglip2-so400m-patch16-naflex`](https://huggingface.co/google/siglip2-so400m-patch16-naflex). Vision encoder + text encoder + multilingual tokenizer + sigmoid image-text scoring, packaged for **lean GPU residency** and **Q8_0 quantization**.

> **Status:** early scaffolding. Milestone 1 (vision-only fp16 parity) in progress.

## Why

PyTorch reference uses ~2.4 GiB VRAM peak per the HF `transformers` runtime. This port targets ~650 MiB Q8_0 with no quality regression on cosine ranking and minor drift acceptable for sigmoid scoring, letting you keep the encoder resident alongside other GPU consumers (LLM, STT, TTS) on a 12 GB card.

## Goals

- 1:1 feature parity with `transformers`' `Siglip2Model` (within fuzzy reason — bilinear pos-embed interp without antialias may drift cosine ~1-2%).
- Q8_0 weights via `ggml`'s block-quantized format.
- Reproduce the 4-endpoint surface that exists today in PyTorch services like kobbler's `kobbler-vision`: `/v1/embeddings`, `/v1/classify`, `/v1/classify_from_embeddings`, `/v1/text_embeddings`.
- Standalone binary + Python parity harness alongside.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

GGML backend options pass through (e.g. `-DGGML_CUDA=ON`).

## Status by Milestone

- [ ] **M1**: vision encoder fp16 parity — one cosine ≥ 0.999 vs HF on a fixed image
- [ ] **M2**: probe pooling head + Q8_0 + NaFlex preprocessing
- [ ] **M3**: text encoder + multilingual tokenizer + sigmoid scoring
- [ ] **M4**: HTTP server with 4 endpoints

## Acknowledgements

- Scaffolding cribbed from [`qwen3-tts.cpp`](https://github.com/dbrain/qwen3-tts.cpp) (sibling project, same author).
- Vision graph template from [llama.cpp clip.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/clip.cpp).
- Multimodal tensor naming from [llama.cpp gguf-py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/tensor_mapping.py).
