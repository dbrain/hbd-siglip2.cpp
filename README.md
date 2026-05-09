# siglip2.cpp

> ⚠️ PSAs/TLDRs from the potato that told Claude to see if we could make siglip2 friendlier to run after poking at [dbrain/qwen3-tts.cpp](https://github.com/dbrain/qwen3-tts.cpp) which is a fork of a fork I also potatoed
> - LLM generated noise - I'm a software engineer but I won't pretend to know anything about this space.
> - Entirely tested and targetted at my hardware (RTX 3060 12GB, AMD misc), may explode on anything else or run slower. Likely any CUDA device would benefit but I'm no nvidiaologist, 0 attention paid outside of "works for me".
> - Performance gains are ok - but thin margins - VRAM in my case was the more important bit, although my actual python server was running at around 2400 MiB after doing some magic so saving are overplayed based on "copy paste siglip2 example" VRAM
> - Rest of the README.md is Claude pretending I'm a real human

Runs Google's [`siglip2-so400m-patch16-naflex`](https://huggingface.co/google/siglip2-so400m-patch16-naflex) in pure C++/CUDA via ggml. Same outputs as the HF PyTorch reference, in roughly a third of the GPU memory at Q8. Ships an HTTP server.

## Why this exists

PyTorch + HF `transformers` is currently the only first-class runtime for SigLIP2-naflex. That's a fine answer for batch jobs and notebooks, but it's a heavy tenant if you're trying to share a single consumer GPU between an LLM, an STT model, a TTS model, and a vision encoder.

This is **not** a "fix the broken upstream" port — the HF runtime works fine. It's a *"there wasn't an alternative"* port: leaner VRAM, faster cold-load, and a worker-isolation model that returns to **0 MiB** on idle.

## At a glance

RTX 3060 12 GB · Q8_0 · 729-patch image · 5 prompts · 50-run p50.

| Metric | HF transformers (fp16) | siglip2.cpp (Q8_0) | Δ |
|---|---|---|---|
| Peak VRAM (whole device, model active) | 4285 MiB | 1785 MiB | **2.4× tighter** |
| Worker resident (per-process) | n/a — model lives in the python process | **1666 MiB** | — |
| Idle resident after `/v1/admin/unload` | 1956 MiB *(PyTorch CUDA context stays)* | **0 MiB** *(worker SIGKILLed, full reclaim)* | — |
| Cold load (unload + first inference) | 11.1 s | **0.69 s** | **16× faster** |
| `/v1/embeddings` p50 | 54.6 ms | 43.9 ms | 0.80× |
| `/v1/text_embeddings` p50 | 28.7 ms | 22.3 ms | 0.78× |
| `/v1/classify` p50 | 71.1 ms | 57.5 ms | 0.81× |
| `/v1/classify_from_embeddings` p50 | 22.3 ms | 21.0 ms | 0.94× |
| Vision cosine vs HF | (reference) | **≥ 0.9995** on real images | < 0.001 drift |
| Sigmoid score MAE vs HF | (reference) | 0.020 | ranking-stable |

The latency wins are mild — 15–20 % on most endpoints, parity on `classify_from_embeddings`. **The real wins are footprint, cold-load, and unload semantics.** If you're running pure batch embedding extraction across thousands of images, this isn't more compelling than HF; the pitch is for the *resident-on-shared-GPU* case.

## Who is this for?

- Self-hosted apps coexisting on a single 12 GB GPU (LLM + STT + TTS + vision encoder, all on one card).
- Lazy-loaded vision: a 0.7 s cold-load makes "unload between requests" actually viable.
- Anyone who wants zero-shot image–text scoring without shipping `python` + `torch` + `transformers` + CUDA libs as part of the deploy.
- Edge / embedded scenarios where the PyTorch CUDA tax (~1.8 GiB resident before you load anything) is a non-starter.

## What is SigLIP2-naflex?

[SigLIP2](https://arxiv.org/abs/2502.14786) is an image–text dual encoder trained with a sigmoid loss (rather than softmax-contrastive). The `naflex` variant accepts a **variable patch count** — preprocessing decides how many patches an image gets based on its native resolution and aspect ratio, instead of padding to a fixed grid like CLIP / SigLIP-v1. This codebase ports `so400m-patch16-naflex` (1.1 B params); see HF for variants.

## Quick start

### 1. Get the source

```bash
git clone --recurse-submodules https://github.com/dbrain/hbd-siglip2.cpp.git
cd siglip2.cpp
```

### 2. Convert the model

The HF snapshot needs to be converted to GGUF once. You'll also need the SentencePiece `tokenizer.model` from the same snapshot at runtime.

```bash
pip install transformers torch huggingface_hub safetensors numpy

huggingface-cli download google/siglip2-so400m-patch16-naflex \
  --local-dir hf-snapshot

python scripts/convert_siglip2_to_gguf.py \
  --input  hf-snapshot \
  --output models/siglip2-so400m-naflex-q8_0.gguf \
  --type   q8_0

cp hf-snapshot/tokenizer.model models/tokenizer.model
```

`--type` accepts `f16`, `f32`, `q8_0`, `q4_k_m`. For `q5_k_m` or arbitrary requantization, use the `siglip2-quantize` binary built alongside the server.

### 3a. Run with Docker (recommended)

A `Dockerfile` and `docker-compose.yml` are included.

```bash
# Optional: trim CUDA arch list to your card to shrink the image / context VRAM.
# Default builds for Turing→Hopper (75;80;86;89;90).
export SIGLIP2_CUDA_ARCHS=86   # 86=Ampere, 89=Ada, 90=Hopper, 120=Blackwell

docker compose up -d --build
curl -sf http://localhost:8890/health
```

The compose file mounts `./models:/models:ro` and sets `LAZY_LOAD=1`, `IDLE_UNLOAD_SECONDS=300`, and `SIGLIP2_WORKER_ISOLATION=1` for the "no GPU at idle" pattern. Override env or `command:` to taste — see the comments inside `docker-compose.yml`.

### 3b. Build & run manually

Requires CMake ≥ 3.14, a C++17 compiler, and (for CUDA) nvcc 12.x with toolkit + drivers. `sentencepiece`, `cpp-httplib`, and `nlohmann/json` are pulled by `FetchContent` at configure time.

```bash
# CUDA build
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda -j$(nproc)

# Or CPU-only (supported but unmeasured — perf claims here are CUDA-only)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run
build-cuda/siglip2-server \
  --model     models/siglip2-so400m-naflex-q8_0.gguf \
  --tokenizer models/tokenizer.model \
  --port      8890
```

Useful env: `SIGLIP2_WORKER_ISOLATION=0` to keep the encoder in-process; `IDLE_UNLOAD_SECONDS=N` to auto-kill the worker after N idle seconds; `LAZY_LOAD=1` to skip loading at startup.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/embeddings` | Image → embedding |
| `POST /v1/text_embeddings` | Text(s) → embedding(s), batched |
| `POST /v1/classify` | Image + texts → sigmoid scores |
| `POST /v1/classify_from_embeddings` | Pre-computed image embedding + texts → scores |
| `POST /v1/admin/load`, `/v1/admin/unload` | Spawn / kill the GPU worker |
| `GET  /health` | Health + worker pid |

Endpoints that take images use `multipart/form-data` (field name `images`, supports batched); responses are JSON.

## Quants

| Quant | On-disk | Notes |
|---|---|---|
| `f16` | 2.2 GB | Reference. |
| `q8_0` | 1.4 GB | **Production default.** Cosine ≥ 0.9995 vs HF on real images. |
| `q5_k_m` | 809 MB | Untested at scale. |
| `q4_k_m` | 734 MB | Acceptable for ranking; absolute scores drift more than Q8. |

VRAM is mostly insensitive to weight-quant choice — activations dominate at 1.1 B params. Q8 vs Q4 is a download-size decision, not a residency decision.

## Architecture notes

- **Worker isolation** (default ON): the HTTP parent process holds no GPU state. A forked worker owns the CUDA primary context, ggml backend buffers, and encoder weights. `/v1/admin/unload` SIGKILLs the worker; the kernel reclaims every byte. Subsequent requests respawn (~700 ms cold-load). IPC overhead is < 0.3 % on warm calls. Disable with `SIGLIP2_WORKER_ISOLATION=0`.
- **Custom CUDA fusions** (`siglip2_megakernel`): three op-hook fusions over ggml-cuda — LayerNorm-with-affine, QKV split-permute-pad-cast, and pointwise `(mm + 1D bias) + 2D residual`. Drop ~880 kernel launches per `/v1/classify`. Toggle with `SIGLIP2_DISABLE_MEGAKERNEL=1`.
- **Bypass-scheduler graph cache**: per-shape `ggml_gallocr_t` + direct `ggml_backend_graph_compute`. Tensor pointers are stable across calls, so ggml-cuda's CUDA-graphs path warms up after 2 calls per shape.
- **Multilingual tokenizer**: SentencePiece, Gemma-style `.model`. Batched encode for text embeddings.
- **Concurrent vision + text streams** (server-side): each encoder gets its own CUDA stream so `/v1/classify` overlaps the vision encode with text encodes.

## Hardware tested

All numbers in this README come from a single **RTX 3060 12 GB** (Ampere GA106). The code targets generic ggml-cuda; Turing / Ada / Hopper / Blackwell should work but aren't measured. No multi-GPU, no NVLink. CPU-only build is supported but not benchmarked.

## Status

| | |
|---|---|
| Vision encoder | ✅ HF parity, all quants |
| Text encoder | ✅ HF parity, batched |
| Tokenizer | ✅ multilingual SentencePiece |
| Sigmoid scoring | ✅ MAE ~0.02 vs HF |
| HTTP server | ✅ 4 endpoints + admin |
| Worker isolation | ✅ default ON |
| CPU-only build | ✅ supported, untuned |
| CI | ❌ |
| Non-CUDA accelerators (Metal, Vulkan) | ❌ |

Parity is verified by `scripts/parity_check_{vision,text,score,image,tokenizer}.py` against the HF reference.

## Acknowledgements

- Build scaffolding lifted from sibling project [`qwen3-tts.cpp`](https://github.com/dbrain/qwen3-tts.cpp) — same author.
- Vision graph template from [llama.cpp `clip.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/clip.cpp).
- Tensor naming conventions from [llama.cpp `gguf-py`](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/tensor_mapping.py).
- Built on the [`dbrain/ggml`](https://github.com/dbrain/ggml) fork (a few op-hook + kernel additions on top of upstream).

## License

MIT — see [LICENSE](LICENSE).
