# syntax=docker/dockerfile:1.7
#
# Multi-stage CUDA build for siglip2-server. Compiles in stage 1, ships a
# slim runtime in stage 2.
#
# Build:
#   docker build --build-arg SIGLIP2_CUDA_ARCHS=86 -t siglip2.cpp:local .
#
# Run (assuming you've already converted a GGUF and have it in ./models/):
#   docker run --gpus all -p 8890:8890 \
#     -v $PWD/models:/models:ro \
#     siglip2.cpp:local
#
# See README.md and docker-compose.yml for the recommended operational
# setup (lazy-load + idle-unload + worker-isolation).

# ─── build stage ──────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates curl ccache pkg-config

# CUDA archs to compile kernels for. Each arch baked in adds ~15-20 MiB of
# resident PTX/cubin to every CUDA process's primary context — i.e. ~80 MiB
# of "stuck at idle" VRAM unless you're running with
# SIGLIP2_WORKER_ISOLATION=1 (default ON, which kills the whole subprocess
# on /unload, dropping all that to zero).
#
# Default covers Turing through Hopper. Override for a single-arch host:
#   75;80;86;89;90  default — Turing/Ampere/Ada/Hopper
#   86              Ampere — RTX 3000-series, A-series datacenter
#   89              Ada    — RTX 4000-series, L40
#   90              Hopper — H100 / H200
#
# Pre-Volta excluded: ggml's wmma/mma kernels need __CUDA_ARCH__ >= 700.
# sm_100 (Blackwell) needs CUDA 12.8+ — bump the base image first if you
# add it.
ARG SIGLIP2_CUDA_ARCHS="75;80;86;89;90"

WORKDIR /src
COPY . /src/siglip2.cpp/

ENV CCACHE_DIR=/root/.ccache CCACHE_MAXSIZE=20G

# Wire ccache in front of g++/nvcc. cmake prepends ccache to compile
# invocations; ccache hashes (compiler, flags, preprocessed source) so
# flag-only changes that don't affect a TU give cache hits.
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S /src/siglip2.cpp -B /src/siglip2.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DSIGLIP2_SERVER=ON \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${SIGLIP2_CUDA_ARCHS}" \
    && cmake --build /src/siglip2.cpp/build --target siglip2-server -j"$(nproc)" \
    && ccache -s


# ─── runtime stage ────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# nvidia/cuda:runtime carries libcudart + libcublas + libcublasLt + libcurand.
# Extras:
#   libgomp1        — ggml uses OpenMP; libgomp.so.1 isn't in cuda:runtime
#   curl            — healthcheck (curl -sf /health)
#   ca-certificates — kept for any HTTPS the runtime may want
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends \
        curl libgomp1 ca-certificates

COPY --from=builder /src/siglip2.cpp/build/siglip2-server /usr/local/bin/siglip2-server

# All siglip2-server config can come from env (see `siglip2-server --help`).
# Defaults below assume a host bind-mount of converted weights at /models.
ENV HOST=0.0.0.0 \
    PORT=8890 \
    MODEL_PATH=/models/siglip2-so400m-naflex-q8_0.gguf \
    TOKENIZER_PATH=/models/tokenizer.model \
    SIGLIP2_WORKER_ISOLATION=1

EXPOSE 8890

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -sf http://localhost:8890/health || exit 1

ENTRYPOINT ["/usr/local/bin/siglip2-server"]
