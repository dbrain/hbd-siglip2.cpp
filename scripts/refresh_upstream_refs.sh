#!/usr/bin/env bash
# Refresh local copies of upstream reference source used to port SigLIP2.
#
# Pulls a small subset of llama.cpp (clip.cpp + SigLIP2 graph + gguf tensor
# map + convert_hf_to_gguf YoutuVL reference) and HF transformers SigLIP2
# model sources into reference/{llama.cpp,transformers}/ for offline study.
#
# These references are NOT built and are gitignored.
#
# Usage: bash scripts/refresh_upstream_refs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REF_DIR="$REPO_ROOT/reference"

LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp.git"
TRANSFORMERS_REPO="https://github.com/huggingface/transformers.git"

llama_cpp_files=(
  tools/mtmd/clip.h
  tools/mtmd/clip.cpp
  tools/mtmd/clip-graph.h
  tools/mtmd/models/siglip.cpp
  tools/mtmd/models/youtuvl.cpp
  gguf-py/gguf/tensor_mapping.py
  convert_hf_to_gguf.py
)

transformers_files=(
  src/transformers/models/siglip2/modeling_siglip2.py
  src/transformers/models/siglip2/image_processing_siglip2.py
  src/transformers/models/siglip2/configuration_siglip2.py
  src/transformers/models/siglip2/processing_siglip2.py
)

pull_subset() {
  local repo="$1"
  local target="$2"
  shift 2
  local files=("$@")

  local tmp
  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' RETURN

  echo "==> cloning $repo (depth=1) into $tmp"
  git clone --depth 1 --filter=blob:none --quiet "$repo" "$tmp"
  local sha
  sha=$(git -C "$tmp" rev-parse HEAD)

  rm -rf "$target"
  mkdir -p "$target"

  for f in "${files[@]}"; do
    local dst="$target/$f"
    mkdir -p "$(dirname "$dst")"
    cp "$tmp/$f" "$dst"
  done

  cat > "$target/MANIFEST.md" <<EOF
# Upstream reference excerpts: $(basename "$repo" .git)

Pulled from $repo
SHA: $sha
At:  $(date -u +%Y-%m-%dT%H:%M:%SZ)

Files:
$(printf '  - %s\n' "${files[@]}")

Refresh: bash scripts/refresh_upstream_refs.sh
EOF
  echo "==> $target ready (SHA $sha)"
}

pull_subset "$LLAMA_CPP_REPO"    "$REF_DIR/llama.cpp"    "${llama_cpp_files[@]}"
pull_subset "$TRANSFORMERS_REPO" "$REF_DIR/transformers" "${transformers_files[@]}"

echo "Done. References at $REF_DIR/{llama.cpp,transformers}/"
