#!/usr/bin/env python3
"""
Vision encoder parity: HuggingFace Siglip2VisionModel vs siglip2-cli (M1).

Generates a deterministic random pixel_values tensor at the native 16x16 patch
grid, runs both paths, and reports cosine + MSE between the mean-pooled
embeddings. Exits non-zero if cosine < threshold.

Both paths feed the same pixel_values, so this isolates the model's forward
pass (preprocessing parity is a separate concern, M2+).

Usage (from the siglip2.cpp worktree root):
    python3 scripts/parity_check_vision.py \\
        --hf-model reference/hf/siglip2-so400m-patch16-naflex \\
        --gguf models/siglip2-so400m-naflex-f16.gguf \\
        --cli ./build/siglip2-cli \\
        --threshold 0.997
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import Siglip2VisionModel


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, type=Path,
                   help="Local HF SigLIP2 model directory (config + safetensors)")
    p.add_argument("--gguf", required=True, type=Path,
                   help="Converted GGUF file matching the HF model")
    p.add_argument("--cli", required=True, type=Path,
                   help="Path to siglip2-cli binary")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--pooling", choices=["mean", "probe"], default="probe",
                   help="probe = matches HF outputs.pooler_output (default); "
                        "mean = matches outputs.last_hidden_state.mean(dim=1)")
    p.add_argument("--shape", type=str, default=None,
                   help="non-square patch grid as H,W (e.g. 24,16 for a 24x16=384 patch image). "
                        "Default: native square grid from config.num_patches.")
    p.add_argument("--threshold", default=0.997, type=float,
                   help="Minimum cosine similarity for pass (default 0.999)")
    p.add_argument("--keep-tmp", action="store_true",
                   help="Keep intermediate .bin files for debugging")
    args = p.parse_args()

    if not args.cli.exists():
        sys.exit(f"CLI not found: {args.cli}  (build it with: cmake --build build)")
    if not args.gguf.exists():
        sys.exit(f"GGUF not found: {args.gguf}")
    if not args.hf_model.exists():
        sys.exit(f"HF model dir not found: {args.hf_model}")

    print(f"[parity] loading HF model from {args.hf_model}")
    model = Siglip2VisionModel.from_pretrained(
        str(args.hf_model), torch_dtype=torch.float32
    ).eval()

    cfg = model.config
    feat = 3 * cfg.patch_size * cfg.patch_size  # 768

    if args.shape:
        h_str, w_str = args.shape.split(",")
        n_h, n_w = int(h_str), int(w_str)
    else:
        side = int(round(cfg.num_patches ** 0.5))
        n_h, n_w = side, side
    n_patches = n_h * n_w
    print(f"[parity] grid={n_h}x{n_w} ({n_patches} patches) patch_size={cfg.patch_size} hidden={cfg.hidden_size}")

    # Deterministic random pixel_values (already-rescaled+normalized range).
    g = torch.Generator().manual_seed(args.seed)
    pixel_values = torch.randn(1, n_patches, feat, generator=g, dtype=torch.float32)
    pixel_attention_mask = torch.ones(1, n_patches, dtype=torch.long)
    spatial_shapes = torch.tensor([[n_h, n_w]], dtype=torch.long)

    print(f"[parity] running HF forward (pooling={args.pooling})")
    with torch.no_grad():
        out = model(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
    if args.pooling == "mean":
        hf_emb = out.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
    else:  # probe
        if out.pooler_output is None:
            sys.exit("HF model did not produce pooler_output (vision_use_head=False?)")
        hf_emb = out.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        if args.keep_tmp:
            td = Path("/tmp/siglip2-parity")
            td.mkdir(exist_ok=True)
        pv_path = td / "pixel_values.bin"
        cpp_path = td / "cpp_embedding.bin"

        # Save pixel_values as raw fp32 [n_patches, feat] (CLI expects ne[0]=feat innermost,
        # which matches numpy's row-major when shape is (n_patches, feat)).
        pixel_values.squeeze(0).cpu().numpy().astype(np.float32).tofile(pv_path)

        cmd = [
            str(args.cli),
            "--model", str(args.gguf),
            "--pixel-values", str(pv_path),
            "--shape", f"{n_h},{n_w}",
            "--pooling", args.pooling,
            "--out", str(cpp_path),
        ]
        print("[parity] running C++:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("--- CLI stdout ---")
            print(r.stdout)
            print("--- CLI stderr ---")
            print(r.stderr)
            sys.exit(f"siglip2-cli failed (exit {r.returncode})")
        if r.stderr.strip():
            print("[parity] cli stderr:")
            for line in r.stderr.strip().splitlines():
                print(f"  {line}")

        cpp_emb = np.fromfile(cpp_path, dtype=np.float32)
        if cpp_emb.size != hf_emb.size:
            sys.exit(f"size mismatch: HF={hf_emb.size} CPP={cpp_emb.size}")

    cos = float(np.dot(hf_emb, cpp_emb) / (np.linalg.norm(hf_emb) * np.linalg.norm(cpp_emb) + 1e-12))
    diff = hf_emb - cpp_emb
    mse = float(np.mean(diff ** 2))
    max_abs = float(np.max(np.abs(diff)))

    print()
    print(f"[parity] cosine    = {cos:.6f}  (threshold {args.threshold})")
    print(f"[parity] MSE       = {mse:.6e}")
    print(f"[parity] max |dif| = {max_abs:.6e}")
    print(f"[parity] HF first 8: {hf_emb[:8]}")
    print(f"[parity] CPP first 8: {cpp_emb[:8]}")

    if cos < args.threshold:
        print(f"[parity] FAIL: cosine {cos:.6f} < threshold {args.threshold}")
        return 1
    print(f"[parity] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
