#!/usr/bin/env python3
"""
End-to-end image parity: HF (image -> Siglip2ImageProcessor -> Siglip2VisionModel)
                vs.       C++ (image -> siglip2-cli --image)

Generates a deterministic synthetic image, runs both stacks, compares cosine.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Siglip2VisionModel


def make_test_image(seed: int, w: int, h: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base[..., 0] = (xx / max(w - 1, 1)) * 255.0
    base[..., 1] = (yy / max(h - 1, 1)) * 255.0
    base[..., 2] = ((xx + yy) / max(w + h - 2, 1)) * 255.0
    base += rng.standard_normal((h, w, 3)).astype(np.float32) * 8.0
    base = np.clip(base, 0, 255).astype(np.uint8)
    return Image.fromarray(base, "RGB")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, type=Path)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--cli", required=True, type=Path)
    p.add_argument("--width", default=320, type=int)
    p.add_argument("--height", default=180, type=int)
    p.add_argument("--max-num-patches", default=256, type=int)
    p.add_argument("--pooling", choices=["mean", "probe"], default="probe")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--threshold", default=0.997, type=float)
    p.add_argument("--keep-tmp", action="store_true")
    args = p.parse_args()

    print(f"[image-parity] image={args.width}x{args.height} max_num_patches={args.max_num_patches}")

    img = make_test_image(args.seed, args.width, args.height)

    print(f"[image-parity] loading HF processor + model from {args.hf_model}")
    processor = AutoProcessor.from_pretrained(str(args.hf_model))
    model = Siglip2VisionModel.from_pretrained(str(args.hf_model), torch_dtype=torch.float32).eval()

    # HF preproc — variable resolution, padded to max_num_patches
    inputs = processor(images=[img], return_tensors="pt", max_num_patches=args.max_num_patches)
    n_h, n_w = inputs["spatial_shapes"][0].tolist()
    n_active = n_h * n_w
    print(f"[image-parity] HF spatial_shapes = {n_h}x{n_w} ({n_active} active patches)")

    with torch.no_grad():
        out = model(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs["pixel_attention_mask"],
            spatial_shapes=inputs["spatial_shapes"],
        )
    if args.pooling == "mean":
        # apply attention-mask-aware mean per HF Siglip2ForImageClassification path
        seq = out.last_hidden_state
        mask = inputs["pixel_attention_mask"][..., None].to(seq.dtype)
        hf_emb = (torch.sum(seq * mask, dim=1) / torch.sum(mask, dim=1)).squeeze(0).cpu().numpy()
    else:
        if out.pooler_output is None:
            sys.exit("HF model did not produce pooler_output (vision_use_head=False?)")
        hf_emb = out.pooler_output.squeeze(0).cpu().numpy()
    hf_emb = hf_emb.astype(np.float32)

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        if args.keep_tmp:
            td = Path("/tmp/siglip2-image-parity")
            td.mkdir(exist_ok=True)
        img_path = td / "test.png"
        img.save(img_path)
        cpp_path = td / "cpp_embedding.bin"

        cmd = [
            str(args.cli),
            "--model", str(args.gguf),
            "--image", str(img_path),
            "--max-num-patches", str(args.max_num_patches),
            "--pooling", args.pooling,
            "--out", str(cpp_path),
        ]
        print("[image-parity] running C++:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("--- stdout ---")
            print(r.stdout)
            print("--- stderr ---")
            print(r.stderr)
            sys.exit(f"siglip2-cli failed (exit {r.returncode})")
        if r.stderr.strip():
            for line in r.stderr.strip().splitlines():
                print(f"[cli] {line}")
        cpp_emb = np.fromfile(cpp_path, dtype=np.float32)

    if cpp_emb.size != hf_emb.size:
        sys.exit(f"size mismatch: HF={hf_emb.size} CPP={cpp_emb.size}")

    cos = float(np.dot(hf_emb, cpp_emb) / (np.linalg.norm(hf_emb) * np.linalg.norm(cpp_emb) + 1e-12))
    diff = hf_emb - cpp_emb
    mse = float(np.mean(diff ** 2))
    max_abs = float(np.max(np.abs(diff)))

    print()
    print(f"[image-parity] cosine    = {cos:.6f}  (threshold {args.threshold})")
    print(f"[image-parity] MSE       = {mse:.6e}")
    print(f"[image-parity] max |dif| = {max_abs:.6e}")
    print(f"[image-parity] HF first 8: {hf_emb[:8]}")
    print(f"[image-parity] CPP first 8: {cpp_emb[:8]}")

    if cos < args.threshold:
        print(f"[image-parity] FAIL: cosine {cos:.6f} < {args.threshold}")
        return 1
    print(f"[image-parity] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
