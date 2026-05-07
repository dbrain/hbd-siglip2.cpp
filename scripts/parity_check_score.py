#!/usr/bin/env python3
"""
End-to-end image-text scoring parity (Siglip2Model contrastive head).

Generates a synthetic image + a list of prompts, runs:
  - HF: AutoProcessor + Siglip2Model -> logits_per_image + probs
  - C++: siglip2-cli --image (probe pool) + siglip2-text-cli (per prompt) +
         score_image_text in Python (mirroring siglip2_score.cpp)

Compares logits + probs.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoTokenizer, Siglip2Model


def make_test_image(seed: int, w: int, h: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base[..., 0] = (xx / max(w - 1, 1)) * 255.0
    base[..., 1] = (yy / max(h - 1, 1)) * 255.0
    base[..., 2] = ((xx + yy) / max(w + h - 2, 1)) * 255.0
    base += rng.standard_normal((h, w, 3)).astype(np.float32) * 8.0
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def cpp_score(scale: float, bias: float, img_emb: np.ndarray, txt_embs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    img_n = img_emb / (np.linalg.norm(img_emb) + 1e-12)
    txt_n = txt_embs / (np.linalg.norm(txt_embs, axis=1, keepdims=True) + 1e-12)
    logits = (txt_n @ img_n) * math.exp(scale) + bias  # (n_text,) for single image
    probs = 1.0 / (1.0 + np.exp(-logits))
    return logits, probs


def run_cli(cli: Path, args: list[str]) -> bytes:
    r = subprocess.run([str(cli), *args], capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", errors="replace"))
        raise SystemExit(f"{cli.name} failed: rc={r.returncode}")
    return r.stdout


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, type=Path)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--cli", required=True, type=Path, help="siglip2-cli")
    p.add_argument("--text-cli", required=True, type=Path, help="siglip2-text-cli")
    p.add_argument("--max-num-patches", default=729, type=int)
    p.add_argument("--threshold-cos", default=0.997, type=float)
    p.add_argument("--threshold-mae", default=5e-3, type=float,
                   help="Max acceptable mean-abs-error on probabilities")
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    prompts = [
        "a colorful gradient",
        "a photo of a cat",
        "a photo of a dog",
        "an abstract painting",
        "white noise on a TV",
    ]

    print(f"[score-parity] loading HF processor + tokenizer + model")
    processor = AutoProcessor.from_pretrained(str(args.hf_model))
    tokenizer = AutoTokenizer.from_pretrained(str(args.hf_model))
    model = Siglip2Model.from_pretrained(str(args.hf_model), torch_dtype=torch.float32).eval()

    img = make_test_image(args.seed, 320, 180)

    # HF path
    image_inputs = processor(images=[img], return_tensors="pt", max_num_patches=args.max_num_patches)
    text_inputs = tokenizer(prompts, padding="max_length", max_length=model.config.text_config.max_position_embeddings,
                            truncation=True, return_tensors="pt", return_attention_mask=True)
    if "attention_mask" not in text_inputs:
        pad_id = tokenizer.pad_token_id or 0
        text_inputs["attention_mask"] = (text_inputs["input_ids"] != pad_id).long()

    with torch.no_grad():
        out = model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=image_inputs["pixel_values"],
            pixel_attention_mask=image_inputs["pixel_attention_mask"],
            spatial_shapes=image_inputs["spatial_shapes"],
        )
    hf_logits = out.logits_per_image.squeeze(0).cpu().numpy().astype(np.float32)  # (n_text,)
    hf_probs = 1.0 / (1.0 + np.exp(-hf_logits))

    # C++ path
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        img_path = td / "img.png"
        img.save(img_path)
        img_out = td / "img_emb.bin"
        run_cli(args.cli, [
            "--model", str(args.gguf),
            "--image", str(img_path),
            "--max-num-patches", str(args.max_num_patches),
            "--pooling", "probe",
            "--out", str(img_out),
        ])
        img_emb = np.fromfile(img_out, dtype=np.float32)

        txt_embs = []
        for prompt in prompts:
            enc = tokenizer([prompt], padding="max_length",
                            max_length=model.config.text_config.max_position_embeddings,
                            truncation=True, return_tensors="pt", return_attention_mask=True)
            if "attention_mask" not in enc:
                pad_id = tokenizer.pad_token_id or 0
                enc["attention_mask"] = (enc["input_ids"] != pad_id).long()
            tok_bin = td / "tok.bin"
            mask_bin = td / "mask.bin"
            txt_out = td / "txt.bin"
            enc["input_ids"][0].cpu().numpy().astype(np.int32).tofile(tok_bin)
            enc["attention_mask"][0].cpu().numpy().astype(np.int32).tofile(mask_bin)
            run_cli(args.text_cli, [
                "--model", str(args.gguf),
                "--token-ids", str(tok_bin),
                "--attention-mask", str(mask_bin),
                "--out", str(txt_out),
            ])
            txt_embs.append(np.fromfile(txt_out, dtype=np.float32))
        txt_embs = np.stack(txt_embs)  # (n_text, H)

    # Read logit_scale + logit_bias from HF model (we'll cross-check with GGUF reader from C++ side)
    scale = float(model.logit_scale.detach().cpu().item())
    bias  = float(model.logit_bias.detach().cpu().item())
    cpp_logits, cpp_probs_local = cpp_score(scale, bias, img_emb, txt_embs)

    # Compare
    print(f"[score-parity] prompts: {prompts}")
    print(f"[score-parity] logit_scale={scale:.6f} logit_bias={bias:.6f}")
    print(f"[score-parity] HF probs  : {hf_probs}")
    print(f"[score-parity] CPP probs : {cpp_probs_local}")
    print(f"[score-parity] HF logits : {hf_logits}")
    print(f"[score-parity] CPP logits: {cpp_logits}")

    cos = float(np.dot(hf_logits, cpp_logits) /
                (np.linalg.norm(hf_logits) * np.linalg.norm(cpp_logits) + 1e-12))
    mae_probs = float(np.mean(np.abs(hf_probs - cpp_probs_local)))
    max_diff_probs = float(np.max(np.abs(hf_probs - cpp_probs_local)))

    print()
    print(f"[score-parity] logits cosine = {cos:.6f}  (threshold {args.threshold_cos})")
    print(f"[score-parity] probs MAE     = {mae_probs:.6e}  (threshold {args.threshold_mae})")
    print(f"[score-parity] probs maxdiff = {max_diff_probs:.6e}")

    fail = False
    if cos < args.threshold_cos:
        print(f"[score-parity] FAIL (logits cosine)")
        fail = True
    if mae_probs > args.threshold_mae:
        print(f"[score-parity] FAIL (probs MAE)")
        fail = True
    if not fail:
        print(f"[score-parity] PASS")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
