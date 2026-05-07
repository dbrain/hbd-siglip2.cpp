#!/usr/bin/env python3
"""
Text encoder parity: HuggingFace Siglip2TextModel vs siglip2-text-cli.

Tokenizes a fixed prompt with HF (padding='max_length' to 64), runs both
paths, compares cosine on the projection_size-d embedding.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, Siglip2TextModel


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, type=Path)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--cli", required=True, type=Path,
                   help="Path to siglip2-text-cli binary")
    p.add_argument("--prompt", default="a photo of two cats sleeping on a couch", type=str)
    p.add_argument("--threshold", default=0.997, type=float)
    args = p.parse_args()

    print(f"[text-parity] loading HF tokenizer + model from {args.hf_model}")
    tok = AutoTokenizer.from_pretrained(str(args.hf_model))
    model = Siglip2TextModel.from_pretrained(str(args.hf_model), torch_dtype=torch.float32).eval()

    max_len = model.config.max_position_embeddings  # 64 for so400m-naflex
    enc = tok([args.prompt], padding="max_length", max_length=max_len, truncation=True,
              return_tensors="pt", return_attention_mask=True)
    print(f"[text-parity] prompt='{args.prompt}'")
    input_ids = enc["input_ids"]
    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"]
    else:
        # Tokenizer didn't emit a mask — derive from pad_token_id.
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        attention_mask = (input_ids != pad_id).long()
    print(f"[text-parity] input_ids shape={tuple(input_ids.shape)} mask sum={attention_mask.sum().item()}")

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    if out.pooler_output is None:
        sys.exit("HF model did not produce pooler_output for text")
    hf_emb = out.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        tok_bin = td / "tokens.bin"
        mask_bin = td / "mask.bin"
        out_bin = td / "embed.bin"

        input_ids[0].cpu().numpy().astype(np.int32).tofile(tok_bin)
        attention_mask[0].cpu().numpy().astype(np.int32).tofile(mask_bin)

        cmd = [
            str(args.cli),
            "--model", str(args.gguf),
            "--token-ids", str(tok_bin),
            "--attention-mask", str(mask_bin),
            "--out", str(out_bin),
        ]
        print("[text-parity] running C++:", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("--- stdout ---"); print(r.stdout)
            print("--- stderr ---"); print(r.stderr)
            sys.exit(f"siglip2-text-cli failed (exit {r.returncode})")
        if r.stderr.strip():
            for line in r.stderr.strip().splitlines():
                print(f"[cli] {line}")
        cpp_emb = np.fromfile(out_bin, dtype=np.float32)

    if cpp_emb.size != hf_emb.size:
        sys.exit(f"size mismatch: HF={hf_emb.size} CPP={cpp_emb.size}")

    cos = float(np.dot(hf_emb, cpp_emb) / (np.linalg.norm(hf_emb) * np.linalg.norm(cpp_emb) + 1e-12))
    diff = hf_emb - cpp_emb
    mse = float(np.mean(diff ** 2))
    max_abs = float(np.max(np.abs(diff)))

    print()
    print(f"[text-parity] cosine    = {cos:.6f}  (threshold {args.threshold})")
    print(f"[text-parity] MSE       = {mse:.6e}")
    print(f"[text-parity] max |dif| = {max_abs:.6e}")
    print(f"[text-parity] HF first 8: {hf_emb[:8]}")
    print(f"[text-parity] CPP first 8: {cpp_emb[:8]}")

    if cos < args.threshold:
        print(f"[text-parity] FAIL")
        return 1
    print(f"[text-parity] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
