#!/usr/bin/env python3
"""
Tokenizer parity: HuggingFace AutoTokenizer vs siglip2.cpp's sentencepiece wrapper.

Tests a battery of prompts (English + multilingual). For each:
- HF: AutoTokenizer.encode with padding="max_length", max_length=64
- C++: siglip2-text-cli --text on the same prompt; reads back the embedding
       (we can't easily intercept token_ids in the CLI, but we can check
        end-to-end embedding parity which subsumes tokenizer correctness)

Strategy: feed each prompt through both pipelines (HF model+tokenizer, C++ CLI
with C++ tokenizer) and compare embedding cosine. If tokenizers diverge,
embeddings will diverge.
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


PROMPTS = [
    "a photo of a cat",
    "a photo of a dog sitting on a bench in the park",
    "an abstract painting in the style of Picasso",
    "white noise on an old TV screen",
    "走在京都的小巷里",                           # Chinese
    "おにぎりを食べながら本を読む",              # Japanese
    "Ein Foto eines roten Autos auf einer Brücke", # German
    "Una persona caminando por la playa al atardecer", # Spanish
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", required=True, type=Path)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--text-cli", required=True, type=Path)
    p.add_argument("--threshold", default=0.997, type=float)
    args = p.parse_args()

    print("[tok-parity] loading HF model + tokenizer")
    tok = AutoTokenizer.from_pretrained(str(args.hf_model))
    model = Siglip2TextModel.from_pretrained(str(args.hf_model), torch_dtype=torch.float32).eval()
    spm_path = args.hf_model / "tokenizer.model"
    if not spm_path.exists():
        sys.exit(f"sentencepiece model not found: {spm_path}")

    max_len = model.config.max_position_embeddings  # 64

    failures = 0
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        for prompt in PROMPTS:
            # HF
            enc = tok([prompt], padding="max_length", max_length=max_len, truncation=True,
                      return_tensors="pt", return_attention_mask=True)
            input_ids = enc["input_ids"]
            if "attention_mask" in enc:
                attn_mask = enc["attention_mask"]
            else:
                pad_id = tok.pad_token_id or 0
                attn_mask = (input_ids != pad_id).long()
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask)
            hf_emb = out.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)

            # C++
            cpp_out = td / "embed.bin"
            r = subprocess.run([
                str(args.text_cli),
                "--model", str(args.gguf),
                "--tokenizer", str(spm_path),
                "--text", prompt,
                "--out", str(cpp_out),
            ], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  CLI stderr: {r.stderr.strip()}")
                sys.exit(f"siglip2-text-cli failed for prompt: {prompt!r}")
            cpp_emb = np.fromfile(cpp_out, dtype=np.float32)

            cos = float(np.dot(hf_emb, cpp_emb) /
                        (np.linalg.norm(hf_emb) * np.linalg.norm(cpp_emb) + 1e-12))
            tag = "PASS" if cos >= args.threshold else "FAIL"
            print(f"[tok-parity] cosine={cos:.6f} {tag}  prompt={prompt!r}")
            if cos < args.threshold:
                failures += 1
                # Diagnostic: dump HF token_ids to see what the C++ side might have produced differently
                print(f"               HF tokens (first 20): {input_ids[0][:20].tolist()}")

    if failures > 0:
        print(f"[tok-parity] {failures}/{len(PROMPTS)} FAIL")
        return 1
    print(f"[tok-parity] all {len(PROMPTS)} prompts PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
