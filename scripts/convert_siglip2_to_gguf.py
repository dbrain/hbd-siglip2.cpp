#!/usr/bin/env python3
"""
Convert HuggingFace SigLIP2 to GGUF.

Tested against `google/siglip2-so400m-patch16-naflex`.

Tensor naming (siglip2.cpp convention, distinct from clip.cpp's `v.blk.*`):
  Vision tower:
    v.patch_embd.{weight,bias}            <- vision_model.embeddings.patch_embedding.*
    v.position_embd.weight                <- vision_model.embeddings.position_embedding.weight
    v.blk.{i}.ln1.{weight,bias}           <- vision_model.encoder.layers.{i}.layer_norm1.*
    v.blk.{i}.attn_qkv.{weight,bias}      <- concat([q_proj, k_proj, v_proj], out-dim) for layer i
    v.blk.{i}.attn_o.{weight,bias}        <- self_attn.out_proj.*
    v.blk.{i}.ln2.{weight,bias}           <- layer_norm2.*
    v.blk.{i}.ffn_up.{weight,bias}        <- mlp.fc1.*
    v.blk.{i}.ffn_down.{weight,bias}      <- mlp.fc2.*
    v.post_ln.{weight,bias}               <- vision_model.post_layernorm.*
  Vision pooling head (Siglip2MultiheadAttentionPoolingHead, "probe"):
    v.head.probe                          <- vision_model.head.probe (shape [1,1,H])
    v.head.attn_q.{weight,bias}           <- split from head.attention.in_proj_{weight,bias}[:H]
    v.head.attn_k.{weight,bias}           <- split [H:2H]
    v.head.attn_v.{weight,bias}           <- split [2H:3H]
    v.head.attn_o.{weight,bias}           <- head.attention.out_proj.*
    v.head.ln.{weight,bias}               <- head.layernorm.*
    v.head.ffn_up.{weight,bias}           <- head.mlp.fc1.*
    v.head.ffn_down.{weight,bias}         <- head.mlp.fc2.*
  Text tower:
    t.token_embd.weight                   <- text_model.embeddings.token_embedding.weight
    t.position_embd.weight                <- text_model.embeddings.position_embedding.weight
    t.blk.{i}.{ln1,attn_qkv,attn_o,ln2,ffn_up,ffn_down}.{weight,bias}
    t.final_ln.{weight,bias}              <- text_model.final_layer_norm.*
    t.head.{weight,bias}                  <- text_model.head.* (Linear projection)
  Top-level:
    mm.logit_scale                        <- logit_scale
    mm.logit_bias                         <- logit_bias

Quantization policy:
  - Layer norms (.ln*, .final_ln, .post_ln, .ffn_norm)  -> F32
  - Biases                                              -> F32
  - Tiny tensors (probe, logit_scale, logit_bias, position_embd) -> F32
  - Token embedding (256K x H)                          -> output_type (Q8_0 verified parity-safe; ~145 MiB VRAM win vs F16)
  - All other 2D weights                                -> output_type (F16 / Q8_0)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm

import gguf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


ARCH = "siglip2"
HIDDEN_SIZE_DEFAULT = 1152


class Siglip2Converter:
    def __init__(
        self,
        input_dir: Path,
        output_path: Path,
        output_type: str,
        include_vision: bool,
        include_text: bool,
    ):
        self.input_dir = input_dir
        self.output_path = output_path
        self.output_type = output_type
        self.include_vision = include_vision
        self.include_text = include_text

        with open(self.input_dir / "config.json", "r", encoding="utf-8") as f:
            self.config: dict[str, Any] = json.load(f)
        with open(self.input_dir / "preprocessor_config.json", "r", encoding="utf-8") as f:
            self.preproc: dict[str, Any] = json.load(f)

        self.vision_cfg = self.config.get("vision_config", {})
        self.text_cfg = self.config.get("text_config", {})

    # -- name mapping ---------------------------------------------------------

    def _map_name(self, hf_name: str) -> str | None:
        # Vision encoder layers
        if hf_name.startswith("vision_model.encoder.layers."):
            if not self.include_vision:
                return None
            parts = hf_name.split(".")
            # parts: ["vision_model","encoder","layers", "{i}", ...rest]
            i = parts[3]
            rest = ".".join(parts[4:])
            sub = self._map_block(rest)
            if sub is None:
                return None
            return f"v.blk.{i}.{sub}"
        # Text encoder layers
        if hf_name.startswith("text_model.encoder.layers."):
            if not self.include_text:
                return None
            parts = hf_name.split(".")
            i = parts[3]
            rest = ".".join(parts[4:])
            sub = self._map_block(rest)
            if sub is None:
                return None
            return f"t.blk.{i}.{sub}"

        # Vision non-block tensors
        if hf_name.startswith("vision_model."):
            if not self.include_vision:
                return None
            return self._map_vision_non_block(hf_name)
        # Text non-block tensors
        if hf_name.startswith("text_model."):
            if not self.include_text:
                return None
            return self._map_text_non_block(hf_name)

        # Top-level
        if hf_name == "logit_scale":
            return "mm.logit_scale"
        if hf_name == "logit_bias":
            return "mm.logit_bias"

        return None

    @staticmethod
    def _map_block(rest: str) -> str | None:
        # rest is e.g. "self_attn.q_proj.weight", "layer_norm1.bias", "mlp.fc1.weight"
        # NOTE: q_proj/k_proj/v_proj are intentionally NOT mapped here. They get
        # concatenated into a single fused `attn_qkv.{weight,bias}` tensor at
        # convert() time so the encoder runs one mul_mat instead of three per
        # layer (≈108 fewer kernel launches across both towers).
        m = {
            "layer_norm1.weight": "ln1.weight",
            "layer_norm1.bias": "ln1.bias",
            "layer_norm2.weight": "ln2.weight",
            "layer_norm2.bias": "ln2.bias",
            "self_attn.out_proj.weight": "attn_o.weight",
            "self_attn.out_proj.bias": "attn_o.bias",
            "mlp.fc1.weight": "ffn_up.weight",
            "mlp.fc1.bias": "ffn_up.bias",
            "mlp.fc2.weight": "ffn_down.weight",
            "mlp.fc2.bias": "ffn_down.bias",
        }
        return m.get(rest)

    @staticmethod
    def _map_vision_non_block(hf_name: str) -> str | None:
        m = {
            "vision_model.embeddings.patch_embedding.weight": "v.patch_embd.weight",
            "vision_model.embeddings.patch_embedding.bias": "v.patch_embd.bias",
            "vision_model.embeddings.position_embedding.weight": "v.position_embd.weight",
            "vision_model.post_layernorm.weight": "v.post_ln.weight",
            "vision_model.post_layernorm.bias": "v.post_ln.bias",
            # Probe head: layernorm + mlp; attention parts are split in convert(), not mapped here
            "vision_model.head.layernorm.weight": "v.head.ln.weight",
            "vision_model.head.layernorm.bias": "v.head.ln.bias",
            "vision_model.head.mlp.fc1.weight": "v.head.ffn_up.weight",
            "vision_model.head.mlp.fc1.bias": "v.head.ffn_up.bias",
            "vision_model.head.mlp.fc2.weight": "v.head.ffn_down.weight",
            "vision_model.head.mlp.fc2.bias": "v.head.ffn_down.bias",
            "vision_model.head.probe": "v.head.probe",
            "vision_model.head.attention.out_proj.weight": "v.head.attn_o.weight",
            "vision_model.head.attention.out_proj.bias": "v.head.attn_o.bias",
            # in_proj_{weight,bias} handled specially (split into Q/K/V) in convert()
        }
        return m.get(hf_name)

    @staticmethod
    def _map_text_non_block(hf_name: str) -> str | None:
        m = {
            "text_model.embeddings.token_embedding.weight": "t.token_embd.weight",
            "text_model.embeddings.position_embedding.weight": "t.position_embd.weight",
            "text_model.final_layer_norm.weight": "t.final_ln.weight",
            "text_model.final_layer_norm.bias": "t.final_ln.bias",
            "text_model.head.weight": "t.head.weight",
            "text_model.head.bias": "t.head.bias",
        }
        return m.get(hf_name)

    # -- dtype policy ---------------------------------------------------------

    def _is_keep_f32(self, ggml_name: str) -> bool:
        if ggml_name.endswith(".bias"):
            return True
        if ".ln" in ggml_name or ".post_ln" in ggml_name or ".final_ln" in ggml_name:
            return True
        if ggml_name in ("v.position_embd.weight", "t.position_embd.weight", "v.head.probe",
                         "mm.logit_scale", "mm.logit_bias"):
            return True
        return False

    def _is_keep_f16(self, ggml_name: str) -> bool:
        # Currently no tensors are forced to F16. The token embedding was previously
        # kept F16 out of caution about per-row Q8 lookup noise; parity check shows
        # Q8_0 holds (cosine ≥ 0.999 vs HF), so we let it ride at output_type.
        del ggml_name
        return False

    def _convert_dtype(self, t: torch.Tensor, ggml_name: str) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
        if t.dtype == torch.bfloat16:
            data = t.float().numpy()
        else:
            data = t.numpy()

        if self._is_keep_f32(ggml_name):
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32

        if self._is_keep_f16(ggml_name):
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16

        if self.output_type == "f32":
            return data.astype(np.float32), gguf.GGMLQuantizationType.F32

        if self.output_type == "f16":
            return data.astype(np.float16), gguf.GGMLQuantizationType.F16

        if self.output_type == "q8_0":
            data = data.astype(np.float32)
            try:
                quantized = gguf.quants.quantize(data, gguf.GGMLQuantizationType.Q8_0)
                return quantized, gguf.GGMLQuantizationType.Q8_0
            except Exception as e:
                logger.warning("Q8_0 failed for %s: %s; falling back to F16", ggml_name, e)
                return data.astype(np.float16), gguf.GGMLQuantizationType.F16

        raise ValueError(f"unknown output_type: {self.output_type}")

    # -- tensor iteration -----------------------------------------------------

    def _iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        files = sorted(self.input_dir.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors in {self.input_dir}")
        for sf in files:
            logger.info("Loading %s", sf.name)
            with safe_open(sf, framework="pt", device="cpu") as f:
                for name in f.keys():
                    yield name, f.get_tensor(name)

    # -- metadata -------------------------------------------------------------

    def _add_metadata(self, w: gguf.GGUFWriter) -> None:
        w.add_name("siglip2-so400m-patch16-naflex")
        w.add_type(gguf.GGUFType.MODEL)

        ftype = {
            "f32": gguf.LlamaFileType.ALL_F32,
            "f16": gguf.LlamaFileType.MOSTLY_F16,
            "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        }[self.output_type]
        w.add_file_type(ftype)
        w.add_quantization_version(gguf.GGML_QUANT_VERSION)

        # Towers included in this file
        w.add_bool(f"{ARCH}.has_vision", self.include_vision)
        w.add_bool(f"{ARCH}.has_text", self.include_text)

        # Vision config
        if self.include_vision:
            v = self.vision_cfg
            w.add_uint32(f"{ARCH}.vision.embedding_length", v.get("hidden_size", HIDDEN_SIZE_DEFAULT))
            w.add_uint32(f"{ARCH}.vision.feed_forward_length", v.get("intermediate_size", 4304))
            w.add_uint32(f"{ARCH}.vision.attention.head_count", v.get("num_attention_heads", 16))
            w.add_uint32(f"{ARCH}.vision.block_count", v.get("num_hidden_layers", 27))
            w.add_uint32(f"{ARCH}.vision.patch_size", v.get("patch_size", self.preproc.get("patch_size", 16)))
            w.add_uint32(f"{ARCH}.vision.num_channels", v.get("num_channels", 3))
            w.add_uint32(f"{ARCH}.vision.num_patches", v.get("num_patches", 256))
            w.add_float32(f"{ARCH}.vision.layer_norm_eps", v.get("layer_norm_eps", 1e-6))
            w.add_string(f"{ARCH}.vision.hidden_act", v.get("hidden_act", "gelu_pytorch_tanh"))

            # Preprocessor defaults (max_num_patches in config is the trained-against default)
            w.add_uint32(f"{ARCH}.preproc.default_max_num_patches", self.preproc.get("max_num_patches", 256))
            w.add_array(f"{ARCH}.preproc.image_mean", self.preproc.get("image_mean", [0.5, 0.5, 0.5]))
            w.add_array(f"{ARCH}.preproc.image_std", self.preproc.get("image_std", [0.5, 0.5, 0.5]))
            w.add_float32(f"{ARCH}.preproc.rescale_factor", self.preproc.get("rescale_factor", 1.0 / 255.0))

        # Text config
        if self.include_text:
            t = self.text_cfg
            w.add_uint32(f"{ARCH}.text.embedding_length", t.get("hidden_size", HIDDEN_SIZE_DEFAULT))
            w.add_uint32(f"{ARCH}.text.feed_forward_length", t.get("intermediate_size", 4304))
            w.add_uint32(f"{ARCH}.text.attention.head_count", t.get("num_attention_heads", 16))
            w.add_uint32(f"{ARCH}.text.block_count", t.get("num_hidden_layers", 27))
            w.add_uint32(f"{ARCH}.text.vocab_size", t.get("vocab_size", 256000))
            w.add_uint32(f"{ARCH}.text.max_position_embeddings", t.get("max_position_embeddings", 64))
            w.add_uint32(f"{ARCH}.text.projection_size", t.get("projection_size", HIDDEN_SIZE_DEFAULT))
            w.add_float32(f"{ARCH}.text.layer_norm_eps", t.get("layer_norm_eps", 1e-6))
            w.add_string(f"{ARCH}.text.hidden_act", t.get("hidden_act", "gelu_pytorch_tanh"))

    # -- main convert ---------------------------------------------------------

    def convert(self) -> None:
        logger.info("Converting %s -> %s [%s]", self.input_dir, self.output_path, self.output_type)
        logger.info("include_vision=%s include_text=%s", self.include_vision, self.include_text)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = gguf.GGUFWriter(path=None, arch=ARCH)
        self._add_metadata(writer)

        # Cache the head's in_proj tensors so we can split Q/K/V at write time
        head_in_proj_weight: torch.Tensor | None = None
        head_in_proj_bias: torch.Tensor | None = None

        # Per-layer Q/K/V buffer for the QKV fusion path. Indexed by
        # (tower, layer_idx) -> {"q.w": ..., "q.b": ..., ...}. Once all three
        # weights and three biases are seen for a layer we emit one fused
        # attn_qkv.weight + attn_qkv.bias and drop the partials.
        qkv_buf: dict[tuple[str, int], dict[str, torch.Tensor]] = {}

        n_written = 0
        n_skipped = 0
        unmapped: list[str] = []

        H = self.vision_cfg.get("hidden_size", HIDDEN_SIZE_DEFAULT)

        def maybe_flush_qkv(tower: str, idx: int):
            """If all six Q/K/V weight+bias parts are present for this layer,
            stack them along output dim and emit a single fused tensor."""
            nonlocal n_written
            key = (tower, idx)
            slot = qkv_buf.get(key)
            if not slot:
                return
            need = {"q.w", "k.w", "v.w", "q.b", "k.b", "v.b"}
            if not need.issubset(slot.keys()):
                return
            # Stack [q,k,v] along out-axis (axis 0 in HF (out, in) convention).
            qkv_w = torch.cat([slot["q.w"], slot["k.w"], slot["v.w"]], dim=0)
            qkv_b = torch.cat([slot["q.b"], slot["k.b"], slot["v.b"]], dim=0)
            base = f"{tower}.blk.{idx}.attn_qkv"
            wd, wt = self._convert_dtype(qkv_w.contiguous(), f"{base}.weight")
            bd, bt = self._convert_dtype(qkv_b.contiguous(), f"{base}.bias")
            writer.add_tensor(f"{base}.weight", wd, raw_dtype=wt)
            writer.add_tensor(f"{base}.bias",   bd, raw_dtype=bt)
            n_written += 2
            qkv_buf.pop(key)

        # HF block paths look like vision_model.encoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
        def parse_qkv(hf_name: str) -> tuple[str, int, str] | None:
            for hf_prefix, tower in (("vision_model.encoder.layers.", "v"),
                                     ("text_model.encoder.layers.",   "t")):
                if not hf_name.startswith(hf_prefix):
                    continue
                if (tower == "v" and not self.include_vision) or \
                   (tower == "t" and not self.include_text):
                    return None
                rest = hf_name[len(hf_prefix):]
                parts = rest.split(".")
                if len(parts) < 4:
                    continue
                idx = int(parts[0])
                # parts: [idx, "self_attn", "{q|k|v}_proj", "{weight|bias}"]
                if parts[1] != "self_attn" or parts[2] not in ("q_proj", "k_proj", "v_proj"):
                    continue
                qkv_letter = parts[2][0]                # 'q'|'k'|'v'
                wb         = "w" if parts[3] == "weight" else "b"
                return tower, idx, f"{qkv_letter}.{wb}"
            return None

        for hf_name, tensor in tqdm(list(self._iter_tensors()), desc="convert"):
            if hf_name == "vision_model.head.attention.in_proj_weight":
                if self.include_vision:
                    head_in_proj_weight = tensor
                continue
            if hf_name == "vision_model.head.attention.in_proj_bias":
                if self.include_vision:
                    head_in_proj_bias = tensor
                continue

            qkv = parse_qkv(hf_name)
            if qkv is not None:
                tower, idx, slot_key = qkv
                qkv_buf.setdefault((tower, idx), {})[slot_key] = tensor
                maybe_flush_qkv(tower, idx)
                continue

            ggml_name = self._map_name(hf_name)
            if ggml_name is None:
                # vision tensors when not include_vision (and vice versa) shouldn't be unmapped
                if (hf_name.startswith("vision_model.") and not self.include_vision) or \
                   (hf_name.startswith("text_model.") and not self.include_text):
                    n_skipped += 1
                else:
                    unmapped.append(hf_name)
                    n_skipped += 1
                continue

            data, dtype = self._convert_dtype(tensor, ggml_name)
            writer.add_tensor(ggml_name, data, raw_dtype=dtype)
            n_written += 1

        # Sanity: every buffered layer must have flushed.
        if qkv_buf:
            keys = ", ".join(f"{t}.blk.{i}" for (t, i) in sorted(qkv_buf.keys()))
            raise RuntimeError(f"Unflushed QKV buffers (incomplete attn weights): {keys}")

        # Split + write the probe head's packed in_proj
        if self.include_vision:
            if head_in_proj_weight is None or head_in_proj_bias is None:
                raise RuntimeError(
                    "Expected vision_model.head.attention.in_proj_{weight,bias} for the probe head; not found."
                )
            assert head_in_proj_weight.shape == (3 * H, H), \
                f"unexpected in_proj_weight shape {head_in_proj_weight.shape}, want (3*{H},{H})"
            q_w, k_w, v_w = head_in_proj_weight.split(H, dim=0)
            q_b, k_b, v_b = head_in_proj_bias.split(H, dim=0)
            for tag, w_t, b_t in (("q", q_w, q_b), ("k", k_w, k_b), ("v", v_w, v_b)):
                wd, wt = self._convert_dtype(w_t.contiguous(), f"v.head.attn_{tag}.weight")
                bd, bt = self._convert_dtype(b_t.contiguous(), f"v.head.attn_{tag}.bias")
                writer.add_tensor(f"v.head.attn_{tag}.weight", wd, raw_dtype=wt)
                writer.add_tensor(f"v.head.attn_{tag}.bias", bd, raw_dtype=bt)
                n_written += 2

        if unmapped:
            logger.warning("Unmapped tensors (%d):", len(unmapped))
            for u in unmapped[:20]:
                logger.warning("  %s", u)
            if len(unmapped) > 20:
                logger.warning("  ... (+%d more)", len(unmapped) - 20)
            raise RuntimeError(f"{len(unmapped)} tensors had no mapping; refusing to write incomplete GGUF")

        logger.info("Writing %d tensors to %s (skipped %d)", n_written, self.output_path, n_skipped)
        writer.write_header_to_file(path=self.output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        logger.info("Done.")


def main():
    p = argparse.ArgumentParser(description="Convert HF SigLIP2 to GGUF")
    p.add_argument("--input", "-i", type=Path, required=True, help="HF model directory")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output .gguf path")
    p.add_argument("--type", "-t", choices=["f16", "f32", "q8_0"], default="f16",
                   help="Output dtype for matmul weights (default: f16)")
    p.add_argument("--vision-only", action="store_true", help="Skip text tower")
    p.add_argument("--text-only", action="store_true", help="Skip vision tower")
    args = p.parse_args()

    if args.vision_only and args.text_only:
        sys.exit("--vision-only and --text-only are mutually exclusive")

    include_vision = not args.text_only
    include_text = not args.vision_only

    Siglip2Converter(
        input_dir=args.input,
        output_path=args.output,
        output_type=args.type,
        include_vision=include_vision,
        include_text=include_text,
    ).convert()


if __name__ == "__main__":
    main()
