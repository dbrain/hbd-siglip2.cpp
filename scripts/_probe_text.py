import requests
import numpy as np
import torch
from transformers import AutoTokenizer, Siglip2TextModel

prompt = "a photo of a cat"

py = requests.post("http://localhost:8890/v1/text_embeddings", data=[("prompts", prompt)]).json()["embeddings"][0]
cp = requests.post("http://localhost:18890/v1/text_embeddings", data=[("prompts", prompt)]).json()["embeddings"][0]

py = np.asarray(py)
cp = np.asarray(cp)
print(f"py first 8: {py[:8]}")
print(f"cp first 8: {cp[:8]}")
print(f"py L2 norm: {np.linalg.norm(py):.4f}")
print(f"cp L2 norm: {np.linalg.norm(cp):.4f}")
print(f"py vs cp cosine: {np.dot(py, cp) / (np.linalg.norm(py) * np.linalg.norm(cp)):.6f}")
print()

tok = AutoTokenizer.from_pretrained("/work/reference/hf/siglip2-so400m-patch16-naflex")
print(f"tokenizer.model_max_length = {tok.model_max_length}")
print(f"tokenizer.pad_token_id = {tok.pad_token_id}")

# What does padding="max_length" without explicit max_length do?
enc_default = tok([prompt], padding="max_length", return_tensors="pt", return_attention_mask=True)
print(f"default padding shape: {enc_default['input_ids'].shape}")
print(f"default first 8 ids: {enc_default['input_ids'][0][:8].tolist()}")

# Explicit 64
enc_64 = tok([prompt], padding="max_length", max_length=64, truncation=True, return_tensors="pt", return_attention_mask=True)
print(f"max_length=64 shape: {enc_64['input_ids'].shape}")
print(f"max_length=64 first 8: {enc_64['input_ids'][0][:8].tolist()}")

model = Siglip2TextModel.from_pretrained("/work/reference/hf/siglip2-so400m-patch16-naflex", torch_dtype=torch.float32).eval()

with torch.no_grad():
    out64 = model(input_ids=enc_64["input_ids"], attention_mask=enc_64["attention_mask"])
hf64 = out64.pooler_output.squeeze(0).numpy()
hf64_n = hf64 / np.linalg.norm(hf64)
print(f"HF(64) first 8: {hf64_n[:8]}")
print(f"py vs HF(64) cosine: {np.dot(py, hf64_n):.6f}")
print(f"cp vs HF(64) cosine: {np.dot(cp, hf64_n):.6f}")

# What about default-length?
try:
    with torch.no_grad():
        out_def = model(input_ids=enc_default["input_ids"], attention_mask=enc_default["attention_mask"])
    hf_def = out_def.pooler_output.squeeze(0).numpy()
    hf_def_n = hf_def / np.linalg.norm(hf_def)
    print(f"HF(default-len) first 8: {hf_def_n[:8]}")
    print(f"py vs HF(default) cosine: {np.dot(py, hf_def_n):.6f}")
except Exception as e:
    print(f"HF(default-len) FAILED: {type(e).__name__}: {e}")
