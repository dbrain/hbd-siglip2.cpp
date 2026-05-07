#!/usr/bin/env python3
"""
End-to-end benchmark: kobbler-vision (Python+PyTorch) vs siglip2-server (C++/ggml).

Measures three axes per the user spec:
  1. Endpoint latency: /v1/embeddings, /v1/text_embeddings, /v1/classify,
     /v1/classify_from_embeddings — p50/p95 over N runs.
  2. VRAM trajectory: poll nvidia-smi at startup, post-load, during burst,
     idle steady, post-unload.
  3. Time-to-reload (POST /unload + first inference cold-path).

Quality: replays the same payload through both, reports cosine on embeddings
and probs MAE on classify scores.

Run each server separately (sequential, not concurrent — we want clean VRAM
numbers per side). Both must be reachable via host IP+port.

Usage:
    python3 scripts/bench_vs_python.py \\
        --python-url http://localhost:8890 \\
        --cpp-url    http://localhost:18890 \\
        --image-path models/test_image.png \\
        --runs 20 \\
        --max-num-patches 729
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import requests


PROMPTS = [
    "a photo of a cat",
    "a photo of a dog",
    "an abstract gradient",
    "white noise on a TV",
    "a colorful landscape painting",
]


@dataclass
class Stats:
    p50_ms: float
    p95_ms: float
    mean_ms: float
    n: int


def stat(samples: list[float]) -> Stats:
    if not samples:
        return Stats(0, 0, 0, 0)
    return Stats(
        p50_ms=float(statistics.median(samples)),
        p95_ms=float(np.percentile(samples, 95)),
        mean_ms=float(statistics.mean(samples)),
        n=len(samples),
    )


def gpu_memory_mib() -> int | None:
    """Return current GPU 0 memory used (MiB), or None if nvidia-smi missing."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            timeout=5,
        ).decode().strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def wait_for(url: str, timeout: float = 60.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def time_call(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    dt = (time.perf_counter() - t0) * 1000.0
    return out, dt


def call_embeddings(base: str, image_bytes: bytes, max_num_patches: int) -> dict:
    files = {"images": ("img.png", image_bytes, "image/png")}
    data = {"max_num_patches": str(max_num_patches), "pooling": "pooler"}
    r = requests.post(f"{base}/v1/embeddings", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def call_text_embeddings(base: str, prompts: list[str]) -> dict:
    data = [("prompts", p) for p in prompts]
    r = requests.post(f"{base}/v1/text_embeddings", data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def call_classify(base: str, image_bytes: bytes, prompts: list[str], max_num_patches: int) -> dict:
    files = {"images": ("img.png", image_bytes, "image/png")}
    data = [("max_num_patches", str(max_num_patches))] + [("prompts", p) for p in prompts]
    r = requests.post(f"{base}/v1/classify", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def call_classify_from_embeddings(base: str, embs: list[list[float]], prompts: list[str]) -> dict:
    body = {"image_embeddings": embs, "prompts": prompts}
    r = requests.post(f"{base}/v1/classify_from_embeddings", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def call_unload(base: str) -> None:
    requests.post(f"{base}/unload", timeout=30)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bench_endpoints(base: str, image_bytes: bytes, runs: int, max_num_patches: int) -> dict[str, Any]:
    """Hit each endpoint `runs` times, return latency stats + a single sample of each result."""
    print(f"  warmup...", end=" ", flush=True)
    call_embeddings(base, image_bytes, max_num_patches)
    print(f"done")

    samples = {"embeddings": [], "text_embeddings": [], "classify": [], "classify_from_embeddings": []}
    last = {}

    print(f"  /v1/embeddings ({runs} runs)...", end=" ", flush=True)
    for _ in range(runs):
        out, dt = time_call(call_embeddings, base, image_bytes, max_num_patches)
        samples["embeddings"].append(dt)
    last["embeddings"] = out
    print(f"done")

    print(f"  /v1/text_embeddings ({runs} runs)...", end=" ", flush=True)
    for _ in range(runs):
        out, dt = time_call(call_text_embeddings, base, PROMPTS)
        samples["text_embeddings"].append(dt)
    last["text_embeddings"] = out
    print(f"done")

    print(f"  /v1/classify ({runs} runs)...", end=" ", flush=True)
    for _ in range(runs):
        out, dt = time_call(call_classify, base, image_bytes, PROMPTS, max_num_patches)
        samples["classify"].append(dt)
    last["classify"] = out
    print(f"done")

    print(f"  /v1/classify_from_embeddings ({runs} runs)...", end=" ", flush=True)
    img_emb = last["embeddings"]["embeddings"]
    for _ in range(runs):
        out, dt = time_call(call_classify_from_embeddings, base, img_emb, PROMPTS)
        samples["classify_from_embeddings"].append(dt)
    last["classify_from_embeddings"] = out
    print(f"done")

    return {
        "stats": {k: asdict(stat(v)) for k, v in samples.items()},
        "samples": last,
    }


def measure_reload_time(base: str, image_bytes: bytes, max_num_patches: int) -> float:
    """POST /unload, then time-first-inference."""
    print(f"  unloading...", end=" ", flush=True)
    call_unload(base)
    time.sleep(2)
    print(f"done")
    print(f"  cold inference...", end=" ", flush=True)
    out, dt = time_call(call_embeddings, base, image_bytes, max_num_patches)
    print(f"done ({dt:.1f}ms)")
    return dt


def measure_vram_trajectory(base: str, image_bytes: bytes, max_num_patches: int, runs: int) -> dict[str, int]:
    """Capture VRAM at several points: post-startup, post-warmup, post-burst, idle, post-unload."""
    out = {}
    out["post_startup"] = gpu_memory_mib() or -1
    print(f"  post-startup: {out['post_startup']} MiB")

    # Warmup forces lazy-load
    call_embeddings(base, image_bytes, max_num_patches)
    time.sleep(0.5)
    out["post_warmup"] = gpu_memory_mib() or -1
    print(f"  post-warmup:  {out['post_warmup']} MiB")

    peak = out["post_warmup"]
    for _ in range(runs):
        call_classify(base, image_bytes, PROMPTS, max_num_patches)
        v = gpu_memory_mib() or -1
        if v > peak: peak = v
    out["burst_peak"] = peak
    print(f"  burst peak:   {out['burst_peak']} MiB")

    time.sleep(2)
    out["idle_steady"] = gpu_memory_mib() or -1
    print(f"  idle steady:  {out['idle_steady']} MiB")

    call_unload(base)
    time.sleep(3)
    out["post_unload"] = gpu_memory_mib() or -1
    print(f"  post-unload:  {out['post_unload']} MiB")

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--python-url", default="http://localhost:8890",
                   help="kobbler-vision base URL")
    p.add_argument("--cpp-url", default="http://localhost:18890",
                   help="siglip2-server base URL")
    p.add_argument("--image-path", required=True, type=Path)
    p.add_argument("--runs", default=20, type=int)
    p.add_argument("--max-num-patches", default=729, type=int)
    p.add_argument("--out", type=Path, default=Path("bench_results.json"))
    p.add_argument("--skip-python", action="store_true")
    p.add_argument("--skip-cpp", action="store_true")
    p.add_argument("--lite", action="store_true",
                   help="VRAM trajectory + quality cross-check only; skip the latency loop "
                        "(useful when the GPU is shared with other workloads)")
    args = p.parse_args()

    image_bytes = args.image_path.read_bytes()
    print(f"Test image: {args.image_path} ({len(image_bytes)} bytes)")
    print(f"Prompts: {PROMPTS}")
    print(f"Runs/endpoint: {args.runs}, max_num_patches: {args.max_num_patches}")
    print()

    results: dict[str, Any] = {"config": vars(args).copy()}
    results["config"]["image_path"] = str(args.image_path)
    results["config"]["out"] = str(args.out)

    backends = []
    if not args.skip_python:
        backends.append(("python", args.python_url))
    if not args.skip_cpp:
        backends.append(("cpp", args.cpp_url))

    for name, url in backends:
        print(f"=== {name} backend at {url} ===")
        if not wait_for(f"{url}/health", timeout=60):
            print(f"  ERROR: {url}/health did not become ready")
            continue
        print(f"  /health OK")

        # NOTE: VRAM trajectory only makes sense if the server is using GPU
        # (siglip2-server CPU build will show ~0). The Python service is GPU.
        # In --lite mode we burst with a small number of runs to capture peak VRAM
        # without hammering a shared GPU.
        burst_runs = max(3, args.runs // 4) if args.lite else args.runs

        print(f"  --- VRAM trajectory ---")
        vram = measure_vram_trajectory(url, image_bytes, args.max_num_patches, burst_runs)
        results.setdefault(name, {})["vram_mib"] = vram

        if not args.lite:
            print(f"  --- endpoint latency ---")
            eb = bench_endpoints(url, image_bytes, args.runs, args.max_num_patches)
            results[name]["endpoints"] = eb["stats"]
            results[name]["sample_outputs"] = {
                k: ("<embeddings>" if "embeddings" in v else v)
                for k, v in eb["samples"].items()
            }
            print(f"  --- reload time ---")
            reload_ms = measure_reload_time(url, image_bytes, args.max_num_patches)
            results[name]["reload_ms"] = reload_ms
        else:
            print(f"  --- skipping endpoint-latency loop (--lite) ---")
        print()

    # Quality comparison: same image -> embeddings on both; cosine compare
    if "python" in results and "cpp" in results:
        print(f"=== quality cross-check ===")
        py_emb = call_embeddings(args.python_url, image_bytes, args.max_num_patches)["embeddings"][0]
        cp_emb = call_embeddings(args.cpp_url,    image_bytes, args.max_num_patches)["embeddings"][0]
        results["quality"] = {
            "embedding_cosine": cos(np.asarray(py_emb), np.asarray(cp_emb)),
        }
        py_scores = call_classify(args.python_url, image_bytes, PROMPTS, args.max_num_patches)["scores"][0]
        cp_scores = call_classify(args.cpp_url,    image_bytes, PROMPTS, args.max_num_patches)["scores"][0]
        results["quality"]["scores_mae"] = float(np.mean(np.abs(np.asarray(py_scores) - np.asarray(cp_scores))))
        results["quality"]["scores_max_diff"] = float(np.max(np.abs(np.asarray(py_scores) - np.asarray(cp_scores))))
        print(f"  embedding cosine: {results['quality']['embedding_cosine']:.6f}")
        print(f"  scores MAE      : {results['quality']['scores_mae']:.6e}")
        print(f"  scores max diff : {results['quality']['scores_max_diff']:.6e}")
        print()

    # Pretty summary
    print("=== SUMMARY ===")
    if "python" in results and "cpp" in results:
        py = results["python"]
        cp = results["cpp"]
        print(f"\n{'metric':35} {'python':>14} {'cpp':>14}")
        print("-" * 65)
        if not args.lite:
            for ep in ["embeddings", "text_embeddings", "classify", "classify_from_embeddings"]:
                ps = py["endpoints"][ep]["p50_ms"]
                cs = cp["endpoints"][ep]["p50_ms"]
                ratio = f"{ps/cs:.2f}x" if cs else "?"
                print(f"{'p50 ' + ep:35} {ps:>11.1f}ms {cs:>11.1f}ms ({ratio})")
            print()
        for k in ["post_startup", "post_warmup", "burst_peak", "idle_steady", "post_unload"]:
            pv = py["vram_mib"][k]
            cv = cp["vram_mib"][k]
            delta = (pv - cv) if pv > 0 and cv > 0 else 0
            print(f"{'VRAM ' + k + ' (MiB)':35} {pv:>14} {cv:>14}  Δ={delta:+d}")
        print()
        if not args.lite:
            print(f"{'reload time (ms)':35} {py['reload_ms']:>11.1f}ms {cp['reload_ms']:>11.1f}ms")
        if "quality" in results:
            print(f"{'embedding cosine':35} {results['quality']['embedding_cosine']:>14.6f}")
            print(f"{'scores MAE':35} {results['quality']['scores_mae']:>14.6e}")

    args.out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nFull results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
