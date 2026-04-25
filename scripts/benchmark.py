"""
Benchmark: เปรียบเทียบ latency และ size ของ
    1) PyTorch original (FP32)
    2) ONNX FP32
    3) ONNX INT8 (dynamic quantized)

Usage:
    python -m scripts.benchmark [--num-runs 50]

ผลลัพธ์ออกมาเป็นตาราง markdown ที่เอาไปใส่ในรายงานได้เลย
และเซฟเป็น docs/benchmark_results.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import (  # noqa: E402
    HF_MODEL_ID,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    ONNX_FP32_PATH,
    ONNX_QUANTIZED_PATH,
)

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)


def make_dummy_input() -> np.ndarray:
    """สุ่มภาพ 224x224 RGB และ normalize"""
    rng = np.random.default_rng(42)
    arr = rng.random((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    arr = (arr - np.array(NORMALIZE_MEAN, dtype=np.float32)) / np.array(NORMALIZE_STD, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0).astype(np.float32)


def bench_pytorch(num_runs: int) -> dict:
    print("[PyTorch] โหลดโมเดล ...")
    model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID).eval()
    dummy = torch.from_numpy(make_dummy_input())

    # warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    # หา size ของไฟล์ pytorch_model.bin / model.safetensors ใน HF cache
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    size_mb = 0.0
    for fname in ["model.safetensors", "pytorch_model.bin"]:
        try:
            info = fs.info(f"{HF_MODEL_ID}/{fname}")
            size_mb = info["size"] / (1024 * 1024)
            break
        except Exception:
            continue

    return summarize("PyTorch FP32", times, size_mb)


def bench_onnx(model_path: Path, name: str, num_runs: int) -> dict:
    if not model_path.exists():
        print(f"[{name}] ข้าม — ไม่พบ {model_path}")
        return {"name": name, "skipped": True}

    print(f"[{name}] โหลด ONNX session ...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = make_dummy_input()

    # warmup
    for _ in range(3):
        sess.run(None, {input_name: dummy})

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    return summarize(name, times, size_mb)


def summarize(name: str, times: list[float], size_mb: float) -> dict:
    return {
        "name": name,
        "size_mb": round(size_mb, 2),
        "mean_ms": round(statistics.mean(times), 2),
        "median_ms": round(statistics.median(times), 2),
        "p95_ms": round(np.percentile(times, 95), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "runs": len(times),
    }


def render_table(results: list[dict]) -> str:
    """สร้างตาราง markdown"""
    rows = [
        "| Model | Size (MB) | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |",
        "|-------|-----------|-----------|-------------|----------|----------|----------|",
    ]
    for r in results:
        if r.get("skipped"):
            rows.append(f"| {r['name']} | _skipped_ | - | - | - | - | - |")
            continue
        rows.append(
            f"| {r['name']} | {r['size_mb']} | {r['mean_ms']} | {r['median_ms']} "
            f"| {r['p95_ms']} | {r['min_ms']} | {r['max_ms']} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=50)
    args = parser.parse_args()

    results = [
        bench_pytorch(args.num_runs),
        bench_onnx(ONNX_FP32_PATH, "ONNX FP32", args.num_runs),
        bench_onnx(ONNX_QUANTIZED_PATH, "ONNX INT8 (Quantized)", args.num_runs),
    ]

    table = render_table(results)
    print("\n" + table)

    # คำนวณ speedup vs baseline
    baseline = next((r for r in results if r["name"] == "PyTorch FP32" and not r.get("skipped")), None)
    if baseline:
        print("\n=== Speedup vs PyTorch FP32 ===")
        for r in results:
            if r.get("skipped") or r["name"] == "PyTorch FP32":
                continue
            speedup = baseline["mean_ms"] / r["mean_ms"]
            size_ratio = baseline["size_mb"] / r["size_mb"] if r["size_mb"] > 0 else 0
            print(f"  {r['name']:30s} -> {speedup:.2f}x faster, {size_ratio:.2f}x smaller")

    # เซฟเป็นไฟล์
    out_md = DOCS_DIR / "benchmark_results.md"
    out_md.write_text(
        f"# Benchmark Results\n\n"
        f"Runs per model: {args.num_runs}\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    out_json = DOCS_DIR / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nบันทึกไปที่ {out_md} และ {out_json}")


if __name__ == "__main__":
    main()
