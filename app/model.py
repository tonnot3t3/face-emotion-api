"""
Model inference module.

ฟังก์ชันที่ใช้ใน worker process ของ ProcessPoolExecutor
- โหลด ONNX session แบบ lazy (เพราะ process ใหม่ไม่มี state จาก parent)
- preprocess รูปภาพ
- รัน inference
- postprocess softmax + argmax

หมายเหตุ: ต้องไม่ import torch หรือ transformers ในไฟล์นี้
เพื่อให้ image ของ container เล็กที่สุด — ใช้แค่ onnxruntime + numpy + Pillow
"""
from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.config import (
    ACTIVE_MODEL_PATH,
    EMOTION_LABELS,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)

# Global session ต่อ process (ProcessPoolExecutor reuse worker จึงโหลดครั้งเดียวต่อ worker)
_session: ort.InferenceSession | None = None
_input_name: str | None = None


def _get_session(model_path: str | Path | None = None) -> ort.InferenceSession:
    """โหลด ONNX session แบบ singleton ต่อ process"""
    global _session, _input_name
    if _session is None:
        path = Path(model_path) if model_path else ACTIVE_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"ไม่พบไฟล์โมเดลที่ {path}. "
                "กรุณารัน scripts/convert_to_onnx.py และ scripts/quantize.py ก่อน"
            )
        # ปรับ session option ให้เร็วที่สุดบน CPU
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # intra_op = thread ภายใน operator, inter_op = thread ระหว่าง operator
        # ตั้งเป็น 1 เพราะเราใช้ multi-process แล้ว ไม่ต้องการให้แต่ละ process แย่ง CPU กัน
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            str(path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        _input_name = _session.get_inputs()[0].name
    return _session


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """
    แปลงรูปภาพ bytes -> tensor [1, 3, 224, 224] แบบ float32

    ใช้ algorithm เดียวกับ ViTImageProcessor แต่ implement เองเพื่อไม่ต้องใช้ transformers
    """
    img = Image.open(BytesIO(image_bytes))
    # บางภาพอาจเป็น RGBA, L (grayscale) ฯลฯ — แปลงเป็น RGB ทั้งหมด
    if img.mode != "RGB":
        img = img.convert("RGB")
    # resize ให้เป็น 224x224 ด้วย bilinear (ตรงกับ default ของ ViTImageProcessor)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    # normalize: pixel/255 -> (x - mean) / std
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(NORMALIZE_MEAN, dtype=np.float32)) / np.array(
        NORMALIZE_STD, dtype=np.float32
    )
    # HWC -> CHW -> NCHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr.astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """numerically stable softmax"""
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def predict_from_bytes(image_bytes: bytes) -> dict:
    """
    Public API ที่ถูกเรียกใน worker process
    คืนค่า dict ที่ประกอบไปด้วย:
      - predicted_label
      - confidence
      - scores: list ของ {label, score}
      - inference_time_ms
      - total_time_ms
    """
    start_total = time.perf_counter()

    session = _get_session()

    # Preprocess
    tensor = _preprocess(image_bytes)

    # Inference
    start_inf = time.perf_counter()
    outputs = session.run(None, {_input_name: tensor})
    inference_time_ms = (time.perf_counter() - start_inf) * 1000.0

    # Postprocess
    logits = outputs[0][0]  # shape: (num_classes,)
    probs = _softmax(logits)
    top_idx = int(np.argmax(probs))

    scores = [
        {"label": EMOTION_LABELS[i], "score": float(probs[i])}
        for i in range(len(EMOTION_LABELS))
    ]
    # เรียงจาก score มากไปน้อยให้ใช้งานง่าย
    scores.sort(key=lambda x: x["score"], reverse=True)

    total_time_ms = (time.perf_counter() - start_total) * 1000.0

    return {
        "predicted_label": EMOTION_LABELS[top_idx],
        "confidence": float(probs[top_idx]),
        "scores": scores,
        "inference_time_ms": round(inference_time_ms, 3),
        "total_time_ms": round(total_time_ms, 3),
    }


def warmup() -> None:
    """
    เรียกครั้งเดียวตอน worker process boot เพื่อ load โมเดลเข้า memory
    ทำให้ request แรกไม่ช้า
    """
    session = _get_session()
    dummy = np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    session.run(None, {_input_name: dummy})


def get_model_meta() -> Tuple[str, float]:
    """คืนค่า (path, size_mb) ของโมเดลปัจจุบัน"""
    path = ACTIVE_MODEL_PATH
    if not path.exists():
        return str(path), 0.0
    size_mb = path.stat().st_size / (1024 * 1024)
    return str(path), round(size_mb, 2)
