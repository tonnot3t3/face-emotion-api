"""
Model inference module.

ฟังก์ชันที่ใช้ใน worker process ของ ProcessPoolExecutor
- โหลด ONNX session แบบ lazy (เพราะ process ใหม่ไม่มี state จาก parent)
- ตรวจจับใบหน้า (Haar cascade) ก่อน inference
- crop ใบหน้าก่อนส่งเข้าโมเดลเพื่อความแม่นยำ
- TTA (horizontal flip) เพื่อลด noise
- preprocess รูปภาพ
- รัน inference
- postprocess softmax + argmax

หมายเหตุ: ต้องไม่ import torch หรือ transformers ในไฟล์นี้
เพื่อให้ image ของ container เล็กที่สุด — ใช้แค่ onnxruntime + numpy + Pillow + opencv
"""
from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from app.config import (
    ACTIVE_MODEL_PATH,
    EMOTION_LABELS,
    ENABLE_TTA,
    FACE_CROP_MARGIN,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)

# Global session ต่อ process (ProcessPoolExecutor reuse worker จึงโหลดครั้งเดียวต่อ worker)
_session: ort.InferenceSession | None = None
_input_name: str | None = None

# Global face cascade ต่อ process (lazy load)
_face_cascade: cv2.CascadeClassifier | None = None


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
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            str(path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        _input_name = _session.get_inputs()[0].name
    return _session


def _get_face_cascade() -> cv2.CascadeClassifier:
    """โหลด Haar cascade สำหรับ face detection แบบ singleton ต่อ process"""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError(f"ไม่สามารถโหลด Haar cascade ที่ {cascade_path}")
        _face_cascade = cascade
    return _face_cascade


def _detect_faces(image_bytes: bytes) -> Tuple[Image.Image, list]:
    """
    ตรวจจับใบหน้าในภาพ คืน (PIL image RGB ขนาดเดิม, list ของ (x, y, w, h) ใน original coords)
    ใช้ Haar cascade frontal face detector ของ OpenCV
    ทำการ detect บนภาพที่ย่อแล้วถ้ามันใหญ่ — แล้ว scale box กลับเป็น coord เดิม
    """
    img_full = Image.open(BytesIO(image_bytes))
    if img_full.mode != "RGB":
        img_full = img_full.convert("RGB")

    max_side = 800
    w, h = img_full.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img_small = img_full.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    else:
        img_small = img_full

    arr = np.asarray(img_small)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if scale != 1.0 and len(faces) > 0:
        inv = 1.0 / scale
        faces = [
            (int(x * inv), int(y * inv), int(w * inv), int(h * inv))
            for (x, y, w, h) in faces
        ]
    else:
        faces = [tuple(map(int, b)) for b in faces]

    return img_full, faces


def _detect_face(image_bytes: bytes) -> bool:
    """Backward-compat: คืน True ถ้าพบใบหน้าอย่างน้อย 1 ใบ"""
    _, faces = _detect_faces(image_bytes)
    return len(faces) > 0


def _crop_face_with_margin(img: Image.Image, box: tuple, margin_ratio: float) -> Image.Image:
    """
    Crop ใบหน้าจากภาพต้นฉบับพร้อม margin (เก็บหน้าผาก/คาง/ขมับ)
    margin_ratio ต่ำกว่า 0 → คืนภาพเดิม (disable cropping)
    """
    if margin_ratio < 0:
        return img
    x, y, w, h = box
    iw, ih = img.size
    mx = int(round(w * margin_ratio))
    my = int(round(h * margin_ratio))
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(iw, x + w + mx)
    y2 = min(ih, y + h + my)
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def _preprocess_pil(img: Image.Image) -> np.ndarray:
    """แปลง PIL Image -> tensor [1, 3, 224, 224] float32"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(NORMALIZE_MEAN, dtype=np.float32)) / np.array(
        NORMALIZE_STD, dtype=np.float32
    )
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr.astype(np.float32)


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """แปลง bytes -> tensor (backward-compat สำหรับ test/benchmark)"""
    return _preprocess_pil(Image.open(BytesIO(image_bytes)))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """numerically stable softmax"""
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def predict_from_bytes(image_bytes: bytes) -> dict:
    """
    Public API ที่ถูกเรียกใน worker process
    คืนค่า dict ที่ประกอบไปด้วย:
      - face_detected: bool — พบใบหน้าหรือไม่
      - predicted_label: emotion ที่มี probability สูงสุด (None ถ้าไม่พบใบหน้า)
      - confidence: ความน่าจะเป็นสูงสุด (None ถ้าไม่พบใบหน้า)
      - scores: list ของ {label, score} (ว่างถ้าไม่พบใบหน้า)
      - inference_time_ms
      - total_time_ms
      - message: ข้อความเพิ่มเติม
    """
    start_total = time.perf_counter()

    # 1) ตรวจจับใบหน้าก่อน — ถ้าไม่พบ ไม่ต้องรัน inference
    img_full, faces = _detect_faces(image_bytes)
    if not faces:
        total_time_ms = (time.perf_counter() - start_total) * 1000.0
        return {
            "face_detected": False,
            "predicted_label": None,
            "confidence": None,
            "scores": [],
            "inference_time_ms": 0.0,
            "total_time_ms": round(total_time_ms, 3),
            "message": "Not Emotional Try Again",
        }

    # 2) เลือกใบหน้าใหญ่ที่สุด แล้ว crop พร้อม margin
    #    การ crop ช่วยให้โมเดลเห็นเฉพาะใบหน้า (ไม่มีพื้นหลัง/เสื้อ) → แม่นขึ้นมาก
    faces.sort(key=lambda b: b[2] * b[3], reverse=True)
    face_img = _crop_face_with_margin(img_full, faces[0], FACE_CROP_MARGIN)

    # 3) Preprocess + inference (TTA: original + horizontal flip → average probs)
    session = _get_session()
    tensor_orig = _preprocess_pil(face_img)

    start_inf = time.perf_counter()
    outputs_orig = session.run(None, {_input_name: tensor_orig})
    probs_orig = _softmax(outputs_orig[0][0])

    if ENABLE_TTA:
        face_flipped = face_img.transpose(Image.FLIP_LEFT_RIGHT)
        tensor_flip = _preprocess_pil(face_flipped)
        outputs_flip = session.run(None, {_input_name: tensor_flip})
        probs_flip = _softmax(outputs_flip[0][0])
        probs = (probs_orig + probs_flip) / 2.0
    else:
        probs = probs_orig

    inference_time_ms = (time.perf_counter() - start_inf) * 1000.0
    top_idx = int(np.argmax(probs))

    scores = [
        {"label": EMOTION_LABELS[i], "score": float(probs[i])}
        for i in range(len(EMOTION_LABELS))
    ]
    scores.sort(key=lambda x: x["score"], reverse=True)

    total_time_ms = (time.perf_counter() - start_total) * 1000.0

    return {
        "face_detected": True,
        "predicted_label": EMOTION_LABELS[top_idx],
        "confidence": float(probs[top_idx]),
        "scores": scores,
        "inference_time_ms": round(inference_time_ms, 3),
        "total_time_ms": round(total_time_ms, 3),
        "message": None,
    }


def warmup() -> None:
    """
    เรียกครั้งเดียวตอน worker process boot เพื่อ load โมเดลเข้า memory
    ทำให้ request แรกไม่ช้า (โหลดทั้ง ONNX session และ Haar cascade)
    """
    _get_face_cascade()
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
