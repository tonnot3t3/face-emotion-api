"""Unit tests สำหรับ app/model.py — ใช้ ONNX session โดยตรง"""
from __future__ import annotations

import pytest

from app.config import ACTIVE_MODEL_PATH, EMOTION_LABELS

# ข้ามทั้งไฟล์ถ้ายังไม่ได้ build โมเดล (เช่น ใน CI ขั้น lint-only)
pytestmark = pytest.mark.skipif(
    not ACTIVE_MODEL_PATH.exists(),
    reason=f"โมเดลที่ {ACTIVE_MODEL_PATH} ยังไม่ถูกสร้าง — รัน scripts/convert + quantize ก่อน",
)


def test_predict_returns_valid_structure(sample_image_bytes):
    """ผลลัพธ์ต้องมีคีย์ครบและค่าอยู่ใน range ที่ถูกต้อง"""
    from app.model import predict_from_bytes

    result = predict_from_bytes(sample_image_bytes)

    # มีคีย์ครบ
    expected_keys = {"predicted_label", "confidence", "scores", "inference_time_ms", "total_time_ms"}
    assert expected_keys.issubset(result.keys())

    # label ต้องเป็นหนึ่งใน 7 emotion
    assert result["predicted_label"] in EMOTION_LABELS

    # confidence ต้องเป็น probability
    assert 0.0 <= result["confidence"] <= 1.0

    # scores ต้องครบทุก class
    assert len(result["scores"]) == len(EMOTION_LABELS)
    labels_in_scores = {s["label"] for s in result["scores"]}
    assert labels_in_scores == set(EMOTION_LABELS)

    # ผลรวม probability ใกล้ 1
    total_prob = sum(s["score"] for s in result["scores"])
    assert abs(total_prob - 1.0) < 1e-3

    # latency ต้องเป็นบวก
    assert result["inference_time_ms"] > 0
    assert result["total_time_ms"] >= result["inference_time_ms"]


def test_scores_sorted_descending(sample_image_bytes):
    """scores ต้องเรียงจากมากไปน้อย และตัวบนสุดต้องตรงกับ predicted_label"""
    from app.model import predict_from_bytes

    result = predict_from_bytes(sample_image_bytes)
    scores = result["scores"]
    for i in range(len(scores) - 1):
        assert scores[i]["score"] >= scores[i + 1]["score"]
    assert scores[0]["label"] == result["predicted_label"]
    assert abs(scores[0]["score"] - result["confidence"]) < 1e-6


def test_predict_is_deterministic(sample_image_bytes):
    """รันสองรอบบน input เดียวกันต้องได้ผลเหมือนกัน"""
    from app.model import predict_from_bytes

    r1 = predict_from_bytes(sample_image_bytes)
    r2 = predict_from_bytes(sample_image_bytes)
    assert r1["predicted_label"] == r2["predicted_label"]
    assert abs(r1["confidence"] - r2["confidence"]) < 1e-6
