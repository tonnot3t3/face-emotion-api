"""
Integration tests สำหรับ FastAPI endpoints
ใช้ TestClient ของ FastAPI/Starlette
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import ACTIVE_MODEL_PATH, EMOTION_LABELS, MAX_UPLOAD_SIZE_BYTES
from app.main import app


@pytest.fixture(scope="module")
def client():
    """TestClient — ใช้ context manager เพื่อให้ lifespan ถูก trigger"""
    with TestClient(app) as c:
        yield c


# ===== Meta endpoints (ทำงานได้เสมอ ไม่ต้องมีโมเดล) =====

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "num_workers" in body
    assert body["num_workers"] >= 1


def test_model_info_endpoint(client):
    res = client.get("/model-info")
    assert res.status_code == 200
    body = res.json()
    assert body["model_id"] == "trpakov/vit-face-expression"
    assert body["num_classes"] == 7
    assert set(body["labels"]) == set(EMOTION_LABELS)
    assert body["input_size"] == 224


def test_index_returns_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_openapi_docs_available(client):
    res = client.get("/docs")
    assert res.status_code == 200


# ===== Predict endpoint validation (ทำงานได้แม้ไม่มีโมเดล เพราะ validate ก่อน) =====

def test_predict_rejects_text_file(client, text_bytes):
    """ไฟล์ที่ไม่ใช่รูปต้องตอบ 415"""
    res = client.post(
        "/predict",
        files={"file": ("hello.txt", text_bytes, "text/plain")},
    )
    assert res.status_code == 415
    body = res.json()
    assert body["status_code"] == 415


def test_predict_rejects_empty_file(client):
    """ไฟล์ว่างต้อง 400"""
    res = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert res.status_code == 400


def test_predict_rejects_corrupted_image(client, corrupted_bytes):
    """ไฟล์ที่อ้าง MIME ว่ารูปแต่ decode ไม่ได้ ต้อง 400 (ไม่ใช่ 500)"""
    res = client.post(
        "/predict",
        files={"file": ("broken.jpg", corrupted_bytes, "image/jpeg")},
    )
    assert res.status_code == 400


def test_predict_rejects_too_large_file(client):
    """ไฟล์ใหญ่เกินต้อง 413"""
    big = b"\x00" * (MAX_UPLOAD_SIZE_BYTES + 1024)
    res = client.post(
        "/predict",
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert res.status_code == 413


def test_predict_requires_file_field(client):
    """ไม่ส่ง field file มา ต้อง 400 (เราแปลง 422 -> 400)"""
    res = client.post("/predict")
    assert res.status_code == 400


# ===== Predict endpoint success cases (ต้องมีโมเดล) =====

requires_model = pytest.mark.skipif(
    not ACTIVE_MODEL_PATH.exists(),
    reason=f"โมเดลที่ {ACTIVE_MODEL_PATH} ยังไม่ถูกสร้าง",
)


@requires_model
def test_predict_success_returns_valid_json(client, sample_image_bytes):
    """กรณี happy path — รูปจริง ต้องตอบ 200 พร้อม JSON ถูกต้อง"""
    res = client.post(
        "/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert res.status_code == 200, res.text
    body = res.json()

    # โครงสร้าง response ตรงกับ Pydantic schema
    assert body["predicted_label"] in EMOTION_LABELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["scores"]) == 7
    assert body["filename"] == "test.jpg"
    assert body["inference_time_ms"] > 0


@requires_model
def test_predict_supports_png(client, sample_image_bytes):
    """รองรับ PNG ด้วย"""
    import io
    from PIL import Image
    img = Image.open(io.BytesIO(sample_image_bytes))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    res = client.post(
        "/predict",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
    )
    assert res.status_code == 200


@requires_model
def test_predict_concurrent_requests(client, sample_image_bytes):
    """รัน 5 request ติดกัน — ต้องไม่ค้าง / pass ทั้งหมด"""
    from concurrent.futures import ThreadPoolExecutor

    def call():
        return client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda _: call(), range(5)))

    assert all(r.status_code == 200 for r in results)
