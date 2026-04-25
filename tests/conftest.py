"""Shared pytest fixtures."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


def _make_face_like_image(size: int = 224) -> Image.Image:
    """
    สร้างรูป "หน้าคน" จำลองสำหรับเทส (ไม่ใช่หน้าจริง — แค่มีโครงสร้างพอให้โมเดลรันได้)
    เป็น vertical gradient + วงรีตรงกลาง
    """
    rng = np.random.default_rng(42)
    arr = rng.integers(80, 200, (size, size, 3), dtype=np.uint8)
    # วาดวงรีกลางภาพ (เหมือนหน้า)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2
    mask = ((xx - cx) ** 2 / (size * 0.3) ** 2 + (yy - cy) ** 2 / (size * 0.4) ** 2) < 1
    arr[mask] = [220, 180, 150]
    return Image.fromarray(arr, "RGB")


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """JPEG bytes ของรูปทดสอบ"""
    img = _make_face_like_image()
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    """เซฟไฟล์ทดสอบให้ใช้ใน Postman/JMeter ด้วย"""
    path = FIXTURES_DIR / "test_face.jpg"
    if not path.exists():
        img = _make_face_like_image()
        img.save(path, format="JPEG", quality=85)
    return path


@pytest.fixture
def corrupted_bytes() -> bytes:
    """bytes ที่ดูเหมือนรูปแต่เสีย"""
    return b"\xff\xd8\xff\xe0not_a_real_image_just_random_bytes_to_break_decoder"


@pytest.fixture
def text_bytes() -> bytes:
    """bytes ที่เป็น text ธรรมดา"""
    return b"Hello, this is not an image."
