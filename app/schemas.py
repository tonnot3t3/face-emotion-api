"""
Pydantic schemas สำหรับ request/response ของ API
ใช้ Pydantic v2 syntax
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class EmotionScore(BaseModel):
    """คะแนน probability ของแต่ละ emotion class"""

    label: str = Field(..., description="ชื่อ emotion class")
    score: float = Field(..., ge=0.0, le=1.0, description="ความน่าจะเป็น 0-1")


class PredictionResponse(BaseModel):
    """response ของ /predict endpoint"""

    face_detected: bool = Field(
        True, description="ตรวจพบใบหน้าในภาพหรือไม่ (ถ้าไม่พบ, predicted_label/scores จะเป็น null/[])"
    )
    predicted_label: Optional[str] = Field(
        None, description="emotion ที่มี probability สูงสุด (null ถ้าไม่พบใบหน้า)"
    )
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    scores: List[EmotionScore] = Field(
        default_factory=list, description="probability ของทุก class (ว่างถ้าไม่พบใบหน้า)"
    )
    inference_time_ms: float = Field(..., ge=0.0, description="เวลา inference เฉพาะตัวโมเดล")
    total_time_ms: float = Field(..., ge=0.0, description="เวลารวมทั้ง preprocess + inference")
    filename: str | None = None
    message: Optional[str] = Field(
        None, description="ข้อความเพิ่มเติม เช่น เหตุผลที่ไม่พบใบหน้า"
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_path: str
    num_workers: int
    version: str


class ErrorResponse(BaseModel):
    """รูปแบบ error response มาตรฐานของ API"""

    error: str
    detail: str
    status_code: int


class ModelInfoResponse(BaseModel):
    """ข้อมูลโมเดลปัจจุบันที่กำลังรันอยู่"""

    model_id: str
    model_path: str
    model_size_mb: float
    num_classes: int
    labels: List[str]
    input_size: int
