"""
Pydantic schemas สำหรับ request/response ของ API
ใช้ Pydantic v2 syntax
"""
from typing import Dict, List
from pydantic import BaseModel, Field


class EmotionScore(BaseModel):
    """คะแนน probability ของแต่ละ emotion class"""

    label: str = Field(..., description="ชื่อ emotion class")
    score: float = Field(..., ge=0.0, le=1.0, description="ความน่าจะเป็น 0-1")


class PredictionResponse(BaseModel):
    """response ของ /predict endpoint"""

    predicted_label: str = Field(..., description="emotion ที่มี probability สูงสุด")
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: List[EmotionScore] = Field(..., description="probability ของทุก class")
    inference_time_ms: float = Field(..., ge=0.0, description="เวลา inference เฉพาะตัวโมเดล")
    total_time_ms: float = Field(..., ge=0.0, description="เวลารวมทั้ง preprocess + inference")
    filename: str | None = None


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
