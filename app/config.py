"""
Application configuration.

ค่าคงที่ทั้งหมดของแอป รวมถึง path ของโมเดล, ขนาดไฟล์ที่ยอมรับได้,
และจำนวน worker process ที่ใช้รันโมเดล
"""
import os
from pathlib import Path

# ===== Paths =====
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Hugging Face model id (ใช้ตอน export ครั้งแรก)
HF_MODEL_ID = "trpakov/vit-face-expression"

# Path ของไฟล์โมเดลแต่ละแบบ
ONNX_FP32_PATH = MODELS_DIR / "vit_face_expression.onnx"
ONNX_QUANTIZED_PATH = MODELS_DIR / "vit_face_expression_quantized.onnx"

# โมเดลที่ใช้ใน production (ค่า default = quantized เพื่อความเร็วและขนาดเล็ก)
ACTIVE_MODEL_PATH = Path(
    os.getenv("ACTIVE_MODEL_PATH", str(ONNX_QUANTIZED_PATH))
)

# ===== API limits =====
# ขนาดไฟล์สูงสุด: 5 MB
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "5"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# นามสกุลไฟล์ภาพที่อนุญาต
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ===== Concurrency =====
# จำนวน worker process สำหรับรัน inference
# ใช้ ProcessPoolExecutor เพราะการรันโมเดลเป็น CPU-bound
# ค่า default = จำนวน CPU - 1 (เผื่อให้ event loop มี thread ทำงาน) แต่ไม่น้อยกว่า 1
NUM_INFERENCE_WORKERS = int(
    os.getenv("NUM_INFERENCE_WORKERS", str(max(1, (os.cpu_count() or 2) - 1)))
)

# ===== Model meta =====
# 7 emotion classes ของโมเดล vit-face-expression
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ขนาด input ของ ViT
IMAGE_SIZE = 224
# ค่า normalization ของ ViT (ImageNet stats ที่ HuggingFace processor ใช้)
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# ===== Misc =====
APP_TITLE = "Face Emotion Classification API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "High-throughput Image Classification service powered by an ONNX-quantized "
    "Vision Transformer (trpakov/vit-face-expression). "
    "Detects 7 facial emotions: angry, disgust, fear, happy, neutral, sad, surprise."
)
