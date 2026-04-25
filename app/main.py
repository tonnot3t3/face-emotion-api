"""
FastAPI application entry point.

Highlights:
- async def endpoints (ไม่บล็อก event loop)
- ProcessPoolExecutor สำหรับงาน CPU-bound (ONNX inference)
- Pydantic validation ของ response
- Error handling ที่ตอบ HTTP status code ถูกต้อง (400 vs 413 vs 415 vs 500)
- เสิร์ฟหน้าเว็บ UI ที่ /
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.config import (
    ACTIVE_MODEL_PATH,
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    EMOTION_LABELS,
    IMAGE_SIZE,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    NUM_INFERENCE_WORKERS,
)
from app.model import get_model_meta, predict_from_bytes, warmup
from app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

logger = logging.getLogger("uvicorn.error")

# ProcessPoolExecutor: shared ทั้งแอป สร้างใน lifespan
_executor: ProcessPoolExecutor | None = None


def _init_worker() -> None:
    """initializer ที่รันใน worker process แต่ละตัว เพื่อ warmup โมเดล"""
    try:
        warmup()
    except Exception as exc:  # pragma: no cover
        # ถ้า warmup fail แต่ inference จริงสำเร็จก็ยังใช้ได้ จึงแค่ log
        logging.getLogger().warning("Worker warmup failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """startup/shutdown ของแอป"""
    global _executor
    logger.info(
        "Starting %s v%s with %d worker process(es)",
        APP_TITLE, APP_VERSION, NUM_INFERENCE_WORKERS,
    )
    if not ACTIVE_MODEL_PATH.exists():
        logger.warning("ไม่พบโมเดลที่ %s — endpoint /predict จะ error 503", ACTIVE_MODEL_PATH)
    _executor = ProcessPoolExecutor(
        max_workers=NUM_INFERENCE_WORKERS,
        initializer=_init_worker,
    )
    try:
        yield
    finally:
        if _executor is not None:
            _executor.shutdown(wait=True, cancel_futures=True)
            logger.info("ProcessPoolExecutor shut down")


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

# CORS เปิดให้หน้าเว็บ static เรียก API ได้ (และเผื่อกรณี deploy แยก domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Exception handlers =====

@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    """ทำให้ทุก HTTPException ตอบในรูปแบบ ErrorResponse"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail if isinstance(exc.detail, str) else "HTTP error",
            detail=str(exc.detail),
            status_code=exc.status_code,
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    """422 -> 400 (เพราะส่วนใหญ่ผู้ใช้คุ้นกับ 400 มากกว่า)"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation error",
            detail=str(exc.errors()),
            status_code=status.HTTP_400_BAD_REQUEST,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):  # pragma: no cover
    """fallback สำหรับ error ที่ไม่ได้ดักไว้ อย่าเผย stack trace ออกไป"""
    logger.exception("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred. Please try again later.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ).model_dump(),
    )


# ===== Helpers =====

def _validate_upload(file: UploadFile, content: bytes) -> None:
    """
    ตรวจสอบไฟล์ที่อัปโหลดและ raise HTTPException ที่เหมาะสม
    - 413 ถ้าใหญ่เกิน
    - 415 ถ้า mime/ext ไม่รองรับ
    - 400 ถ้าเสียหรือ decode ไม่ได้
    """
    # ขนาดไฟล์
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {MAX_UPLOAD_SIZE_MB} MB.",
        )

    # MIME / นามสกุล: ยอมรับถ้าผ่านอย่างใดอย่างหนึ่งจาก content_type หรือ ext
    ext = Path(file.filename or "").suffix.lower()
    mime_ok = file.content_type in ALLOWED_CONTENT_TYPES
    ext_ok = ext in ALLOWED_EXTENSIONS
    if not (mime_ok or ext_ok):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}. "
                f"Got content_type={file.content_type!r}, ext={ext!r}."
            ),
        )

    # ตรวจสอบว่าเปิดเป็นรูปได้จริง (กันไฟล์เสีย / ไฟล์ปลอม extension)
    from io import BytesIO
    try:
        with Image.open(BytesIO(content)) as img:
            img.verify()  # ตรวจสอบ structure โดยไม่ decode ทั้งภาพ
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decode image — file may be corrupted: {exc}",
        )


# ===== Endpoints =====

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    """liveness/readiness probe"""
    return HealthResponse(
        status="ok",
        model_loaded=ACTIVE_MODEL_PATH.exists(),
        model_path=str(ACTIVE_MODEL_PATH),
        num_workers=NUM_INFERENCE_WORKERS,
        version=APP_VERSION,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["meta"])
async def model_info() -> ModelInfoResponse:
    """ข้อมูลโมเดลปัจจุบัน"""
    path, size_mb = get_model_meta()
    return ModelInfoResponse(
        model_id="trpakov/vit-face-expression",
        model_path=path,
        model_size_mb=size_mb,
        num_classes=len(EMOTION_LABELS),
        labels=EMOTION_LABELS,
        input_size=IMAGE_SIZE,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request / corrupted image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["inference"],
)
async def predict(file: UploadFile = File(..., description="ไฟล์ภาพใบหน้า .jpg/.png/.webp")) -> PredictionResponse:
    """
    ทำนาย emotion จากภาพใบหน้า 1 รูป

    คืนค่า:
    - **predicted_label**: emotion ที่ความน่าจะเป็นสูงสุด
    - **confidence**: ความน่าจะเป็นของ label นั้น
    - **scores**: ความน่าจะเป็นของทุก class (เรียงจากมากไปน้อย)
    - **inference_time_ms / total_time_ms**: สำหรับ profiling
    """
    if _executor is None or not ACTIVE_MODEL_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please contact administrator.",
        )

    # อ่านเนื้อไฟล์ทั้งหมด — ทำใน async context ได้เพราะเป็น I/O
    content = await file.read()

    # validate ก่อนส่งเข้า worker (เร็วกว่าและ error message ชัดเจน)
    _validate_upload(file, content)

    # offload งาน CPU-bound เข้า ProcessPoolExecutor เพื่อไม่ให้ event loop ค้าง
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_executor, predict_from_bytes, content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return PredictionResponse(filename=file.filename, **result)


# ===== Static UI =====

# เสิร์ฟไฟล์ static (เช่น CSS, JS เพิ่มเติม) ที่ /static
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """หน้าเว็บ UI สำหรับให้ผู้ใช้อัปโหลดภาพและดูผลลัพธ์"""
    index_path = _static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            "<h1>Face Emotion API</h1><p>UI not found. ดู docs ที่ <a href='/docs'>/docs</a></p>"
        )
    return HTMLResponse(index_path.read_text(encoding="utf-8"))
