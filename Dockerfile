# =============================================================================
# Stage 1: Builder — convert HF model -> ONNX FP32 -> ONNX INT8
# มี torch + transformers (หนัก) แต่จะไม่ติดมาใน final image
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# ติดตั้ง dependencies สำหรับ build เท่านั้น
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# คัดลอกโค้ด + script
COPY app ./app
COPY scripts ./scripts

# Convert + quantize โมเดล (จะดาวน์โหลดจาก HF แล้ว save ที่ /build/models)
RUN python -m scripts.convert_to_onnx && \
    python -m scripts.quantize && \
    rm -f models/vit_face_expression.onnx
# ลบไฟล์ FP32 ทิ้งใน final image (เก็บแค่ INT8 เพื่อให้ image เล็กที่สุด)


# =============================================================================
# Stage 2: Runtime — มีแค่ที่จำเป็นต่อการรัน inference
# ไม่มี torch / transformers / pytest -> image เล็กลงมาก
# =============================================================================
FROM python:3.11-slim AS runtime

# สร้าง user ไม่ใช่ root (security best practice + Hugging Face Spaces ต้องการ user 1000)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ติดตั้งเฉพาะ runtime deps
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# คัดลอกโค้ดและโมเดลที่ build ไว้แล้ว
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser static ./static
COPY --chown=appuser:appuser --from=builder /build/models ./models

USER appuser

# Hugging Face Spaces จะส่ง $PORT มา (default 7860) — ต้องอ่านจาก env
ENV PORT=7860
EXPOSE 7860

# ใช้ shell form เพื่อให้ ${PORT} ถูก expand
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
