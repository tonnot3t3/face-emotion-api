"""
Dynamic Quantization: vit_face_expression.onnx -> vit_face_expression_quantized.onnx

ใช้ onnxruntime.quantization.quantize_dynamic เพื่อแปลง weight INT8
- ไม่ต้องใช้ calibration dataset (ต่างจาก static)
- ลดขนาดโมเดล ~4x และเร็วขึ้นบน CPU
- accuracy drop เล็กน้อย (โดยปกติ < 1%) บน task แบบนี้

Usage:
    python -m scripts.quantize
"""
from __future__ import annotations

import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import ONNX_FP32_PATH, ONNX_QUANTIZED_PATH  # noqa: E402


def main() -> None:
    if not ONNX_FP32_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบ {ONNX_FP32_PATH}. กรุณารัน scripts/convert_to_onnx.py ก่อน"
        )

    print(f"[1/2] กำลัง quantize dynamic (INT8 weights) ...")
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_QUANTIZED_PATH),
        weight_type=QuantType.QInt8,
    )

    fp32_mb = ONNX_FP32_PATH.stat().st_size / (1024 * 1024)
    int8_mb = ONNX_QUANTIZED_PATH.stat().st_size / (1024 * 1024)
    reduction = (1 - int8_mb / fp32_mb) * 100

    print(f"[2/2] เสร็จแล้ว -> {ONNX_QUANTIZED_PATH}")
    print(f"    FP32 size: {fp32_mb:.2f} MB")
    print(f"    INT8 size: {int8_mb:.2f} MB")
    print(f"    ลดลง:      {reduction:.1f}%")


if __name__ == "__main__":
    main()
