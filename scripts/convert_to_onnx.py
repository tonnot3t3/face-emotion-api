"""
Convert HuggingFace ViT image classifier -> ONNX (FP32).

Usage:
    python -m scripts.convert_to_onnx

จะดาวน์โหลดโมเดลจาก HuggingFace แล้วเซฟเป็น
    models/vit_face_expression.onnx
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageClassification

# ทำให้ import app ได้แม้รันจาก root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import HF_MODEL_ID, IMAGE_SIZE, ONNX_FP32_PATH  # noqa: E402


def main() -> None:
    print(f"[1/3] กำลังโหลดโมเดล {HF_MODEL_ID} จาก HuggingFace ...")
    model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
    model.eval()

    print(f"[2/3] กำลัง trace และ export เป็น ONNX (opset 14) ...")
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    ONNX_FP32_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_FP32_PATH),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    size_mb = ONNX_FP32_PATH.stat().st_size / (1024 * 1024)
    print(f"[3/3] เสร็จแล้ว -> {ONNX_FP32_PATH}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
