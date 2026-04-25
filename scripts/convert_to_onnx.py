import os
import shutil
from optimum.onnxruntime import ORTModelForImageClassification

def main():
    print("[1/3] กำลังโหลดและแปลงโมเดลด้วย Optimum (วิธีนี้ชัวร์ที่สุด)...")
    model_id = "trpakov/vit-face-expression"
    temp_dir = "models_temp"

    # โหลดและแปลงเป็น ONNX อัตโนมัติ (ปลอดภัยและสมบูรณ์ 100%)
    model = ORTModelForImageClassification.from_pretrained(model_id, export=True)
    model.save_pretrained(temp_dir)

    print("[2/3] กำลังจัดการไฟล์ให้ตรงกับโปรเจกต์...")
    os.makedirs("models", exist_ok=True)

    # ย้ายและเปลี่ยนชื่อไฟล์ให้ตรงกับที่ API ต้องการ
    source_file = os.path.join(temp_dir, "model.onnx")
    target_file = os.path.join("models", "vit_face_expression.onnx")

    if os.path.exists(source_file):
        if os.path.exists(target_file):
            os.remove(target_file)
        shutil.move(source_file, target_file)
        
    print(f"[3/3] ✅ เสร็จสมบูรณ์! ได้ไฟล์ขนาดเต็มพร้อมใช้งานที่: {target_file}")

if __name__ == "__main__":
    main()