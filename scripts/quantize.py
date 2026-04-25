import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    print("[1/2] เริ่มต้นการบีบอัดโมเดล (Quantize INT8)...")
    
    model_input = os.path.join("models", "vit_face_expression.onnx")
    # สมมติฐานชื่อไฟล์ Output ตามมาตรฐาน ถ้าโค้ดคุณใช้ชื่ออื่น สามารถแก้ตรงนี้ได้เลยครับ
    model_output = os.path.join("models", "vit_face_expression_quantized.onnx")
    
    if not os.path.exists(model_input):
        print(f"❌ Error: ไม่พบไฟล์ {model_input}")
        return

    print("[2/2] กำลังบีบอัดเฉพาะส่วน Transformer (ข้าม Conv เพื่อหลบ Error)...")
    
    # หัวใจสำคัญคือ op_types_to_quantize=['MatMul']
    quantize_dynamic(
        model_input=model_input,
        model_output=model_output,
        op_types_to_quantize=['MatMul'], 
        weight_type=QuantType.QUInt8
    )
    
    print(f"✅ เสร็จสมบูรณ์! ได้ไฟล์บีบอัดที่: {model_output}")

if __name__ == "__main__":
    main()