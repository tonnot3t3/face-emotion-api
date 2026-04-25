# 🚀 Quick Start Guide

คู่มือ 5 นาที ตั้งแต่ clone จนถึง deploy บน Hugging Face Spaces

---

## 📦 ก่อนเริ่ม
ต้องมี:
- **Python 3.11** ขึ้นไป
- **Git**
- **Docker** (สำหรับทดสอบ container)
- บัญชี **GitHub**
- บัญชี **Hugging Face**
- (ถ้ารัน load test) **Apache JMeter** + Java 11+

---

## 🧪 1. รันบนเครื่อง (3 นาที)

```bash
# Clone
git clone <YOUR_REPO_URL>
cd face-emotion-api

# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Build model (ดาวน์โหลด HF + convert + quantize)
python -m scripts.convert_to_onnx
python -m scripts.quantize

# รัน
uvicorn app.main:app --port 7860 --reload
```

เปิดเบราว์เซอร์ไปที่ <http://localhost:7860> เพื่อใช้งาน UI

---

## 🐳 2. ทดสอบ Docker (1 นาที)

```bash
docker build -t face-emotion-api .
docker run -d -p 7860:7860 --name face-emotion-api face-emotion-api
```

ตรวจสอบ:
```bash
curl http://localhost:7860/health
```

---

## ☁️ 3. Deploy ไป Hugging Face Spaces (5 นาที)

### Step 1: สร้าง Hugging Face Space

1. ไปที่ <https://huggingface.co/new-space>
2. ตั้งชื่อ Space เช่น `face-emotion`
3. **License**: MIT
4. **Space SDK**: เลือก **Docker**
5. **Space hardware**: CPU basic (ฟรี)
6. กด Create Space

### Step 2: เอา Token จาก Hugging Face

1. ไปที่ <https://huggingface.co/settings/tokens>
2. กด **New token** ตั้งชื่ออะไรก็ได้
3. เลือก **Role: Write** (สำคัญ! ต้อง write permission)
4. คัดลอก token ไว้

### Step 3: ตั้งค่า GitHub Secrets

ใน GitHub repo ของคุณ:

1. ไปที่ **Settings → Secrets and variables → Actions → New repository secret**
2. เพิ่ม secrets 3 ตัว:

| Name | Value |
|------|-------|
| `HF_TOKEN` | token ที่คัดลอกมา |
| `HF_USERNAME` | username บน Hugging Face ของคุณ |
| `HF_SPACE_NAME` | ชื่อ Space (เช่น `face-emotion`) |

### Step 4: Push code

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<YOUR>/<REPO>.git
git push -u origin main
```

### Step 5: ดู GitHub Actions ทำงาน

1. ไปที่แท็บ **Actions** ใน GitHub repo
2. รอให้ workflow รันเสร็จ (~5-10 นาที)
3. ถ้า test ผ่าน 100% → จะ auto-deploy ไป HF Space

### Step 6: เข้าใช้งาน

หลัง deploy เสร็จ:
- **หน้าเว็บ UI**: `https://huggingface.co/spaces/<USERNAME>/<SPACE_NAME>`
- **API**: `https://<USERNAME>-<SPACE_NAME>.hf.space`
- **API docs**: `https://<USERNAME>-<SPACE_NAME>.hf.space/docs`

> หมายเหตุ: ครั้งแรกที่ Space build จะใช้เวลา ~10-15 นาที (เพราะต้องดาวน์โหลด HF model + quantize)

---

## 🔥 4. รัน JMeter Load Test

```bash
# Local
./jmeter/run_loadtest.sh http://localhost:7860 50 60

# Cloud (เปลี่ยน URL เป็นของคุณ)
./jmeter/run_loadtest.sh https://YOUR_USERNAME-face-emotion.hf.space 30 120
```

ดู HTML dashboard ที่ `jmeter/dashboard/index.html`

---

## 🎯 ลองใช้ API ทันที (ตัวอย่าง cURL)

```bash
# Local
curl -X POST -F "file=@my_face.jpg" http://localhost:7860/predict

# Cloud
curl -X POST -F "file=@my_face.jpg" https://YOUR_USERNAME-face-emotion.hf.space/predict
```

ผลลัพธ์ที่ได้:
```json
{
  "predicted_label": "happy",
  "confidence": 0.952,
  "scores": [...],
  "inference_time_ms": 48.2,
  "total_time_ms": 55.1,
  "filename": "my_face.jpg"
}
```

---

## ❓ Troubleshooting

| ปัญหา | วิธีแก้ |
|-------|---------|
| `ModelNotFoundError` | รัน `python -m scripts.convert_to_onnx && python -m scripts.quantize` |
| Workflow fail "secrets undefined" | ยังไม่ได้ตั้ง `HF_TOKEN/HF_USERNAME/HF_SPACE_NAME` ใน GitHub |
| HF Space build error | ดู logs ใน Space → Settings → Logs; ปกติคือ disk quota เต็มเพราะ model ใหญ่ |
| API 503 "Model is not loaded" | โมเดลยังไม่ build เสร็จ; รัน script convert_to_onnx ใหม่ |
| JMeter timeout | เพิ่ม `--connect-timeout` หรือใช้ `--throughput` ลด rate |
