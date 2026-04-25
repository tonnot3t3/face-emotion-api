#!/usr/bin/env bash
# รัน JMeter test แบบ headless (CLI) และ generate HTML dashboard
#
# Usage:
#   ./jmeter/run_loadtest.sh [base_url] [threads] [duration_sec]
#
# ตัวอย่าง:
#   # Local
#   ./jmeter/run_loadtest.sh http://localhost:7860 50 60
#
#   # Cloud (Hugging Face Spaces)
#   ./jmeter/run_loadtest.sh https://YOURNAME-face-emotion.hf.space 30 120

set -euo pipefail

BASE_URL="${1:-http://localhost:7860}"
THREADS="${2:-50}"
DURATION="${3:-60}"
RAMP_UP="${4:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
DASHBOARD_DIR="$SCRIPT_DIR/dashboard"
JTL_FILE="$RESULTS_DIR/results.jtl"
LOG_FILE="$RESULTS_DIR/jmeter.log"
IMAGE_PATH="$ROOT_DIR/tests/fixtures/test_face.jpg"

# เคลียร์ผลเก่า (JMeter dashboard generator ต้องใช้ folder ว่าง)
rm -rf "$RESULTS_DIR" "$DASHBOARD_DIR"
mkdir -p "$RESULTS_DIR"

# สร้างรูปทดสอบถ้ายังไม่มี
if [ ! -f "$IMAGE_PATH" ]; then
    echo "[setup] สร้างรูปทดสอบที่ $IMAGE_PATH ..."
    cd "$ROOT_DIR"
    python -c "from tests.conftest import _make_face_like_image; img = _make_face_like_image(); img.save('$IMAGE_PATH')"
fi

echo "================================================================"
echo " JMeter Load Test"
echo "----------------------------------------------------------------"
echo " Target URL : $BASE_URL"
echo " Threads    : $THREADS"
echo " Ramp-up    : ${RAMP_UP}s"
echo " Duration   : ${DURATION}s"
echo " Image      : $IMAGE_PATH"
echo "================================================================"

jmeter -n \
    -t "$SCRIPT_DIR/load_test.jmx" \
    -Jbase_url="$BASE_URL" \
    -Jimage_path="$IMAGE_PATH" \
    -Jthreads="$THREADS" \
    -Jramp_up="$RAMP_UP" \
    -Jduration="$DURATION" \
    -l "$JTL_FILE" \
    -j "$LOG_FILE" \
    -e -o "$DASHBOARD_DIR"

echo ""
echo "================================================================"
echo " เสร็จสิ้น!"
echo " - Raw results : $JTL_FILE"
echo " - Dashboard   : $DASHBOARD_DIR/index.html"
echo "================================================================"
