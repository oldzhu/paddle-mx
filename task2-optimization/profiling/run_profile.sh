#!/usr/bin/env bash
# run_profile.sh — Profile PaddleOCR-VL-1.5 inference on Metax GPU with FastDeploy
#
# Prerequisites:
#   - FastDeploy release/2.5 installed
#   - PaddleOCR-VL-1.5 model downloaded
#   - MACA environment variables set (source from 02_build_fastdeploy.sh)
#
# Usage:
#   bash task2-optimization/profiling/run_profile.sh [--quick]
#     --quick : run 10 iterations only (for testing); default runs 50
#
# Output:
#   traces/profile_<timestamp>/  — trace file + per-op timing CSV
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRACES_DIR="${SCRIPT_DIR}/traces"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${TRACES_DIR}/profile_${TIMESTAMP}"
NUM_ITERS=50

# Parse args
for arg in "$@"; do
    case "$arg" in
        --quick) NUM_ITERS=10 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"

echo "===== [0] Environment check ====="
MACA_PATH="${MACA_PATH:-/opt/maca}"
export MACA_VISIBLE_DEVICES="${MACA_VISIBLE_DEVICES:-0}"
export PADDLE_XCCL_BACKEND="${PADDLE_XCCL_BACKEND:-metax_gpu}"

if command -v maca-smi &>/dev/null; then
    maca-smi | tee "${OUTPUT_DIR}/maca-smi.txt"
else
    echo "WARNING: maca-smi not found"
fi
python --version
python -c "import fastdeploy; print('FastDeploy:', fastdeploy.__version__)"

echo ""
echo "===== [1] Discover profiling tool ====="
# MACA profiling tool: mcTracer (the MACA equivalent of nvprof/nsys)
MCTRACER="/opt/maca/bin/mcTracer"
if [ -x "${MCTRACER}" ]; then
    echo "Found mcTracer: ${MCTRACER}"
    ${MCTRACER} --help 2>&1 | head -5
else
    echo "WARNING: mcTracer not found at ${MCTRACER}"
    MCTRACER=""
fi

echo ""
echo "===== [2] Prepare model path ====="
# Default model download location — update if model is elsewhere
MODEL_DIR="${HOME}/models/PaddleOCR-VL-1.5"
if [ ! -d "${MODEL_DIR}" ]; then
    echo "Model not found at ${MODEL_DIR}. Downloading from HuggingFace..."
    pip install -q huggingface_hub 2>/dev/null || true
    python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='PaddlePaddle/PaddleOCR-VL-1.5',
    local_dir=os.path.expanduser('${MODEL_DIR}'),
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*'],
)
print('Download complete:', '${MODEL_DIR}')
"
fi
echo "Model directory: ${MODEL_DIR}"
ls "${MODEL_DIR}" | head -20

echo ""
echo "===== [3] Generate profiling inference script ====="
INFER_SCRIPT="${OUTPUT_DIR}/run_infer.py"
cat > "${INFER_SCRIPT}" << 'PYEOF'
"""
Profiling inference script for PaddleOCR-VL-1.5 on Metax GPU + FastDeploy.
Runs NUM_ITERS iterations and records per-iteration latency.
"""
import os
import sys
import time
import json
import csv

import numpy as np

NUM_ITERS = int(os.environ.get("NUM_ITERS", "50"))
MODEL_DIR = os.environ.get("MODEL_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
WARMUP = 5

# ---- Import FastDeploy / PaddleOCR pipeline ----
try:
    from paddleocr import PaddleOCR
    print("Using PaddleOCR pipeline")
    USE_PADDLEOCR = True
except ImportError:
    print("PaddleOCR not available — using raw FastDeploy")
    USE_PADDLEOCR = False

# A small test image (1x1 white PNG in base64, for pipeline smoke test)
TEST_IMAGE_PATH = os.path.join(OUTPUT_DIR, "test_img.jpg")
if not os.path.exists(TEST_IMAGE_PATH):
    import urllib.request
    # Download a sample document image for OCR testing
    sample_url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/test_image/rec_text_image.jpg"
    try:
        urllib.request.urlretrieve(sample_url, TEST_IMAGE_PATH)
        print(f"Downloaded test image to {TEST_IMAGE_PATH}")
    except Exception as e:
        print(f"Could not download test image: {e}")
        # Create a minimal dummy image
        import cv2
        import numpy as np as np_cv
        dummy = np_cv.ones((64, 512, 3), dtype=np_cv.uint8) * 255
        cv2.putText(dummy, "Hello OCR Test", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(TEST_IMAGE_PATH, dummy)
        print(f"Created dummy test image at {TEST_IMAGE_PATH}")

# ---- Initialize pipeline ----
print(f"\nInitializing pipeline (model_dir={MODEL_DIR or 'default'})...")
init_start = time.perf_counter()
if USE_PADDLEOCR:
    ocr_engine = PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        use_gpu=True,
        **({"model_dir": MODEL_DIR} if MODEL_DIR else {}),
    )
init_end = time.perf_counter()
print(f"Pipeline initialized in {init_end - init_start:.2f}s")

# ---- Warmup ----
print(f"\nWarmup ({WARMUP} iterations)...")
for _ in range(WARMUP):
    if USE_PADDLEOCR:
        ocr_engine.ocr(TEST_IMAGE_PATH, cls=True)

# ---- Timed inference ----
print(f"\nTimed inference ({NUM_ITERS} iterations)...")
latencies = []
for i in range(NUM_ITERS):
    t0 = time.perf_counter()
    if USE_PADDLEOCR:
        result = ocr_engine.ocr(TEST_IMAGE_PATH, cls=True)
    t1 = time.perf_counter()
    lat_ms = (t1 - t0) * 1000
    latencies.append(lat_ms)
    if i % 10 == 0:
        print(f"  iter {i:3d}: {lat_ms:.1f} ms")

# ---- Statistics ----
lats = np.array(latencies)
stats = {
    "num_iters": NUM_ITERS,
    "mean_ms": float(np.mean(lats)),
    "median_ms": float(np.median(lats)),
    "p90_ms": float(np.percentile(lats, 90)),
    "p99_ms": float(np.percentile(lats, 99)),
    "min_ms": float(np.min(lats)),
    "max_ms": float(np.max(lats)),
    "std_ms": float(np.std(lats)),
}
print("\n===== Latency Statistics =====")
for k, v in stats.items():
    print(f"  {k:15s}: {v:.2f}")

# Save results
stats_path = os.path.join(OUTPUT_DIR, "latency_stats.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nStats saved to: {stats_path}")

csv_path = os.path.join(OUTPUT_DIR, "per_iter_latency.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iter", "latency_ms"])
    for i, lat in enumerate(latencies):
        writer.writerow([i, f"{lat:.3f}"])
print(f"Per-iter CSV saved to: {csv_path}")
PYEOF

echo ""
echo "===== [4] Run inference with profiling ====="
export NUM_ITERS="${NUM_ITERS}"
export MODEL_DIR="${MODEL_DIR}"
export OUTPUT_DIR="${OUTPUT_DIR}"

if [ -n "${PROF_TOOL}" ]; then
    echo "Running with ${PROF_TOOL}..."
    TRACE_FILE="${OUTPUT_DIR}/trace.json"
    # Command varies by tool — adjust flags as needed after identifying the tool
    case "${PROF_TOOL}" in
        mxprof)
            ${PROF_TOOL} --output "${TRACE_FILE}" python "${INFER_SCRIPT}" \
                2>&1 | tee "${OUTPUT_DIR}/profile_run.log"
            ;;
        *)
            echo "Unknown tool ${PROF_TOOL} — running without profiling wrapper"
            python "${INFER_SCRIPT}" 2>&1 | tee "${OUTPUT_DIR}/profile_run.log"
            ;;
    esac
else
    echo "No profiling tool found — running Python-level timing only."
    python "${INFER_SCRIPT}" 2>&1 | tee "${OUTPUT_DIR}/profile_run.log"
fi

echo ""
echo "===== [5] Summarize output ====="
echo "Output directory: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"
if [ -f "${OUTPUT_DIR}/latency_stats.json" ]; then
    echo ""
    echo "Latency summary:"
    cat "${OUTPUT_DIR}/latency_stats.json"
fi

echo ""
echo "===== [DONE] Profiling complete ====="
echo "Traces and logs saved to: ${OUTPUT_DIR}"
echo "Next: analyze traces and fill in task2-optimization/profiling/rfcs/perf-analysis-report_001.md"
