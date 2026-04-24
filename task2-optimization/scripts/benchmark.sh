#!/usr/bin/env bash
# benchmark.sh — Before/after benchmark for Task 2 Stage 2 optimization
#
# Usage:
#   bash task2-optimization/scripts/benchmark.sh [--baseline | --optimized | --compare]
#     --baseline  : run benchmark and save as baseline results
#     --optimized : run benchmark and save as optimized results
#     --compare   : print side-by-side comparison of saved results
#
# Output: task2-optimization/results/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
NUM_ITERS="${NUM_ITERS:-100}"
MODE="${1:---baseline}"

mkdir -p "${RESULTS_DIR}"

run_benchmark() {
    local label="$1"
    local out_file="${RESULTS_DIR}/${label}_${TIMESTAMP}.json"

    echo "===== Running benchmark: ${label} (${NUM_ITERS} iters) ====="
    python - <<PYEOF
import time, json, os, numpy as np
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)
    TEST_IMG = os.path.expanduser("~/models/PaddleOCR-VL-1.5/test_img.jpg")
    if not os.path.exists(TEST_IMG):
        TEST_IMG = "/tmp/test_ocr.jpg"
        import cv2, numpy as np2
        img = np2.ones((64, 512, 3), dtype=np2.uint8) * 255
        cv2.imwrite(TEST_IMG, img)

    # Warmup
    for _ in range(5):
        ocr.ocr(TEST_IMG, cls=True)

    # Timed
    lats = []
    for i in range(int(os.environ.get("NUM_ITERS", "100"))):
        t0 = time.perf_counter()
        ocr.ocr(TEST_IMG, cls=True)
        lats.append((time.perf_counter() - t0) * 1000)

    lats = np.array(lats)
    stats = {
        "label": "${label}",
        "timestamp": "${TIMESTAMP}",
        "num_iters": len(lats),
        "mean_ms": float(np.mean(lats)),
        "median_ms": float(np.median(lats)),
        "p90_ms": float(np.percentile(lats, 90)),
        "p99_ms": float(np.percentile(lats, 99)),
        "min_ms": float(np.min(lats)),
        "max_ms": float(np.max(lats)),
        "throughput_iter_per_s": float(1000 / np.mean(lats)),
    }
    with open("${out_file}", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nSaved: ${out_file}")
except Exception as e:
    print(f"Benchmark error: {e}")
    raise
PYEOF
}

compare_results() {
    echo "===== Benchmark Comparison ====="
    python - <<PYEOF
import json, glob, os

results_dir = "${RESULTS_DIR}"
baseline_files = sorted(glob.glob(os.path.join(results_dir, "baseline_*.json")))
optimized_files = sorted(glob.glob(os.path.join(results_dir, "optimized_*.json")))

if not baseline_files:
    print("No baseline results found.")
    exit(0)
if not optimized_files:
    print("No optimized results found.")
    exit(0)

b = json.load(open(baseline_files[-1]))
o = json.load(open(optimized_files[-1]))

print(f"{'Metric':<20} {'Baseline':>12} {'Optimized':>12} {'Improvement':>14}")
print("-" * 62)
for key in ["mean_ms", "median_ms", "p90_ms", "p99_ms", "throughput_iter_per_s"]:
    bv = b[key]; ov = o[key]
    if "throughput" in key:
        pct = (ov - bv) / bv * 100
        direction = "+" if pct > 0 else ""
        flag = " ✅" if pct >= 20 else (" ⚠️" if pct > 0 else " ❌")
        print(f"{key:<20} {bv:>12.2f} {ov:>12.2f} {direction}{pct:>12.1f}%{flag}")
    else:
        pct = (bv - ov) / bv * 100  # lower latency = improvement
        direction = "+" if pct > 0 else ""
        flag = " ✅" if pct >= 20 else (" ⚠️" if pct > 0 else " ❌")
        print(f"{key:<20} {bv:>12.2f} {ov:>12.2f} {direction}{pct:>12.1f}%{flag}")

print()
mean_improvement = (b["mean_ms"] - o["mean_ms"]) / b["mean_ms"] * 100
if mean_improvement >= 20:
    print(f"✅ Mean latency improvement {mean_improvement:.1f}% — MEETS ≥20% target")
else:
    print(f"⚠️  Mean latency improvement {mean_improvement:.1f}% — does NOT meet ≥20% target")
PYEOF
}

case "${MODE}" in
    --baseline)  run_benchmark "baseline" ;;
    --optimized) run_benchmark "optimized" ;;
    --compare)   compare_results ;;
    *)
        echo "Usage: $0 [--baseline | --optimized | --compare]"
        exit 1
        ;;
esac
