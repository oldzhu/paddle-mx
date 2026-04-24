#!/usr/bin/env bash
# 01_install_deps.sh — Install Python dependencies for Metax GPU + PaddleOCR-VL-1.5
# Run this script once on a fresh GiteeAI instance (Pytorch/2.8.0/Python 3.12/maca 3.3.0.4)
set -euo pipefail

# Use --break-system-packages to work around PEP 668 on Debian/Ubuntu-based images
PIP_FLAGS="--break-system-packages --no-cache-dir"

echo "===== [1/4] Install PaddlePaddle (CPU wheel, nightly) ====="
pip install paddlepaddle==3.4.0.dev20251223 \
    -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/ \
    ${PIP_FLAGS}

echo "===== [2/4] Install Paddle Metax GPU backend ====="
# Discover the latest available version at the maca index
MACA_INDEX="https://www.paddlepaddle.org.cn/packages/nightly/maca/"
echo "Querying available paddle-metax-gpu versions at: ${MACA_INDEX}"
AVAIL=$(pip index versions paddle-metax-gpu --index-url "${MACA_INDEX}" 2>/dev/null | head -3 || true)
echo "Available: ${AVAIL}"
# Install latest (no pinned version) — use --find-links fallback if index fails
pip install paddle-metax-gpu \
    -i "${MACA_INDEX}" \
    ${PIP_FLAGS} || \
pip install paddle-metax-gpu==3.3.0.dev20251224 \
    --find-links "${MACA_INDEX}" \
    ${PIP_FLAGS} || \
echo "WARNING: paddle-metax-gpu not available from pip index (may already be installed by the image)"

echo "===== [3/4] Install PaddleOCR with doc-parser extras ====="
python3 -m pip install -U "paddleocr[doc-parser]" ${PIP_FLAGS}

echo "===== [4/4] Install OpenCV headless ====="
pip install opencv-contrib-python-headless==4.10.0.84 ${PIP_FLAGS}

echo ""
echo "===== Verification ====="
python3 -c "
import paddle
print('PaddlePaddle version:', paddle.__version__)
import paddle_metax_gpu  # noqa – verify plugin loads
print('paddle-metax-gpu loaded OK')
import paddleocr
print('PaddleOCR version:', paddleocr.__version__)
import cv2
print('OpenCV version:', cv2.__version__)
"

echo ""
echo "===== [DONE] All dependencies installed successfully ====="
