#!/usr/bin/env bash
# 01_install_deps.sh — Install Python dependencies for Metax GPU + PaddleOCR-VL-1.5
# Run this script once on a fresh GiteeAI instance (Pytorch/2.8.0/Python 3.12/maca 3.3.0.4)
set -euo pipefail

echo "===== [1/4] Install PaddlePaddle (CPU wheel, nightly) ====="
pip install paddlepaddle==3.4.0.dev20251223 \
    -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/ \
    --no-cache-dir

echo "===== [2/4] Install Paddle Metax GPU backend ====="
pip install paddle-metax-gpu==3.3.0.dev20251224 \
    -i https://www.paddlepaddle.org.cn/packages/nightly/maca/ \
    --no-cache-dir

echo "===== [3/4] Install PaddleOCR with doc-parser extras ====="
python -m pip install -U "paddleocr[doc-parser]"

echo "===== [4/4] Install OpenCV headless ====="
pip install opencv-contrib-python-headless==4.10.0.84

echo ""
echo "===== Verification ====="
python -c "
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
