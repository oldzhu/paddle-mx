#!/usr/bin/env bash
# 02_build_fastdeploy.sh — Clone FastDeploy release/2.5 and build on Metax GPU
#
# IMPORTANT: Run this inside a tmux session to survive SSH disconnections:
#   tmux new -s build
#   bash task1-warmup/scripts/02_build_fastdeploy.sh
#   Ctrl-b d  (to detach; reconnect with: tmux attach -t build)
#
# Build time: approximately 30–60 minutes on C500
set -euo pipefail

FASTDEPLOY_DIR="${HOME}/FastDeploy"
MACA_PATH=/opt/maca

echo "===== [0/6] Pre-flight checks ====="
if ! command -v maca-smi &>/dev/null; then
    echo "WARNING: maca-smi not found in PATH. Check MACA installation."
else
    maca-smi
fi
python --version

echo ""
echo "===== [1/6] Clone FastDeploy release/2.5 from Gitee ====="
if [ -d "${FASTDEPLOY_DIR}" ]; then
    echo "Directory ${FASTDEPLOY_DIR} already exists — pulling latest instead."
    cd "${FASTDEPLOY_DIR}"
    git fetch origin
    git checkout release/2.5
    git reset --hard origin/release/2.5
else
    git clone https://gitee.com/paddlepaddle/FastDeploy.git "${FASTDEPLOY_DIR}"
    cd "${FASTDEPLOY_DIR}"
    git checkout release/2.5
fi

echo "FastDeploy branch: $(git branch --show-current)  commit: $(git rev-parse --short HEAD)"

echo ""
echo "===== [2/6] Set up cu-bridge (MACA CUDA compatibility layer) ====="
if [ ! -d "${HOME}/cu-bridge" ]; then
    echo "Running pre_make to create cu-bridge..."
    "${MACA_PATH}/tools/cu-bridge/tools/pre_make"
else
    echo "cu-bridge already exists at ${HOME}/cu-bridge — skipping pre_make"
fi

echo ""
echo "===== [3/6] Export MACA environment variables ====="
export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUCC_CMAKE_ENTRY=2
export CUDA_PATH="${HOME}/cu-bridge/CUDA_DIR"
export PATH="${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"

# FastDeploy-specific flags
export MACA_VISIBLE_DEVICES="0"
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_MOE_BACKEND=cutlass
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_ENC_DEC_BLOCK_NUM=2
export FD_SAMPLING_CLASS=rejection

echo "MACA_PATH:              ${MACA_PATH}"
echo "CUCC_PATH:              ${CUCC_PATH}"
echo "CUDA_PATH:              ${CUDA_PATH}"
echo "PADDLE_XCCL_BACKEND:    ${PADDLE_XCCL_BACKEND}"
echo "FD_MOE_BACKEND:         ${FD_MOE_BACKEND}"

echo ""
echo "===== [4/6] Verify build.sh exists ====="
if [ ! -f "build.sh" ]; then
    echo "ERROR: build.sh not found in $(pwd). Listing repo root:"
    ls -la
    exit 1
fi
echo "build.sh found at $(pwd)/build.sh"

echo ""
echo "===== [5/6] Run build.sh ====="
BUILD_START=$(date +%s)
echo "Build started at: $(date)"
bash build.sh
BUILD_END=$(date +%s)
echo "Build finished at: $(date) — elapsed $((BUILD_END - BUILD_START))s"

echo ""
echo "===== [6/6] List build artifacts ====="
DIST_CANDIDATES=("${HOME}/fastdeploy/dist" "dist" "build/dist" "python/dist")
FOUND_DIST=""
for candidate in "${DIST_CANDIDATES[@]}"; do
    # Resolve relative paths against FASTDEPLOY_DIR
    if [[ "${candidate}" != /* ]]; then
        candidate="${FASTDEPLOY_DIR}/${candidate}"
    fi
    if [ -d "${candidate}" ]; then
        echo "Found dist directory: ${candidate}"
        ls -lh "${candidate}"
        FOUND_DIST="${candidate}"
        break
    fi
done
if [ -z "${FOUND_DIST}" ]; then
    echo "WARNING: Could not find dist directory. Searching for .whl files..."
    find "${FASTDEPLOY_DIR}" -name "*.whl" 2>/dev/null | head -20
fi

echo ""
echo "===== [DONE] FastDeploy build complete ====="
echo "Next step: bash task1-warmup/scripts/03_verify_install.sh"
