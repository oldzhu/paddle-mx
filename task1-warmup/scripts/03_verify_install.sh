#!/usr/bin/env bash
# 03_verify_install.sh — Install the FastDeploy wheel and verify the installation
# Run this after 02_build_fastdeploy.sh completes successfully.
set -euo pipefail

FASTDEPLOY_DIR="${HOME}/FastDeploy"

echo "===== [1/4] Locate built wheel ====="
# Try common output locations; expand from most to least likely
WHL_FILE=""
for candidate in \
    "${HOME}/fastdeploy/dist/"*.whl \
    "${FASTDEPLOY_DIR}/dist/"*.whl \
    "${FASTDEPLOY_DIR}/build/dist/"*.whl \
    "${FASTDEPLOY_DIR}/python/dist/"*.whl; do
    if [ -f "${candidate}" ]; then
        WHL_FILE="${candidate}"
        echo "Found wheel: ${WHL_FILE}"
        break
    fi
done

if [ -z "${WHL_FILE}" ]; then
    echo "ERROR: No .whl file found. Build may have failed or the output path is different."
    echo "Searching for .whl files under ${HOME}..."
    find "${HOME}" -name "fastdeploy*.whl" 2>/dev/null | head -10
    exit 1
fi

ls -lh "${WHL_FILE}"

echo ""
echo "===== [2/4] Install wheel ====="
pip install "${WHL_FILE}" --force-reinstall

echo ""
echo "===== [3/4] Verify import and version ====="
python -c "
import fastdeploy
print('fastdeploy version:', fastdeploy.__version__)
print('fastdeploy location:', fastdeploy.__file__)
"

echo ""
echo "===== [4/4] Show installed location ====="
pip show fastdeploy

echo ""
echo "===== GPU info ====="
if command -v maca-smi &>/dev/null; then
    maca-smi
else
    echo "maca-smi not in PATH; checking nvidia-smi as fallback..."
    nvidia-smi 2>/dev/null || echo "No GPU query tool found."
fi

echo ""
echo "===== Environment summary (for email) ====="
echo "OS:              $(uname -sr)"
echo "Python:          $(python --version)"
echo "PaddlePaddle:    $(python -c 'import paddle; print(paddle.__version__)' 2>/dev/null || echo 'not installed')"
echo "paddle-metax:    $(pip show paddle-metax-gpu 2>/dev/null | grep Version || echo 'not installed')"
echo "FastDeploy:      $(python -c 'import fastdeploy; print(fastdeploy.__version__)' 2>/dev/null || echo 'not installed')"
echo "MACA:            ${MACA_PATH:-/opt/maca} ($(ls /opt/maca/lib/libmaca.so* 2>/dev/null | head -1 || echo 'lib not found'))"

echo ""
echo "===== [DONE] Verification complete ====="
echo "Screenshot this terminal output for the check-in email."
echo "Next: fill in task1-warmup/email_template.md and send the email."
