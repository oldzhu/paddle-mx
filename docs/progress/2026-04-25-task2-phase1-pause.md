# Task 2 Phase 1 — Pause & Resume Document

**Date**: 2026-04-25 (updated end-of-day)
**Status**: TASK 1 COMPLETE — resuming tomorrow at Task 2 Phase 1 (server launch)

---

## TASK 1 — DONE ✅

### What was accomplished
- Fixed `~/cu-bridge` broken symlinks (all point to `/opt/maca-3.3.0/tools/cu-bridge/bin/`)
- Compiled GPU custom ops via `setup_ops.py` → `fastdeploy_ops_pd_.so` (33 MB, MetaX `mxcc 1.0.0`)
- Built wheel: `/root/FastDeploy/dist/fastdeploy_metax_gpu-2.5.0-py3-none-any.whl` (9.5 MB)
- Installed with `pip install --no-deps --force-reinstall`
- Patched `__init__.py` line 63: `raise ImportError` → `logging.warning` (non-fatal model registry)
- `pip show fastdeploy-metax-gpu` confirmed: **Name: fastdeploy-metax-gpu, Version: 2.5.0**
- GPU ops `.so` in place: `/opt/conda/lib/python3.10/site-packages/fastdeploy/model_executor/ops/gpu/fastdeploy_ops/fastdeploy_ops_pd_.so`

### Key env fix (CRITICAL — do not forget)
This GiteeAI instance has MACA at `/opt/maca-3.3.0`, NOT `/opt/maca`:
```bash
export MACA_PATH=/opt/maca-3.3.0   # NOT /opt/maca
export CUCC_PATH=/root/cu-bridge
export CUCC_CMAKE_ENTRY=2
export CUDA_PATH=/root/cu-bridge/CUDA_DIR
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export MACA_VISIBLE_DEVICES=0
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_MOE_BACKEND=cutlass
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_ENC_DEC_BLOCK_NUM=2
export FD_SAMPLING_CLASS=rejection
```

### Task 1 打卡 (if not yet sent)
Run in JupyterLab terminal, screenshot the output:
```bash
/opt/conda/bin/pip install /root/FastDeploy/dist/fastdeploy_metax_gpu-2.5.0-py3-none-any.whl --no-deps --force-reinstall 2>&1 | grep -v WARNING && \
echo "" && \
/opt/conda/bin/pip show fastdeploy-metax-gpu
```
Email to: ext_paddle_oss@baidu.com, kaichuang.gao@metax-tech.com, yang.yang2@metax-tech.com
Subject: `文心伙伴赛道-沐曦-打卡-【GithubID:oldzhu】`

---

---

## Summary of What Was Done (This Session)

### Step 1 — Environment Setup (COMPLETE)
All required packages installed in conda Python 3.10 (`/opt/conda/bin/python`):

| Package | Version | Notes |
|---------|---------|-------|
| paddlepaddle | 3.4.0.dev20251223 | CPU wheel from paddle nightly |
| paddle-metax-gpu | 3.3.0.dev20251224 | MetaX GPU plugin for Paddle |
| fastdeploy-cpu | 2.5.0 | ⚠️ **WRONG WHEEL** — Task 1 requires MetaX GPU build → `fastdeploy-metax-gpu`, not CPU-only wheel |
| paddleocr | 3.5.0 | with `[doc-parser]` extras |
| paddlex | 3.5.1 | auto-installed with paddleocr |
| opencv-contrib-python-headless | 4.10.0.84 | pinned version |
| flash-mask | 4.0.post20260128 | required by FastDeploy deepseek_v3 module |
| aiozmq | 1.0.0 | required by FastDeploy api_server |

GPU device detection confirmed working:
```
CustomDevice: metax_gpu, visible devices count: 1
paddle: 3.4.0.dev20251223
devices: ['metax_gpu']
```

### Step 2 — Model Downloaded (COMPLETE)
PaddleOCR-VL model (0.9B) downloaded to `/data/models/PaddlePaddle/PaddleOCR-VL`  
Size: ~2.0 GB, 22 files confirmed.

### Step 3 — FastDeploy Server Debugging (BLOCKED)

Worked through several import/startup errors in sequence:

| Error | Root Cause | Fix Applied |
|-------|-----------|-------------|
| `expected str, bytes or os.PathLike object, not NoneType` in triton metax driver | `MACA_PATH` env var not set; `maca_home_dirs()` returned `None` | Set `MACA_PATH=/opt/maca-3.3.0` in all launch commands |
| `ImportError: module_file='deepseek_v3'` (fatal) | `auto_models_registry()` re-raises any single model import failure, blocking ALL models including PaddleOCR-VL | **Patched** `/opt/conda/lib/python3.10/site-packages/fastdeploy/model_executor/models/__init__.py` line 63: `raise ImportError` → `logging.warning` (non-fatal) |
| `No module named 'aiozmq'` | missing dependency | `pip install aiozmq` |
| Worker crash: `cannot import name 'fused_rotary_position_encoding' from fastdeploy.model_executor.ops.gpu` | Root cause confirmed: **wrong wheel installed**. Task 1 build requires MetaX GPU env (cu-bridge + MACA compiler) to produce `fastdeploy_metax_gpu-2.5.0-*.whl`. We only built `fastdeploy_cpu` (CPU-only). | **NOT YET FIXED — real fix is re-running `bash build.sh` with MetaX env** |

### Current Blocker: Wrong FastDeploy Wheel — Must Rebuild with MetaX GPU Support

Task 1 requires the **MetaX GPU wheel** (`fastdeploy-metax-gpu`), not the CPU-only wheel we built. The correct `bash build.sh` with MetaX env compiles all `gpu_ops/*.cu` + `metax_ops/*.cu` and produces a wheel that includes the native ops `.so`.

**Verification from Task 1 page**: `pip show fastdeploy-metax-gpu` (not fastdeploy-cpu)

**Correct build procedure** (from Task 1 page):
```bash
export MACA_PATH=/opt/maca   # NOTE: /opt/maca, NOT /opt/maca-3.3.0

if [ ! -d ${HOME}/cu-bridge ]; then
  `${MACA_PATH}/tools/cu-bridge/tools/pre_make`
fi

export CUCC_PATH=/opt/maca/tools/cu-bridge
export CUCC_CMAKE_ENTRY=2
export CUDA_PATH=${HOME}/cu-bridge/CUDA_DIR
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export MACA_VISIBLE_DEVICES="0"
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_MOE_BACKEND=cutlass
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_ENC_DEC_BLOCK_NUM=2
export FD_SAMPLING_CLASS=rejection

cd /root/FastDeploy
bash build.sh
```
Then install: `pip install dist/fastdeploy_metax_gpu-2.5.0-*.whl`

---

## TASK 2 — Resume Plan (start here tomorrow)

### Pre-conditions (all satisfied)
- [x] `fastdeploy-metax-gpu 2.5.0` installed with GPU ops `.so`
- [x] `paddle-metax-gpu 3.3.0.dev20251224` installed
- [x] Model at `/data/models/PaddlePaddle/PaddleOCR-VL/` (2GB, 22 files)
- [x] `__init__.py` patched (non-fatal model registry)
- [x] Task 1 打卡 email sent (or send first thing)

### Step B — Launch FastDeploy server with MetaX env

```bash
export MACA_PATH=/opt/maca-3.3.0
export MACA_VISIBLE_DEVICES=0
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_METAX_KVCACHE_MEM=6
export FD_MOE_BACKEND=cutlass
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_ENC_DEC_BLOCK_NUM=2
export FD_SAMPLING_CLASS=rejection
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

nohup /opt/conda/bin/python -m fastdeploy.entrypoints.openai.api_server \
    --model /data/models/PaddlePaddle/PaddleOCR-VL \
    --port 8118 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4 \
    --workers 1 \
    > /tmp/fd_server.log 2>&1 &

tail -f /tmp/fd_server.log  # wait for "Application startup complete"
```

### Step C — Test inference

```bash
curl http://127.0.0.1:8118/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"PaddleOCR-VL","messages":[{"role":"user","content":"Hello, please introduce yourself."}],"max_tokens":100}'
```

### Step D — mcTracer profiling

```bash
mkdir -p /data/trace
export MACA_PATH=/opt/maca-3.3.0
/opt/maca-3.3.0/bin/mcTracer --mctx --odname /data/trace/paddleocr_vl_001 \
  /opt/conda/bin/python /root/FastDeploy/benchmarks/paddleocr_vl/benchmark.py
```
If no dedicated benchmark script, profile the curl inference instead using mcTracer wrapping.

### Step E — Write Phase 1 RFC report

Fill actual profiling data into:
- `task2-optimization/profiling/rfcs/perf-analysis-report_001.md` (EN)
- `task2-optimization/profiling/rfcs/perf-analysis-report_001.zh.md` (ZH)

Required content: kernel names, timings, GPU utilization, ≥5 bottleneck analyses, optimization suggestions.

### Step F — User reviews → submit Phase 1 PR

**CRITICAL**: Show user the draft report before submitting.
Target repo: `PaddlePaddle/community`, path: `rfcs/FastDeploy/`

---

## Remote Server Info

| Item | Value |
|------|-------|
| SSH host | `140.207.205.81:32222` |
| SSH user | `root+vm-1Fe2g2PVUjoRh4Zq` |
| SSH password | `$GITEEAI_PASS` env var (local) |
| Python | `/opt/conda/bin/python` (3.10) |
| MACA SDK | `/opt/maca-3.3.0/` ← always use this, NOT `/opt/maca` |
| Model | `/data/models/PaddlePaddle/PaddleOCR-VL/` |
| FastDeploy source | `/root/FastDeploy/` (release/2.5) |
| GPU ops .so | `.../ops/gpu/fastdeploy_ops/fastdeploy_ops_pd_.so` (33MB) |
| Server log | `/tmp/fd_server.log` |
| Worker log | `/root/log/workerlog.0` |
