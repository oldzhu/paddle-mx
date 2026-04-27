# Task 2 Phase 1 — 暂停与恢复文档

**日期**: 2026-04-25（当日结束时更新）
**状态**: Task 1 已完成 ✅ — 明天从 Task 2 Phase 1（启动服务器）继续

---

## 本次会话完成内容

### 步骤一 — 环境配置（已完成）

在 conda Python 3.10（`/opt/conda/bin/python`）中安装所有依赖：

| 包名 | 版本 | 说明 |
|------|------|------|
| paddlepaddle | 3.4.0.dev20251223 | CPU wheel，来自 paddle nightly |
| paddle-metax-gpu | 3.3.0.dev20251224 | MetaX GPU 插件 |
| fastdeploy-cpu | 2.5.0 | ⚠️ **误用 wheel** — Task 1 需要 MetaX GPU 构建 → `fastdeploy-metax-gpu`，不是 CPU-only wheel |
| paddleocr | 3.5.0 | 含 `[doc-parser]` 扩展 |
| paddlex | 3.5.1 | 随 paddleocr 自动安装 |
| opencv-contrib-python-headless | 4.10.0.84 | 固定版本 |
| flash-mask | 4.0.post20260128 | FastDeploy deepseek_v3 模块依赖 |
| aiozmq | 1.0.0 | FastDeploy api_server 依赖 |

GPU 设备检测确认正常：
```
CustomDevice: metax_gpu, visible devices count: 1
paddle: 3.4.0.dev20251223
devices: ['metax_gpu']
```

### 步骤二 — 模型下载（已完成）

PaddleOCR-VL 模型（0.9B）已下载至 `/data/models/PaddlePaddle/PaddleOCR-VL`  
大小约 2.0 GB，共 22 个文件，确认完整。

### 步骤三 — FastDeploy 服务启动调试（遇阻）

依次排查并修复了多个启动错误：

| 错误 | 根本原因 | 解决措施 |
|------|---------|---------|
| triton metax driver 中 `NoneType` 路径错误 | `MACA_PATH` 环境变量未设置，`maca_home_dirs()` 返回 `None` | 所有启动命令中设置 `MACA_PATH=/opt/maca-3.3.0` |
| `ImportError: module_file='deepseek_v3'`（致命错误） | `auto_models_registry()` 将任意单个模型的导入失败抛出为致命错误，导致包括 PaddleOCR-VL 在内的所有模型都无法加载 | **已打补丁**：将 `__init__.py` 第 63 行的 `raise ImportError` 改为 `logging.warning`（非致命）|
| `No module named 'aiozmq'` | 缺少依赖 | `pip install aiozmq` |
| Worker 崩溃：`cannot import name 'fused_rotary_position_encoding'` | 根本原因已确认：**安装了错误的 wheel**。Task 1 需要用 MetaX GPU 环境（cu-bridge + MACA 编译器）编译，产出 `fastdeploy_metax_gpu-2.5.0-*.whl`。我们只编译了 `fastdeploy_cpu`（纯 CPU）。| **尚未解决 — 真正修复方案是用 MetaX 环境重新运行 `bash build.sh`** |

### 当前阻塞点：误用了 FastDeploy wheel — 必须重新编译 MetaX GPU 版本

Task 1 需要 **MetaX GPU wheel**（`fastdeploy-metax-gpu`），不是我们编译的 CPU-only wheel。使用 MetaX 环境运行 `bash build.sh`，会编译所有 `gpu_ops/*.cu` + `metax_ops/*.cu`，并产出含有原生算子 `.so` 的 wheel。

**Task 1 页面验证命令**：`pip show fastdeploy-metax-gpu`（不是 fastdeploy-cpu）

**正确编译步骤**（来自 Task 1 页面）：
```bash
export MACA_PATH=/opt/maca   # 注意：/opt/maca，不是 /opt/maca-3.3.0

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
安装：`pip install dist/fastdeploy_metax_gpu-2.5.0-*.whl`

---

## 远程服务器已修改的文件

| 文件 | 修改内容 |
|------|---------|
| `/opt/conda/lib/python3.10/site-packages/fastdeploy/model_executor/models/__init__.py` 第 63 行 | `raise ImportError(...)` → `logging.warning(...)` — 使模型注册表导入失败非致命化 |

---

## 所有服务/编译操作所需环境变量（编译）

```bash
export MACA_PATH=/opt/maca   # 官方规范用 /opt/maca（/opt/maca-3.3.0 的软链接）
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
```

## 所有服务/编译操作所需环境变量（服务启动 — GPU wheel 安装后）

```bash
export MACA_PATH=/opt/maca
export MACA_VISIBLE_DEVICES=0
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_METAX_KVCACHE_MEM=6
export FD_MOE_BACKEND=cutlass
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
```

---

## 恢复计划（分步骤）

### 步骤 A — 重新编译 FastDeploy MetaX GPU 版本（真正修复方案）

先卸载 CPU wheel，再重新编译：
```bash
/opt/conda/bin/pip uninstall fastdeploy-cpu -y

export MACA_PATH=/opt/maca
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
bash build.sh 2>&1 | tee /tmp/build_metax.log

# 安装 MetaX GPU wheel
/opt/conda/bin/pip install dist/fastdeploy_metax_gpu-2.5.0-*.whl
/opt/conda/bin/pip show fastdeploy-metax-gpu  # 验证
```

### 步骤 B — 启动 FastDeploy 服务

```bash
export MACA_PATH=/opt/maca-3.3.0
export MACA_VISIBLE_DEVICES=0
export PADDLE_XCCL_BACKEND=metax_gpu
export FLAGS_weight_only_linear_arch=80
export FD_METAX_KVCACHE_MEM=6
export LD_LIBRARY_PATH=/opt/maca-3.3.0/lib:/opt/maca-3.3.0/lib64:$LD_LIBRARY_PATH

nohup /opt/conda/bin/python -m fastdeploy.entrypoints.openai.api_server \
    --model /data/models/PaddlePaddle/PaddleOCR-VL \
    --port 8118 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4 \
    --workers 1 \
    > /tmp/fd_server.log 2>&1 &
```

### 步骤 C — 测试推理

```bash
curl http://127.0.0.1:8118/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"PaddleOCR-VL","messages":[{"role":"user","content":"你好，请介绍一下自己。"}],"max_tokens":100}'
```

### 步骤 D — mcTracer 性能采集

```bash
/opt/maca/bin/mcTracer --mctx --odname /data/trace/paddleocr_vl_001 \
  '/opt/conda/bin/python /root/FastDeploy/benchmarks/paddleocr_vl/benchmark.py ...'
```

### 步骤 E — 撰写 Phase 1 RFC 报告

填充 `task2-optimization/profiling/rfcs/perf-analysis-report_001.zh.md`，须包含：
- 实际 profiling 数据（kernel 名称、耗时、GPU 利用率）
- ≥5 个 kernel 函数分析
- 瓶颈识别与优化建议

### 步骤 F — 用户审阅 → 提交 Phase 1 PR

**重要**：提交前务必让用户审阅报告草稿。  
提交目标：`https://github.com/PaddlePaddle/community/tree/master/rfcs/FastDeploy`

---

## 关键文件路径（远程服务器）

| 路径 | 说明 |
|------|------|
| `/data/models/PaddlePaddle/PaddleOCR-VL/` | 模型权重（2GB）|
| `/root/FastDeploy/` | FastDeploy 2.5 源码 |
| `/root/FastDeploy/custom_ops/` | GPU 算子源码（MetaX + CUDA）|
| `/root/FastDeploy/benchmarks/paddleocr_vl/` | 基准测试脚本 + YAML |
| `/opt/maca-3.3.0/` | MACA SDK（即 MACA_PATH）|
| `/opt/maca/bin/mcTracer` | GPU profiler 工具 |
| `/opt/conda/bin/python` | Conda Python 3.10（必须用此路径！）|
| `/tmp/fd_server.log` | FastDeploy 服务日志 |
| `/root/log/workerlog.0` | Worker 进程日志 |

---

## SSH 连接信息

```
Host: 140.207.205.81
Port: 32222
User: root+vm-1Fe2g2PVUjoRh4Zq
Password: $GITEEAI_PASS 环境变量
```
