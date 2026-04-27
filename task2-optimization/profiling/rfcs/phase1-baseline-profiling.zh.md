# FastDeploy 2.5 — PaddleOCR-VL-1.5 在 MetaX C500 上的 Phase 1 基线性能分析

**状态：** 草稿  
**日期：** 2026-04-27  
**作者：** paddle-mx 团队  
**目标：** 提交至 PaddlePaddle/community rfcs/FastDeploy/

---

## 1. 背景

本文档报告了在 **MetaX C500** GPU（MACA 3.3.0）上通过 **FastDeploy 2.5** 运行 **PaddleOCR-VL-1.5**（0.9B 参数视觉-语言模型）的基线性能数据。

Phase 1 的目标是建立可复现的基线，识别主要瓶颈，并提出优化计划，目标是端到端吞吐量提升 ≥ 20%。

---

## 2. 环境信息

### 2.1 硬件配置

| 组件 | 规格 |
|------|------|
| GPU | MetaX C500 |
| GPU 显存 | 65,536 MiB（64 GB） |
| GPU TDP | 350 W |
| MACA 驱动 | 3.3.0.15 |
| CPU | Intel Core i7-8550U @ 1.80 GHz（4C/8T） |
| 系统内存 | 128 GB |

### 2.2 软件环境

| 组件 | 版本 |
|------|------|
| FastDeploy | 2.5.0（metax-gpu wheel） |
| MACA Runtime | 3.3.0.15 |
| Python | 3.10（Miniconda） |
| PaddlePaddle | 3.0.0b2 |
| 操作系统 | Linux（Docker 容器） |

### 2.3 服务启动配置

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model /data/models/PaddlePaddle/PaddleOCR-VL \
  --port 8118 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --workers 1 \
  --graph-optimization-config '{"use_cudagraph": false}'
```

关键参数说明：
- `use_cudagraph: false` — MetaX 不支持 MACA 图捕获，必须禁用
- `do_profile` — 启用 KV 缓存自动调优（分配了 5,461 个 GPU 块）
- `enable_chunked_prefill` — 启用分块预填充（改善长提示的批处理效率）

---

## 3. 模型架构

**模型：** PaddleOCR-VL-1.5（`paddleocr_vl` 架构）

| 组件 | 规格 |
|------|------|
| 总参数量 | ~0.9B |
| 磁盘大小 | 1,828 MB（bfloat16 safetensors） |
| 文本解码器层数 | 18 |
| 文本隐藏层维度 | 1,024 |
| 文本注意力头数 | 16（Q）/ 2（KV）— GQA 分组查询注意力 |
| 词表大小 | 103,424 |
| 最大序列长度 | 131,072 |
| 视觉编码器层数 | 27（SigLIP 架构） |
| 视觉隐藏层维度 | 1,152 |
| KV 缓存精度 | bfloat16 |

---

## 4. KV 缓存分配

FastDeploy 在启动时对 GPU 内存进行了自动调优：

| 指标 | 数值 |
|------|------|
| 可用 KV 缓存内存 | 6.0 GB |
| 每块内存占用 | ~1.1 MB（64 tokens × 18 层 × 2 头 × 128 维 × BF16） |
| 分配的 KV 块总数 | 5,461 |
| Worker 进程 GPU 内存使用 | 9,555 MiB（~9.3 GB） |
| — 模型权重 | ~1.8 GB |
| — KV 缓存 | ~5.8 GB |
| — 框架开销 | ~1.7 GB |

---

## 5. 基线延迟 / 吞吐量测量结果

所有测量均在 **JIT 预热之后** 进行（单请求并发，温缓存状态）。

### 5.1 按输出长度的解码吞吐量

| 测试用例 | 输入 Token 数 | 最大输出 | 平均延迟（s） | P50 延迟（s） | 解码速度（tok/s） |
|----------|:------------:|:--------:|:------------:|:------------:|:-----------------:|
| 短文本（16 tok 输出） | 14 | 16 | 1.823 | 1.859 | **8.78** |
| 中文本（32 tok 输出） | 35 | 32 | 2.139 | 2.246 | **11.46** |
| 长文本（64 tok 输出） | ~60 | 64 | 6.239 | 6.160 | **9.82** |

**预热后平均解码速度：~10 tok/s**（batch size = 1）

### 5.2 首次请求（冷启动）延迟

| 阶段 | 耗时 |
|------|------|
| JIT/SOT 内核编译（首次请求 TTFT） | **~230 秒** |
| 预热后 TTFT（14 token 提示） | ~0.5 秒 |
| 预热后 TTFT（60 token 提示） | ~1.5 秒 |

首次冷启动 230 秒的延迟由 MACA JIT 编译计算内核引起。后续请求的延迟恢复正常。

### 5.3 GPU 利用率

- **空闲状态：** GPU 利用率 0%，功耗 ~66 W
- **推理期间：** 功耗升至 ~70 W（通过 `mx-smi` 测量）
- **显存使用：** 推理过程中稳定在 ~9,553 MiB

---

## 6. 瓶颈分析

### 6.1 JIT 编译开销（关键瓶颈 — 冷启动 230 秒）

**根本原因：** MetaX MACA 不预编译静态计算内核。在第一次前向传播时，SOT（静态运算树）框架会 JIT 编译所有等效 CUDA 内核，导致约 230 秒的启动延迟。

**影响：** 服务重启后，第一个请求会阻塞约 4 分钟才能正常响应。

**证据：**
- 首次请求 TTFT：08:35:24 → 08:39:14 = 230 秒
- 预热后同类请求 TTFT：~1.5 秒

### 6.2 解码吞吐量偏低（~10 tok/s）

**可能的根本原因：**
1. **无 CUDA 图**（`use_cudagraph: false`）— 每个解码步骤均有内核启动开销。在 NVIDIA GPU 上，CUDA 图可降低 30–50% 的每步开销。
2. **Flash Attention 回退** — Worker 日志显示 `"Only support CUDA version flash attention."`，说明 MetaX **未运行**优化的 Flash Attention 内核，而是退回到标准注意力实现。
3. **SOT 每步动态图开销** — `graph_opt_level=0` 时，每个解码步骤都需要重新执行完整的动态计算图。
4. **GQA 2 个 KV 头** — 虽然节省内存带宽，但注意力内核可能未针对 MetaX 计算单元完全优化。

### 6.3 功耗效率

- GPU 推理期间功耗约 70 W（TDP 的 20%）
- 这与显存带宽受限行为一致：解码阶段受限于 KV 缓存内存读取速度，而非计算能力
- 模型使用 GQA（2 个 KV 头），减少了 KV 内存流量，但 batch=1 时注意力仍是带宽受限

### 6.4 视觉编码器路径

- 当前测试仅使用文本输入；视觉编码器（SigLIP，27 层）未被调用
- 图像输入的额外 TTFT 预计为数秒（视觉编码阶段）

---

## 7. 优化计划（Phase 2 目标）

### 7.1 启用 MACA 内核 AOT 预编译（优先级：高）

**预期收益：冷启动时间减少 90%**

FastDeploy/MACA 应支持内核缓存序列化。通过实现 `KernelCache` 预热功能，将已编译内核保存到磁盘，可消除服务重启后的 230 秒冷启动代价。

### 7.2 启用 SOT 图优化（优先级：高）

**预期收益：解码吞吐量提升 20–30%**

设置 `graph_opt_level=1` 可启用 `sot_warmup()`，对静态解码图进行预追踪和优化。即使禁用 CUDA 图，SOT 预热仍能减少每步的 Python 层开销。

### 7.3 优化 MetaX Flash Attention（优先级：高）

**预期收益：解码吞吐量提升 20–40%**

Worker 报告"只支持 CUDA 版本 flash attention"——MetaX 原生注意力内核未被启用。将 `fastdeploy_ops_pd_.so` 中的 MetaX 自定义 Flash Attention 集成到解码路径，应能恢复完整的 MetaX 注意力性能。

### 7.4 批量大小扩展（优先级：中）

**预期收益：随批量线性扩展吞吐量**

batch=1 时 GPU 利用率低（~20% TDP）。增加并发请求数（配置的最大 batch size 为 4）应能按比例提升 GPU 利用率和聚合吞吐量。

### 7.5 验证推测解码效果（优先级：中）

**预期收益：有效吞吐量提升 15–30%**

FastDeploy 已配置推测解码（`ngram_match`、`mtp`、`suffix`），验证其在 MetaX 上的实际运行状态并测量接受率，可量化实际收益。

---

## 8. 总结

| 指标 | 当前基线 |
|------|---------|
| 冷启动 TTFT | ~230 秒 |
| 预热后 TTFT（14-token 提示） | ~0.5 秒 |
| 解码速度（batch=1） | ~10 tok/s |
| GPU 显存使用量 | 9.3 GB / 64 GB |
| GPU TDP 利用率 | ~20% |
| Flash Attention | ❌ CUDA 回退（未用 MetaX 原生） |
| CUDA 图 | ❌ 已禁用（MACA 不支持） |

**主要瓶颈：** Flash Attention 回退 + 无图优化 → 解码速度约 10 tok/s。  
**优化目标：** 通过 SOT 图优化和 MetaX Flash Attention 内核，解码速度达到 ≥ 12 tok/s（提升 ≥ 20%）。

---

## 9. 附录：复现命令

### 启动服务
```bash
export MACA_PATH=/opt/maca-3.3.0
cd /root
python -m fastdeploy.entrypoints.openai.api_server \
  --model /data/models/PaddlePaddle/PaddleOCR-VL \
  --port 8118 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --workers 1 \
  --graph-optimization-config '{"use_cudagraph": false}' \
  > /tmp/fd_server.log 2>&1 &
```

### 预热（首次启动后必须执行）
```bash
# 首次请求触发 JIT 编译（约 230 秒）
curl -X POST http://localhost:8118/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "/data/models/PaddlePaddle/PaddleOCR-VL", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5}'
```

### 延迟基准测试
```bash
python3 /tmp/bench_test.py
```

### 必要补丁（pip 重装后需重新应用）
```bash
# 补丁 1：post_process() 签名修复
python3 /tmp/patch_post_process.py
# 补丁 2：模型注册表修复（raise→warning）
python3 /tmp/patch_init.py
```
