# FastDeploy 2.5 — PaddleOCR-VL-1.5 在 MetaX C500 上的阶段一基线性能分析

**状态：** 草稿  
**日期：** 2026-04-28  
**作者：** paddle-mx 团队  
**目标仓库：** PaddlePaddle/community rfcs/FastDeploy/

---

## 1. 背景

本文档报告了通过 **FastDeploy 2.5** 在 **MetaX C500** GPU（MACA 3.3.0）上运行 **PaddleOCR-VL-1.5**（0.9B 参数视觉-语言模型）的基线性能数据。

本阶段一报告的目标是建立**文本推理和图像推理**两条路径的可复现基线，通过 **mcTracer** GPU 内核级性能分析工具定位主要瓶颈，并提出针对端到端吞吐量提升 ≥20% 的优化方案。

---

## 2. 环境配置

### 2.1 硬件

| 组件 | 规格 |
|------|------|
| GPU | MetaX C500 |
| 显存 | 65,536 MiB（64 GB） |
| 功耗上限 | 350 W |
| MACA 驱动 | 3.3.0.15 |
| 系统内存 | 128 GB |

### 2.2 软件

| 组件 | 版本 |
|------|------|
| FastDeploy | 2.5.0（metax-gpu wheel） |
| MACA 运行时 | 3.3.0.15 |
| Python | 3.10（Miniconda） |
| PaddlePaddle | 3.0.0b2 |
| 性能分析工具 | mcTracer 3.3.0.15（附加模式） |
| 操作系统 | Linux（Docker 容器） |

### 2.3 服务器启动配置

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
- `use_cudagraph: false` — MetaX 平台不支持 MACA 图捕获，必须禁用
- `do_profile` — 启用 KV 缓存分析（自动分配 5,461 个 GPU 块）
- `enable_chunked_prefill` — 启用分块预填充（优化长序列批处理）

---

## 3. 模型架构

**模型：** PaddleOCR-VL-1.5（`paddleocr_vl` 架构）

| 组件 | 规格 |
|------|------|
| 总参数量 | ~0.9B |
| 磁盘模型大小 | 1,828 MB（bfloat16 safetensors） |
| 文本解码器层数 | 18 |
| 文本隐藏层维度 | 1,024 |
| 文本注意力头 | 16（Q）/ 2（KV）— GQA |
| 文本词表大小 | 103,424 |
| 最大序列长度 | 131,072 |
| 视觉编码器 | SigLIP-L，27 层 Transformer |
| 视觉隐藏层维度 | 1,152 |
| 图像块大小 | 14×14，每张图输出 609 个 token |
| KV 缓存类型 | bfloat16 |

---

## 4. KV 缓存分配

FastDeploy 在启动时完成 GPU 内存分析：

| 指标 | 数值 |
|------|------|
| 可用 KV 缓存内存 | 6.0 GB |
| 每块内存占用 | ~1.1 MB（64 token，18 层，2 KV 头，维度 128，BF16） |
| 总分配 KV 块数 | 5,461 |
| 启动时 GPU 内存 | ~826 MiB（框架基线） |
| 模型加载后 GPU 内存 | 48,323 MiB（~47.2 GB） |
| — 模型权重 | ~1.8 GB |
| — KV 缓存 | ~5.8 GB |
| — 视觉编码器权重 | ~0.9 GB |
| — 框架 + 激活值 | ~38.7 GB |

---

## 5. 基线延迟与吞吐量测量

以下测量均在 **JIT 预热后**（内核缓存已预热，单请求并发）进行，另有注明者除外。

### 5.1 纯文本解码吞吐量

| 测试场景 | Prompt Token 数 | 平均输出 Token 数 | 平均延迟（秒） | 解码速度（tok/s） |
|---------|:--------------:|:----------------:|:-------------:|:----------------:|
| 短序列 | 14 | 16 | 1.82 | **8.8** |
| 中序列 | 35 | 32 | 2.14 | **11.5** |
| 长序列 | ~60 | 64 | 6.24 | **9.8** |

**预热后平均解码速度（纯文本，batch=1）：~10 tok/s**

### 5.2 图像推理（视觉编码器路径）

测试输入：800×600 发票文档图像，以 `data:image/jpeg;base64,...` 格式编码。

| 阶段 | 测量值 |
|------|--------|
| JIT 冷启动 TTFT（首次图像请求） | **135.2 秒** |
| 预热后图像推理延迟（第1次） | **4.38 秒** |
| 预热后图像推理延迟（第2次） | **4.33 秒** |
| 输入 Prompt Token 数（图像+文本） | 628 |
| — 图像 Token 数（SigLIP 输出） | **609** |
| — 文本 Token 数 | 19 |
| 生成输出 Token 数 | 165 |
| 有效解码速度（预热后） | **37.7 tok/s** |

预热后图像推理吞吐量（37.7 tok/s）显著高于纯文本（~10 tok/s），原因在于图像输入通过单次大型预填充步骤处理（609 个 token），KV 缓存一次性填充后解码即可高效进行。

### 5.3 启动与冷启动延迟

| 阶段 | 时长 |
|------|------|
| 模型权重加载 | ~12 秒 |
| KV 缓存分析与分配 | ~65 秒 |
| 服务器就绪总时间（首次启动） | ~80 秒 |
| JIT 编译（首次文本请求） | ~0.5 秒 |
| JIT 编译（首次图像请求） | **135.2 秒** |
| 完全预热后 TTFT（文本，14 token） | ~0.5 秒 |
| 完全预热后 TTFT（图像，628 token） | **4.33 秒** |

135 秒的图像冷启动源于 MACA 对 SigLIP 视觉编码器特定张量形状的 JIT 编译，这些形状在纯文本预热阶段未被触发。

### 5.4 图像推理期间 GPU 利用率

通过 mcTracer 内核跟踪测量（详见第 6 节），采集于预热后图像推理（4.38 秒挂钟时间）：

| 指标 | 数值 |
|------|------|
| 捕获 GPU 内核总执行时间 | **854 µs** |
| 挂钟推理时间 | 4,380 ms |
| GPU 内核利用率 | **19.5%** |
| GPU 功耗（空闲） | 38 W |
| GPU 功耗（推理中） | 67 W |
| GPU TDP 利用率 | 19.1%（67/350 W） |

GPU 利用率低（~20%）表明**瓶颈在于 Python/CPU 开销**，而非 GPU 算力。

---

## 6. 内核级性能分析（mcTracer）

### 6.1 分析方法

**工具：** mcTracer 3.3.0.15，随 MACA SDK 提供，路径：`/opt/maca-3.3.0/bin/mcTracer`

**方法：** 附加模式——无需重启服务器：
```bash
# 必须从 /root/ 目录运行（mcTracer 会将 --odname 拼接到当前目录）
cd /root
mcTracer --mctx --attach <worker_pid> --odname mctrace_out &
# 同时发送推理请求
curl http://localhost:8118/v1/chat/completions -d '{...图像载荷...}'
# 停止追踪
kill -INT <tracer_pid>
```

**输出格式：** Chrome Trace Event JSON（`tracer_out-<pid>.json`）。注意：mcTracer 的 `ts`（时间戳）和 `dur`（持续时间）字段均以**纳秒**为单位。

**采集范围：** 单次预热后图像推理请求，628 个输入 token（609 图像 + 19 文本），165 个输出 token，挂钟 4.38 秒。跟踪文件：133 MB，501,974 个事件。

### 6.2 GPU 内核执行时间排名（Top 14）

**捕获 GPU 内核总执行时间：854,286 µs（~854 ms）**

| 排名 | 内核函数 | 调用次数 | 总时间（µs） | GPU 占比 | 平均/次（µs） |
|------|---------|--------:|-----------:|--------:|------------:|
| 1 | `flash_fwd_splitkv_kernel<128,16,16,bf16>` | 2,952 | 285,111 | 33.4% | 96.6 |
| 2 | `b16gemvn_splitk_kernel<256,4,4,bf16>` | 5,903 | 105,134 | 12.3% | 17.8 |
| 3 | `b16gemvn_kernel<64,4,4,bf16>` | 2,952 | 54,025 | 6.3% | 18.3 |
| 4 | `phi::RmsNormBlockSMemImpl<bf16>` | 5,903 | 49,777 | 5.8% | 8.4 |
| 5 | `b16gemvn_kernel<64,4,8,bf16>` | 2,952 | 41,656 | 4.9% | 14.1 |
| 6 | `phi::KeMatrixTopPBeamTopKFt<float>` | 164 | 28,226 | 3.3% | 172.1 |
| 7 | `b16gemv_splitk_combine_kernel<bf16>` | 5,903 | 27,152 | 3.2% | 4.6 |
| 8 | `b16gemvn_row_double_buffer_kernel<256,4,8>` | 164 | 23,129 | 2.7% | 141.0 |
| 9 | `DispatchCacheKVWithRopeVecKernel<bf16>` | 2,952 | 21,059 | 2.5% | 7.1 |
| 10 | `cub::DeviceSegmentedRadixSortKernelLarge` | 656 | 20,546 | 2.4% | 31.3 |
| 11 | `phi::fusion::ActFFNGlu<bf16,SwiGLU>` | 2,969 | 18,399 | 2.2% | 6.2 |
| 12 | `memcpy DTOD(device,device)` | 2,483 | 16,952 | 2.0% | 6.8 |
| 13 | `flash_fwd_kernel<96,128,64,4>`（**SigLIP 视觉编码器**） | **27** | **12,939** | **1.5%** | **479.2** |
| 14 | `mcdnn::KernelSoftmaxForwardInstanceLdgB128` | 164 | 12,531 | 1.5% | 76.4 |

### 6.3 关键内核详细分析

#### 内核 1：`flash_fwd_splitkv_kernel` — LLM 逐 Token 解码注意力（GPU 占 33.4%）

**功能：** FlashAttention-2 Split-KV 变体，用于自回归 token 生成。每次调用处理一个解码步骤的一个注意力层：读取当前序列的 KV 缓存块，计算缩放点积注意力。

**执行模式：**
- 调用次数：2,952 = 165 解码步 × 18 LLM 层（理论 2,970）
- Block 形状：`{x:64, y:1, z:1}` — 64 线程 warp 级执行
- 模板参数：`headdim=128, kBlockM=16, kBlockN=16`，BF16 累加
- 平均每次：**96.6 µs**
- 合计：**285 ms**（165 步解码总量）

**关键发现：** 96.6 µs × 18 层 = 1.74 ms/步注意力时间，而每步实际挂钟 ~26.5 ms，注意力内核**仅占每步 6.6%**，其余 93.4% 为 Python 开销。

**优化方向：** 增大并发 batch 大小提高每次注意力的计算强度；启用 MACA 图优化（SOT 预编译）减少逐步 Python 调度开销。

---

#### 内核 2：`b16gemvn_splitk_kernel<256,4,4,bf16>` — LLM 线性投影 GEMV Split-K（GPU 占 12.3%）

**功能：** BF16 通用矩阵向量乘（GEMV）配合 Split-K 并行，用于 LLM 注意力（Q/K/V 投影）和 MLP 层的权重矩阵投影。Split-K 将 K 维度分解到多个线程块，提升小批量（M=1）解码时的并行度。

**执行模式：**
- 调用次数：5,903 ≈ 165 步 × 18 层 × 2（SwiGLU 的 gate + up 投影）
- 平均每次：**17.8 µs**
- 合计：**105 ms**（第二大贡献者）

**优化方向：** 将 split-K GEMV + combine + 激活函数融合为单个内核，减少第 2、7、11 号内核的总 kernel launch 次数。

---

#### 内核 3：`phi::RmsNormBlockSMemImpl<bf16>` — 层归一化（GPU 占 5.8%）

**功能：** 每个注意力和 FFN 子层前应用的 RMSNorm。使用共享内存分块处理隐藏维度（1,024），单次遍历完成 `x / sqrt(mean(x²) + ε) * weight`。

**执行模式：**
- 调用次数：5,903 = 165 步 × 18 层 × 2（注意力前 + FFN 前 RMSNorm）
- 平均每次：**8.4 µs**
- 合计：**50 ms**（占 GPU 时间 5.8%）

**关键发现：** 每次 kernel launch 仅传输 2 KB 数据（1,024 × 2 字节），有效带宽仅 0.24 GB/s——远低于硬件峰值，说明**开销由 kernel launch 延迟主导**，而非实际数据传输。

**优化方向：** 将 RMSNorm 与后续线性层融合为单个内核，消除 5,903 次冗余 kernel launch 和全局内存往返。

---

#### 内核 4：`DispatchCacheKVWithRopeVecKernel<bf16>` — KV 缓存写入 + RoPE（GPU 占 2.5%）

**功能：** 融合内核，同时完成：（1）对 Query 和 Key 张量施加旋转位置编码（RoPE）；（2）将新的 K/V 张量写入分页 KV 缓存块。每个解码步骤的每个注意力层调用一次。

**执行模式：**
- 调用次数：2,952 = 165 步 × 18 层
- 平均每次：**7.1 µs**
- 合计：**21 ms**（占 GPU 时间 2.5%）

**优化方向：** 该内核已充分融合（RoPE + 缓存写入）。对于更长序列（更多 KV 块），FlashAttention 内置的 KV 缓存访问效率更高。当前规模下无需优化。

---

#### 内核 5：`phi::fusion::ActFFNGlu<bf16,SwiGLU>` — 融合 SwiGLU FFN 激活（GPU 占 2.2%）

**功能：** 融合的逐元素计算：`output = silu(gate) * up_proj`，将 SwiGLU 激活与逐元素乘法合并在单次遍历中完成（本模型 FFN 中间层 ~2,816 维）。

**执行模式：**
- 调用次数：2,969 ≈ 165 步 × 18 LLM 层 + 27 SigLIP FFN 层
- 平均每次：**6.2 µs**
- 合计：**18 ms**（占 GPU 时间 2.2%）

**关键发现：** 调用次数 2,969 确认此内核同时在 LLM（~2,970 次）和视觉编码器（~27 次）中执行，覆盖两条推理路径。

---

#### 内核 6：`flash_fwd_kernel<96,128,64,4>` — SigLIP 视觉编码器注意力（GPU 占 1.5%）

**功能：** 用于 SigLIP-L 视觉编码器的标准 FlashAttention-2 预填充内核。与 LLM 解码注意力（Split-KV）不同，此内核一次性并行处理全部 609 个图像块 token——单层 Transformer 的完整预填充。

**区分视觉编码器与 LLM 解码器的关键标识：**
- `flash_fwd_kernel`（无 `splitkv` 后缀）→ 预填充模式
- 模板参数 `headdim=64`（SigLIP 使用 64 维注意力头，LLM 使用 128 维）
- 恰好 **27 次调用** = 对应 SigLIP-L 的 27 个 Transformer 层（已证实）

**执行模式：**
- 调用次数：**27**（每层一次）
- 平均每次：**479 µs**——因处理 609 token 序列，比 LLM 解码注意力（96 µs）慢 5 倍
- 合计：**12.9 ms**（完整视觉编码，27 层全部注意力）

**关键发现：** SigLIP 编码器仅耗费 12.9 ms GPU 时间即完成图像编码，占总 GPU 内核时间的 1.5%。与 165 步解码（~841 ms GPU 时间）相比，**视觉编码器不是性能瓶颈**。

**优化方向：** 对于处理多图的批量 OCR 工作负载，批量化 SigLIP 编码可显著摊薄 27 层注意力开销；动态分辨率（简单图像使用更少 patch）可减少视觉 GPU 时间。

---

### 6.4 GPU 时间预算汇总

```
挂钟总时间：                 4,380 ms (100%)
│
├── GPU 内核时间：              854 ms ( 19.5%)
│   ├─ LLM 解码注意力：         285 ms ( 33.4% of GPU)
│   ├─ LLM GEMV 投影合计：      228 ms ( 26.7% of GPU)
│   │   ├─ b16gemvn_splitk：   105 ms
│   │   ├─ b16gemvn（2种）：     96 ms
│   │   └─ splitk_combine：     27 ms
│   ├─ RMSNorm：                 50 ms (  5.8% of GPU)
│   ├─ TopK/TopP 采样：          29 ms (  3.3% of GPU)
│   ├─ KV 缓存 + RoPE：          21 ms (  2.5% of GPU)
│   ├─ 融合 ActFFNGlu：          18 ms (  2.2% of GPU)
│   ├─ 设备间内存拷贝：          17 ms (  2.0% of GPU)
│   ├─ 视觉编码器（SigLIP）：    13 ms (  1.5% of GPU)
│   ├─ Softmax（采样）：         13 ms (  1.5% of GPU)
│   └─ 其他内核：               180 ms ( 21.1% of GPU)
│
└── CPU/Python 开销：          3,526 ms ( 80.5%)
    ├─ HTTP/JSON 解析：~10 ms
    ├─ IPC 队列（引擎 ↔ Worker）：~50 ms
    ├─ Tokenizer + 调度器：~30 ms
    └─ 每解码步 Python 调度（×165）：~3,400 ms
```

**核心发现：GPU 80.5% 的时间处于空闲状态。** 每步解码约 21 ms 的 Python 调度开销是主要瓶颈，而非 GPU 算力或内存带宽。

---

## 7. 瓶颈总结

### 7.1 Python/CPU 序列化开销（关键——占挂钟时间 80.5%）

**根本原因：** `use_cudagraph: false` 且 `graph_opt_level: 0` 的情况下，165 个解码步骤中的每一步都需要 Python 完成 IPC 队列轮询、完整计算图重新调度和独立 kernel launch。导致每 token 约 21 ms 的 Python 开销。

**影响：** 将此开销降低 50%（通过 `graph_opt_level=1` 的 SOT 预编译）可实现 >30% 的吞吐量提升。

### 7.2 新形状的 JIT 编译（关键——图像冷启动 135 秒）

**根本原因：** MACA 在每种新张量形状首次执行时进行 JIT 编译。图像推理引入了 SigLIP 专属形状（609 token 预填充，headdim=64，27 层），纯文本预热阶段未触发这些形状。

**影响：** 服务器重启或首次图像请求需等待 135 秒。通过 AOT 内核缓存可消除此惩罚（仅首次安装时触发）。

### 7.3 GPU 利用率偏低（~20%）

**根本原因：** 短时高频 kernel launch（注意力 96 µs，RMSNorm 8 µs）之间存在 Python 调度间隙。缺少 CUDA Graph 批量捕获内核序列。

**影响：** GPU TDP 350W，实际仅运行在 67W——81% 的算力闲置。

---

## 8. 优化方案（阶段二目标）

### 8.1 SOT 图预编译（优先级：高，预期增益 ≥20%）

启用 `graph_opt_level=1` 激活 SOT 静态操作树预追踪，为所有预热大小（`[1,2,...,128]`）缓存解码计算图，消除 Python 逐步重新调度开销。

**目标：** 每步 Python 开销从 ~21 ms 降至 <10 ms → 吞吐量：10 → ≥13 tok/s（+30%）。

### 8.2 视觉编码器 AOT 内核缓存（优先级：高）

将首次编译的 MACA 内核序列化到磁盘，后续服务器启动直接加载缓存，跳过 135 秒 JIT 编译。

**目标：** 图像冷启动 TTFT：135 秒 → 初次安装后 <5 秒。

### 8.3 并发请求批处理（优先级：中）

增大活跃并发数（`--max-num-seqs`）。batch=2 时每次注意力处理 2 个解码位置，对内存带宽受限的工作负载实现近线性吞吐量扩展。

**目标：** batch=2 时聚合吞吐量 ~18–20 tok/s（+80–100% 系统总吞吐）。

### 8.4 RMSNorm + 线性层融合（优先级：中）

将 5,903 次 RMSNorm 调用与后续线性投影内核融合，消除冗余 kernel launch 和全局内存往返（当前占 GPU 时间 5.8%）。

**目标：** 减少 kernel launch 开销带来 +5–8% GPU 吞吐提升。

---

## 9. 总结表

| 指标 | 基线测量值 |
|------|-----------|
| 服务器就绪时间 | ~80 秒 |
| 图像 JIT 冷启动（首次请求） | **135.2 秒** |
| 文本 JIT 冷启动 | ~0.5 秒 |
| 预热后 TTFT（文本，14 token） | ~0.5 秒 |
| 预热后 TTFT（图像，628 token） | **4.38 秒** |
| 解码速度——纯文本（batch=1） | **~10 tok/s** |
| 解码速度——图像输入（batch=1） | **37.7 tok/s** |
| 每次图像请求的图像 token 数 | **609**（SigLIP，27 层） |
| 视觉编码器 GPU 时间 | **12.9 ms**（GPU 时间占比 1.5%） |
| GPU 内核利用率 | **19.5%** |
| GPU TDP 利用率 | **19.1%** |
| 主要瓶颈 | Python 调度（占挂钟时间 80.5%） |
| 最大 GPU 内核 | FlashAttention 解码（33.4%） |
| CUDA Graph | ❌ 已禁用（MACA 不支持） |
| SOT 图优化 | ❌ 已禁用（`graph_opt_level=0`） |
| mcTracer 跟踪文件 | `tracer_out-3423.json`（133 MB，501,974 事件） |

**阶段二目标：** 通过 SOT 图预编译实现解码速度 ≥12 tok/s（≥20% 提升）。

---

## 10. 附录：复现命令

### 启动服务器
```bash
export MACA_PATH=/opt/maca-3.3.0
export PATH=/opt/maca-3.3.0/bin:/opt/conda/bin:$PATH

# 安装 pymxsml（GPU 内存分析依赖，一次性安装）
pip install /opt/maca-3.3.0/share/mxsml/pymxsml-2.2.9-py3-none-any.whl

/opt/conda/bin/python -m fastdeploy.entrypoints.openai.api_server \
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

### 图像推理测试（Python）
```python
import base64, requests, time

with open("/tmp/test_doc.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "/data/models/PaddlePaddle/PaddleOCR-VL",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "提取此文档图像中的所有文字。"}
        ]
    }],
    "max_tokens": 200
}

t0 = time.time()
r = requests.post("http://localhost:8118/v1/chat/completions", json=payload)
print(f"状态: {r.status_code}, 耗时: {time.time()-t0:.2f}s")
print(r.json()["choices"][0]["message"]["content"])
```

### mcTracer 内核分析（附加模式）
```bash
# 1. 找到 Worker 进程 PID
WORKER_PID=$(pgrep -f "worker_process.py" | head -1)

# 2. 在后台启动 mcTracer（必须从 /root/ 运行）
cd /root
/opt/maca-3.3.0/bin/mcTracer --mctx --attach $WORKER_PID --odname mctrace_out &
TRACER_PID=$!
sleep 2

# 3. 发送推理请求
/opt/conda/bin/python /tmp/infer_image.py

# 4. 停止追踪
kill -INT $TRACER_PID
sleep 3

# 5. 修复截断的 JSON（SIGINT 可能导致 JSON 不完整）
echo "]}" >> /root/mctrace_out/tracer_out-${WORKER_PID}.json
```

### 必要补丁（pip 重装后需重新应用）
```bash
python3 /tmp/patch_post_process.py   # post_process() 函数签名修复
python3 /tmp/patch_init.py           # 模型注册 raise→warning 修复
```
