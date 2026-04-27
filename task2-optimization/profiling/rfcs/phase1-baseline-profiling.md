# FastDeploy 2.5 — PaddleOCR-VL-1.5 on MetaX C500: Phase 1 Baseline Profiling

**Status:** Draft  
**Date:** 2026-04-27  
**Author:** paddle-mx team  
**Target:** PaddlePaddle/community rfcs/FastDeploy/

---

## 1. Background

This document reports baseline performance of running **PaddleOCR-VL-1.5** (0.9 B parameter vision-language model) through **FastDeploy 2.5** on a **MetaX C500** GPU under **MACA 3.3.0**.

The goal of this Phase 1 report is to establish a reproducible baseline, identify primary bottlenecks, and propose an optimization plan targeting ≥ 20% end-to-end throughput improvement.

---

## 2. Environment

### 2.1 Hardware

| Component | Spec |
|-----------|------|
| GPU | MetaX C500 |
| GPU Memory | 65,536 MiB (64 GB) |
| GPU TDP | 350 W |
| MACA Driver | 3.3.0.15 |
| CPU | Intel Core i7-8550U @ 1.80 GHz (4C/8T) |
| System RAM | 128 GB |

### 2.2 Software

| Component | Version |
|-----------|---------|
| FastDeploy | 2.5.0 (metax-gpu wheel) |
| MACA Runtime | 3.3.0.15 |
| Python | 3.10 (Miniconda) |
| PaddlePaddle | 3.0.0b2 |
| OS | Linux (Docker container) |

### 2.3 Server Launch Configuration

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

Key flags:
- `use_cudagraph: false` — required on MetaX (MACA graph capture unsupported)
- `do_profile` — KV cache profiling enabled (auto-tuned 5,461 GPU blocks)
- `enable_chunked_prefill` — active (improves batching for long prompts)

---

## 3. Model Architecture

**Model:** PaddleOCR-VL-1.5 (`paddleocr_vl` architecture)

| Component | Spec |
|-----------|------|
| Total Parameters | ~0.9 B |
| Model Size on Disk | 1,828 MB (bfloat16 safetensors) |
| Text Decoder Layers | 18 |
| Text Hidden Size | 1,024 |
| Text Attention Heads | 16 (Q) / 2 (KV) — GQA |
| Text Vocab Size | 103,424 |
| Max Sequence Length | 131,072 |
| Vision Encoder Layers | 27 (SigLIP-style) |
| Vision Hidden Size | 1,152 |
| KV Cache dtype | bfloat16 |

---

## 4. KV Cache Allocation

FastDeploy profiled GPU memory at startup:

| Metric | Value |
|--------|-------|
| Available KV cache memory | 6.0 GB |
| Per-block memory | ~1.1 MB (64 tokens, 18 layers, 2 heads, dim 128, BF16) |
| Total KV blocks allocated | 5,461 |
| GPU memory used by worker | 9,555 MiB (~9.3 GB) |
| — Model weights | ~1.8 GB |
| — KV cache | ~5.8 GB |
| — Framework overhead | ~1.7 GB |

---

## 5. Baseline Latency / Throughput Measurements

All measurements taken **after JIT warmup** (warm cache, single-request concurrency).

### 5.1 Decode Throughput by Output Length

| Test Case | Prompt Tokens | Max Output | Avg Latency (s) | P50 Latency (s) | Decode Speed (tok/s) |
|-----------|:------------:|:----------:|:---------------:|:---------------:|:--------------------:|
| Short (16-tok output) | 14 | 16 | 1.823 | 1.859 | **8.78** |
| Medium (32-tok output) | 35 | 32 | 2.139 | 2.246 | **11.46** |
| Long (64-tok output) | ~60 | 64 | 6.239 | 6.160 | **9.82** |

**Average decode speed (warmed up): ~10 tok/s** at batch size 1.

### 5.2 First-Request (Cold Start) Latency

| Stage | Duration |
|-------|----------|
| JIT/SOT kernel compilation (first request TTFT) | **~230 seconds** |
| TTFT after warmup (14-token prompt) | ~0.5 s |
| TTFT after warmup (60-token prompt) | ~1.5 s |

The 230-second cold-start is caused by MACA JIT compilation of compute kernels on first execution. Subsequent requests show normal latency.

### 5.3 GPU Utilization

- **Idle:** 0% GPU utilization, 66 W power draw
- **During inference:** Power draw increases to ~70 W (measured via `mx-smi`)
- **Memory usage:** Stable at ~9,553 MiB throughout inference

---

## 6. Bottleneck Analysis

### 6.1 JIT Compilation Overhead (Critical — 230s cold start)

**Root cause:** MetaX MACA does not pre-compile static compute kernels. On the first forward pass, all CUDA-equivalent kernels are JIT-compiled by the SOT (Static Operation Tree) framework. This creates a ~230s startup penalty before any token is produced.

**Impact:** Any fresh server restart requires a "warmup" request that blocks for ~4 minutes before normal operation resumes.

**Evidence:** 
- First request TTFT: 08:35:24 → 08:39:14 = 230s
- Same request after warmup: ~1.5s TTFT

### 6.2 Low Decode Throughput (~10 tok/s)

**Root cause candidates:**
1. **No CUDA graph** (`use_cudagraph: false`) — Each decode step incurs kernel launch overhead. On NVIDIA GPUs, CUDA graphs reduce per-step overhead by 30–50%.
2. **GQA with only 2 KV heads** — While memory-efficient, the attention kernel may not be fully optimized for MetaX's compute units.
3. **Flash Attention fallback** — The worker log shows `"Only support CUDA version flash attention."` This means MetaX is **not** running the optimized flash attention kernel; it falls back to a standard attention implementation.
4. **SOT overhead per step** — With `graph_opt_level=0`, each decode step re-runs through the full dynamic computation graph.

### 6.3 Power Efficiency

- GPU operates at ~70 W during inference (vs 350 W TDP) — **20% TDP utilization**
- This is consistent with memory-bandwidth-bound behavior: decode is limited by KV cache memory reads, not compute
- The model has GQA (2 KV heads), reducing KV memory traffic, but attention is still memory-bound at batch-1

### 6.4 Vision Encoder Path

- Current tests used text-only prompts; vision encoder (SigLIP, 27 layers) was not exercised
- Expected additional TTFT for image inputs: several seconds for vision encoding

---

## 7. Optimization Plan (Phase 2 Targets)

### 7.1 Enable MACA Kernel AOT Pre-compilation (Priority: High)

**Expected gain: -90% cold start time**

FastDeploy/MACA should support kernel cache serialization. Implementing `KernelCache` pre-warm saves compiled kernels to disk, eliminating the 230s cold-start penalty on subsequent server restarts.

### 7.2 Enable SOT Graph Optimization (Priority: High)

**Expected gain: +20–30% decode throughput**

Setting `graph_opt_level=1` enables `sot_warmup()` which pre-traces and optimizes the static decode graph. With CUDA graph disabled, SOT warmup still reduces per-step Python overhead.

### 7.3 Optimize Flash Attention for MetaX (Priority: High)

**Expected gain: +20–40% decode throughput**

The worker reports "Only support CUDA version flash attention" — the MetaX-native attention kernel is not being used. Integrating `fastdeploy_ops_pd_.so` flash attention (the MetaX custom ops) for the decode path should restore full MetaX attention performance.

### 7.4 Batch-Size Scaling (Priority: Medium)

**Expected gain: linear throughput scaling with batch**

With batch size 1, GPU utilization is low (~20% TDP). Increasing concurrent requests (batch size 4, the configured max) should improve GPU utilization and aggregate throughput proportionally.

### 7.5 Speculative Decoding Validation (Priority: Medium)

**Expected gain: +15–30% effective throughput**

FastDeploy is launched with speculative decoding configured (`ngram_match`, `mtp`, `suffix`). Validating that the speculative path is active on MetaX and measuring acceptance rate would quantify the actual gain.

---

## 8. Summary

| Metric | Current Baseline |
|--------|-----------------|
| Cold-start TTFT | ~230 s |
| Warm TTFT (14-tok prompt) | ~0.5 s |
| Decode speed (batch=1) | ~10 tok/s |
| GPU memory used | 9.3 GB / 64 GB |
| GPU TDP utilization | ~20% |
| Flash Attention | ❌ CUDA fallback |
| CUDA graph | ❌ Disabled (MACA unsupported) |

**Primary bottleneck:** Flash attention fallback + no graph optimization → ~10 tok/s decode speed.  
**Optimization target:** ≥ 12 tok/s decode speed (≥ 20% improvement) via SOT graph optimization and MetaX flash attention kernel.

---

## 9. Appendix: Reproduction Commands

### Start Server
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

### Warmup (required after first start)
```bash
# First request triggers JIT compilation (~230s)
curl -X POST http://localhost:8118/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "/data/models/PaddlePaddle/PaddleOCR-VL", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 5}'
```

### Latency Benchmark
```bash
python3 /tmp/bench_test.py
```

### Required Patches (must re-apply after pip reinstall)
```bash
# Patch 1: post_process() signature
python3 /tmp/patch_post_process.py
# Patch 2: model registry (raise→warning)
python3 /tmp/patch_init.py
```
