# [Metax GPU] PaddleOCR-VL-1.5 Performance Analysis Report

> **Chinese version**: [perf-analysis-report_001.zh.md](perf-analysis-report_001.zh.md)  
> **PR target**: https://github.com/PaddlePaddle/community/tree/master/rfcs/FastDeploy  
> **Author**: oldzhu  
> **Date**: 2026-04-24 _(update with actual submission date)_  
> **Status**: 🔲 Draft — to be filled after profiling run

---

## 0. Executive Summary

> _(Fill in after profiling: one paragraph summarizing the key bottleneck and its impact)_

PaddleOCR-VL-1.5 inference on a Metax C500 GPU via FastDeploy `release/2.5` was profiled over [N] iterations.
The dominant bottleneck is **[TBD — e.g., attention kernel memory bandwidth utilization]**, accounting for approximately
**[X]%** of total inference time. Targeting this bottleneck is projected to yield **>20%** end-to-end speedup.

---

## 1. Environment

| Item | Value |
|------|-------|
| Platform | GiteeAI 算力广场 |
| GPU | Metax 曦云C500, 64 GB |
| MACA version | 3.3.0.4 |
| Driver | [TBD] |
| OS | [TBD] |
| Python | 3.12.x |
| PaddlePaddle | 3.4.0.dev20251223 |
| paddle-metax-gpu | 3.3.0.dev20251224 |
| FastDeploy | release/2.5 |
| PaddleOCR-VL-1.5 | [TBD — model commit/version] |

---

## 2. Profiling Setup

### 2.1 Tool

```
Profiling tool: [TBD — e.g., mxprof, or Python-level timing]
Command:        [TBD — fill in exact command used]
```

### 2.2 Test Input

| Property | Value |
|----------|-------|
| Input type | Document image (JPEG) |
| Image resolution | [TBD — e.g., 1024×768] |
| Number of warmup iters | 5 |
| Number of timed iters | 50 |
| Batch size | 1 |

### 2.3 Trace File

> Trace file location: `task2-optimization/profiling/traces/profile_<TIMESTAMP>/trace.json`  
> _(Upload to the PR or provide a download link here)_

---

## 3. Inference Framework Scheduling Analysis

### 3.1 Pipeline Stages

PaddleOCR-VL-1.5 inference consists of the following stages:

| Stage | Description | Approx. % of total time |
|-------|-------------|------------------------|
| Text detection | Layout detection model forward pass | [TBD] |
| Text recognition (OCR) | VL recognition model forward pass | [TBD] |
| Pre/post processing | Image decode, resize, NMS, decode text | [TBD] |

### 3.2 CPU↔GPU Synchronization Points

> _(Identify synchronization boundaries visible in the trace — e.g., `cudaDeviceSynchronize` equivalents, host-device copies)_

Identified synchronization points:

1. **[TBD]** — after detection model inference, blocking copy of bounding-box results to CPU for NMS
2. **[TBD]** — _(add more as found)_

### 3.3 Dispatch Scheduling

> _(Describe how FastDeploy dispatches operators — is there pipeline parallelism between stages? Are there idle GPU gaps?)_

[TBD — analyze from trace]

---

## 4. GPU Utilization

### 4.1 Timeline Overview

> _(Insert a timeline screenshot or ASCII approximation from the profiling tool)_

```
Time (ms): 0           50          100         150         200
GPU util:  [████████░░][████████████████░░░░░░][████████████]
           Detection   VL Recognition          Postproc
```

### 4.2 Utilization Statistics

| Stage | GPU Utilization | Memory Bandwidth Utilization |
|-------|----------------|------------------------------|
| Text detection | [TBD]% | [TBD]% |
| VL recognition — prefill | [TBD]% | [TBD]% |
| VL recognition — decode | [TBD]% | [TBD]% |
| Idle / sync gaps | [TBD] ms total | — |

---

## 5. Kernel Analysis (≥5 kernels)

> _(Fill in after profiling. Sort by descending total time contribution.)_

### Kernel 1: [TBD kernel name]

| Property | Value |
|----------|-------|
| Kernel name | [TBD — e.g., `ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn`] |
| Total time | [TBD] ms ([TBD]% of end-to-end) |
| Calls | [TBD] |
| Avg. duration | [TBD] μs |
| Theoretical occupancy | [TBD]% |
| Achieved occupancy | [TBD]% |
| Bottleneck type | ☐ Compute-bound  ☐ Memory-bandwidth-bound  ☐ Latency-bound |
| Analysis | [TBD — why is this a bottleneck? what limits it?] |

### Kernel 2: [TBD kernel name]

| Property | Value |
|----------|-------|
| Kernel name | [TBD] |
| Total time | [TBD] ms ([TBD]%) |
| Calls | [TBD] |
| Avg. duration | [TBD] μs |
| Theoretical occupancy | [TBD]% |
| Achieved occupancy | [TBD]% |
| Bottleneck type | ☐ Compute-bound  ☐ Memory-bandwidth-bound  ☐ Latency-bound |
| Analysis | [TBD] |

### Kernel 3: [TBD kernel name]

| Property | Value |
|----------|-------|
| Kernel name | [TBD] |
| Total time | [TBD] ms ([TBD]%) |
| Calls | [TBD] |
| Bottleneck type | ☐ Compute-bound  ☐ Memory-bandwidth-bound  ☐ Latency-bound |
| Analysis | [TBD] |

### Kernel 4: [TBD kernel name]

| Property | Value |
|----------|-------|
| Kernel name | [TBD] |
| Total time | [TBD] ms ([TBD]%) |
| Calls | [TBD] |
| Bottleneck type | ☐ Compute-bound  ☐ Memory-bandwidth-bound  ☐ Latency-bound |
| Analysis | [TBD] |

### Kernel 5: [TBD kernel name]

| Property | Value |
|----------|-------|
| Kernel name | [TBD] |
| Total time | [TBD] ms ([TBD]%) |
| Calls | [TBD] |
| Bottleneck type | ☐ Compute-bound  ☐ Memory-bandwidth-bound  ☐ Latency-bound |
| Analysis | [TBD] |

---

## 6. Memory Bandwidth Analysis

| Kernel | Theoretical BW (GB/s) | Achieved BW (GB/s) | Efficiency |
|--------|----------------------|---------------------|-----------|
| [TBD] | [TBD] | [TBD] | [TBD]% |
| [TBD] | [TBD] | [TBD] | [TBD]% |

Metax C500 peak memory bandwidth: **[TBD — confirm from maca-smi or specs]** GB/s

---

## 7. End-to-End Latency Baseline

| Metric | Value |
|--------|-------|
| Mean latency | [TBD] ms |
| Median latency | [TBD] ms |
| P90 latency | [TBD] ms |
| P99 latency | [TBD] ms |
| Throughput | [TBD] iter/s |

---

## 8. Identified Top Bottleneck & Stage 2 Proposal

### 8.1 Primary Bottleneck

> _(State the single most impactful target for optimization)_

**Bottleneck**: [TBD]  
**Impact**: [TBD]% of total end-to-end time  
**Root cause**: [TBD — memory bandwidth? kernel launch overhead? sync point? operator fusion opportunity?]

### 8.2 Proposed Optimization for Stage 2

> _(Describe the optimization approach at a high level)_

| Approach | Expected Speedup | PR Target |
|----------|-----------------|-----------|
| [TBD] | ~[TBD]% | FastDeploy/develop |

---

## Appendix: Raw Profiling Commands

```bash
# Environment setup
export MACA_PATH=/opt/maca
export MACA_VISIBLE_DEVICES=0
export PADDLE_XCCL_BACKEND=metax_gpu

# Profiling run
[TBD — fill in exact command used]
```

## Appendix: Trace File

> Trace file: `traces/profile_<TIMESTAMP>/trace.json`  
> Tool to open: [TBD — e.g., Perfetto UI at https://ui.perfetto.dev, or mxprof viewer]
