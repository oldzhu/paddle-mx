# Progress — 2026-04-27 Phase 1 Complete

**Date:** 2026-04-27  
**Status:** Phase 1 DONE — baseline profiling complete, RFC written  
**Resume point for next session:** Start Phase 2 optimizations

---

## What Was Accomplished (This Session)

### Task 1: ✅ Complete (since 2026-04-25)
- FastDeploy 2.5.0 metax-gpu wheel built and installed
- 打卡 email sent

### Task 2, Phase 1: ✅ Complete (2026-04-27)
- Server running at port 8118 on MetaX C500
- Inference confirmed working (was blocked by 230s JIT cold-start, not a code bug)
- Baseline profiling completed
- RFC report written (EN + ZH)

---

## Key Baseline Metrics

| Metric | Value |
|--------|-------|
| Model | PaddleOCR-VL-1.5 (0.9B, BF16) |
| GPU | MetaX C500 (64 GB) |
| MACA | 3.3.0.15 |
| FastDeploy | 2.5.0 |
| Cold-start TTFT | ~230 s (JIT compilation) |
| Warm TTFT (14-tok) | ~0.5 s |
| Decode speed (batch=1) | ~10 tok/s |
| GPU memory used | 9.3 GB |
| GPU TDP utilization | ~20% |
| Flash Attention | ❌ CUDA fallback (not MetaX native) |

---

## Server State (as of 2026-04-27)

- Server PID: 5216 (gunicorn), 5617 (worker)
- Port: 8118
- JIT cache: warmed (ready for fast responses)
- Required patches:
  - `/tmp/patch_post_process.py` — `post_process()` signature fix
  - `/tmp/patch_init.py` — model registry raise→warning

**SSH:** `ssh root+vm-1Fe2g2PVUjoRh4Zq@140.207.205.81 -p 32222`

---

## RFC Documents Written

- `task2-optimization/profiling/rfcs/phase1-baseline-profiling.md` (EN)
- `task2-optimization/profiling/rfcs/phase1-baseline-profiling.zh.md` (ZH)

---

## Next Session: Phase 2 Optimizations

**Goal:** ≥ 20% improvement in decode throughput (≥ 12 tok/s from 10 tok/s baseline)

**Priority actions:**
1. Review RFC with user → submit PR to `PaddlePaddle/community/rfcs/FastDeploy/`
2. Enable SOT graph optimization: `--graph-optimization-config '{"graph_opt_level": 1, "use_cudagraph": false}'`
3. Measure batch=4 throughput (concurrent requests)
4. Investigate MetaX Flash Attention integration in `fastdeploy_ops_pd_.so`
5. Validate speculative decoding acceptance rate on MetaX

**Resume command:**
```bash
# On MACA instance — check server status
ps aux | grep fastdeploy
curl -s http://localhost:8118/health
```
