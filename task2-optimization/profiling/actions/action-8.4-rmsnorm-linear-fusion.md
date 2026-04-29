# Action 8.4 — RMSNorm + Linear Projection Fusion

**Status:** 🔬 INVESTIGATED — BLOCKED (no fused kernel available; logged as Future Work)  
**Date Started:** 2026-04-29  
**Priority:** Medium (expected +5–8% GPU throughput from reduced kernel launch overhead)  
**Hypothesis:** Fusing the 5,903 RMSNorm calls (8.4 µs avg each, 50 ms total) with subsequent linear projection kernels will reduce total kernel launches by ~30% for normalization operations, eliminate 5,903 redundant HBM round-trips, and yield ~5–8% end-to-end throughput improvement.

---

## 1. Background

From Phase 1 profiling:
- **`phi::RmsNormBlockSMemImpl<bf16>`**: 5,903 calls, 50 ms total, 5.8% of GPU time
- Each call reads+writes 2 KB (1,024 hidden dim × BF16) from/to global memory
- Total HBM round-trip for normalization: 5,903 × 2 KB × 2 = **23.6 MB**
- At 8.4 µs/call, effective bandwidth: 2 KB / 8.4 µs = **0.24 GB/s** — far below C500 capability
- This means **kernel launch overhead (~5-8 µs) dominates**, not actual data movement

**Why fusion helps:**
- Fused `RMSNorm + Linear` reads input once, applies norm, immediately feeds into GEMV
- Eliminates separate global write (RMSNorm output) + separate global read (Linear input)
- Reduces kernel count from 2N → N per attention/FFN sublayer

**Estimated data saved per fused call:**
- RMSNorm output write: 1,024 × 2 = 2 KB
- Linear input read: 1,024 × 2 = 2 KB
- Savings: **4 KB × 5,903 calls = 23.6 MB** of HBM traffic eliminated

---

## 2. Baseline (Before)

| Kernel | Count | Total (µs) | GPU % | Avg (µs) |
|--------|------:|----------:|------:|--------:|
| `phi::RmsNormBlockSMemImpl<bf16>` | 5,903 | 49,777 | 5.8% | 8.4 |
| `b16gemvn_splitk_kernel` (follows norm) | 5,903 | 105,134 | 12.3% | 17.8 |
| `b16gemvn_kernel<64,4,4>` (follows norm) | 2,952 | 54,025 | 6.3% | 18.3 |

Effective per-step overhead: 2 × RMSNorm × 18 layers × 165 steps = 5,940 kernel launches just for normalization.

**Total GPU time baseline:** 854 ms  
**RMSNorm contribution:** 50 ms (5.8%)

---

## 3. Investigation: Does FastDeploy/PaddlePaddle Already Support Fused RMSNorm+Linear?

Before implementing, check if this fusion is already available as a flag:

```bash
# Check FastDeploy model executor config options
/opt/conda/bin/python -c "
import fastdeploy
help(fastdeploy.serving.server.fastdeploy_server.FastDeployServer)
" 2>&1 | grep -i "fus\|norm\|linear\|opt" | head -20

# Check if PaddlePaddle has fused RMSNorm op
/opt/conda/bin/python -c "
import paddle
print([op for op in dir(paddle.nn.functional) if 'norm' in op.lower()])
print([op for op in dir(paddle.nn) if 'norm' in op.lower()])
"

# Check model graph for existing fusion passes
grep -r "rmsnorm\|rms_norm\|RmsNorm" \
  /opt/conda/lib/python3.10/site-packages/fastdeploy/ 2>/dev/null | \
  grep -i "fuse\|pass\|graph" | head -20
```

### 3.1 Approach A — Enable Existing Fusion Pass (if available)

If FastDeploy already has a fusion pass for RMSNorm+Linear:
```bash
--graph-optimization-config '{"use_cudagraph": false, "graph_opt_level": 1, "fuse_norm_linear": true}'
```

### 3.2 Approach B — Enable via Environment Variable

```bash
# Check MACA/PaddlePaddle fusion flags
export FLAGS_fuse_norm_add_relu=1
export FLAGS_use_mkldnn=0
# or PaddlePaddle graph pass flags
export FLAGS_enable_auto_fusion=1
```

### 3.3 Approach C — Verify via Kernel Count Change

The most reliable test: after enabling any fusion flag, re-run mcTracer and count `phi::RmsNormBlockSMemImpl` calls. If fusion worked:
- Before: 5,903 RMSNorm calls visible
- After: 0 (or very few) RMSNorm calls (absorbed into fused kernels)
- New fused kernel appears with name like `RmsNormGemv` or `FusedNormLinear`

### 3.4 mcTracer Re-profiling to Verify Fusion

```bash
WORKER_PID=$(pgrep -f "worker_process.py" | head -1)
cd /root
/opt/maca-3.3.0/bin/mcTracer --mctx --attach $WORKER_PID --odname mctrace_norm_fused &
TRACER_PID=$!
sleep 2
/opt/conda/bin/python /tmp/infer_image.py
kill -INT $TRACER_PID
sleep 3
echo "]}" >> /root/mctrace_norm_fused/tracer_out-${WORKER_PID}.json

# Quick analysis: count RMSNorm vs fused calls
/opt/conda/bin/python3 - << 'PYEOF'
import json, collections

with open(f"/root/mctrace_norm_fused/tracer_out-$(pgrep -f worker_process.py | head -1).json") as f:
    data = json.load(f)

kernel_counts = collections.Counter()
kernel_time = collections.Counter()
for e in data.get("traceEvents", []):
    if e.get("ph") == "X" and e.get("cat") in ("gpu_op", "kernel"):
        name = e.get("name", "")
        if "norm" in name.lower() or "gemv" in name.lower() or "fuse" in name.lower():
            kernel_counts[name] += 1
            kernel_time[name] += e.get("dur", 0) / 1000  # ns -> us

print("Normalization + GEMV kernels after fusion attempt:")
for name, count in kernel_counts.most_common(20):
    print(f"  {count:5d}x  {kernel_time[name]/1000:8.1f}ms  {name}")
PYEOF
```

---

## 4. Results (After Investigation)

### 4.1 Investigation: Fusion Support Findings (2026-04-29)

**Method:** Inspected FastDeploy 2.5 source + MetaX custom device `.so` symbols.

**Findings:**

1. **FastDeploy normalization layer** (`normalization.py` line 84):
   - Non-GCU path (used by MetaX): `self.norm_func = fused_rms_norm` from `paddle.incubate.nn.functional`
   - The `fused_rms_norm` includes residual-add fusion (reads residual + x, applies RMSNorm, returns both norm output and updated residual in one op)
   - This is already the most optimized single-op norm available in Paddle

2. **MetaX custom device `.so` symbols** (`libpaddle-metax-gpu.so`):
   - `fused_rms_norm_ext_metax_gpu_ALL_LAYOUT` ✅ (present — MetaX-accelerated norm)
   - `fused_rms_norm_quant_metax_gpu_ALL_LAYOUT` ✅ (present — quantized variant)
   - `norm_linear_*` ❌ **NOT found** — no fused RMSNorm+Linear kernel
   - `fused_norm_linear_*` ❌ **NOT found**
   - `rms_norm_gemv_*` ❌ **NOT found**

3. **FastDeploy ops `.so`** (`fastdeploy_ops_pd_.so`):
   - Only `TopKRenorm` found — no normalization fusion kernel

4. **MetaX backend directory** (`layers/backends/metax/`):
   - Contains: `attention/`, `moe/` — **no `linear.py` or `normalization.py` in MetaX backend**
   - MetaX does not override the linear or normalization layers with fused variants

5. **`fused_rms_norm` is already dispatched to MetaX custom kernel:**
   - The PaddlePaddle `fused_rms_norm` op dispatches to `fused_rms_norm_ext_metax_gpu` on MetaX GPU
   - The Phase 1 profiler showed `phi::RmsNormBlockSMemImpl<bf16>` which is the MACA translation of this kernel (MACA runs CUDA PTX via JIT translation; kernel names retain original CUDA naming)

**Conclusion:** `fused_rms_norm` is already active and dispatching to the MetaX-accelerated kernel. A **deeper fusion** (RMSNorm + immediately-following GEMV in a single kernel) does **not exist** in:
- FastDeploy 2.5.0
- PaddlePaddle (incubate)
- MetaX custom device plugin (MACA 3.3.0)

Implementing this fusion would require writing a custom MACA/CUDA kernel that computes `output = (x / RMS(x)) * w @ W_linear` in a single pass — significant engineering effort beyond the scope of this optimization phase.

### 4.2 Throughput Results

**Not measured** — no config-level change available. The investigation confirmed there is no flag, environment variable, or existing code path to enable RMSNorm+Linear fusion on this stack.

---

## 5. Analysis

### 5.1 Hypothesis vs. Reality

| Hypothesis | Reality |
|------------|---------|
| FastDeploy has a fusion flag to enable RMSNorm+Linear | ❌ No such flag or config exists |
| MetaX has a fused RMSNorm+Linear kernel | ❌ Not present in `libpaddle-metax-gpu.so` |
| `fused_rms_norm` is not being used (optimization gap) | ❌ Already dispatching to `fused_rms_norm_ext_metax_gpu` |
| 5,903 separate kernel launches can be eliminated | ⚠️ Only through custom MACA kernel (future work) |

### 5.2 Actual Outcome

The action was investigated but is **not executable at configuration level**.

**What is already optimized:**
- FastDeploy already calls `paddle.incubate.nn.functional.fused_rms_norm` (not raw RMSNorm)
- `fused_rms_norm` includes residual-add fusion: `x + residual → RMSNorm(x+residual)` in one op
- MetaX custom device dispatches this to `fused_rms_norm_ext_metax_gpu` — a MetaX-accelerated kernel
- The 50 ms / 5,903 calls observed in Phase 1 represents the **already-optimized** state

**Why further fusion isn't possible at this stage:**
- A true RMSNorm+GEMV fused kernel (one kernel that normalizes and then matrix-multiplies) would eliminate one complete HBM write/read cycle per layer per token
- Estimated savings: 23.6 MB HBM traffic × 2 (read+write) at 900 GB/s = ~52 µs — close to our 50 ms total
- But implementing this requires a custom MACA kernel, not a config change

### 5.3 Comparison to RFC Targets

Phase 2 RFC target was ≥20% end-to-end improvement. Actions 8.1–8.3 already achieved this:

| Action | Result | Status |
|--------|--------|--------|
| 8.1 SOT pre-compilation | CRASHED on MACA | ❌ Discarded |
| 8.2 MACA shader cache | -97% cold start, +360% decode | ✅ KEEP |
| 8.3 Concurrent batching | +91% aggregate @ batch=4 | ✅ KEEP |
| **8.4 RMSNorm+Linear fusion** | **No config path exists** | **🔮 Future Work** |

---

## 6. Decision

- [ ] **KEEP** — Not applicable (nothing to enable)
- [ ] **CONDITIONAL** — Not applicable
- [x] **FUTURE WORK** — Requires custom MACA kernel development; no config-level change available

**Reason:** The MetaX MACA 3.3.0 ecosystem does not include a fused RMSNorm+Linear kernel. FastDeploy 2.5 already uses `fused_rms_norm` dispatched to the MetaX-accelerated `fused_rms_norm_ext_metax_gpu` kernel. Further fusion (RMSNorm + GEMV in a single pass) would require a new MACA kernel registered via PaddlePaddle's custom op API — out of scope for this phase.

**Future Work Proposal:**

> **"MACA Fused RMSNorm+GEMV Kernel for FastDeploy 2.5"**
>
> Implement a custom MACA kernel using `paddle.utils.cpp_extension` that computes:
> ```
> norm_out = x * rsqrt(mean(x^2) + eps) * weight
> proj_out = norm_out @ W  # in same kernel, no HBM write for norm_out
> ```
> Register as `fused_rms_norm_linear` in the MetaX custom device plugin.  
> Estimated effort: 3–5 days. Expected gain: +3–5% end-to-end throughput.

**Next step if discarded:** Open as a GitHub issue on the FastDeploy or MetaX paddle-custom-device repository.
