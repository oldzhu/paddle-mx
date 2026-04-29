# Action 8.1 — SOT Graph Pre-compilation

**Status:** ❌ DISCARDED — SOT incompatible with PaddleOCR-VL on MetaX MACA  
**Date Started:** 2026-04-29  
**Priority:** High (expected ≥ 20% throughput gain)  
**Hypothesis:** Enabling `graph_opt_level=1` (SOT static operation tree pre-compilation) will reduce per-decode-step Python dispatch overhead from ~21 ms to < 10 ms, improving text decode throughput from ~10 tok/s to ≥ 13 tok/s.

---

## 1. Background

From Phase 1 profiling:
- GPU kernel time: **854 ms** out of **4,380 ms** wall clock → 19.5% GPU utilization
- Python/CPU overhead: **3,526 ms** (80.5%)
- Per decode step overhead: ~21 ms Python dispatch per token
- Root cause: `graph_opt_level=0` (default) means Python re-dispatches the full computation graph on every decode step

SOT (Static Operation Tree) pre-compilation, when enabled via `graph_opt_level=1`, pre-traces the decode computation graph for a set of sequence lengths during server warmup. Subsequent decode steps replay the pre-compiled graph without Python re-dispatch.

---

## 2. Baseline (Before)

Measured with `graph-optimization-config '{"use_cudagraph": false}'` (default, no SOT):

### Decode Speed

| Test Case | Prompt Tokens | Output Tokens | Latency (s) | Decode Speed (tok/s) |
|-----------|:------------:|:------------:|:-----------:|:--------------------:|
| Short (text) | 14 | 16 | 1.82 | 8.8 |
| Medium (text) | 35 | 32 | 2.14 | 11.5 |
| Long (text) | ~60 | 64 | 6.24 | 9.8 |
| Image input | 628 | 165 | 4.38 | 37.7 |

**Baseline average decode speed (text): ~10 tok/s**

### GPU Profiling (mcTracer)

| Metric | Value |
|--------|-------|
| Total GPU kernel time | 854 ms |
| Wall clock time | 4,380 ms |
| GPU utilization | 19.5% |
| Python overhead | 80.5% |
| Per-decode-step Python cost | ~21 ms |
| Trace events | 501,974 |
| Top kernel | flash_fwd_splitkv (33.4% GPU) |

---

## 3. Action: Configuration Change

### 3.1 Server Config Change

**Before:**
```bash
--graph-optimization-config '{"use_cudagraph": false}'
```

**After:**
```bash
--graph-optimization-config '{"use_cudagraph": false, "graph_opt_level": 1}'
```

### 3.2 Commands to Apply

```bash
# Step 1: Kill current server
pkill -f "api_server" || true
sleep 5

# Step 2: Re-apply required patches (lost if server reimports)
/opt/conda/bin/python /tmp/patch_init.py
/opt/conda/bin/python /tmp/patch_post_process.py

# Step 3: Start server with SOT enabled
export MACA_PATH=/opt/maca-3.3.0
export PATH=/opt/maca-3.3.0/bin:/opt/conda/bin:$PATH
nohup /opt/conda/bin/python -m fastdeploy.entrypoints.openai.api_server \
  --model /data/models/PaddlePaddle/PaddleOCR-VL \
  --port 8118 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 4 \
  --workers 1 \
  --graph-optimization-config '{"use_cudagraph": false, "graph_opt_level": 1}' \
  > /tmp/fd_server_sot.log 2>&1 &

# Step 4: Wait for server ready
echo "Waiting for server..."
for i in $(seq 1 120); do
  if curl -s http://localhost:8118/health > /dev/null 2>&1; then
    echo "Server ready at step $i (${i}s)"; break
  fi
  sleep 1
done

# Step 5: Check log for SOT-related messages
grep -i "sot\|graph_opt\|static\|pre.compil\|warmup" /tmp/fd_server_sot.log | head -30
```

### 3.3 Warmup and Benchmark Script

```bash
# /tmp/bench_sot.py
# Run text decode benchmark (short/medium/long) + image inference
# Compare against baseline numbers

cat > /tmp/bench_sot.py << 'EOF'
import requests, time, base64, statistics

BASE_URL = "http://localhost:8118/v1/chat/completions"
MODEL = "/data/models/PaddlePaddle/PaddleOCR-VL"

def chat(prompt, max_tokens=64):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }
    t0 = time.time()
    r = requests.post(BASE_URL, json=payload, timeout=300)
    elapsed = time.time() - t0
    data = r.json()
    out_tokens = data["usage"]["completion_tokens"]
    return elapsed, out_tokens

def image_chat(img_path, prompt, max_tokens=200):
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": max_tokens
    }
    t0 = time.time()
    r = requests.post(BASE_URL, json=payload, timeout=300)
    elapsed = time.time() - t0
    data = r.json()
    out_tokens = data["usage"]["completion_tokens"]
    return elapsed, out_tokens

print("=== Action 8.1 SOT Benchmark ===")
print()

# Warmup (first request triggers JIT)
print("[Warmup] First text request (JIT)...")
t0 = time.time()
chat("Hello, what is 2+2?", max_tokens=16)
print(f"  JIT cold start: {time.time()-t0:.1f}s")

# Short text: 3 runs
print("[Short text, ~14 prompt tokens]")
times = []
for i in range(5):
    lat, tok = chat("What is the capital of France?", max_tokens=16)
    speed = tok / lat
    times.append(speed)
    print(f"  Run {i+1}: {lat:.2f}s, {tok} tokens, {speed:.1f} tok/s")
print(f"  Avg: {statistics.mean(times):.1f} tok/s  (baseline: 8.8)")

# Medium text: 3 runs
print("[Medium text, ~35 prompt tokens]")
times = []
for i in range(5):
    lat, tok = chat(
        "You are a helpful assistant. Please provide a brief explanation of "
        "what machine learning is and how it works in simple terms.",
        max_tokens=32
    )
    speed = tok / lat
    times.append(speed)
    print(f"  Run {i+1}: {lat:.2f}s, {tok} tokens, {speed:.1f} tok/s")
print(f"  Avg: {statistics.mean(times):.1f} tok/s  (baseline: 11.5)")

# Long text: 3 runs
print("[Long text, ~60 prompt tokens]")
times = []
for i in range(5):
    lat, tok = chat(
        "You are an expert in natural language processing. Explain the transformer "
        "architecture in detail, covering self-attention, multi-head attention, "
        "positional encoding, feed-forward layers, and residual connections.",
        max_tokens=64
    )
    speed = tok / lat
    times.append(speed)
    print(f"  Run {i+1}: {lat:.2f}s, {tok} tokens, {speed:.1f} tok/s")
print(f"  Avg: {statistics.mean(times):.1f} tok/s  (baseline: 9.8)")

# Image inference: 3 runs
print("[Image inference, 628 tokens]")
import os
IMG = "/tmp/test_doc.jpg"
if os.path.exists(IMG):
    # Warmup image
    print("  [Image warmup (JIT)]...")
    t0 = time.time()
    image_chat(IMG, "Extract all text.", max_tokens=200)
    print(f"  Image JIT cold start: {time.time()-t0:.1f}s  (baseline: 135.2s)")
    
    times = []
    for i in range(3):
        lat, tok = image_chat(IMG, "Extract all text.", max_tokens=200)
        speed = tok / lat
        times.append(speed)
        print(f"  Run {i+1}: {lat:.2f}s, {tok} tokens, {speed:.1f} tok/s")
    print(f"  Avg: {statistics.mean(times):.1f} tok/s  (baseline: 37.7)")
else:
    print(f"  SKIP: {IMG} not found")

print()
print("=== Benchmark Complete ===")
EOF

/opt/conda/bin/python /tmp/bench_sot.py
```

### 3.4 mcTracer Re-profiling Script (After Benchmark)

Same method as baseline:
```bash
WORKER_PID=$(pgrep -f "worker_process.py" | head -1)
echo "Worker PID: $WORKER_PID"
cd /root
/opt/maca-3.3.0/bin/mcTracer --mctx --attach $WORKER_PID --odname mctrace_sot &
TRACER_PID=$!
sleep 2
/opt/conda/bin/python /tmp/infer_image.py
kill -INT $TRACER_PID
sleep 3
echo "]}" >> /root/mctrace_sot/tracer_out-${WORKER_PID}.json
echo "Trace saved to /root/mctrace_sot/"
wc -c /root/mctrace_sot/tracer_out-${WORKER_PID}.json
```

---

## 4. Results (After)

### 4.1 Server Startup — CRASHED

Date: 2026-04-29 08:31  
Command used:
```bash
/opt/conda/bin/python -m fastdeploy.entrypoints.openai.api_server \
  --model /data/models/PaddlePaddle/PaddleOCR-VL \
  --port 8118 --max-model-len 4096 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --max-num-seqs 4 --workers 1 \
  --graph-optimization-config '{"use_cudagraph": false, "graph_opt_level": 1}'
```

Server started, loaded weights (100%, ~12s), then crashed at "Loading Layers: 0%" after 18s.
Worker log: `/root/log/workerlog.0`

**Crash sequence in workerlog.0:**

1. **Attempt 1** — Worker tried `post_process()` with `line_break_id` kwarg → `TypeError` (patch not applied)
2. **Attempt 2** — FastDeploy retried with cudagraph backend → `mcErrorStreamCaptureUnsupported` (MACA does not support stream capture needed for cudagraph)
3. **Attempt 3** — Sub-process tried `pymxsml` → `ModuleNotFoundError` (transient)
4. **Final fatal crash** — SOT warmup in `metax_worker.py:sot_warmup()` called `_dummy_run()` → forward pass → `dynamic_dims_marker.py` tried to mark dynamic dims in all `Tensor`-annotated fields, but found `attention_metadata.max_len_kv = None`:

```
TypeError: data forward_meta.attn_backend.attention_metadata.max_len_kv
  has type annotation Tensor but got type <class 'NoneType'>
```

Full traceback path:
```
worker_process.py:1274 run_worker_proc()
  → metax_worker.py:229 graph_optimize_and_warm_up_model() → sot_warmup()
  → metax_model_runner.py:2117 sot_warmup() → _dummy_run()
  → metax_model_runner.py:1876 _dummy_run() → self.model()
  → paddleocr_vl.py:261 forward()
  → decorator.py:68 → graph_optimization_backend.py:131 → :91 static_forward
  → graph_optimization_backend.py:62 warmup_impl → resolve_dynamic_dims
  → dynamic_dims_marker.py:185 resolve()
    raise TypeError: max_len_kv is None, expected Tensor
```

### 4.2 Decode Speed Results — N/A (server never started)

| Test Case | Baseline (tok/s) | SOT (tok/s) | Delta |
|-----------|:----------------:|:-----------:|:-----:|
| Short text | 8.8 | — | — |
| Medium text | 11.5 | — | — |
| Long text | 9.8 | — | — |
| Image inference | 37.7 | — | — |

### 4.3 GPU Profiling — N/A

| Metric | Baseline | SOT | Delta |
|--------|----------|-----|-------|
| GPU kernel time | 854 ms | — | — |
| Wall clock time | 4,380 ms | — | — |
| GPU utilization | 19.5% | — | — |
| Python overhead | 80.5% | — | — |
| Trace events | 501,974 | — | — |

---

## 5. Analysis

> *To be filled after results are collected*

### 5.1 Expected Outcome
- Server log should show lines like: `[SOT] pre-compiling graph for batch_size=1, seq_len=X`
- Decode speed should increase ≥ 20% if Python dispatch overhead is reduced
- GPU utilization should increase from 19.5% toward ~30%+
- If SOT is NOT supported on MetaX MACA → server may fail to start or silently ignore the flag → measure identical performance → action fails

### 5.2 Actual Outcome

**SOT is incompatible with PaddleOCR-VL on MetaX MACA 3.3.0.**

Root cause: `dynamic_dims_marker.py` in FastDeploy's SOT backend introspects ALL type-annotated fields in the forward pass arguments to mark dynamic tensor dimensions. The `AttentionMetadata` dataclass has `max_len_kv: Tensor` type annotation, but during the dummy warmup run the MetaX attention backend sets `max_len_kv = None` (because it is only populated during real inference, not during dummy shape-tracing runs).

This causes a hard `TypeError` that kills the worker process. FastDeploy retries multiple times with different backends (cudagraph → SOT) but all fail for MACA-specific reasons:
- **cudagraph**: MACA does not support `mcStreamCapture` (`mcErrorStreamCaptureUnsupported`)
- **SOT**: `max_len_kv=None` annotation mismatch

**Fix would require:** Modifying either `dynamic_dims_marker.py` (to treat `Optional[Tensor]` as skippable) or the MetaX attention metadata provider (to supply a dummy zero-tensor during warmup). Both require FastDeploy source changes outside the scope of config-level optimization.

---

## 6. Decision

> *To be filled after analysis*

- [ ] **KEEP** — ≥ 5% improvement confirmed, include in final config
- [x] **DISCARD** — Server crashes; SOT not supported on MetaX MACA 3.3.0 with PaddleOCR-VL
- [ ] **PARTIAL** — Improvement but with side effects (e.g., longer startup), document tradeoffs

**Reason:** `graph_opt_level=1` (SOT) causes a hard worker crash due to `TypeError: max_len_kv has type annotation Tensor but got NoneType` in `dynamic_dims_marker.py` during dummy warmup. MetaX MACA also lacks `mcStreamCapture` support (cudagraph path also fails). Neither graph optimization backend is usable on this platform without FastDeploy source-level fixes.

**Final config recommendation:** Keep `--graph-optimization-config '{"use_cudagraph": false}'` (default, `graph_opt_level=0`). Do NOT set `graph_opt_level=1`.

**Next action:** Proceed to Action 8.2 (AOT Kernel Cache / Startup Warmup) to address the 135s image cold-start.
