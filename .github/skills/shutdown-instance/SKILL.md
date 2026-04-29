---
name: shutdown-instance
description: 'Shut down the remote Gitee AI / JupyterLab MACA instance via the JupyterLab shutdown API to save cost. USE after every session that runs commands on the remote MACA instance at 140.207.205.81. Trigger keywords: shutdown, done with instance, save cost, stop instance, finish remote work.'
---

# Shutdown MACA Instance

## When to Use

Call this skill **at the end of every session** that interacted with the remote MACA instance
(`140.207.205.81:32222`). This includes any session where:
- The FastDeploy server was started or stopped
- Benchmarks / profiling were run
- Files were modified on the remote instance
- Any SSH/paramiko commands were executed

Do NOT skip this if the user says "we're done" or the conversation is winding down.

## Procedure

### Step 1 — Present Review Block (required by project Workflow Rule #1)

```
📋 Review Request — Shutdown MACA Instance

Current status:
  Remote work complete. Shutting down instance to save cost.

Planned action:
  POST http://140.207.205.81:38080/vm-1Fe2g2PVUjoRh4Zq/api/shutdown

Why:
  Gitee AI charges by instance-uptime. Shutdown after each session prevents
  idle billing.

Expected outcome:
  HTTP 200 response; JupyterLab server terminates; instance stops.

✅ Reply "go" to execute, or "skip" to leave instance running.
```

### Step 2 — Execute Shutdown (after user approval)

Run the following Python script:

```python
import requests

JUPYTER_BASE = "http://140.207.205.81:38080/vm-1Fe2g2PVUjoRh4Zq"

try:
    # JupyterLab shutdown endpoint
    r = requests.post(f"{JUPYTER_BASE}/api/shutdown", timeout=10)
    if r.status_code in (200, 204):
        print("✅ Instance shutdown initiated successfully.")
    else:
        print(f"⚠️  Shutdown returned HTTP {r.status_code}: {r.text[:200]}")
        print("Manual check: visit the URL above or wait for SSH timeout.")
except requests.exceptions.ConnectionError:
    # If the instance already shut down, connection will be refused — that's OK
    print("✅ Connection refused — instance is already down or shutting down.")
except Exception as e:
    print(f"❌ Shutdown error: {e}")
    print(f"Manual shutdown: POST {JUPYTER_BASE}/api/shutdown")
```

### Step 3 — Confirm

After the script runs:
- If `✅` → report to user: *"Instance shutdown initiated. SSH access will become unavailable shortly."*
- If `⚠️` with non-200 → note it and check if the instance is still reachable via SSH
- If connection refused → instance already down — no action needed

## Notes

- **Token**: The Gitee AI JupyterLab instance uses path-based routing (`/vm-<id>/`) without a separate token. No `Authorization` header needed.
- **XSRF**: Standard JupyterLab requires a `_xsrf` cookie for POST. If the API returns 403, fetch the login page first to obtain the cookie, then POST with it.
- **SSH timeout**: Even after JupyterLab shuts down, the underlying VM may stay alive briefly. If the user needs to confirm billing stop, check the Gitee AI console.
- **Do not shutdown** if the user explicitly says they want to continue or resume work in the same session.
