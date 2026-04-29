---
name: shutdown-instance
description: 'Shut down the remote Gitee AI MACA instance via the compute REST API to save cost. USE after every session that runs commands on the remote MACA instance at 140.207.205.81. Trigger keywords: shutdown, done with instance, save cost, stop instance, finish remote work, start instance, power on.'
---

# Shutdown / Start MACA Instance

## When to Use

Call this skill **at the end of every session** that interacted with the remote MACA instance
(`140.207.205.81:32222`). This includes any session where:
- The FastDeploy server was started or stopped
- Benchmarks / profiling were run
- Files were modified on the remote instance
- Any SSH/paramiko commands were executed

Also use this skill when the user asks to **start** the instance at the beginning of a session.

Do NOT skip shutdown if the user says "we're done" or the conversation is winding down.

## Prerequisites

The Gitee AI API token must be available as environment variable `$GITEEAI_API_TOKEN`.

To create one: visit https://ai.gitee.com → 工作台 → 设置 → 访问令牌 → 新建令牌.
Store it on the local machine: `export GITEEAI_API_TOKEN=<your_token>` (add to `~/.bashrc`).

## Procedure — Shutdown

### Step 1 — Present Review Block (required by project Workflow Rule #1)

```
📋 Review Request — Shutdown MACA Instance

Current status:
  Remote work complete. Shutting down instance to save cost.

Planned action:
  1. GET  https://ai.gitee.com/v1/compute/instances            ← list instances
  2. POST https://ai.gitee.com/v1/compute/instances/shutdown?ids=<id>  ← shut down

Auth: Bearer $GITEEAI_API_TOKEN

Why:
  Gitee AI charges by instance-uptime. Shutdown after each session prevents
  idle billing.

Expected outcome:
  Instance status changes to "stopped". SSH access becomes unavailable.

✅ Reply "go" to execute, or "skip" to leave instance running.
```

### Step 2 — Execute Shutdown (after user approval)

Run the following Python script in the local terminal:

```python
import os, requests

TOKEN = os.environ["GITEEAI_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
BASE = "https://ai.gitee.com/v1/compute"

# Step A: list instances to find the correct ID
r = requests.get(f"{BASE}/instances", headers=HEADERS, timeout=15)
r.raise_for_status()
instances = r.json()

print("Current instances:")
for inst in instances:
    print(f"  id={inst['id']}  gpu={inst.get('gpu_model','')}  "
          f"zone={inst.get('zone','')}  status={inst.get('status','')}  "
          f"ssh={inst.get('ssh_props',{})}")

# Step B: shut down all running instances (or filter by zone/id if needed)
running = [i["id"] for i in instances if i.get("status") not in ("stopped", "deleted")]
if not running:
    print("✅ No running instances — nothing to shut down.")
else:
    ids_param = ",".join(running)
    r2 = requests.post(f"{BASE}/instances/shutdown", params={"ids": ids_param},
                       headers=HEADERS, timeout=15)
    r2.raise_for_status()
    results = r2.json()
    for res in results:
        if res.get("success"):
            print(f"✅ Instance {res['id']} shutdown initiated.")
        else:
            print(f"❌ Instance {res['id']} failed: {res.get('error','unknown')}")
```

### Step 3 — Confirm

After the script runs:
- If all `✅` → report: *"Instance shutdown initiated. SSH access will become unavailable shortly."*
- If any `❌` → show the error, advise user to check the Gitee AI console at https://ai.gitee.com/oldzhu99/dashboard/compute/instances

---

## Procedure — Start Instance

Use when the user wants to begin a new MACA session.

### Step 1 — Present Review Block

```
📋 Review Request — Start MACA Instance

Planned action:
  1. GET  https://ai.gitee.com/v1/compute/instances            ← list to find ID
  2. POST https://ai.gitee.com/v1/compute/instances/start?ids=<id>&with_gpu=true

Auth: Bearer $GITEEAI_API_TOKEN

Expected outcome:
  Instance powers on; SSH available at 140.207.205.81:32222 within ~2 min.

✅ Reply "go" to execute.
```

### Step 2 — Execute Start

```python
import os, requests

TOKEN = os.environ["GITEEAI_API_TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
BASE = "https://ai.gitee.com/v1/compute"

r = requests.get(f"{BASE}/instances", headers=HEADERS, timeout=15)
r.raise_for_status()
instances = r.json()

stopped = [i["id"] for i in instances if i.get("status") == "stopped"]
if not stopped:
    print("No stopped instances found. Already running?")
    for inst in instances:
        print(f"  id={inst['id']}  status={inst.get('status','')}")
else:
    ids_param = ",".join(stopped)
    r2 = requests.post(f"{BASE}/instances/start",
                       params={"ids": ids_param, "with_gpu": "true"},
                       headers=HEADERS, timeout=15)
    r2.raise_for_status()
    for res in r2.json():
        if res.get("success"):
            print(f"✅ Instance {res['id']} start initiated.")
        else:
            print(f"❌ Instance {res['id']} failed: {res.get('error','unknown')}")
    print("\nWait ~2 minutes, then SSH: 140.207.205.81:32222")
```

---

## Notes

- **API base**: `https://ai.gitee.com/v1/compute/instances`
- **Auth**: `Authorization: Bearer <token>` — token from Gitee AI 工作台 → 设置 → 访问令牌
- **Instance label**: The JupyterLab URL path `vm-1Fe2g2PVUjoRh4Zq` may differ from the API `id` field; always use `GET /compute/instances` to discover the correct ID.
- **Do not shutdown** if the user explicitly says they want to continue in the same session.
- **Console fallback**: https://ai.gitee.com/oldzhu99/dashboard/compute/instances for manual control.
