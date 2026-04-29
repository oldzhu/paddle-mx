# GitHub Copilot Instructions — paddle-mx

## Project Purpose
FastDeploy 2.5 adaptation for MetaX GPU (MACA 3.3.0 / C500).
Goal: profile PaddleOCR-VL-1.5 inference on MetaX GPU via FastDeploy, then implement ≥20% optimization.

## Workflow Rules

### 1. Review-Before-Execute on MACA Instance (CRITICAL)
For ANY command to be run on the remote MACA instance, always present a review block first:

```
📋 Review Request — <step name>

Current status / last result:
  <what happened last, or current state>

Planned action:
  <exact commands>

Why (learning context):
  <what this achieves at the PaddlePaddle/MACA level>

Expected outcome:
  <what success looks like>

✅ Reply "go" to execute, or give feedback to adjust.
```

Wait for explicit user approval before executing. Local read/edit actions do NOT need approval.

### 2. Interaction Logging (CRITICAL)
After every meaningful round of interaction, create a log document in `docs/chat-logs/` with filename:
`chat-YYYYMMDD-HHMMSS-NNN.md`
where `NNN` is a zero-padded 3-digit incrementing number (001, 002, 003, ...) scoped per day.

Each log must include:
- Date/time
- Summary of what was discussed or decided
- Commands executed (if any) and their output/result
- Issues encountered and how resolved
- Current status at end of round
- Next planned step

### 3. Resume Point
Always read `docs/progress/` latest file first when resuming a session.

### 4. Critical Environment Facts (Remote MACA Instance)
- SSH: `140.207.205.81:32222`, user `root+vm-1Fe2g2PVUjoRh4Zq`, password via `$GITEEAI_PASS`
- Python: `/opt/conda/bin/python` (3.10) — always use this
- MACA_PATH: `/opt/maca-3.3.0` (NOT `/opt/maca`)
- FastDeploy source: `/root/FastDeploy/` (release/2.5)
- Model: `/data/models/PaddlePaddle/PaddleOCR-VL/`
- GPU ops .so: `.../ops/gpu/fastdeploy_ops/fastdeploy_ops_pd_.so` (33MB)
- Patch note: `__init__.py` model registry patch (raise→warning) is lost on pip reinstall — must re-apply

### 5. PR Submission Rule
NEVER submit a PR without explicit user review and approval of the content first.

### 6. Instance Lifecycle — Start, Test, Shutdown (CRITICAL — Cost Saving)

**Starting for a test:**
When testing requires the remote MACA instance, start it autonomously WITHOUT asking for
permission first — just present what you're doing, run the start, then proceed with the test:
```
🚀 Starting MACA instance for testing...
source ~/.bashrc && python3 -c "
import os,requests; TOKEN=os.environ['GITEEAI_API_TOKEN']
HEADERS={'Authorization':f'Bearer {TOKEN}'}
BASE='https://ai.gitee.com/v1/compute'
ids=','.join(i['id'] for i in requests.get(f'{BASE}/instances',headers=HEADERS,timeout=15).json() if i.get('status')=='stopped')
if ids: r=requests.post(f'{BASE}/instances/start',params={'ids':ids,'with_gpu':'true'},headers=HEADERS,timeout=15); print(r.json())
"
```
Wait ~2 minutes, then SSH.

**Shutdown after test (MANDATORY):**
After EVERY test session on the remote MACA instance, ALWAYS shut it down via the
Gitee AI Compute REST API to stop billing. Do this autonomously WITHOUT asking — just
present what you're doing and execute:
```
🔴 Shutting down MACA instance to save cost...
source ~/.bashrc && python3 -c "
import os,requests; TOKEN=os.environ['GITEEAI_API_TOKEN']
HEADERS={'Authorization':f'Bearer {TOKEN}'}
BASE='https://ai.gitee.com/v1/compute'
ids=','.join(i['id'] for i in requests.get(f'{BASE}/instances',headers=HEADERS,timeout=15).json() if i.get('status') not in ('stopped','deleted'))
if ids: r=requests.post(f'{BASE}/instances/shutdown',params={'ids':ids},headers=HEADERS,timeout=15); print(r.json())
"
```
- Instance ID: `GGKZLAER540LR91L` (MetaX C500, verified 2026-04-30)
- Token: `$GITEEAI_API_TOKEN` in `~/.bashrc` — source with `source ~/.bashrc` before use
- Full skill: `.github/skills/shutdown-instance/SKILL.md`
- Skip shutdown ONLY if user explicitly says they want to keep the instance running

### 7. Bilingual Documents
Maintain both English (`.md`) and Chinese (`.zh.md`) versions of all project documents.
