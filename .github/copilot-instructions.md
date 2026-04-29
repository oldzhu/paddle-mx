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

### 6. Instance Shutdown (CRITICAL — Cost Saving)
After every session that ran commands on the remote MACA instance, ALWAYS invoke the
`shutdown-instance` skill (`.github/skills/shutdown-instance/SKILL.md`) to **fully stop the VM**
via the Gitee AI Compute REST API (`POST https://ai.gitee.com/v1/compute/instances/shutdown?ids=<id>`).
- Requires `$GITEEAI_API_TOKEN` (Bearer token from Gitee AI 工作台 → 设置 → 访问令牌)
- First call `GET https://ai.gitee.com/v1/compute/instances` to discover the instance ID
- Present the Review Block from the skill before executing
- Skip only if the user explicitly says they want to continue working on the instance
- This applies even if the session only read files — if SSH/paramiko was used, shut down

### 7. Bilingual Documents
Maintain both English (`.md`) and Chinese (`.zh.md`) versions of all project documents.
