# Metax GPU + PaddleOCR-VL-1.5 + FastDeploy

> **Chinese version**: [README.zh.md](README.zh.md)

## Project Overview

This project participates in **PaddlePaddle Hackathon 10th — 沐曦 (Metax) GPU track**, completing two sequential tasks:

| # | Task | Status | Deliverable |
|---|------|--------|-------------|
| 1 | [Warm-up Check-in: Compile FastDeploy on Metax GPU](#task-1-warm-up-check-in) | 🔲 In Progress | Email + screenshots |
| 2 | [Optimize PaddleOCR-VL-1.5 on Metax GPU](#task-2-optimize-paddleocr-vl-15-on-metax-gpu) | ⏳ Pending Task 1 | Stage 1: RFC PR; Stage 2: FastDeploy PR |

## Environment

| Item | Value |
|------|-------|
| Platform | GiteeAI 算力广场 — 曦云C500 单卡 64G |
| Image | Pytorch/2.8.0 / Python 3.12 / maca 3.3.0.4 |
| PaddlePaddle | 3.4.0.dev20251223 (CPU wheel) |
| Paddle-Metax backend | paddle-metax-gpu==3.3.0.dev20251224 |
| FastDeploy (Task 1 build) | release/2.5 (Gitee) |
| FastDeploy (Task 2 base) | release/2.4 (GitHub) |
| GitHub ID | oldzhu |

## Repository Structure

```
paddle-mx/
├── README.md                              # This file (EN)
├── README.zh.md                           # Chinese version
├── docs/
│   ├── plan/
│   │   ├── plan-2026-04-24_001.md         # Full project plan (EN)
│   │   └── plan-2026-04-24_001.zh.md      # Full project plan (ZH)
│   └── progress/
│       ├── progress-2026-04-24_001.md     # Progress log – session 1 (EN)
│       └── progress-2026-04-24_001.zh.md  # Progress log – session 1 (ZH)
├── task1-warmup/
│   ├── scripts/
│   │   ├── 01_install_deps.sh             # Install Paddle + PaddleOCR deps
│   │   ├── 02_build_fastdeploy.sh         # Clone + set env + build FastDeploy
│   │   └── 03_verify_install.sh           # Verify wheel install
│   └── email_template.md                  # Check-in email template
└── task2-optimization/
    ├── profiling/
    │   ├── run_profile.sh                 # Launch profiling run
    │   └── rfcs/
    │       ├── perf-analysis-report_001.md     # Stage 1 RFC report (EN)
    │       └── perf-analysis-report_001.zh.md  # Stage 1 RFC report (ZH)
    ├── scripts/
    │   └── benchmark.sh                   # Before/after benchmark runner
    └── optimization/                      # (Stage 2 patches/notes added here)
```

## Task 1: Warm-up Check-in

**Goal**: Compile FastDeploy `release/2.5` from source on a Metax GPU instance, install the built wheel, and submit a check-in email with environment info and screenshots.

**Reference doc**: [沐曦-FastDeploy-编译验证-打卡任务.md](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-hardware/%E6%B2%90%E6%9B%A6-FastDeploy-%E7%BC%96%E8%AF%91%E9%AA%8C%E8%AF%81-%E6%89%93%E5%8D%A1%E4%BB%BB%E5%8A%A1.md)

**Quick start**:
```bash
# Step 1 – Install Python dependencies
bash task1-warmup/scripts/01_install_deps.sh

# Step 2 – Clone, configure, and build FastDeploy (may take 30-60 min; run in tmux)
tmux new -s build
bash task1-warmup/scripts/02_build_fastdeploy.sh

# Step 3 – Verify the installed wheel
bash task1-warmup/scripts/03_verify_install.sh

# Step 4 – Fill in email_template.md with screenshots and send
```

## Task 2: Optimize PaddleOCR-VL-1.5 on Metax GPU

**Goal**: Profile PaddleOCR-VL-1.5 inference on Metax GPU → produce a performance bottleneck analysis report (Stage 1 RFC) → implement targeted optimizations achieving ≥20% speedup (Stage 2 PR).

**Reference doc**: [Hackathon 10th — 沐曦赛题](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E3%80%90Hackathon_10th%E3%80%91%E6%96%87%E5%BF%83%E5%90%88%E4%BD%9C%E4%BC%99%E4%BC%B4%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#%E6%B2%90%E6%9B%A6%E4%BC%98%E5%8C%96-paddleocr-vl-15metax-gpu)

**Stage 1 submission**: PR to [PaddlePaddle/community rfcs/FastDeploy](https://github.com/PaddlePaddle/community/tree/master/rfcs/FastDeploy)

**Stage 2 submission**: PR to [PaddlePaddle/FastDeploy develop](https://github.com/PaddlePaddle/FastDeploy/tree/develop)

## Progress Tracking

Latest progress log: [docs/progress/progress-2026-04-24_001.md](docs/progress/progress-2026-04-24_001.md)

> **Resume policy**: When resuming after a break, **always read the latest file in `docs/progress/` first**. It is the canonical resume point containing current status, blockers, and next steps.
