# 沐曦 GPU + PaddleOCR-VL-1.5 + FastDeploy 优化项目

> **英文版本**: [README.md](README.md)

## 项目概述

本项目参与 **飞桨黑客松第10期 — 沐曦 GPU 赛道**，依次完成以下两个任务：

| # | 任务 | 状态 | 交付物 |
|---|------|------|--------|
| 1 | [热身打卡：Metax GPU 上编译 FastDeploy](#task-1-热身打卡) | 🔲 进行中 | 邮件 + 截图 |
| 2 | [优化 PaddleOCR-VL-1.5 + Metax GPU](#task-2-优化-paddleocr-vl-15--metax-gpu) | ⏳ 待 Task 1 完成后开始 | 阶段一：RFC PR；阶段二：FastDeploy PR |

## 运行环境

| 项目 | 值 |
|------|----|
| 平台 | GiteeAI 算力广场 — 曦云C500 单卡 64G |
| 镜像 | Pytorch/2.8.0 / Python 3.12 / maca 3.3.0.4 |
| PaddlePaddle | 3.4.0.dev20251223（CPU wheel） |
| Paddle-Metax 后端 | paddle-metax-gpu==3.3.0.dev20251224 |
| FastDeploy（任务1编译） | release/2.5（Gitee） |
| FastDeploy（任务2基准） | release/2.4（GitHub） |
| GitHub ID | oldzhu |

## 目录结构

```
paddle-mx/
├── README.md                              # 英文版说明
├── README.zh.md                           # 本文件（中文版）
├── docs/
│   ├── plan/
│   │   ├── plan-2026-04-24_001.md         # 完整项目计划（英文）
│   │   └── plan-2026-04-24_001.zh.md      # 完整项目计划（中文）
│   └── progress/
│       ├── progress-2026-04-24_001.md     # 进度日志 – 第1次工作（英文）
│       └── progress-2026-04-24_001.zh.md  # 进度日志 – 第1次工作（中文）
├── task1-warmup/
│   ├── scripts/
│   │   ├── 01_install_deps.sh             # 安装 Paddle + PaddleOCR 依赖
│   │   ├── 02_build_fastdeploy.sh         # 克隆 + 设置环境变量 + 编译 FastDeploy
│   │   └── 03_verify_install.sh           # 验证 wheel 安装
│   └── email_template.md                  # 打卡邮件模板
└── task2-optimization/
    ├── profiling/
    │   ├── run_profile.sh                 # 启动 profiling 推理
    │   └── rfcs/
    │       ├── perf-analysis-report_001.md     # 阶段一 RFC 报告（英文）
    │       └── perf-analysis-report_001.zh.md  # 阶段一 RFC 报告（中文）
    ├── scripts/
    │   └── benchmark.sh                   # 优化前后对比 benchmark
    └── optimization/                      # （阶段二补丁/笔记）
```

## Task 1: 热身打卡

**目标**：在 Metax GPU 实例上从源码编译 FastDeploy `release/2.5`，安装编译产物 wheel 包，并发送打卡邮件（附环境信息截图）。

**参考文档**: [沐曦-FastDeploy-编译验证-打卡任务.md](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-hardware/%E6%B2%90%E6%9B%A6-FastDeploy-%E7%BC%96%E8%AF%91%E9%AA%8C%E8%AF%81-%E6%89%93%E5%8D%A1%E4%BB%BB%E5%8A%A1.md)

**快速开始**：
```bash
# 步骤1 – 安装 Python 依赖
bash task1-warmup/scripts/01_install_deps.sh

# 步骤2 – 克隆、配置并编译 FastDeploy（耗时约 30-60 分钟，建议在 tmux 中运行）
tmux new -s build
bash task1-warmup/scripts/02_build_fastdeploy.sh

# 步骤3 – 验证 wheel 安装
bash task1-warmup/scripts/03_verify_install.sh

# 步骤4 – 填写 email_template.md 并附截图发送
```

## Task 2: 优化 PaddleOCR-VL-1.5 + Metax GPU

**目标**：对 PaddleOCR-VL-1.5 在 Metax GPU 上的推理进行 profiling → 提交性能瓶颈分析报告（阶段一 RFC） → 实现 ≥20% 性能提升优化（阶段二 PR）。

**参考文档**: [黑客松10期 — 沐曦赛题](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E3%80%90Hackathon_10th%E3%80%91%E6%96%87%E5%BF%83%E5%90%88%E4%BD%9C%E4%BC%99%E4%BC%B4%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#%E6%B2%90%E6%9B%A6%E4%BC%98%E5%8C%96-paddleocr-vl-15metax-gpu)

**阶段一提交位置**: PR 至 [PaddlePaddle/community rfcs/FastDeploy](https://github.com/PaddlePaddle/community/tree/master/rfcs/FastDeploy)

**阶段二提交位置**: PR 至 [PaddlePaddle/FastDeploy develop](https://github.com/PaddlePaddle/FastDeploy/tree/develop)

## 进度追踪

最新进度日志：[docs/progress/progress-2026-04-24_001.md](docs/progress/progress-2026-04-24_001.md)

> **恢复策略**：从中断点恢复时，**务必先阅读 `docs/progress/` 目录中最新的文件**。该文件是规范的恢复入口，记录了当前状态、阻塞点和下一步操作。
