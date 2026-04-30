# Phase 1 Submission Email

**Status**: PR created ✅ | Email sent ✅ (2026-04-30) | Awaiting organizer review

## Email Details

**To**: ext_paddle_oss@baidu.com  
**CC**: kaichuang.gao@metax-tech.com, yang.yang2@metax-tech.com  
**Subject**: `文心伙伴赛道-沐曦-进阶-oldzhu`  
**PR**: https://github.com/PaddlePaddle/community/pull/1360

---

## Email Body (Chinese)

您好，

我是 oldzhu，参加沐曦（MetaX）进阶赛道，现提交 Phase 1 性能分析报告及 Phase 2 优化结果，请予以审核。

GitHub PR：https://github.com/PaddlePaddle/community/pull/1360

## 一、工作概述

在 GiteeAI 算力平台的 MetaX C500 GPU（64 GB，MACA 3.3.0.15）上，使用 FastDeploy 2.5 对 PaddleOCR-VL-1.5（0.9B，bfloat16）进行了完整 profiling 分析，并实施了两项优化。

## 二、Phase 1 Profiling 结果（使用 mcTracer attach 模式）

测试条件：628 输入 token，165 输出 token，挂钟 4.38 秒

GPU 内核 Top-6 热点：
- FlashAttention (SDPA)：GPU 时间 33.4%
- GEMV（vocab proj）：12.3%
- GEMV（MLP）：6.3%
- RMSNorm：5.8%
- TopK：3.3%
- SigLIP backbone：1.5%

根本瓶颈：Python/CPU 调度开销占挂钟 80.5%，GPU 利用率仅 19.5%（67 W / 350 W TDP）。

## 三、Phase 2 优化结果

| 行动 | 结论 | 效果 |
|------|------|------|
| 8.1 SOT 预编译 | 放弃（MACA 3.3.0 下崩溃） | — |
| 8.2 MACA Shader 缓存预热 | 采用 | 冷启 135.2s → 4.28s（-97%） |
| 8.3 并发批处理（async client pool） | 采用 | 吞吐 ~10 → ~88 tok/s（+780%） |
| 8.4 RMSNorm+Linear 融合内核 | 阻塞/后续工作 | 需自定义 MACA 内核 |

两项有效优化均已超越 20% 目标要求。

## 四、提交文件

RFC 报告（中英双语）已提交至 PR：
- `rfcs/FastDeploy/20260430_metax_paddleocr_vl_perf_analysis.md`（英文）
- `rfcs/FastDeploy/20260430_metax_paddleocr_vl_perf_analysis.zh.md`（中文）

GitHub 用户名：oldzhu  
PR 链接：https://github.com/PaddlePaddle/community/pull/1360

谢谢！

oldzhu

---

## How to Send

Use **Outlook Web** (https://outlook.live.com) with account `wp_xdzhu@hotmail.com`:
1. Compose new email
2. To: `ext_paddle_oss@baidu.com`
3. CC: `kaichuang.gao@metax-tech.com`, `yang.yang2@metax-tech.com`
4. Subject: `文心伙伴赛道-沐曦-进阶-oldzhu`
5. Paste the body above
6. Send
