# 进度记录 — 2026-04-27 Phase 1 完成

**日期：** 2026-04-27  
**状态：** Phase 1 已完成 — 基线性能分析完成，RFC 已撰写  
**下次会话恢复点：** 开始 Phase 2 优化

---

## 本次会话完成内容

### Task 1：✅ 已完成（2026-04-25）
- 构建并安装 FastDeploy 2.5.0 metax-gpu wheel
- 发送打卡邮件

### Task 2，Phase 1：✅ 已完成（2026-04-27）
- 服务在 MetaX C500 上的 8118 端口运行
- 确认推理正常工作（之前被 230 秒 JIT 冷启动误判为代码 bug）
- 完成基线性能测量
- RFC 报告已撰写（中英文）

---

## 关键基线指标

| 指标 | 数值 |
|------|------|
| 模型 | PaddleOCR-VL-1.5（0.9B，BF16） |
| GPU | MetaX C500（64 GB） |
| MACA | 3.3.0.15 |
| FastDeploy | 2.5.0 |
| 冷启动 TTFT | ~230 秒（JIT 编译） |
| 预热后 TTFT（14 token 提示） | ~0.5 秒 |
| 解码速度（batch=1） | ~10 tok/s |
| GPU 显存使用 | 9.3 GB |
| GPU TDP 利用率 | ~20% |
| Flash Attention | ❌ CUDA 回退（未用 MetaX 原生） |

---

## 服务状态（截至 2026-04-27）

- 服务 PID：5216（gunicorn 主进程），5617（worker 进程）
- 端口：8118
- JIT 缓存：已预热（可快速响应）
- 必要补丁：
  - `/tmp/patch_post_process.py` — `post_process()` 签名修复
  - `/tmp/patch_init.py` — 模型注册表 raise→warning

**SSH：** `ssh root+vm-1Fe2g2PVUjoRh4Zq@140.207.205.81 -p 32222`

---

## 已撰写的 RFC 文档

- `task2-optimization/profiling/rfcs/phase1-baseline-profiling.md`（英文）
- `task2-optimization/profiling/rfcs/phase1-baseline-profiling.zh.md`（中文）

---

## 下次会话：Phase 2 优化

**目标：** 解码吞吐量提升 ≥ 20%（从基线 10 tok/s 提升到 ≥ 12 tok/s）

**优先行动：**
1. 与用户确认 RFC → 提交 PR 至 `PaddlePaddle/community/rfcs/FastDeploy/`
2. 启用 SOT 图优化：`--graph-optimization-config '{"graph_opt_level": 1, "use_cudagraph": false}'`
3. 测量 batch=4 吞吐量（并发请求）
4. 调研 MetaX Flash Attention 在 `fastdeploy_ops_pd_.so` 中的集成方式
5. 验证推测解码在 MetaX 上的接受率

**恢复命令：**
```bash
# 在 MACA 实例上 — 检查服务状态
ps aux | grep fastdeploy
curl -s http://localhost:8118/health
```
