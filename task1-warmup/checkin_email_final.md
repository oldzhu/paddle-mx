# Task 1 Check-in Email — READY TO SEND

---

## Recipients

- **To**: ext_paddle_oss@baidu.com; kaichuang.gao@metax-tech.com; yang.yang2@metax-tech.com

## Subject

```
文心伙伴赛道-沐曦-打卡-oldzhu
```

---

## Email Body

飞桨团队你好，

**【GitHub ID】**: oldzhu  
**【打卡任务仓库地址】**: https://github.com/oldzhu/paddle-mx  
**【打卡内容】**: 在沐曦 Metax GPU 服务器上从源码编译安装 FastDeploy wheel 包 (release/2.5)

**【环境信息】**:

| 项目 | 值 |
|------|----|
| OS | Ubuntu 22.04.2 LTS (Linux 5.15.0-58-generic x86_64) |
| GPU | Metax MACA GPU (arch: xcore1000) |
| MACA 版本 | 3.3.0.15 |
| Python | 3.12.3 |
| PaddlePaddle | 3.4.0.dev20251223 |
| FastDeploy | 2.5.0 (branch: release/2.5, commit: 6d0d404a9) |

**【打卡内容详情】**:

1. 从 Gitee 克隆 FastDeploy release/2.5 分支：
   ```
   git clone https://gitee.com/paddlepaddle/FastDeploy.git -b release/2.5
   ```
2. 解决 Paddle 3.4 API 兼容问题（`PD_BUILD_STATIC_OP` → `PD_BUILD_OP`）
3. 适配 MACA GPU 环境（cu-bridge 路径，is_maca 检测）
4. 编译成功，生成 wheel 包：
   ```
   fastdeploy_cpu-2.5.0-py3-none-any.whl  (1.5MB)
   ```
5. 安装并验证导入：
   ```python
   import fastdeploy
   print(fastdeploy.__version__)  # → 2.5.0
   ```

**【关键日志节选】**:
```
[build] build fastdeploy_ops success
[build] build fastdeploy wheel success
fastdeploy wheel compiled and checked success
          fastdeploy branch: release/2.5 (6d0d404a9)
wheel saved under ./dist
```

```python
>>> import fastdeploy
>>> fastdeploy.__version__
'2.5.0'
```

---

如有任何问题，欢迎回复邮件。

谢谢！  
oldzhu
