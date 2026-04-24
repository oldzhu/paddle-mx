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

**【截图1 — 编译成功 (build log关键节选)】**:

```
# /root/fastdeploy_build.log  (2026-04-24 07:16, lines 575-339)

[build] build fastdeploy_ops success
...
[build] build fastdeploy wheel success

fastdeploy wheel compiled and checked success
          Python version: 3.12.3
          Paddle version: 3.4.0.dev20251223 (c8f9b3d44f61dd80d2c06b40000d1c21f7dca427)
          fastdeploy branch: release/2.5 (6d0d404a9)

wheel saved under ./dist
```

**【截图2 — wheel 文件】**:

```
root@d6c63c7b996e:~/FastDeploy/custom_ops# ls -lh /root/FastDeploy/dist/
total 1.5M
-rw-r--r-- 1 root root 1.5M Apr 24 07:16 fastdeploy_cpu-2.5.0-py3-none-any.whl
```

**【截图3 — pip install 成功】**:

```
root@d6c63c7b996e:~# pip3 install --no-deps /root/FastDeploy/dist/fastdeploy_cpu-2.5.0-py3-none-any.whl
...
Successfully installed fastdeploy-cpu-2.5.0
```

**【截图4 — import 验证】**:

```
root@d6c63c7b996e:~# pip3 show fastdeploy-cpu
Name: fastdeploy-cpu
Version: 2.5.0
Summary: FastDeploy: Large Language Model Serving.
Home-page: https://github.com/PaddlePaddle/FastDeploy
Author: PaddlePaddle
Location: /usr/local/lib/python3.12/dist-packages

root@d6c63c7b996e:~# python3 -c "import fastdeploy; print(fastdeploy.__version__)"
2.5.0
```

---

如有任何问题，欢迎回复邮件。

谢谢！  
oldzhu
