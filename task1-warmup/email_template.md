# Task 1 Check-in Email Template

Fill in the `[PLACEHOLDER]` sections and attach the screenshots before sending.

---

## Recipients

- **To**: ext_paddle_oss@baidu.com; kaichuang.gao@metax-tech.com; yang.yang2@metax-tech.com
- **CC**: _(optional — add if relevant)_

## Subject

```
文心伙伴赛道-沐曦-打卡-oldzhu
```

---

## Email Body

飞桨团队你好，

**【GitHub ID】**: oldzhu  
**【打卡任务仓库地址】**: https://github.com/oldzhu/paddle-mx _(update with actual repo URL once pushed)_  
**【打卡内容】**: 编译/安装 FastDeploy wheel 包 (release/2.5, Metax GPU)

**【环境信息】**:

| 项目 | 值 |
|------|----|
| OS | [TBD — e.g., Ubuntu 22.04] |
| CPU | [TBD — e.g., Intel Xeon XX] |
| GPU | Metax 曦云C500 64G |
| MACA 版本 | maca 3.3.0.4 |
| Python | 3.12.x |
| PaddlePaddle | 3.4.0.dev20251223 |
| paddle-metax-gpu | 3.3.0.dev20251224 |
| FastDeploy | [TBD — paste version from `python -c "import fastdeploy; print(fastdeploy.__version__)"`] |

**【打卡截图】**:

> _(Attach or paste the following screenshots inline or as attachments)_

1. **编译成功截图** — terminal showing `bash build.sh` completed without errors, ending with the success message
2. **wheel 文件截图** — `ls -lh ~/fastdeploy/dist/` showing the generated `.whl` file
3. **安装成功截图** — `pip install fastdeploy-*.whl` success output + `python -c "import fastdeploy; print(fastdeploy.__version__)"` result
4. **GPU 信息截图** — `maca-smi` output confirming Metax C500 GPU

---

如有任何问题，欢迎回复邮件或通过 MXMACA 开发者社群联系。

谢谢！  
oldzhu
