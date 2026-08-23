# 实验结果记录

本文档用于持续记录 V-JEPA 2 相关实验。每次实验完成后，在文档末尾新增一条记录，并填写实验时间、代码/模型版本、数据和关键配置，保证结果可以复查。

## 记录规范

- 实验时间使用 `YYYY-MM-DD HH:MM:SS TZ` 格式。
- 记录实际使用的模型权重、分类器权重、输入视频或数据集路径。
- 同时记录成功结果和失败信息；不要只保留最终准确率。
- 如果修改了代码或配置，记录对应文件和修改内容。

## 实验记录

### 001 - V-JEPA 2 Demo 推理复现

- 实验时间：2026-08-21 09:22:17 CST
- 实验目的：运行官方 Notebook Demo，验证 Hugging Face 模型、原生 PyTorch 模型和 SSv2 attentive probe 的推理流程。
- Hugging Face 模型：`facebook/vjepa2-vitg-fpc64-384`
- PyTorch backbone：`vitg-384.pt`
- 分类器权重：`pretrained_model/evals/ssv2-vitg-384-64x2x3.pt`
- 输入：`sample_video.mp4`
- 任务：Something-Something v2 视频动作分类
- 模型加载：成功，分类器输出 `<All keys matched successfully>`
- 分类器输出形状：`torch.Size([1, 174])`
- Top-5 预测：
  - `Pretending to pick [something] up`：68.9932%
  - `Moving away from [something] with your camera`：9.2889%
  - `Letting [something] roll along a flat surface`：7.7445%
  - `Pretending to take [something] from [somewhere]`：7.3050%
  - `Rolling [something] on a flat surface`：6.6685%
- 结论：Demo 推理流程已跑通，模型和分类器权重均能正常加载并输出 174 类预测。
- 备注：Notebook 原代码对 Top-5 logits 单独做 softmax，显示的百分比是 Top-5 内归一化结果，不是全部 174 类上的校准概率。运行时出现了 `softmax` 缺少 `dim` 的弃用警告。

## 新实验模板

### XXX - 实验名称

- 实验时间：YYYY-MM-DD HH:MM:SS TZ
- 实验目的：
- 代码/配置：
- 模型权重：
- 分类器或 predictor 权重：
- 数据集/输入：
- 关键参数：
- 运行设备：
- 训练或推理命令：
- 结果：
- 结论：
- 备注/异常：
