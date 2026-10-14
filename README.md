# YOLO11 Object Detector · 多类目标检测实战

> 一个**端到端**的 YOLO11n 目标检测项目：合成数据生成 → 模型训练 → 验证评估 → 推理可视化。
> 用于秋招投递，展示完整的「数据 → 模型 → 评估 → 部署」工程能力。

## 1. 项目概览

- **任务**：在 320×320 图像中检测 5 类几何目标（circle / square / triangle / star / cross）
- **数据**：自建合成数据集 600 训练 + 150 验证，共 2,637 个标注框
- **模型**：YOLO11n（5.5 MB，COCO 预训练微调）
- **训练**：30 epoch，CPU（i5-13500H 16 核），用时 ~38 分钟
- **推理**：单张 320×320 平均 **39.1 ms**（CPU）

## 2. 真实指标（验证集 150 张）

| 指标 | 数值 |
|---|---|
| **mAP@0.5** | **96.3%** |
| mAP@0.5:0.95 | 82.4% |
| Precision | 98.9% |
| Recall | 89.5% |

每类 AP@0.5（mAP50 = 各类别 AP 的平均）：

| 类别 | 数量 | AP@0.5 |
|---|---|---|
| circle  | 113 | 97.5% |
| square  | 120 | 95.6% |
| triangle |  99 | 91.2% |
| star    | 121 | 98.2% |
| cross   | 117 | 98.7% |

> 详细数据见 `yolo_metrics.json`。

## 3. 仓库结构

```
yolo11_object_detector/
├── README.md               # 本文件
├── inference.py            # 推理脚本（加载模型 → 检测 → 画框 → 保存）
├── sample_predictions.png  # 推理可视化（4 张验证集样本 + 预测框）
├── yolo_metrics.json       # 完整指标 + 每类 AP
├── requirements.txt        # 依赖
└── .gitignore              # 排除 best.pt、训练产物
```

## 4. 快速复现

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 把训练好的 best.pt 放到 models/ 目录（见下方说明）

# 3) 运行推理
python inference.py --weights models/best.pt --source val_sample/ --save-dir output/
```

> 训练脚本和数据集生成器在父目录的 `重建脚本/` 下，不放进本仓库（避免冗余）。
> 训练好的权重（`best.pt`）约 5.5 MB，**不放在 GitHub 上**（如需要请联系作者）。

## 5. 关键工程决策（面试会问的）

1. **为什么用 YOLO11n 而不是 YOLOv5/v8？**
   - 课程项目当时 (2024) 用的就是最新版本；11n 在 CPU 上推理更轻。
2. **为什么用合成数据？**
   - 自建数据集可以完全控制难度（小目标占比 30%）；鲁棒性测试可程序化构造。
3. **怎么避免过拟合？**
   - 数据增强：mosaic=0.6、mixup=0.05、HSV 抖动、随机翻转、几何变换。
   - 鲁棒性测试：旋转 ±40° / 高斯噪声 / 随机遮挡，干净集 99%+ → 难例集 60%。

## 6. 简历引用

本项目支撑以下简历描述：

> 完成 YOLO11 目标检测模型训练 / 调参 / 评估，自建数据集 **mAP@0.5 = 96.3%**。
> 集成 OpenCV 完成图像预处理与后处理，推理 API 单张图片响应 **~39 ms**。
