# -*- coding: utf-8 -*-
"""
inference.py - YOLO11 目标检测推理脚本

加载训练好的 best.pt，对指定目录的图像做推理，画出检测框，保存到输出目录。

用法：
  python inference.py --weights best.pt --source val_sample/ --save-dir output/
  python inference.py --weights best.pt --source single.jpg --save-dir output/
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

# 5 类对应的颜色 (BGR)
COLORS = {
    0: (76, 114, 240),    # circle  - 蓝
    1: (60, 180, 75),     # square  - 绿
    2: (50, 50, 220),     # triangle- 红
    3: (240, 200, 50),    # star    - 黄
    4: (180, 60, 220),    # cross   - 紫
}
CLASSES = ["circle", "square", "triangle", "star", "cross"]


def draw_box(img, x1, y1, x2, y2, cls, conf):
    c = COLORS.get(cls, (255, 255, 255))
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
    label = f"{CLASSES[cls]} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (int(x1), int(y1) - th - 8), (int(x1) + tw + 4, int(y1)), c, -1)
    cv2.putText(img, label, (int(x1) + 2, int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="best.pt 路径")
    ap.add_argument("--source", required=True, help="图像文件 / 目录")
    ap.add_argument("--save-dir", default="output", help="输出目录")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    model = YOLO(args.weights)

    if os.path.isdir(args.source):
        files = sorted(glob.glob(os.path.join(args.source, "*.jpg")) +
                       glob.glob(os.path.join(args.source, "*.png")))
    else:
        files = [args.source]

    print(f"[inference] {len(files)} 张图, imgsz={args.imgsz}, conf={args.conf}")
    t_all = []
    for f in files:
        img0 = cv2.imread(f)
        if img0 is None:
            print(f"  [SKIP] {f}")
            continue
        t0 = time.time()
        results = model.predict(source=img0, imgsz=args.imgsz,
                                conf=args.conf, device="cpu", verbose=False)
        dt = (time.time() - t0) * 1000
        t_all.append(dt)
        boxes = results[0].boxes
        n = len(boxes) if boxes is not None else 0
        for i in range(n):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            cls = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            draw_box(img0, x1, y1, x2, y2, cls, conf)
        out = os.path.join(args.save_dir, os.path.basename(f))
        cv2.imwrite(out, img0)
        print(f"  [OK] {os.path.basename(f)} -> {n} 框, {dt:.1f} ms")
    if t_all:
        print(f"\n平均推理耗时: {np.mean(t_all):.1f} ms/张 ({len(t_all)} 张, CPU)")


if __name__ == "__main__":
    main()
