#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓库货物检测工具 - V7版（修复背景检测）
目标：检测出左右两侧墙壁上的密集货物
"""

import cv2
from ultralytics import YOLO
import argparse
import os
import sys
import numpy as np


def detect_warehouse_goods_v7(image_path, output_path='result_v7.jpg',
                              conf=0.01, iou=0.5, show=False):
    """
    V7版 - 修复背景检测
    1. 极低尺寸阈值 (0.1%)
    2. 重新开启切片
    3. 强化背景提示词
    4. 详细调试输出
    """

    # ==================== V7 配置区域 ====================
    # 自动下载/加载模型
    MODEL_PATH = 'yolov8l-world.pt'

    # V7: 强化背景提示词 + 软词
    CLASSES = [
        'textile bale',        # 工业词
        'woven sack',          # 工业词
        'pillow',              # 软词
        'sandbag',             # 软词
        'wrapped package',     # 工业词
        'stacked white sacks', # 新增：背景白墙
        'wall of bales'        # 新增：背景货堆
    ]

    # V7 关键参数
    MIN_AREA_RATIO = 0.001     # 0.1% - 超低阈值，保留小包 (关键修复!)
    SLICE_MODE = True          # 重新开启切片！(关键修复!)
    SLICE_HEIGHT = 640         # 切片高度
    SLICE_WIDTH = 640          # 切片宽度
    SLICE_OVERLAP = 0.2        # 20%重叠
    AGNOSTIC_NMS = True        # 保持V4的优点，跨类别去重
    IOU_THRESHOLD = iou        # 0.5 - 严格去重
    CONF_THRESHOLD = conf      # 0.01 - 低阈值
    DEDUP_THRESHOLD = 0.5      # 切片合并时的去重阈值
    # =================================================

    print("=" * 70)
    print("  仓库货物检测工具 V7 - 背景检测修复版")
    print("=" * 70)
    print("  目标: 检出左右墙壁的密集货物")
    print("  配置: 极低阈值(0.1%) + 切片开启 + 强化背景词")
    print("=" * 70)

    # 1. 检查文件
    print("\n[步骤1/7] 检查文件...")
    if not os.path.exists(image_path):
        print(f"❌ 错误: 图片文件 '{image_path}' 不存在")
        return None

    print(f"✅ 输入: {image_path}")

    # 2. 加载模型
    print(f"\n[步骤2/7] 加载 YOLO-World 模型...")
    try:
        # YOLO会自动下载模型，无需手动检查文件是否存在
        model = YOLO(MODEL_PATH) 
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

    # 3. 设置类别
    print(f"\n[步骤3/7] 设置检测类别 ({len(CLASSES)}种)...")
    print("   (背景强化: stacked white sacks, wall of bales)")
    # for i, cls in enumerate(CLASSES, 1):
    #     print(f"   {i}. {cls}")
    model.set_classes(CLASSES)
    print("✅ 类别设置完成")

    # 4. 读取图片信息
    print(f"\n[步骤4/7] 图片分析...")
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("❌ 无法读取图片，请检查路径或格式")
        return None
        
    h, w = original_img.shape[:2]
    total_area = w * h
    min_area = total_area * MIN_AREA_RATIO

    print(f"   图片尺寸: {w}x{h}")
    print(f"   总面积: {total_area:,} 像素")
    print(f"   最小过滤面积: {min_area:.2f} 像素 ({MIN_AREA_RATIO*100}%)")
    print(f"   切片配置: {SLICE_HEIGHT}x{SLICE_WIDTH}, 重叠 {SLICE_OVERLAP*100}%")

    # 5. 执行切片检测
    print(f"\n[步骤5/7] 执行切片检测...")
    print("   (SAHI 算法 - 分块检测后合并)")

    # 计算切片网格
    slice_h = SLICE_HEIGHT
    slice_w = SLICE_WIDTH
    overlap_h = int(slice_h * SLICE_OVERLAP)
    overlap_w = int(slice_w * SLICE_OVERLAP)

    # 切片坐标计算
    slices = []
    y_start = 0
    while y_start < h:
        y_end = min(y_start + slice_h, h)
        x_start = 0
        while x_start < w:
            x_end = min(x_start + slice_w, w)

            # 计算带重叠的切割区域
            x1 = max(0, x_start - overlap_w if x_start > 0 else 0)
            y1 = max(0, y_start - overlap_h if y_start > 0 else 0)
            x2 = min(w, x_end + overlap_w if x_end < w else w)
            y2 = min(h, y_end + overlap_h if y_end < h else h)

            slices.append((x1, y1, x2, y2, x_start, y_start))
            x_start += slice_w - overlap_w
        y_start += slice_h - overlap_h

    print(f"   生成 {len(slices)} 个切片")

    all_boxes_before_nms = []
    all_boxes_after_nms = []
    total_raw_detections = 0

    # 创建临时目录
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    try:
        for i, (x1, y1, x2, y2, x_offset, y_offset) in enumerate(slices, 1):
            # 裁剪切片
            slice_img = original_img[y1:y2, x1:x2]

            # 保存临时文件
            temp_path = os.path.join(temp_dir, f'temp_v7_slice_{i}.jpg')
            cv2.imwrite(temp_path, slice_img)

            # 检测
            results = model.predict(
                source=temp_path,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                agnostic_nms=AGNOSTIC_NMS,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes

            # 统计原始检测数
            total_raw_detections += len(boxes)

            # 转换坐标并收集
            if len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()

                    # 加上偏移量 (映射回原图坐标)
                    xyxy[0] += x1
                    xyxy[1] += y1
                    xyxy[2] += x1
                    xyxy[3] += y1

                    all_boxes_before_nms.append({
                        'cls': cls_id,
                        'conf': conf_score,
                        'xyxy': xyxy,
                        'area': (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    })

            # 进度提示
            if i % 4 == 0 or i == len(slices):
                print(f"   已处理切片 {i}/{len(slices)}...", end='\r')

    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

    print(f"\n\n✅ 切片检测完成")
    print(f"   原始检测总数: {len(all_boxes_before_nms)} 个")

    # 6. 全局去重
    print(f"\n[步骤6/7] 全局去重...")

    if not all_boxes_before_nms:
        print("❌ 未检测到任何物体")
        return None

    # 按置信度排序
    all_boxes_before_nms.sort(key=lambda x: x['conf'], reverse=True)

    # 应用agnostic NMS
    unique_boxes = []
    for box in all_boxes_before_nms:
        is_duplicate = False
        x1, y1, x2, y2 = box['xyxy']

        for existing in unique_boxes:
            ex1, ey1, ex2, ey2 = existing['xyxy']

            # 计算IoU
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (ex2 - ex1) * (ey2 - ey1)
                iou = intersection / (area1 + area2 - intersection)

                if iou > DEDUP_THRESHOLD:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_boxes.append(box)

    all_boxes_after_nms = unique_boxes
    print(f"   NMS后数量: {len(all_boxes_after_nms)} 个")

    # 7. 尺寸过滤（极低阈值）
    print(f"\n[步骤7/7] 尺寸过滤...")
    final_boxes = []
    filtered_count = 0

    for box in all_boxes_after_nms:
        if box['area'] >= min_area:
            final_boxes.append(box)
        else:
            filtered_count += 1

    print(f"   过滤前: {len(all_boxes_after_nms)} 个")
    print(f"   过滤后: {len(final_boxes)} 个")
    print(f"   过滤掉: {filtered_count} 个 (<{MIN_AREA_RATIO*100}% 面积)")

    # 打印调试信息
    print("\n" + "=" * 70)
    print("  🔍 调试输出 - 检测流程追踪")
    print("=" * 70)
    print(f"  1. 原始切片检测: {total_raw_detections} 个框")
    print(f"  2. 全局去重(NMS): {len(all_boxes_after_nms)} 个框")
    print(f"  3. 尺寸过滤后:   {len(final_boxes)} 个框")
    print(f"  4. 过滤损失率:    {((total_raw_detections - len(final_boxes)) / (total_raw_detections + 1e-6) * 100):.1f}%")
    print("=" * 70)

    # 生成结果
    if len(final_boxes) > 0:
        print(f"\n📊 最终结果统计:")
        print("=" * 50)

        # 分类统计
        class_counts = {}
        total_calculated = 0

        for box in final_boxes:
            cls_id = box['cls']
            class_name = CLASSES[cls_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_calculated += 1

        for cls_name, count in class_counts.items():
            if count > 0:
                print(f"  {cls_name}: {count} 个")

        print("=" * 50)
        print(f"  视觉检测总计: {total_calculated} 个包裹")
        print("=" * 50)

        # 保存可视化结果
        print(f"\n💾 生成可视化结果...")
        annotated_img = original_img.copy()

        # 不同类别用不同颜色
        import random
        random.seed(42) # 固定颜色
        colors = {}
        for cls_id in range(len(CLASSES)):
            colors[cls_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

        for box in final_boxes:
            x1, y1, x2, y2 = map(int, box['xyxy'])
            cls_id = box['cls']
            conf = box['conf']
            class_name = CLASSES[cls_id]

            color = colors.get(cls_id, (0, 255, 0))
            # 画细一点的框，避免遮挡
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 1)

            # 标签不要遮挡太多
            # label = f"{conf:.2f}"
            # cv2.putText(annotated_img, label, (x1, y1-2),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        cv2.imwrite(output_path, annotated_img)
        print(f"✅ 结果图已保存: {output_path}")

        # 打印结论
        print("\n" + "=" * 70)
        if total_calculated >= 50:
            print(f"✅ 成功！检测到 {total_calculated} 个包裹")
            print(f"   V7 应该已经看见了背景墙上的大部分货物！")
        else:
            print(f"⚠️ 检测数量: {total_calculated} 个")
            print(f"   如果背景还是空的，请检查图片光线是否过暗。")
        print("=" * 70)

        return {
            'raw': total_raw_detections,
            'nms': len(all_boxes_after_nms),
            'final': len(final_boxes),
            'counts': class_counts
        }

    else:
        print("\n❌ 未检测到任何货物")
        return None


def main():
    parser = argparse.ArgumentParser(description='仓库货物检测 V7 - 背景修复版')
    parser.add_argument('--image', type=str, default='test.jpg', help='输入图片路径')
    parser.add_argument('--output', type=str, default='result_v7.jpg', help='输出图片路径')
    parser.add_argument('--conf', type=float, default=0.01, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU 阈值')
    parser.add_argument('--show', action='store_true', help='显示结果')

    args = parser.parse_args()

    detect_warehouse_goods_v7(
        image_path=args.image,
        output_path=args.output,
        conf=args.conf,
        iou=args.iou,
        show=args.show
    )


if __name__ == "__main__":
    main()
