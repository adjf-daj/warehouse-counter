#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓库货物检测 Web App
基于 V7 版本 + V6 库存计算
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import os
import numpy as np
import tempfile
import random

# ==================== V7 核心检测函数 (封装) ====================

def detect_warehouse_goods_v7(image_path, output_path, conf=0.01, iou=0.5):
    """
    V7 核心检测逻辑
    返回: {'final': 检测数量, 'counts': 分类统计}
    """
    MODEL_PATH = 'yolov8l-world.pt'

    CLASSES = [
        'textile bale',
        'woven sack',
        'pillow',
        'sandbag',
        'wrapped package',
        'stacked white sacks',
        'wall of bales'
    ]

    MIN_AREA_RATIO = 0.001
    SLICE_MODE = True
    SLICE_HEIGHT = 640
    SLICE_WIDTH = 640
    SLICE_OVERLAP = 0.2
    AGNOSTIC_NMS = True
    CONF_THRESHOLD = conf
    DEDUP_THRESHOLD = 0.5

    # 检查模型
    if not os.path.exists(MODEL_PATH):
        return None

    # 加载模型
    model = YOLO(MODEL_PATH)
    model.set_classes(CLASSES)

    # 读取图片
    original_img = cv2.imread(image_path)
    h, w = original_img.shape[:2]
    total_area = w * h
    min_area = total_area * MIN_AREA_RATIO

    # 切片计算
    slice_h = SLICE_HEIGHT
    slice_w = SLICE_WIDTH
    overlap_h = int(slice_h * SLICE_OVERLAP)
    overlap_w = int(slice_w * SLICE_OVERLAP)

    slices = []
    y_start = 0
    while y_start < h:
        y_end = min(y_start + slice_h, h)
        x_start = 0
        while x_start < w:
            x_end = min(x_start + slice_w, w)

            x1 = max(0, x_start - overlap_w if x_start > 0 else 0)
            y1 = max(0, y_start - overlap_h if y_start > 0 else 0)
            x2 = min(w, x_end + overlap_w if x_end < w else w)
            y2 = min(h, y_end + overlap_h if y_end < h else h)

            slices.append((x1, y1, x2, y2, x_start, y_start))
            x_start += slice_w - overlap_w
        y_start += slice_h - overlap_h

    all_boxes_before_nms = []

    # 切片检测
    for i, (x1, y1, x2, y2, x_offset, y_offset) in enumerate(slices, 1):
        slice_img = original_img[y1:y2, x1:x2]

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            temp_path = tmp.name

        cv2.imwrite(temp_path, slice_img)

        results = model.predict(
            source=temp_path,
            conf=CONF_THRESHOLD,
            iou=iou,
            agnostic_nms=AGNOSTIC_NMS,
            verbose=False
        )

        os.remove(temp_path)

        result = results[0]
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

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

    # 全局去重
    all_boxes_before_nms.sort(key=lambda x: x['conf'], reverse=True)

    unique_boxes = []
    for box in all_boxes_before_nms:
        is_duplicate = False
        x1, y1, x2, y2 = box['xyxy']

        for existing in unique_boxes:
            ex1, ey1, ex2, ey2 = existing['xyxy']

            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (ex2 - ex1) * (ey2 - ey1)
                iou_val = intersection / (area1 + area2 - intersection)

                if iou_val > DEDUP_THRESHOLD:
                    is_duplicate = True
                    break

        if not is_duplicate:
            unique_boxes.append(box)

    # 尺寸过滤
    final_boxes = []
    for box in unique_boxes:
        if box['area'] >= min_area:
            final_boxes.append(box)

    # 生成可视化结果
    annotated_img = original_img.copy()
    colors = {}
    for cls_id in range(len(CLASSES)):
        colors[cls_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for box in final_boxes:
        x1, y1, x2, y2 = map(int, box['xyxy'])
        cls_id = box['cls']
        conf = box['conf']

        color = colors.get(cls_id, (255, 255, 255))
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

        label = f"{conf:.2f}"
        cv2.putText(annotated_img, label, (x1+2, y2-2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, annotated_img)

    # 统计分类
    class_counts = {}
    for box in final_boxes:
        cls_id = box['cls']
        class_name = CLASSES[cls_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return {
        'final': len(final_boxes),
        'counts': class_counts,
        'result_image': output_path
    }


# ==================== Streamlit Web App ====================

def main():
    st.set_page_config(
        page_title="仓库货物检测系统",
        page_icon="📦",
        layout="wide"
    )

    st.title("📦 仓库货物检测与库存计算系统")
    st.markdown("---")

    # 侧边栏 - 说明
    with st.sidebar:
        st.header("系统说明")
        st.markdown("""
        **基于 YOLO-World V7 版本**

        功能：
        - ✅ 零样本货物检测
        - ✅ 智能去重
        - ✅ 库存计算

        检测类别：
        - textile bale
        - woven sack
        - pillow
        - sandbag
        - wrapped package
        - stacked white sacks
        - wall of bales
        """)

    # 主界面
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 1. 上传图片")
        uploaded_file = st.file_uploader(
            "请选择一张仓库图片",
            type=['jpg', 'jpeg', 'png'],
            help="支持 JPG, JPEG, PNG 格式"
        )

        if uploaded_file is not None:
            # 保存上传的图片
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_input:
                input_path = tmp_input.name
                tmp_input.write(uploaded_file.read())

            # 输出路径
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_output:
                output_path = tmp_output.name

            st.success(f"✅ 图片已上传: {uploaded_file.name}")

            # 检测按钮
            if st.button("🔍 开始检测", type="primary"):
                with st.spinner("正在检测中，请稍候..."):
                    result = detect_warehouse_goods_v7(input_path, output_path)

                if result:
                    # 保存结果到 session_state
                    st.session_state['detection_result'] = result
                    st.session_state['output_path'] = output_path
                    st.session_state['input_path'] = input_path
                    st.success(f"✅ 检测完成！发现 {result['final']} 个包裹")
                else:
                    st.error("❌ 检测失败，请检查模型文件是否存在")

            # 清理临时文件
            try:
                os.remove(input_path)
            except:
                pass

    with col2:
        st.subheader("📊 2. 检测结果")

        if 'detection_result' in st.session_state:
            result = st.session_state['detection_result']
            output_path = st.session_state['output_path']

            # 显示结果图片
            st.image(output_path, caption="检测结果", use_column_width=True)

            # 文字统计
            st.markdown("---")
            st.markdown(f"### 📦 视觉识别到 **{result['final']}** 个可见包裹")

            # 分类详情
            with st.expander("查看分类详情"):
                for cls_name, count in result['counts'].items():
                    st.markdown(f"- **{cls_name}**: {count} 个")

            st.markdown("---")

            # V6 库存计算逻辑
            st.subheader("🧮 3. 库存计算")

            depth = st.number_input(
                "请输入堆叠深度 (Deep)：",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                help="每层堆叠的深度数量"
            )

            total = result['final'] * depth

            st.markdown("---")
            st.markdown(f"""
            <div style="padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; color: #1f77b4;">当前库存总数</h3>
                <p style="margin: 10px 0; font-size: 24px;">
                    <strong>{result['final']} × {depth} = {total}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # 下载结果
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 下载检测结果图",
                    data=file,
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )

        else:
            st.info("👈 请先上传图片并点击检测按钮")

if __name__ == "__main__":
    main()
