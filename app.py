#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓库货物检测 Web App (云端优化版)
基于 V7 核心检测 + V6 库存计算 + Streamlit 界面
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import os
import numpy as np
import tempfile
import random
from PIL import Image

# ==================== V7 核心检测函数 (封装) ====================

def detect_warehouse_goods_v7(image_path, output_path, conf=0.01, iou=0.5):
    """
    V7 核心检测逻辑 - 适配云端环境
    返回: {'final': 检测数量, 'counts': 分类统计, 'nms': 原始数量}
    """
    # 这里的模型路径是相对路径，云端会自动下载到当前目录
    MODEL_PATH = 'yolov8l-world.pt'

    # V7 优化后的类别列表
    CLASSES = [
        'textile bale',
        'woven sack',
        'pillow',
        'sandbag',
        'wrapped package',
        'stacked white sacks',
        'wall of bales'
    ]

    # V7 关键参数配置
    MIN_AREA_RATIO = 0.001  # 0.1% 面积阈值
    SLICE_MODE = True
    SLICE_HEIGHT = 640
    SLICE_WIDTH = 640
    SLICE_OVERLAP = 0.2
    AGNOSTIC_NMS = True
    CONF_THRESHOLD = conf
    DEDUP_THRESHOLD = 0.5

    # --- 关键修改：云端自动下载逻辑 ---
    # 不再因为文件不存在而返回 None，而是让 YOLO 自动处理下载
    try:
        model = YOLO(MODEL_PATH)
        model.set_classes(CLASSES)
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

    # 读取图片
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None

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

    # 切片检测循环
    # 创建临时目录用于保存切片图
    temp_dir = tempfile.mkdtemp()
    
    try:
        for i, (x1, y1, x2, y2, x_offset, y_offset) in enumerate(slices, 1):
            slice_img = original_img[y1:y2, x1:x2]
            
            # 使用 os.path.join 确保路径兼容性
            temp_path = os.path.join(temp_dir, f"slice_{i}.jpg")
            cv2.imwrite(temp_path, slice_img)

            results = model.predict(
                source=temp_path,
                conf=CONF_THRESHOLD,
                iou=iou,
                agnostic_nms=AGNOSTIC_NMS,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                # 映射回原图坐标
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
    finally:
        import shutil
        shutil.rmtree(temp_dir)

    # 全局去重 (NMS)
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
    
    # 固定随机颜色种子，保证每次运行颜色一致
    random.seed(42)
    colors = {}
    for cls_id in range(len(CLASSES)):
        colors[cls_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

    for box in final_boxes:
        x1, y1, x2, y2 = map(int, box['xyxy'])
        cls_id = box['cls']
        # conf = box['conf'] # 暂时不显示置信度，避免遮挡

        color = colors.get(cls_id, (0, 255, 0))
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        # 标签可以选开
        # label = f"{conf:.2f}"
        # cv2.putText(annotated_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(output_path, annotated_img)

    # 统计分类
    class_counts = {}
    for box in final_boxes:
        cls_id = box['cls']
        class_name = CLASSES[cls_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return {
        'final': len(final_boxes),
        'nms': len(unique_boxes),
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
    st.caption("基于 YOLO-World V7 | 支持云端自动部署 | 适配移动端")
    st.markdown("---")

    # 侧边栏 - 说明与配置
    with st.sidebar:
        st.header("⚙️ 参数配置")
        
        # 参数调整
        conf_val = st.slider("置信度阈值 (Conf)", 0.01, 0.5, 0.01, help="越低发现越多，越高越准确")
        iou_val = st.slider("去重阈值 (IoU)", 0.1, 0.9, 0.5, help="控制重叠框的合并程度")

        st.divider()
        st.info("""
        **使用指南：**
        1. 上传仓库照片
        2. 点击"开始检测"
        3. 等待 AI 分析（首次运行需下载模型）
        4. 输入堆叠深度，计算总库存
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
            # 显示上传的原图
            image = Image.open(uploaded_file)
            st.image(image, caption="原始图片", use_container_width=True)
            
            # 保存上传的图片到临时文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_input:
                input_path = tmp_input.name
                # 重置文件指针并保存
                uploaded_file.seek(0)
                tmp_input.write(uploaded_file.read())

            output_path = "result_cloud.jpg"

            # 检测按钮
            if st.button("🔍 开始检测", type="primary", use_container_width=True):
                # 关键提示语，安抚用户等待模型下载
                with st.spinner("🚀 AI 引擎启动中... (首次运行可能需要 3-5 分钟下载模型，请耐心等待，切勿刷新！)"):
                    try:
                        result = detect_warehouse_goods_v7(input_path, output_path, conf=conf_val, iou=iou_val)
                        
                        if result:
                            # 将结果存入 Session State 防止刷新丢失
                            st.session_state['result'] = result
                            st.session_state['has_result'] = True
                            st.rerun() # 强制刷新以显示结果
                        else:
                            st.error("❌ 检测返回为空，请重试")
                    except Exception as e:
                        st.error(f"❌ 运行出错: {str(e)}")
                        st.info("💡 提示: 如果是第一次运行，可能是下载模型超时。请尝试点击右下角 'Manage app' -> 'Reboot app'。")

    with col2:
        st.subheader("📊 2. 检测结果")

        if st.session_state.get('has_result'):
            result = st.session_state['result']
            
            # 显示结果图片
            if os.path.exists("result_cloud.jpg"):
                result_img = Image.open("result_cloud.jpg")
                st.image(result_img, caption=f"检测结果 (发现 {result['final']} 个目标)", use_container_width=True)
            
            # 结果统计卡片
            st.success(f"✅ 检测完成！视觉可见数量: **{result['final']}** 个")
            
            with st.expander("查看详细数据"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("最终计数", result['final'])
                with col_b:
                    st.metric("原始检测", result['nms'])
                
                st.write("分类统计:")
                for cls_name, count in result['counts'].items():
                    st.write(f"- {cls_name}: {count}")

            st.markdown("---")

            # V6 库存计算逻辑
            st.subheader("🧮 3. 库存计算器")
            
            st.info("💡 视觉只能看到表面。请输入货物的堆叠深度来计算总数。")
            
            depth = st.number_input(
                "堆叠深度 (Deep) - 例如里面还藏了几排？",
                min_value=1,
                value=1,
                step=1
            )

            total_stock = result['final'] * depth

            st.markdown(f"""
            <div style="padding: 15px; background-color: #e8f0fe; border-radius: 8px; border: 1px solid #4285f4; text-align: center;">
                <h4 style="margin:0; color:#1967d2;">📦 估算总库存</h4>
                <h2 style="margin:10px 0; color:#1967d2;">{total_stock} 个</h2>
                <small style="color:#666;">(视觉可见 {result['final']} × 深度 {depth})</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            
            # 下载按钮
            if os.path.exists("result_cloud.jpg"):
                with open("result_cloud.jpg", "rb") as file:
                    st.download_button(
                        label="📥 下载识别结果图",
                        data=file,
                        file_name="warehouse_result.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
        else:
            st.info("👈 请在左侧上传图片并开始检测")

if __name__ == "__main__":
    main()
