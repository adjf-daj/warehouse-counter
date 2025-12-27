#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓库货物检测专业版 Web App (V9 Fixed)
修复 IoU 滑块对全局去重失效的 BUG
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import os
import numpy as np
import tempfile
import random
from PIL import Image
import shutil
import time

# 设置页面配置
st.set_page_config(
    page_title="AI 仓库视觉盘点 Pro",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 后端逻辑 ====================

@st.cache_resource(show_spinner=False)
def load_model():
    """缓存加载模型"""
    MODEL_PATH = 'yolov8l-world.pt'
    CLASSES = [
        'textile bale', 'woven sack', 'pillow', 'sandbag',
        'wrapped package', 'stacked white sacks', 'wall of bales'
    ]
    try:
        model = YOLO(MODEL_PATH)
        model.set_classes(CLASSES)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

def detect_warehouse_goods_v7_web(image_path, conf, iou, model):
    """V7 核心检测逻辑 (Web适配版)"""
    # 参数配置
    MIN_AREA_RATIO = 0.001
    SLICE_HEIGHT, SLICE_WIDTH = 640, 640
    SLICE_OVERLAP = 0.2
    AGNOSTIC_NMS = True
    
    # [关键修复] 让全局去重阈值直接等于用户设置的 IoU
    DEDUP_THRESHOLD = iou 

    original_img = cv2.imread(image_path)
    if original_img is None: return None
    h, w = original_img.shape[:2]
    min_area = w * h * MIN_AREA_RATIO

    # 切片计算
    overlap_h, overlap_w = int(SLICE_HEIGHT * SLICE_OVERLAP), int(SLICE_WIDTH * SLICE_OVERLAP)
    slices = []
    y_start = 0
    while y_start < h:
        y_end = min(y_start + SLICE_HEIGHT, h)
        x_start = 0
        while x_start < w:
            x_end = min(x_start + SLICE_WIDTH, w)
            x1, y1 = max(0, x_start - overlap_w if x_start > 0 else 0), max(0, y_start - overlap_h if y_start > 0 else 0)
            x2, y2 = min(w, x_end + overlap_w if x_end < w else w), min(h, y_end + overlap_h if y_end < h else h)
            slices.append((x1, y1, x2, y2, x_start, y_start))
            x_start += SLICE_WIDTH - overlap_w
        y_start += SLICE_HEIGHT - overlap_h

    # 切片检测
    all_boxes = []
    temp_dir = tempfile.mkdtemp()
    progress_bar = st.progress(0)

    try:
        for i, (x1, y1, x2, y2, _, _) in enumerate(slices):
            progress_bar.progress((i + 1) / len(slices), text=f"正在分析切片 {i+1}/{len(slices)}...")
            
            slice_img = original_img[y1:y2, x1:x2]
            temp_path = os.path.join(temp_dir, f"slice_{i}.jpg")
            cv2.imwrite(temp_path, slice_img)
            
            # 这里是局部 NMS
            results = model.predict(source=temp_path, conf=conf, iou=iou, agnostic_nms=AGNOSTIC_NMS, verbose=False)
            
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                # 坐标映射回原图
                xyxy[0] += x1; xyxy[1] += y1; xyxy[2] += x1; xyxy[3] += y1
                all_boxes.append({
                    'cls': int(box.cls[0]), 'conf': float(box.conf[0]),
                    'xyxy': xyxy, 'area': (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])
                })
    finally:
        shutil.rmtree(temp_dir)
        progress_bar.empty()

    # --- 全局 NMS (关键步骤) ---
    # 先按置信度排序
    all_boxes.sort(key=lambda x: x['conf'], reverse=True)
    unique_boxes = []
    
    for box in all_boxes:
        is_duplicate = False
        # 拿当前框去和已经保留的框做对比
        for xb in unique_boxes:
            # 如果重叠度超过了用户设定的 DEDUP_THRESHOLD (例如 0.1)
            if compute_iou(box['xyxy'], xb['xyxy']) > DEDUP_THRESHOLD:
                is_duplicate = True
                break # 只要和一个重叠过高，就丢弃
        
        if not is_duplicate:
            unique_boxes.append(box)

    # 尺寸过滤
    final_boxes = [b for b in unique_boxes if b['area'] >= min_area]

    # 可视化
    annotated_img = original_img.copy()
    random.seed(42)
    colors = {i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in range(len(model.names))}
    class_counts = {}
    
    for box in final_boxes:
        cls_id = box['cls']
        class_name = model.names[cls_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        x1, y1, x2, y2 = map(int, box['xyxy'])
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), colors[cls_id], 2)

    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    return {
        'final_count': len(final_boxes),
        'raw_count': len(unique_boxes), # 这里其实已经是NMS后的了，为了不混淆显示
        'counts_detail': class_counts,
        'result_img_rgb': annotated_img_rgb
    }

def compute_iou(box1, box2):
    """计算两个框的 IoU (重叠度)"""
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if ix1 >= ix2 or iy1 >= iy2: return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection / (area1 + area2 - intersection + 1e-6)

# ==================== 前端 UI ====================

def main():
    with st.spinner("🏭 正在初始化 AI 引擎..."):
        model = load_model()

    if model is None: st.stop()

    with st.sidebar:
        st.title("⚙️ 控制面板")
        st.subheader("参数微调")
        # 默认值设为 0.2，方便你直接测试
        conf_val = st.slider("置信度 (Conf)", 0.01, 0.5, 0.15, help="过滤掉得分低的框")
        iou_val = st.slider("去重阈值 (IoU)", 0.05, 0.8, 0.2, help="越小去重越狠。设为0.1表示只要重叠10%就合并。")
        st.caption("Version: V9 Fixed")

    st.title("🏭 AI 仓库视觉盘点 Pro")
    
    uploaded_file = st.file_uploader("📤 上传照片", type=['jpg', 'png'])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_input:
            input_path = tmp_input.name
            uploaded_file.seek(0)
            tmp_input.write(uploaded_file.read())

        # 只要参数变了，或者图片变了，就重新运行
        trigger = f"{uploaded_file.name}_{conf_val}_{iou_val}"
        
        if 'last_trigger' not in st.session_state or st.session_state['last_trigger'] != trigger:
             with st.status("🚀 正在分析 (应用新参数)...", expanded=True) as status:
                result_data = detect_warehouse_goods_v7_web(input_path, conf_val, iou_val, model)
                if result_data:
                    st.session_state['result_data'] = result_data
                    st.session_state['last_trigger'] = trigger
                    status.update(label="✅ 分析完成", state="complete", expanded=False)
        
        if 'result_data' in st.session_state:
            data = st.session_state['result_data']
            
            st.subheader("📊 分析看板")
            col1, col2, col3 = st.columns(3)
            col1.metric("📦 最终计数", f"{data['final_count']} 个")
            col2.metric("🎯 参数状态", f"IoU={iou_val}")
            
            st.image(data['result_img_rgb'], caption="识别结果", use_container_width=True)
            
            # 下载逻辑
            img_bgr = cv2.cvtColor(data['result_img_rgb'], cv2.COLOR_RGB2BGR)
            is_success, buffer = cv2.imencode(".jpg", img_bgr)
            st.download_button("📥 下载结果图", buffer.tobytes(), "result.jpg", "image/jpeg")

if __name__ == "__main__":
    main()
