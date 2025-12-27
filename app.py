#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓库货物检测系统 V11 (手动触发版)
新增：开始检测按钮、自动清理旧数据、默认置信度0.01
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
import pandas as pd
from datetime import datetime
import time

# 页面配置
st.set_page_config(
    page_title="AI 批量盘点系统 V11",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 后端逻辑 ====================

@st.cache_resource(show_spinner=False)
def load_model():
    MODEL_PATH = 'yolov8l-world.pt'
    CLASSES = ['textile bale', 'woven sack', 'pillow', 'sandbag',
               'wrapped package', 'stacked white sacks', 'wall of bales']
    try:
        model = YOLO(MODEL_PATH)
        model.set_classes(CLASSES)
        return model
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

def detect_image(image_path, conf, iou, model):
    """单张图片检测逻辑"""
    SLICE_HEIGHT, SLICE_WIDTH = 640, 640
    SLICE_OVERLAP = 0.2
    AGNOSTIC_NMS = True
    MIN_AREA_RATIO = 0.001
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
    
    try:
        for i, (x1, y1, x2, y2, _, _) in enumerate(slices):
            slice_img = original_img[y1:y2, x1:x2]
            temp_path = os.path.join(temp_dir, f"slice_{i}.jpg")
            cv2.imwrite(temp_path, slice_img)
            
            results = model.predict(source=temp_path, conf=conf, iou=iou, agnostic_nms=AGNOSTIC_NMS, verbose=False)
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                xyxy[0] += x1; xyxy[1] += y1; xyxy[2] += x1; xyxy[3] += y1
                all_boxes.append({
                    'cls': int(box.cls[0]), 'conf': float(box.conf[0]),
                    'xyxy': xyxy, 'area': (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])
                })
    finally:
        shutil.rmtree(temp_dir)

    # 全局去重
    all_boxes.sort(key=lambda x: x['conf'], reverse=True)
    unique_boxes = []
    for box in all_boxes:
        if not any(compute_iou(box['xyxy'], xb['xyxy']) > DEDUP_THRESHOLD for xb in unique_boxes):
            unique_boxes.append(box)

    final_boxes = [b for b in unique_boxes if b['area'] >= min_area]

    # 绘图
    annotated_img = original_img.copy()
    random.seed(42)
    colors = {i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in range(len(model.names))}
    class_counts = {}
    for box in final_boxes:
        cls = model.names[box['cls']]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        x1, y1, x2, y2 = map(int, box['xyxy'])
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), colors[box['cls']], 2)

    return {
        'count': len(final_boxes),
        'img_rgb': cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
        'counts_detail': class_counts
    }

def compute_iou(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if ix1 >= ix2 or iy1 >= iy2: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter + 1e-6)

# ==================== 前端 UI 逻辑 ====================

def main():
    # 初始化 Session State
    if 'data_store' not in st.session_state: st.session_state['data_store'] = {}
    if 'user_edits' not in st.session_state: st.session_state['user_edits'] = {}

    with st.spinner("正在启动 AI 批量处理引擎..."):
        model = load_model()
    if not model: st.stop()

    # --- 侧边栏：全局控制 ---
    with st.sidebar:
        st.title("🏭 批量盘点控制台")
        st.markdown("---")
        
        # 1. 参数设置 (默认值已修改为 0.01)
        st.subheader("1. AI 参数")
        conf = st.slider("置信度", 0.01, 0.5, 0.01, help="默认0.01以发现更多货物")
        iou = st.slider("去重阈值", 0.05, 0.8, 0.2)
        
        st.markdown("---")
        
        # 2. 上传区域
        st.subheader("2. 图片选择")
        uploaded_files = st.file_uploader(
            "第一步：选择图片 (可多选)", 
            type=['jpg', 'png'], 
            accept_multiple_files=True
        )

        st.markdown("---")
        
        # 3. 执行按钮 (关键修改)
        st.subheader("3. 执行操作")
        start_btn = st.button("🚀 开始批量检测", type="primary", use_container_width=True)
        
        # 如果点击了开始按钮
        if start_btn:
            if not uploaded_files:
                st.warning("⚠️ 请先上传图片！")
            else:
                # 1. 清理旧数据 (实现“换文件不刷新”)
                st.session_state['data_store'] = {}
                st.session_state['user_edits'] = {}
                
                # 2. 开始处理
                st.info(f"📸 开始处理 {len(uploaded_files)} 张图片...")
                progress_bar = st.progress(0)
                
                for idx, file_obj in enumerate(uploaded_files):
                    # 显示当前正在处理的文件名
                    progress_bar.progress((idx) / len(uploaded_files), text=f"正在分析: {file_obj.name}...")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(file_obj.read())
                        tmp_path = tmp.name
                    
                    result = detect_image(tmp_path, conf, iou, model)
                    
                    if result:
                        st.session_state['data_store'][file_obj.name] = result
                        st.session_state['user_edits'][file_obj.name] = {'depth': 1, 'manual': 0}
                    
                    os.remove(tmp_path)
                
                progress_bar.progress(1.0, text="✅ 处理完成！")
                time.sleep(0.5) # 稍微停顿让用户看到完成状态
                st.rerun() # 刷新页面以显示结果

    # --- 主界面 ---
    st.title("🏭 仓库盘点总览")

    # 如果没有数据
    if not st.session_state['data_store']:
        st.info("👈 请在左侧上传图片，并点击【开始批量检测】按钮。")
        st.stop()

    # 1. Dashboard
    total_ai_count = sum([d['count'] for d in st.session_state['data_store'].values()])
    grand_total = 0
    for name, result in st.session_state['data_store'].items():
        edits = st.session_state['user_edits'].get(name, {'depth': 1, 'manual': 0})
        grand_total += (result['count'] + edits['manual']) * edits['depth']

    col1, col2, col3 = st.columns(3)
    col1.metric("📸 本次盘点图片", f"{len(st.session_state['data_store'])} 张")
    col2.metric("📦 视觉检测总和", f"{total_ai_count} 个")
    col3.metric("💰 最终库存总计", f"{grand_total} 个", delta="含深度与修正")
    
    st.markdown("---")

    # 2. 分图校对
    st.subheader("🔍 分图校对与修正")
    file_list = list(st.session_state['data_store'].keys())
    
    # 增加一个左右切换的便捷操作
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected_file = st.selectbox("选择图片进行核对:", file_list, label_visibility="collapsed")
    
    if selected_file:
        data = st.session_state['data_store'][selected_file]
        edits = st.session_state['user_edits'][selected_file]

        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(data['img_rgb'], caption=f"文件名: {selected_file}", use_container_width=True)
        with c2:
            st.markdown(f"### 当前图: **{data['count']}** 个")
            st.markdown("---")
            st.write("🔧 **参数修正**")
            new_depth = st.number_input("堆叠深度", min_value=1, value=edits['depth'], key=f"d_{selected_file}")
            new_manual = st.number_input("人工补差", value=edits['manual'], step=1, key=f"m_{selected_file}")
            
            # 更新数据
            st.session_state['user_edits'][selected_file]['depth'] = new_depth
            st.session_state['user_edits'][selected_file]['manual'] = new_manual
            
            this_total = (data['count'] + new_manual) * new_depth
            st.success(f"小计: {this_total}")

    st.markdown("---")

    # 3. 导出
    st.subheader("📥 导出报表")
    report_data = []
    for name, result in st.session_state['data_store'].items():
        e = st.session_state['user_edits'][name]
        final = (result['count'] + e['manual']) * e['depth']
        report_data.append({
            "文件名": name,
            "AI识别数": result['count'],
            "人工补差": e['manual'],
            "堆叠深度": e['depth'],
            "该图总库存": final,
            "时间": datetime.now().strftime("%H:%M:%S")
        })
    
    df = pd.DataFrame(report_data)
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📊 下载总库存清单 (Excel/CSV)",
            csv,
            f"Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            type="primary"
        )
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
