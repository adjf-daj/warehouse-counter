#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓库货物检测系统 V10 (批量盘点版)
支持多图上传、批量处理、单图深度修正、总库存汇总导出
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
    page_title="AI 批量盘点系统 V10",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 后端逻辑 (保持 V9 核心不变) ====================

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
    # 核心参数 (V9标准)
    SLICE_HEIGHT, SLICE_WIDTH = 640, 640
    SLICE_OVERLAP = 0.2
    AGNOSTIC_NMS = True
    MIN_AREA_RATIO = 0.001
    DEDUP_THRESHOLD = iou # 动态关联

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

# ==================== 前端 UI 逻辑 (批量版) ====================

def main():
    # 初始化
    if 'data_store' not in st.session_state: st.session_state['data_store'] = {} # 存储检测结果
    if 'user_edits' not in st.session_state: st.session_state['user_edits'] = {} # 存储人工修正(深度/误差)

    with st.spinner("正在启动 AI 批量处理引擎..."):
        model = load_model()
    if not model: st.stop()

    # --- 侧边栏：全局控制 ---
    with st.sidebar:
        st.title("🏭 批量盘点控制台")
        st.markdown("---")
        
        # 1. 参数设置
        st.subheader("1. AI 参数")
        conf = st.slider("置信度", 0.01, 0.5, 0.15)
        iou = st.slider("去重阈值", 0.05, 0.8, 0.2)
        
        st.markdown("---")
        
        # 2. 上传区域 (支持多选)
        st.subheader("2. 批量上传")
        uploaded_files = st.file_uploader(
            "按住 Ctrl 可多选图片", 
            type=['jpg', 'png'], 
            accept_multiple_files=True
        )

        # 触发批量处理
        if uploaded_files:
            # 检查是否有新文件需要处理
            new_files = [f for f in uploaded_files if f.name not in st.session_state['data_store']]
            
            if new_files:
                st.info(f"📸 发现 {len(new_files)} 张新图片，开始处理...")
                progress_bar = st.progress(0)
                
                for idx, file_obj in enumerate(new_files):
                    # 保存临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(file_obj.read())
                        tmp_path = tmp.name
                    
                    # AI 检测
                    result = detect_image(tmp_path, conf, iou, model)
                    
                    if result:
                        st.session_state['data_store'][file_obj.name] = result
                        # 初始化这张图的修正数据 (默认深度1，修正0)
                        if file_obj.name not in st.session_state['user_edits']:
                            st.session_state['user_edits'][file_obj.name] = {'depth': 1, 'manual': 0}
                    
                    os.remove(tmp_path)
                    progress_bar.progress((idx + 1) / len(new_files))
                
                progress_bar.empty()
                st.success("✅ 批量处理完成！")

    # --- 主界面 ---
    st.title("🏭 仓库盘点总览")

    # 如果没有数据
    if not st.session_state['data_store']:
        st.info("👈 请在左侧上传一组仓库照片开始盘点。")
        st.stop()

    # 1. 顶部总计卡片 (Dashboard)
    total_ai_count = sum([d['count'] for d in st.session_state['data_store'].values()])
    
    # 计算修正后的总库存
    grand_total = 0
    for name, result in st.session_state['data_store'].items():
        edits = st.session_state['user_edits'].get(name, {'depth': 1, 'manual': 0})
        grand_total += (result['count'] + edits['manual']) * edits['depth']

    col1, col2, col3 = st.columns(3)
    col1.metric("📸 已拍照片数", f"{len(st.session_state['data_store'])} 张")
    col2.metric("📦 视觉检测总和", f"{total_ai_count} 个")
    col3.metric("💰 最终库存总计", f"{grand_total} 个", delta="含深度与修正")
    
    st.markdown("---")

    # 2. 分图校对界面
    st.subheader("🔍 分图校对与修正")
    
    # 选择要查看的图片
    file_list = list(st.session_state['data_store'].keys())
    selected_file = st.selectbox("选择一张图片进行核对:", file_list)

    if selected_file:
        data = st.session_state['data_store'][selected_file]
        edits = st.session_state['user_edits'][selected_file]

        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.image(data['img_rgb'], caption=f"文件名: {selected_file}", use_container_width=True)

        with c2:
            st.write(f"**当前图 AI 计数:** {data['count']}")
            
            # --- 每一张图的独立修正区 ---
            st.markdown("#### 🔧 人工修正")
            
            new_depth = st.number_input(
                "堆叠深度 (层/排)", 
                min_value=1, 
                value=edits['depth'], 
                key=f"depth_{selected_file}"
            )
            
            new_manual = st.number_input(
                "补差价 (AI漏了填正数，多了填负数)", 
                value=edits['manual'],
                step=1,
                key=f"man_{selected_file}"
            )
            
            # 实时更新 Session State
            st.session_state['user_edits'][selected_file]['depth'] = new_depth
            st.session_state['user_edits'][selected_file]['manual'] = new_manual
            
            # 单图计算结果
            this_total = (data['count'] + new_manual) * new_depth
            
            st.success(f"当前图小计: {this_total} 个")
            st.caption(f"公式: ({data['count']} + {new_manual}) × {new_depth}")

    st.markdown("---")

    # 3. 全局导出
    st.subheader("📥 导出报表")
    
    # 准备 Excel 数据
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
            "检测时间": datetime.now().strftime("%H:%M:%S")
        })
    
    # 增加一行总计
    df = pd.DataFrame(report_data)
    if not df.empty:
        # 导出按钮
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📊 下载总库存清单 (Excel/CSV)",
            csv,
            f"Inventory_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            type="primary"
        )
        
        # 简单预览表格
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
