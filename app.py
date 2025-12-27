#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓库货物检测系统 V13 (通用增强版)
核心升级：
1. [GPT建议] 支持自定义 Prompt (不再局限于纺织袋，想数什么填什么)
2. [GPT建议] 改用绝对像素过滤 (防止误删远处小包)
3. 保留 V12 的硬盘缓存与内存优化
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
import gc

# 页面配置
st.set_page_config(
    page_title="AI 通用盘点系统 V13",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 定义缓存目录
CACHE_DIR = "processed_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# ==================== 后端逻辑 ====================

@st.cache_resource(show_spinner=False)
def load_model():
    """加载模型 (YOLO-World)"""
    try:
        # V13修改: 这里只加载模型权重，不绑定具体类别，类别在检测时动态设定
        model = YOLO('yolov8l-world.pt') 
        return model
    except Exception as e:
        st.error(f"模型加载崩溃: {str(e)}")
        return None

def clear_cache():
    """清理缓存"""
    if os.path.exists(CACHE_DIR):
        try:
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR)
        except Exception:
            pass

def detect_and_save(image_path, conf, iou, model, original_filename, target_classes):
    """
    V13 检测逻辑: 接收自定义类别列表 + 绝对像素过滤
    """
    # 动态设置当前要找的目标
    model.set_classes(target_classes)

    SLICE_HEIGHT, SLICE_WIDTH = 640, 640
    SLICE_OVERLAP = 0.2
    AGNOSTIC_NMS = True
    DEDUP_THRESHOLD = iou
    
    # [GPT建议] 使用绝对像素面积过滤，而不是百分比
    # 300像素大约是 17x17 的小方块，小于这个的视为噪点
    MIN_PIXEL_AREA = 300 

    # 读取图片
    original_img = cv2.imread(image_path)
    if original_img is None: return None
    h, w = original_img.shape[:2]

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
            
            # 推理
            results = model.predict(source=temp_path, conf=conf, iou=iou, agnostic_nms=AGNOSTIC_NMS, verbose=False)
            del slice_img
            
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                xyxy[0] += x1; xyxy[1] += y1; xyxy[2] += x1; xyxy[3] += y1
                all_boxes.append({
                    'cls': int(box.cls[0]), 'conf': float(box.conf[0]),
                    'xyxy': xyxy, 'area': (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])
                })
    finally:
        shutil.rmtree(temp_dir)

    # NMS 去重
    all_boxes.sort(key=lambda x: x['conf'], reverse=True)
    unique_boxes = []
    for box in all_boxes:
        if not any(compute_iou(box['xyxy'], xb['xyxy']) > DEDUP_THRESHOLD for xb in unique_boxes):
            unique_boxes.append(box)

    # [GPT建议] 绝对面积过滤
    final_boxes = [b for b in unique_boxes if b['area'] >= MIN_PIXEL_AREA]

    # 绘图
    annotated_img = original_img.copy()
    random.seed(42)
    # 动态生成颜色
    colors = {i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in range(len(target_classes))}
    
    class_counts = {}
    for box in final_boxes:
        # 此时 model.names 已经根据 set_classes 更新
        cls_name = model.names[box['cls']]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        x1, y1, x2, y2 = map(int, box['xyxy'])
        
        color = colors.get(box['cls'], (0, 255, 0))
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

    # 保存缓存
    save_name = f"{int(time.time())}_{original_filename}"
    save_path = os.path.join(CACHE_DIR, save_name)
    cv2.imwrite(save_path, annotated_img)

    # 内存清理
    del original_img
    del annotated_img
    del all_boxes
    gc.collect()

    return {
        'count': len(final_boxes),
        'img_path': save_path,
        'counts_detail': class_counts
    }

def compute_iou(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if ix1 >= ix2 or iy1 >= iy2: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter + 1e-6)

# ==================== 前端 UI ====================

def main():
    if 'data_store' not in st.session_state: st.session_state['data_store'] = {}
    if 'user_edits' not in st.session_state: st.session_state['user_edits'] = {}

    with st.spinner("🚀 AI 引擎加载中..."):
        model = load_model()
    if not model: st.stop()

    with st.sidebar:
        st.title("🏭 智能盘点控制台")
        st.caption("V13: 通用增强版")
        st.markdown("---")
        
        st.subheader("1. 识别目标设置")
        # [GPT建议] 开放 Prompt 接口
        default_prompts = "textile bale, woven sack, wrapped package, stacked white sacks, wall of bales"
        user_prompt = st.text_area(
            "输入你想数的物体 (英文逗号分隔)", 
            value=default_prompts,
            height=100,
            help="YOLO-World 是通用的，你可以输入 box, tire, bottle, helmet 等任何物体"
        )
        # 清洗用户输入
        TARGET_CLASSES = [x.strip() for x in user_prompt.split(',') if x.strip()]
        
        st.subheader("2. 灵敏度参数")
        # 默认值微调
        conf = st.slider("置信度 (Conf)", 0.01, 0.5, 0.05, help="默认0.05，越小发现越多")
        iou = st.slider("去重阈值 (IoU)", 0.05, 0.8, 0.35, help="默认0.35，防止过度合并")
        
        st.markdown("---")
        
        uploaded_files = st.file_uploader(
            "3. 上传图片 (建议分批处理)", 
            type=['jpg', 'png'], 
            accept_multiple_files=True
        )

        st.markdown("---")
        start_btn = st.button("🚀 开始检测", type="primary", use_container_width=True)
        
        if start_btn:
            if not uploaded_files:
                st.warning("⚠️ 请先上传图片！")
            elif not TARGET_CLASSES:
                st.warning("⚠️ 请至少输入一个识别目标！")
            else:
                # 清理
                st.session_state['data_store'] = {}
                st.session_state['user_edits'] = {}
                clear_cache()
                gc.collect()
                
                st.info(f"正在寻找: {TARGET_CLASSES}")
                progress_bar = st.progress(0)
                
                for idx, file_obj in enumerate(uploaded_files):
                    progress_bar.progress((idx) / len(uploaded_files), text=f"正在分析: {file_obj.name}...")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(file_obj.read())
                        tmp_path = tmp.name
                    
                    try:
                        # 传入用户自定义的 CLASSES
                        result = detect_and_save(tmp_path, conf, iou, model, file_obj.name, TARGET_CLASSES)
                        if result:
                            st.session_state['data_store'][file_obj.name] = result
                            st.session_state['user_edits'][file_obj.name] = {'depth': 1, 'manual': 0}
                    except Exception as e:
                        st.error(f"出错: {e}")
                    
                    os.remove(tmp_path)
                    gc.collect()
                
                progress_bar.progress(1.0, text="✅ 完成！")
                time.sleep(0.5)
                st.rerun()

    # --- 主界面 ---
    st.title("🏭 仓库盘点总览")

    if not st.session_state['data_store']:
        st.info(f"👈 准备就绪。当前识别目标: {len(TARGET_CLASSES)} 类。请上传图片并开始。")
        with st.expander("查看当前识别列表"):
            st.write(TARGET_CLASSES)
        st.stop()

    # Dashboard
    total_ai_count = sum([d['count'] for d in st.session_state['data_store'].values()])
    grand_total = 0
    for name, result in st.session_state['data_store'].items():
        edits = st.session_state['user_edits'].get(name, {'depth': 1, 'manual': 0})
        grand_total += (result['count'] + edits['manual']) * edits['depth']

    col1, col2, col3 = st.columns(3)
    col1.metric("📸 图片数量", f"{len(st.session_state['data_store'])} 张")
    col2.metric("📦 AI 计数", f"{total_ai_count} 个")
    col3.metric("💰 库存总计", f"{grand_total} 个")
    
    st.markdown("---")

    # 分图校对
    st.subheader("🔍 校对与修正")
    file_list = list(st.session_state['data_store'].keys())
    
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected_file = st.selectbox("选择图片:", file_list, label_visibility="collapsed")
    
    if selected_file:
        data = st.session_state['data_store'][selected_file]
        edits = st.session_state['user_edits'][selected_file]

        c1, c2 = st.columns([2, 1])
        with c1:
            if os.path.exists(data['img_path']):
                st.image(data['img_path'], caption=f"文件: {selected_file}", use_container_width=True)
            else:
                st.error("图片缓存失效，请重新检测")

        with c2:
            st.markdown(f"### 计数: **{data['count']}**")
            # 显示分类详情
            with st.expander("分类详情"):
                for k, v in data['counts_detail'].items():
                    st.write(f"- {k}: {v}")
            
            st.markdown("---")
            new_depth = st.number_input("堆叠深度", min_value=1, value=edits['depth'], key=f"d_{selected_file}")
            new_manual = st.number_input("人工补差", value=edits['manual'], step=1, key=f"m_{selected_file}")
            
            st.session_state['user_edits'][selected_file]['depth'] = new_depth
            st.session_state['user_edits'][selected_file]['manual'] = new_manual
            
            this_total = (data['count'] + new_manual) * new_depth
            st.success(f"小计: {this_total}")

    st.markdown("---")

    # 导出
    st.subheader("📥 导出报表")
    report_data = []
    for name, result in st.session_state['data_store'].items():
        e = st.session_state['user_edits'][name]
        final = (result['count'] + e['manual']) * e['depth']
        report_data.append({
            "文件名": name,
            "检测目标": str(TARGET_CLASSES), # 记录这批查的是什么
            "AI识别数": result['count'],
            "人工补差": e['manual'],
            "堆叠深度": e['depth'],
            "该图总库存": final,
            "时间": datetime.now().strftime("%H:%M:%S")
        })
    
    df = pd.DataFrame(report_data)
    if not df.empty:
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📊 下载完整报表", csv, f"Report_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

if __name__ == "__main__":
    main()
