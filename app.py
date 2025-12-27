#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓库货物检测系统 V29 (Final Diamond Release) - Patched
本次补丁：
1. [Compat] PIL 版本兼容：DecompressionBombError 可能不存在 -> getattr 兜底
2. [Security+] CSV 注入防护升级：剥离 BOM + 多种零宽字符 + 全部空白(\s含\r\n\t)，更难绕过
其余保持 V29 逻辑不变
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import os
import tempfile
import shutil
import pandas as pd
from datetime import datetime
import time
import gc
import uuid
import re
import logging
import zlib
import math
from PIL import Image
import torch
import warnings

# ==================== 0. 全局常量配置 (SSOT) ====================
CONF_MODEL_NAME = 'yolov8l-world.pt'
CONF_SLICE_SIZE = 640
CONF_SLICE_OVERLAP_RATIO = 0.2
CONF_SLICE_NMS_IOU = 0.5
CONF_AGNOSTIC_NMS = True
CONF_MIN_PIXEL_AREA = 300
CONF_MAX_SLICES = 2000
CONF_GC_FREQUENCY = 5
CONF_PIL_MAX_PIXELS = 100_000_000

# 应用 PIL 全局限制
Image.MAX_IMAGE_PIXELS = CONF_PIL_MAX_PIXELS

# [Compat] PIL 版本差异兜底：有些版本没有 DecompressionBombError
PIL_BOMB_WARNING = getattr(Image, "DecompressionBombWarning", Warning)
PIL_BOMB_ERROR = getattr(Image, "DecompressionBombError", PIL_BOMB_WARNING)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 页面配置
st.set_page_config(
    page_title="AI 智能盘点 V29 Diamond",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 1. 会话、缓存与工具函数 ====================

BASE_CACHE_DIR = "processed_cache"

def cleanup_old_sessions(max_age_seconds=86400):
    """清理旧 Session 目录，绝对跳过当前 Session"""
    if not os.path.exists(BASE_CACHE_DIR):
        try:
            os.makedirs(BASE_CACHE_DIR, exist_ok=True)
        except Exception as e:
            logging.warning(f"Failed to create cache dir {BASE_CACHE_DIR}: {e}")
        return

    now = time.time()
    current_session = st.session_state.get('session_id')

    for item in os.listdir(BASE_CACHE_DIR):
        if item == current_session:
            continue

        item_path = os.path.join(BASE_CACHE_DIR, item)
        if os.path.isdir(item_path):
            try:
                if now - os.path.getmtime(item_path) > max_age_seconds:
                    shutil.rmtree(item_path)
                    logging.info(f"[Auto-Clean] Deleted old session: {item}")
            except Exception as e:
                logging.error(f"Cleanup error for {item}: {e}")

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
    cleanup_old_sessions()

# 当前会话目录
SESSION_CACHE_DIR = os.path.join(BASE_CACHE_DIR, st.session_state['session_id'])
if not os.path.exists(SESSION_CACHE_DIR):
    try:
        os.makedirs(SESSION_CACHE_DIR, exist_ok=True)
    except Exception as e:
        st.error(f"严重错误：无法创建缓存目录，系统停止运行。\n{e}")
        st.stop()

def clear_session_cache():
    """只清理当前会话"""
    if os.path.exists(SESSION_CACHE_DIR):
        try:
            shutil.rmtree(SESSION_CACHE_DIR)
            os.makedirs(SESSION_CACHE_DIR, exist_ok=True)
        except Exception as e:
            logging.warning(f"Failed to clear session cache: {e}")

def sanitize_filename(name):
    name = re.sub(r'[^\w\u4e00-\u9fa5\.-]', '_', name)
    return name

def excel_safe(s):
    """[Security] 防止高级 CSV 注入攻击（更强清洗：BOM/零宽字符/任意空白）"""
    if not isinstance(s, str):
        return s

    # 剥离：BOM、常见零宽字符（ZWSP/ZWNJ/ZWJ/WordJoiner）以及所有空白(\s含\r\n\t)
    s_clean = re.sub(r'^[\ufeff\u200b\u200c\u200d\u2060\s]+', '', s)

    if s_clean.startswith(('=', '+', '-', '@')):
        return "'" + s
    return s

def get_stable_color(cls_name):
    """基于类名 Hash 生成固定颜色 (BGR 格式)"""
    hash_val = zlib.crc32(cls_name.encode('utf-8'))
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = hash_val & 0x0000FF
    return (max(50, b), max(50, g), max(50, r))

# ==================== 2. 核心检测逻辑 ====================

def detect_and_save(image_path, conf, dedup_iou, model, original_filename):
    """
    V29 检测流程: 钻石级防御与鲁棒性
    """
    SLICE_H, SLICE_W = CONF_SLICE_SIZE, CONF_SLICE_SIZE

    # ---------------------------------------------------------
    # [Defense Phase 1] PIL 惰性读取 + 像素策略防御
    # ---------------------------------------------------------
    w, h = 0, 0
    try:
        with warnings.catch_warnings():
            # 将 DecompressionBombWarning 升级为异常（若当前 PIL 版本支持该 Warning）
            warnings.simplefilter("error", PIL_BOMB_WARNING)
            with Image.open(image_path) as pil_img:
                w, h = pil_img.size
                pil_img.verify()
    except (PIL_BOMB_WARNING, PIL_BOMB_ERROR):
        dim_str = f"{w}x{h}" if (w and h) else "未知尺寸"
        pixel_str = f"{w*h}" if (w and h) else "Unknown"
        return {'error': f'图片像素量异常 ({dim_str}, px={pixel_str})，超过阈值 {CONF_PIL_MAX_PIXELS}，已拒绝处理。'}
    except Exception as e:
        return {'error': f'图片文件可能损坏或格式不支持: {e}'}

    # 切片参数计算
    overlap_h = int(SLICE_H * CONF_SLICE_OVERLAP_RATIO)
    overlap_w = int(SLICE_W * CONF_SLICE_OVERLAP_RATIO)

    step_h = max(1, SLICE_H - overlap_h)
    step_w = max(1, SLICE_W - overlap_w)

    # [Circuit Breaker 1] PIL 预估熔断
    est_cols = max(1, math.ceil(w / step_w))
    est_rows = max(1, math.ceil(h / step_h))
    total_slices_est = est_cols * est_rows

    if total_slices_est > CONF_MAX_SLICES:
        logging.warning(f"Slice overflow (PIL): {total_slices_est} slices needed for {original_filename}")
        return {'error': f"图片尺寸过大 ({w}x{h})，需生成 {total_slices_est} 个切片 (上限 {CONF_MAX_SLICES})，已熔断停止。"}

    # ---------------------------------------------------------
    # [Action Phase] 真正加载图片
    # ---------------------------------------------------------
    original_img = cv2.imread(image_path)
    if original_img is None:
        return {'error': 'OpenCV 无法解码该图片，请尝试转换为标准 JPG/PNG。'}

    # [Reliability] 双重确认：OpenCV 真实尺寸
    h, w = original_img.shape[:2]

    # [Circuit Breaker 2] 二次熔断
    est_cols2 = max(1, math.ceil(w / step_w))
    est_rows2 = max(1, math.ceil(h / step_h))
    total_slices_est2 = est_cols2 * est_rows2

    if total_slices_est2 > CONF_MAX_SLICES:
        logging.warning(f"Slice overflow (OpenCV): {total_slices_est2} slices needed for {original_filename} ({w}x{h})")
        del original_img
        gc.collect()
        return {'error': f"图片解码后尺寸异常 ({w}x{h})，需生成 {total_slices_est2} 个切片 (上限 {CONF_MAX_SLICES})，已熔断停止。"}

    slices = []
    y_start = 0
    while y_start < h:
        y_end = min(y_start + SLICE_H, h)
        x_start = 0
        while x_start < w:
            x_end = min(x_start + SLICE_W, w)
            x1, y1 = max(0, x_start - overlap_w if x_start > 0 else 0), max(0, y_start - overlap_h if y_start > 0 else 0)
            x2, y2 = min(w, x_end + overlap_w if x_end < w else w), min(h, y_end + overlap_h if y_end < h else h)
            slices.append((x1, y1, x2, y2, x_start, y_start))
            x_start += step_w
        y_start += step_h

    all_boxes = []

    for x1, y1, x2, y2, _, _ in slices:
        slice_img = original_img[y1:y2, x1:x2]
        slice_img_rgb = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)

        results = model.predict(
            source=slice_img_rgb,
            conf=conf,
            iou=CONF_SLICE_NMS_IOU,
            agnostic_nms=CONF_AGNOSTIC_NMS,
            verbose=False
        )

        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            global_x1 = xyxy[0] + x1
            global_y1 = xyxy[1] + y1
            global_x2 = xyxy[2] + x1
            global_y2 = xyxy[3] + y1
            area = (global_x2 - global_x1) * (global_y2 - global_y1)

            if area >= CONF_MIN_PIXEL_AREA:
                all_boxes.append({
                    'cls': int(box.cls[0]),
                    'conf': float(box.conf[0]),
                    'xyxy': [global_x1, global_y1, global_x2, global_y2],
                    'area': area
                })

        del results
        del slice_img
        del slice_img_rgb

    # 全局去重
    all_boxes.sort(key=lambda x: x['conf'], reverse=True)
    unique_boxes = []

    for box in all_boxes:
        is_duplicate = False
        b_x1, b_y1, b_x2, b_y2 = box['xyxy']

        for xb in unique_boxes:
            xb_x1, xb_y1, xb_x2, xb_y2 = xb['xyxy']

            # 坐标快速剪枝
            if (b_x2 < xb_x1) or (b_x1 > xb_x2) or (b_y2 < xb_y1) or (b_y1 > xb_y2):
                continue

            if compute_iou(box['xyxy'], xb['xyxy']) > dedup_iou:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_boxes.append(box)

    final_boxes = unique_boxes

    # 绘图
    annotated_img = original_img.copy()
    class_counts = {}
    for box in final_boxes:
        cls_name = model.names[box['cls']]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        x1, y1, x2, y2 = map(int, box['xyxy'])

        color = get_stable_color(cls_name)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

    # 保存
    clean_name = sanitize_filename(original_filename)
    unique_suffix = uuid.uuid4().hex[:8]
    save_name = f"{int(time.time())}_{unique_suffix}_{clean_name}"
    save_path = os.path.join(SESSION_CACHE_DIR, save_name)

    success = cv2.imwrite(save_path, annotated_img)

    # 释放内存
    del original_img
    del annotated_img
    del all_boxes

    if not success:
        logging.error(f"Failed to write image to {save_path}")
        gc.collect()
        return {'error': '结果图片写入磁盘失败 (磁盘满或权限不足)'}

    return {
        'count': len(final_boxes),
        'img_path': save_path,
        'counts_detail': class_counts,
        'original_name': original_filename
    }

def compute_iou(box1, box2):
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter + 1e-6)

# ==================== 3. 前端 UI ====================

def main():
    if 'data_store' not in st.session_state: st.session_state['data_store'] = {}
    if 'user_edits' not in st.session_state: st.session_state['user_edits'] = {}
    if 'run_config' not in st.session_state: st.session_state['run_config'] = {}
    if 'run_errors' not in st.session_state: st.session_state['run_errors'] = []
    if 'run_time_str' not in st.session_state: st.session_state['run_time_str'] = ""

    with st.sidebar:
        st.title("🏭 智能盘点控制台")
        st.caption(f"V29: Diamond | ID: {st.session_state['session_id'][:4]}")
        st.markdown("---")

        st.subheader("1. 识别目标")
        default_prompts = "textile bale, woven sack, wrapped package, stacked white sacks, wall of bales"
        raw_prompt = st.text_area("输入目标 (逗号分隔)", value=default_prompts, height=100)
        clean_prompt = raw_prompt.replace('，', ',')
        TARGET_CLASSES = [x.strip() for x in clean_prompt.split(',') if x.strip()]

        st.subheader("2. 核心参数")
        conf = st.slider("置信度 (Conf)", 0.01, 0.5, 0.05)
        dedup_iou = st.slider("全局去重 (Dedup IoU)", 0.05, 0.8, 0.35)

        st.markdown("---")

        uploaded_files = st.file_uploader(
            "3. 上传图片",
            type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
            accept_multiple_files=True
        )
        st.markdown("---")

        start_btn = st.button("🚀 开始检测", type="primary", use_container_width=True)

        if start_btn:
            if not uploaded_files:
                st.warning("⚠️ 请先上传图片")
            elif not TARGET_CLASSES:
                st.warning("⚠️ 请至少输入一个目标")
            else:
                st.session_state['data_store'] = {}
                st.session_state['user_edits'] = {}
                st.session_state['run_errors'] = []
                st.session_state['run_time_str'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.session_state['run_config'] = {
                    'targets': str(TARGET_CLASSES),
                    'conf': conf,
                    'dedup_iou': dedup_iou,
                    'model_name': CONF_MODEL_NAME,
                    'slice_size': CONF_SLICE_SIZE,
                    'slice_overlap': CONF_SLICE_OVERLAP_RATIO,
                    'slice_nms_iou': CONF_SLICE_NMS_IOU,
                    'agnostic_nms': CONF_AGNOSTIC_NMS,
                    'min_pixel_area': CONF_MIN_PIXEL_AREA,
                    'max_slices': CONF_MAX_SLICES,
                    'max_pil_pixels': CONF_PIL_MAX_PIXELS
                }

                clear_session_cache()
                gc.collect()

                try:
                    model = YOLO(CONF_MODEL_NAME)
                    model.set_classes(TARGET_CLASSES)
                except Exception as e:
                    st.error(f"模型加载失败: {e}")
                    st.stop()

                st.info(f"正在处理 {len(uploaded_files)} 张图片...")
                progress_bar = st.progress(0)

                for idx, file_obj in enumerate(uploaded_files):
                    progress_bar.progress(idx / len(uploaded_files), text=f"分析中: {file_obj.name}")

                    _, ext = os.path.splitext(file_obj.name)
                    ext = ext.lower() if ext else ".jpg"
                    if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                        ext = ".jpg"

                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(file_obj.read())
                        tmp_path = tmp.name

                    try:
                        result = detect_and_save(tmp_path, conf, dedup_iou, model, file_obj.name)

                        if isinstance(result, dict) and result.get('error'):
                            error_msg = f"{file_obj.name}: {result['error']}"
                            st.session_state['run_errors'].append(error_msg)
                            logging.error(error_msg)

                        elif isinstance(result, dict) and 'count' in result:
                            safe_key_name = sanitize_filename(file_obj.name)
                            unique_key = f"{idx+1}_{safe_key_name}"

                            st.session_state['data_store'][unique_key] = result
                            st.session_state['user_edits'][unique_key] = {'depth': 1, 'manual': 0}

                        else:
                            logging.warning(f"Unexpected result format for {file_obj.name}")

                    except Exception as e:
                        crash_msg = f"程序崩溃 ({file_obj.name}): {str(e)}"
                        st.session_state['run_errors'].append(crash_msg)
                        logging.error(crash_msg)

                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                        except Exception:
                            pass

                    try:
                        os.remove(tmp_path)
                    except Exception as e:
                        logging.warning(f"Failed to remove temp file {tmp_path}: {e}")

                    # [Perf/GPU] 批次 GC
                    if (idx + 1) % CONF_GC_FREQUENCY == 0:
                        gc.collect()
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                        except Exception:
                            pass

                gc.collect()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass

                progress_bar.progress(1.0, text="✅ 完成！")
                time.sleep(0.5)
                st.rerun()

    # --- 主界面 ---
    st.title("🏭 仓库盘点总览")

    if st.session_state['run_errors']:
        run_ts = st.session_state.get('run_time_str', 'Unknown')
        with st.expander(f"⚠️ 检测警告/错误 (本次运行: {run_ts})", expanded=True):
            for err in st.session_state['run_errors']:
                st.error(err)

    if not st.session_state['data_store']:
        st.info("👈 系统就绪。V29 最终钻石版已加载。")
        st.stop()

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

    col_sel1, _ = st.columns([3, 1])
    with col_sel1:
        selected_key = st.selectbox("选择图片:", file_list, label_visibility="collapsed")

    if selected_key:
        data = st.session_state['data_store'][selected_key]
        edits = st.session_state['user_edits'][selected_key]

        c1, c2 = st.columns([2, 1])
        with c1:
            if os.path.exists(data['img_path']):
                display_name = data.get('original_name', selected_key)
                st.image(data['img_path'], caption=f"文件: {display_name} (ID: {selected_key})", use_container_width=True)
            else:
                st.warning("图片缓存已清理")

        with c2:
            st.markdown(f"### 计数: **{data['count']}**")
            with st.expander("分类详情"):
                for k, v in data['counts_detail'].items():
                    st.write(f"- {k}: {v}")

            st.markdown("---")
            new_depth = st.number_input("堆叠深度", min_value=1, value=edits['depth'], key=f"d_{selected_key}")
            new_manual = st.number_input("人工补差", value=edits['manual'], step=1, key=f"m_{selected_key}")

            st.session_state['user_edits'][selected_key]['depth'] = new_depth
            st.session_state['user_edits'][selected_key]['manual'] = new_manual

            this_total = (data['count'] + new_manual) * new_depth
            st.success(f"小计: {this_total}")

    st.markdown("---")

    # 导出
    st.subheader("📥 导出报表")
    report_data = []
    run_cfg = st.session_state.get('run_config', {})

    for key, result in st.session_state['data_store'].items():
        e = st.session_state['user_edits'][key]
        final = (result['count'] + e['manual']) * e['depth']

        row = {
            "文件Key": key,
            "原始文件名": result.get('original_name', key),
            "检测目标": run_cfg.get('targets', ''),
            "AI识别数": result['count'],
            "人工补差": e['manual'],
            "堆叠深度": e['depth'],
            "该图总库存": final,
            "置信度": run_cfg.get('conf', ''),
            "全局去重IoU": run_cfg.get('dedup_iou', ''),
            "切片尺寸": run_cfg.get('slice_size', ''),
            "切片重叠": run_cfg.get('slice_overlap', ''),
            "切片NMS_IoU": run_cfg.get('slice_nms_iou', ''),
            "最小像素面积": run_cfg.get('min_pixel_area', ''),
            "AgnosticNMS": run_cfg.get('agnostic_nms', ''),
            "切片熔断上限": run_cfg.get('max_slices', ''),
            "像素安全阈值": run_cfg.get('max_pil_pixels', ''),
            "模型版本": run_cfg.get('model_name', ''),
            "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        report_data.append(row)

    df = pd.DataFrame(report_data)
    if not df.empty:
        # [Security] 全量清洗所有文本列（兼容 object + string dtype）
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].apply(excel_safe)

        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            "📊 下载完整报表 (CSV/Excel兼容)",
            csv,
            f"Inventory_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()
