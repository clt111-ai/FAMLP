#!/usr/bin/env python3
"""
评估部位识别器在原始图像和多种扰动下的鲁棒性（部位独立统计）。
支持选择测试数据集和/或训练数据集。
扰动类型: CS, CC, BW, GNC, GB, JPEG，每种 5 个级别，共 30 种扰动 + 原始图像 = 31 次测试。
统计方式：对每个部位单独计算 AUC，不计算 Macro AUC。
结果保存到日志文件，每个部位一个表格。
"""

import os
import sys
import logging
from datetime import datetime
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# 导入项目模块
from config import Config
from model import ForensicModel

# ==================== 配置参数 ====================
PART_DETECTOR_WEIGHT = "./part_detector.pth"   # 部位识别器权重路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 4
SAMPLE_RATIO = 0.01                 # 测试数据集采样比例（0~1）
OUTPUT_LOG = "part_detector_robustness_per_part.log"

# 要评估的测试数据集名称（从 config.test_datasets 中选择）
SELECTED_DATASETS = ["CDFv2", "DFDC", "DFDCP", "DFD", "CDFv1"]

# 是否包含训练数据集进行评估
INCLUDE_TRAIN_DATA = True           # 若为 True，则将训练数据作为一个名为 "Train" 的数据集加入测试
# =================================================

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_LOG, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 扰动函数定义 ====================
def get_distortion_parameter(type, level):
    """返回指定扰动类型和级别（1~5）的参数值"""
    param_dict = {
        'CS': [0.4, 0.3, 0.2, 0.1, 0.0],      # 饱和度系数，越小越淡
        'CC': [0.85, 0.725, 0.6, 0.475, 0.35], # 对比度系数，越小对比度越低
        'BW': [16, 32, 48, 64, 80],            # 块大小，越大块越粗糙
        'GNC': [0.001, 0.002, 0.005, 0.01, 0.05], # 噪声强度（标准差）
        'GB': [7, 9, 13, 17, 21],              # 高斯模糊半径，越大越模糊
        'JPEG': [2, 3, 4, 5, 6]                # 压缩质量映射因子（越大质量越差）
    }
    return param_dict[type][level - 1]

def apply_color_saturation(img, factor):
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def apply_color_contrast(img, factor):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def apply_block_wise(img, block_size):
    img_np = np.array(img)
    h, w, c = img_np.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img_np[i:min(i+block_size, h), j:min(j+block_size, w)]
            if block.size == 0:
                continue
            mean_val = block.mean(axis=(0,1), keepdims=True)
            img_np[i:min(i+block_size, h), j:min(j+block_size, w)] = mean_val
    return Image.fromarray(img_np.astype('uint8'))

def apply_gaussian_noise_color(img, sigma):
    img_np = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma * 255, img_np.shape)
    noisy = img_np + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def apply_gaussian_blur(img, radius):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_jpeg_compression(img, quality_factor):
    quality = max(10, 100 - 10 * quality_factor)
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

def get_distortion_function(type):
    func_dict = {
        'CS': apply_color_saturation,
        'CC': apply_color_contrast,
        'BW': apply_block_wise,
        'GNC': apply_gaussian_noise_color,
        'GB': apply_gaussian_blur,
        'JPEG': apply_jpeg_compression
    }
    return func_dict[type]

# ==================== 辅助函数 ====================
def get_npy_path(img_path):
    return img_path.replace('frames', 'landmarks').replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')

def generate_patch_labels_from_landmarks(landmarks, img_size=224, patch_size=14):
    part_indices = {
        "contour": list(range(0, 17)) + list(range(68, 81)),
        "eyebrows": list(range(17, 27)),
        "eyes": list(range(36, 48)),
        "nose": list(range(27, 36)),
        "mouth": list(range(48, 68)),
    }
    part_order = ["contour", "eyebrows", "eyes", "nose", "mouth"]
    grid_size = img_size // patch_size
    num_patches = grid_size ** 2
    labels = np.zeros((num_patches, 5), dtype=np.float32)
    if landmarks is None or landmarks.shape[0] == 0:
        return torch.from_numpy(labels)
    for part_idx, part_name in enumerate(part_order):
        for idx in part_indices[part_name]:
            if idx >= landmarks.shape[0]:
                continue
            x, y = landmarks[idx]
            if x < 0 or x >= img_size or y < 0 or y >= img_size:
                continue
            px = int(x // patch_size)
            py = int(y // patch_size)
            if 0 <= px < grid_size and 0 <= py < grid_size:
                patch_idx = py * grid_size + px
                labels[patch_idx, part_idx] = 1.0
    return torch.from_numpy(labels)

def load_image_and_labels(img_path, npy_path, target_size=(224,224)):
    pil_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = pil_img.size
    try:
        data = np.load(npy_path)
        if data.shape[-1] == 2 and len(data.shape) == 2:
            landmarks = data.astype(np.float32)
        else:
            landmarks = np.zeros((0,2), dtype=np.float32)
    except:
        landmarks = np.zeros((0,2), dtype=np.float32)
    pil_img_resized = pil_img.resize(target_size, Image.BILINEAR)
    if landmarks.shape[0] > 0:
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y
    patch_labels = generate_patch_labels_from_landmarks(landmarks, img_size=target_size[0], patch_size=14)
    return pil_img_resized, patch_labels

def collect_images_from_dirs(directories, sample_ratio=1.0, recursive=True):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    all_images = []
    for directory in directories:
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist, skipping")
            continue
        if recursive:
            files = list(Path(directory).rglob("*"))
        else:
            files = list(Path(directory).iterdir())
        valid = [str(f) for f in files if f.is_file() and f.suffix.lower() in image_extensions]
        all_images.extend(valid)
    if sample_ratio < 1.0:
        sample_size = max(1, int(len(all_images) * sample_ratio))
        all_images = random.sample(all_images, sample_size)
    return all_images

def collect_dataset_images(real_dirs, fake_dirs, sample_ratio=1.0, recursive=True, dataset_name="unknown"):
    all_images = []
    all_images.extend(collect_images_from_dirs(real_dirs, sample_ratio, recursive))
    all_images.extend(collect_images_from_dirs(fake_dirs, sample_ratio, recursive))
    valid_imgs = []
    npy_paths = []
    for img in all_images:
        npy = get_npy_path(img)
        if os.path.exists(npy):
            valid_imgs.append(img)
            npy_paths.append(npy)
    logger.info(f"Dataset {dataset_name}: total images {len(all_images)}, valid with landmarks {len(valid_imgs)}")
    return valid_imgs, npy_paths

# ==================== 核心评估函数（返回每个部位的 AUC 列表） ====================
def evaluate_on_condition(model, image_list, npy_list, condition_func=None, condition_param=None, device=DEVICE):
    """
    对一组图像应用扰动，返回每个部位的 AUC 值（长度为5的列表，顺序为 part_names）
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    all_probs = []   # 存储所有 patch 的预测概率 [N_patches, 5]
    all_labels = []  # 存储所有 patch 的真实标签 [N_patches, 5]
    model.eval()
    
    for img_path, npy_path in tqdm(zip(image_list, npy_list), total=len(image_list), desc="Processing", leave=False):
        pil_img, patch_labels = load_image_and_labels(img_path, npy_path, target_size=(224,224))
        if condition_func is not None:
            pil_img = condition_func(pil_img, condition_param)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            patch_tokens = model.extract_patch_tokens(img_tensor)
            logits = model.part_detector(patch_tokens)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1, 5)
        all_probs.append(probs)
        all_labels.append(patch_labels.numpy().reshape(-1, 5))
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    part_names = ["contour", "eyebrows", "eyes", "nose", "mouth"]
    per_part_auc = []
    for i in range(5):
        if len(np.unique(all_labels[:, i])) < 2:
            auc_val = float('nan')
        else:
            auc_val = roc_auc_score(all_labels[:, i], all_probs[:, i])
        per_part_auc.append(auc_val)
    return per_part_auc  # 长度5的列表

# ==================== 主程序 ====================
def main():
    logger.info(f"=== 部位识别器鲁棒性评估（按部位独立统计） ===")
    logger.info(f"开始时间: {datetime.now()}")
    logger.info(f"设备: {DEVICE}")
    logger.info(f"部位识别器权重: {PART_DETECTOR_WEIGHT}")
    logger.info(f"采样比例: {SAMPLE_RATIO}")
    logger.info(f"包含训练数据: {INCLUDE_TRAIN_DATA}")
    
    config = Config()
    model = ForensicModel(config).to(DEVICE)
    if not os.path.exists(PART_DETECTOR_WEIGHT):
        logger.error(f"权重文件不存在: {PART_DETECTOR_WEIGHT}")
        sys.exit(1)
    state_dict = torch.load(PART_DETECTOR_WEIGHT, map_location=DEVICE)
    model.part_detector.load_state_dict(state_dict)
    model.eval()
    model.set_part_detector_trainable(False)
    logger.info("部位识别器加载完成")
    
    # 构建数据集列表
    datasets_to_eval = []
    for ds_cfg in config.test_datasets:
        if ds_cfg["name"] in SELECTED_DATASETS:
            datasets_to_eval.append((ds_cfg["name"], ds_cfg["real_dirs"], ds_cfg["fake_dirs"]))
    if INCLUDE_TRAIN_DATA:
        datasets_to_eval.append(("Train", config.train_real_dirs, config.train_fake_dirs))
    
    if not datasets_to_eval:
        logger.error("没有选择任何数据集进行评估")
        sys.exit(1)
    
    part_names = ["contour", "eyebrows", "eyes", "nose", "mouth"]
    
    # 定义所有测试条件
    distortion_types = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG']
    levels = [1,2,3,4,5]
    conditions = [("Original", None, None)]
    for d_type in distortion_types:
        func = get_distortion_function(d_type)
        for lvl in levels:
            param = get_distortion_parameter(d_type, lvl)
            cond_name = f"{d_type}_L{lvl}"
            conditions.append((cond_name, func, param))
    
    # 对每个数据集单独处理
    for ds_name, real_dirs, fake_dirs in datasets_to_eval:
        logger.info(f"\n{'='*80}\n数据集: {ds_name}\n{'='*80}")
        img_list, npy_list = collect_dataset_images(real_dirs, fake_dirs,
                                                     sample_ratio=SAMPLE_RATIO,
                                                     recursive=config.recursive_search,
                                                     dataset_name=ds_name)
        if len(img_list) == 0:
            logger.warning(f"数据集 {ds_name} 无有效图像，跳过")
            continue
        
        # 存储每个条件下各个部位的 AUC：dict[cond_name] -> list of 5 AUCs
        results = {}
        for cond_name, func, param in conditions:
            logger.info(f"  测试条件: {cond_name}")
            auc_list = evaluate_on_condition(model, img_list, npy_list, func, param, DEVICE)
            results[cond_name] = auc_list
        
        # 输出以部位为单位的表格
        logger.info(f"\n--- 数据集 [{ds_name}] 各部位在不同条件下的 AUC ---")
        # 构造表头
        header = f"{'Condition':<15}" + "".join([f"{name:>12}" for name in part_names])
        logger.info(header)
        logger.info("-" * (15 + 12*5))
        for cond_name, auc_list in results.items():
            # 格式化每个 AUC，NaN 显示为 'N/A'
            auc_strs = []
            for auc in auc_list:
                if np.isnan(auc):
                    auc_strs.append("        N/A")
                else:
                    auc_strs.append(f"{auc:12.4f}")
            line = f"{cond_name:<15}" + "".join(auc_strs)
            logger.info(line)
        
        # 也可按部位输出详细数据（可选）
        logger.info(f"\n--- 数据集 [{ds_name}] 按部位整理的 AUC 值 ---")
        for part_idx, part_name in enumerate(part_names):
            logger.info(f"\n部位: {part_name}")
            for cond_name, auc_list in results.items():
                auc_val = auc_list[part_idx]
                if np.isnan(auc_val):
                    logger.info(f"  {cond_name:<15}: N/A")
                else:
                    logger.info(f"  {cond_name:<15}: {auc_val:.4f}")
    
    logger.info(f"\n评估完成，日志保存至: {OUTPUT_LOG}")

if __name__ == "__main__":
    main()