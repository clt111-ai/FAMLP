#!/usr/bin/env python3
"""
Patch级别权重可视化脚本
- 对指定目录下所有图片，调用部位检测器输出256个patch的5个部位概率
- 每个图片生成5张纯权重图（无原图叠加），每张图对应一个部位
- 每个patch格子颜色深浅表示该部位的概率（深红=高概率，浅粉=低概率）
- 输出目录结构：OUTPUT_ROOT/图片名(不带扩展名)/部位名.png
"""

import os
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.cm as cm

from config import Config
from model import ForensicModel

# ==================== 用户配置（直接修改这里） ====================
INPUT_DIR = "/root/autodl-tmp/newmethods/FF1FAKE/Deepfakes/c23/frames/000_003"                    # 输入图片目录（必填）
WEIGHT_PATH = "/root/autodl-tmp/BBBBB-base/part_detector.pth"   # 部位识别器权重
OUTPUT_ROOT = "./patch_weight_viz"                    # 输出根目录
RECURSIVE = True                                      # 是否递归搜索子目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 可视化参数
GRID_SIZE = 16        # 16x16 patches
CELL_SIZE = 16        # 每个格子显示的像素大小（输出图像尺寸 = GRID_SIZE * CELL_SIZE = 256x256）
COLORMAP = 'Reds'     # 颜色映射：'Reds' 从浅粉到深红
# ================================================================

def get_patch_probs(model, img_path, transform, device):
    """加载图片，返回5个部位的概率，shape: (16,16,5)"""
    pil_img = Image.open(img_path).convert('RGB')
    pil_img_resized = pil_img.resize((224,224), Image.BILINEAR)
    img_tensor = transform(pil_img_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        patch_tokens = model.extract_patch_tokens(img_tensor)
        logits = model.part_detector(patch_tokens)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [256,5]
    prob_map = probs.reshape(GRID_SIZE, GRID_SIZE, 5)  # [16,16,5]
    return prob_map

def create_patch_weight_image(prob_grid, cell_size=16, cmap='Reds'):
    """
    将16x16的概率网格可视化为彩色图像（纯权重图）
    prob_grid: (16,16) 概率值 0~1
    cell_size: 每个格子像素大小
    cmap: 颜色映射
    返回: numpy array (H,W,3) RGB, 0-255
    """
    # 获取颜色映射函数
    color_map = cm.get_cmap(cmap)
    # 将概率映射为RGBA，取前3通道
    colored = color_map(prob_grid)[:, :, :3]  # (16,16,3) 0-1 float
    colored_uint8 = (colored * 255).astype(np.uint8)
    # 放大每个格子
    h, w = prob_grid.shape
    img_out = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img_out[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = colored_uint8[i, j]
    return img_out

def collect_images(input_dir, recursive=True):
    """递归收集目录下所有图片文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if recursive:
        files = list(Path(input_dir).rglob("*"))
    else:
        files = list(Path(input_dir).iterdir())
    images = [str(f) for f in files if f.is_file() and f.suffix.lower() in image_extensions]
    return images

def main():
    print("="*60)
    print("Patch-Level Weight Visualization")
    print("="*60)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Weight path: {WEIGHT_PATH}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}, cell size: {CELL_SIZE}px, colormap: {COLORMAP}")
    
    # 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory does not exist: {INPUT_DIR}")
        sys.exit(1)
    
    # 加载模型
    config = Config()
    print("Loading model...")
    model = ForensicModel(config).to(DEVICE)
    if not os.path.exists(WEIGHT_PATH):
        print(f"Error: weight file not found: {WEIGHT_PATH}")
        sys.exit(1)
    state_dict = torch.load(WEIGHT_PATH, map_location=DEVICE)
    model.part_detector.load_state_dict(state_dict)
    model.eval()
    model.set_part_detector_trainable(False)
    print("Model loaded successfully.")
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 收集图片
    img_paths = collect_images(INPUT_DIR, recursive=RECURSIVE)
    if len(img_paths) == 0:
        print("No images found.")
        sys.exit(0)
    print(f"Found {len(img_paths)} images.")
    
    part_names = ["contour", "eyebrows", "eyes", "nose", "mouth"]
    
    # 处理每张图片
    for img_path in tqdm(img_paths, desc="Processing images"):
        basename = Path(img_path).stem  # 文件名不含扩展名
        img_out_dir = os.path.join(OUTPUT_ROOT, basename)
        os.makedirs(img_out_dir, exist_ok=True)
        
        try:
            prob_map = get_patch_probs(model, img_path, transform, DEVICE)  # (16,16,5)
        except Exception as e:
            print(f"Error processing {img_path}: {e}, skipping")
            continue
        
        # 为每个部位生成纯权重图
        for p_idx, p_name in enumerate(part_names):
            prob_grid = prob_map[:, :, p_idx]  # (16,16)
            weight_img = create_patch_weight_image(prob_grid, cell_size=CELL_SIZE, cmap=COLORMAP)
            out_path = os.path.join(img_out_dir, f"{p_name}.png")
            Image.fromarray(weight_img).save(out_path)
    
    print(f"\nVisualization completed. Results saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()