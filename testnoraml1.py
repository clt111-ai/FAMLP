import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import logging
from datetime import datetime

from config import Config
from model import ForensicModel


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_images_with_video_info(directories, recursive=True):
    """
    从目录列表中收集所有图片，并记录每张图片的视频ID。
    视频ID定义为图片所在目录的绝对路径（即同一子目录下的所有图片属于一个视频）。
    返回: list of (image_path, video_id)
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    items = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: {directory} does not exist, skipping")
            continue
        if recursive:
            all_files = list(Path(directory).rglob("*"))
        else:
            all_files = list(Path(directory).iterdir())
        for f in all_files:
            if f.is_file() and f.suffix.lower() in image_extensions:
                video_id = str(f.parent.absolute())   # 直接父目录作为视频ID
                items.append((str(f), video_id))
    return items


class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path, True
        except Exception:
            # 返回空张量，标记为无效
            return torch.zeros(3, 224, 224), img_path, False


def compute_eer(labels, scores):
    """计算等错误率"""
    fpr, tpr, _ = roc_curve(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def evaluate_dataset(model, dataset_config, test_transform, device, output_dir, dataset_name):
    """
    对单个数据集进行评测
    dataset_config: dict with keys "real_dirs", "fake_dirs"
    """
    real_dirs = dataset_config["real_dirs"]
    fake_dirs = dataset_config["fake_dirs"]
    
    # 收集真实和伪造图片及其视频ID
    real_items = collect_images_with_video_info(real_dirs, recursive=True)
    fake_items = collect_images_with_video_info(fake_dirs, recursive=True)
    
    all_items = []
    for path, vid in real_items:
        all_items.append((path, vid, 0))   # 真实标签0
    for path, vid in fake_items:
        all_items.append((path, vid, 1))   # 伪造标签1
    
    print(f"\n[{dataset_name}] Total images found: {len(all_items)}")
    print(f"  Real: {len(real_items)}, Fake: {len(fake_items)}")
    
    if len(all_items) == 0:
        print(f"  Skipping {dataset_name} because no images found.")
        return None
    
    # 构建路径到标签和视频ID的映射
    path_to_label = {item[0]: item[2] for item in all_items}
    path_to_video = {item[0]: item[1] for item in all_items}
    image_paths = [item[0] for item in all_items]
    
    dataset = InferenceDataset(image_paths, test_transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    model.eval()
    all_probs = []
    all_labels = []
    valid_paths = []
    video_probs = {}   # video_id -> list of probs
    video_labels = {}  # video_id -> label (should be consistent)
    
    with torch.no_grad():
        for images, paths, valid_flags in tqdm(loader, desc=f"Inference on {dataset_name}"):
            images = images.to(device)
            valid_flags = valid_flags.numpy()
            valid_indices = [i for i, v in enumerate(valid_flags) if v]
            if not valid_indices:
                continue
            valid_images = images[valid_indices]
            outputs = model(valid_images, training=False)
            probs = outputs['probs'][:, 1].cpu().numpy()  # 取假样本的概率
            for idx, prob in zip(valid_indices, probs):
                img_path = paths[idx]
                label = path_to_label[img_path]
                video_id = path_to_video[img_path]
                all_probs.append(prob)
                all_labels.append(label)
                valid_paths.append(img_path)
                
                if video_id not in video_probs:
                    video_probs[video_id] = []
                    video_labels[video_id] = label
                video_probs[video_id].append(prob)
    
    num_success = len(valid_paths)
    total_images = len(all_items)
    success_rate = num_success / total_images if total_images > 0 else 0
    print(f"  Successfully loaded and inferred: {num_success}/{total_images} ({success_rate:.2%})")
    
    if num_success == 0:
        print(f"  No valid images in {dataset_name}, skipping metrics.")
        return None
    
    # 图片级指标
    img_acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
    img_auc = roc_auc_score(all_labels, all_probs)
    img_eer = compute_eer(all_labels, all_probs)
    
    # 视频级指标
    video_labels_list = []
    video_probs_list = []
    for vid, probs in video_probs.items():
        avg_prob = np.mean(probs)
        # 视频标签：取该视频中出现最多的标签（正常情况下所有图片标签一致）
        labels_in_video = [path_to_label[p] for p in valid_paths if path_to_video[p] == vid]
        if labels_in_video:
            majority_label = max(set(labels_in_video), key=labels_in_video.count)
        else:
            majority_label = video_labels[vid]
        video_labels_list.append(majority_label)
        video_probs_list.append(avg_prob)
    
    if len(video_labels_list) > 0:
        vid_acc = accuracy_score(video_labels_list, (np.array(video_probs_list) > 0.5).astype(int))
        vid_auc = roc_auc_score(video_labels_list, video_probs_list)
        vid_eer = compute_eer(video_labels_list, video_probs_list)
    else:
        vid_acc = vid_auc = vid_eer = 0.0
    
    results = {
        "dataset": dataset_name,
        "total_images": total_images,
        "successful_images": num_success,
        "success_rate": success_rate,
        "image_level": {"acc": img_acc, "auc": img_auc, "eer": img_eer},
        "video_level": {"acc": vid_acc, "auc": vid_auc, "eer": vid_eer},
        "num_videos": len(video_probs)
    }
    
    # 保存详细结果
    out_file = os.path.join(output_dir, f"{dataset_name}_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Results saved to {out_file}")
    
    return results


def main():
    # ==================== 手动配置参数 ====================
    # 模型权重文件路径（必须）
    MODEL_WEIGHTS = "./best_model.pth"
    # 可选：单独的部位检测器权重（如果已包含在 MODEL_WEIGHTS 中，可设为 None）
    PART_DETECTOR_WEIGHTS = "./part_detector.pth"
    
    # 输出目录
    OUTPUT_DIR = "./video"
    
    # 设备
    DEVICE = "cuda"   # "cuda" 或 "cpu"
    
    # ==================== 定义测试数据集 ====================
    # 每个数据集是一个字典，包含 name, real_dirs, fake_dirs
    TEST_DATASETS = [
        {
            "name": "CDFv1",
            "real_dirs": [
                "/root/autodl-tmp/newmethods/Celebv1-real/frames",
                "/root/autodl-tmp/newmethods/YouTube-real/frames"
            ],
            "fake_dirs": [
                "/root/autodl-tmp/newmethods/Celebv1-synthesis/frames"
            ]
        },
        {
            "name": "DFD",
            "real_dirs": [
                "/root/autodl-tmp/newmethods/FF1REAL/actors/c23/frames"
            ],
            "fake_dirs": [
                "/root/autodl-tmp/newmethods/FF1FAKE/DeepFakeDetection/c23/frames"
            ]
        },
        {
            "name": "CDFv2",
            "real_dirs": [
                "/root/autodl-tmp/newmethods/CDFv2/Celeb-real/frames",
                "/root/autodl-tmp/newmethods/CDFv2/YouTube-real/frames"
            ],
            "fake_dirs": [
                "/root/autodl-tmp/newmethods/CDFv2/Celeb-synthesis/frames"
            ]
        },
        {
            "name": "DFDCP",
            "real_dirs": [
                "/root/autodl-tmp/newmethods/DFDCP/original_videos/frames"
            ],
            "fake_dirs": [
                "/root/autodl-tmp/newmethods/DFDCP/method_A/frames",
                "/root/autodl-tmp/newmethods/DFDCP/method_B/frames"
            ]
        },
        # 可以继续添加更多数据集
    ]
    # ======================================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(OUTPUT_DIR, "test_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Test started at {datetime.now()}")
    logger.info(f"Model weights: {MODEL_WEIGHTS}")
    if PART_DETECTOR_WEIGHTS:
        logger.info(f"Part detector weights: {PART_DETECTOR_WEIGHTS}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # 加载配置
    config = Config()
    config.device = DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu"
    set_seed(config.seed)
    
    # 初始化模型
    logger.info("Initializing model...")
    model = ForensicModel(config).to(config.device)
    
    # 加载权重
    logger.info(f"Loading model weights from {MODEL_WEIGHTS}")
    checkpoint = torch.load(MODEL_WEIGHTS, map_location=config.device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    if PART_DETECTOR_WEIGHTS and os.path.exists(PART_DETECTOR_WEIGHTS):
        logger.info(f"Loading part detector weights from {PART_DETECTOR_WEIGHTS}")
        part_state = torch.load(PART_DETECTOR_WEIGHTS, map_location=config.device, weights_only=False)
        model.part_detector.load_state_dict(part_state)
    
    model.eval()
    
    # 测试图像预处理（与训练时测试集一致）
    test_transform = transforms.Compose([
        transforms.Resize((config.clip_img_size, config.clip_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 遍历每个测试集
    all_results = []
    for ds in TEST_DATASETS:
        name = ds["name"]
        # 只保留存在的目录
        real_dirs = [d for d in ds["real_dirs"] if os.path.exists(d)]
        fake_dirs = [d for d in ds["fake_dirs"] if os.path.exists(d)]
        if not real_dirs or not fake_dirs:
            logger.warning(f"Test dataset {name} has missing directories, skipped")
            continue
        ds_config = {"real_dirs": real_dirs, "fake_dirs": fake_dirs}
        logger.info(f"\nEvaluating dataset: {name}")
        results = evaluate_dataset(model, ds_config, test_transform, config.device, OUTPUT_DIR, name)
        if results:
            all_results.append(results)
            logger.info(f"  Image-level: Acc={results['image_level']['acc']:.4f}, AUC={results['image_level']['auc']:.4f}, EER={results['image_level']['eer']:.4f}")
            logger.info(f"  Video-level: Acc={results['video_level']['acc']:.4f}, AUC={results['video_level']['auc']:.4f}, EER={results['video_level']['eer']:.4f}")
    
    # 保存汇总结果
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"\nAll results saved to {OUTPUT_DIR}")
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()