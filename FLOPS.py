import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from thop import profile
from config import Config
from model import ForensicModel
from train import collect_images_from_dirs
from torchvision import transforms
from PIL import Image

# ==================== 用户配置 ====================
CONFIGS_TO_TEST = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
]

SAMPLE_RATIO = 0.001      # 从训练集中抽取多少比例进行推理测试
BATCH_SIZE = 16
# ==================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_data_loader(sample_ratio):
    """从训练数据目录中采样图片，返回 DataLoader"""
    cfg = Config()
    transform = transforms.Compose([
        transforms.Resize((cfg.clip_img_size, cfg.clip_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    real_images = collect_images_from_dirs(cfg.train_real_dirs, sample_ratio, cfg.recursive_search)
    fake_images = collect_images_from_dirs(cfg.train_fake_dirs, sample_ratio, cfg.recursive_search)
    all_paths = real_images + fake_images
    labels = [0] * len(real_images) + [1] * len(fake_images)
    print(f"Loaded {len(all_paths)} images for inference (sampled ratio={sample_ratio})")

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, paths, labels, transform):
            self.paths = paths
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

    dataset = SimpleDataset(all_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return loader

def compute_flops_for_config(cfg, input_size, device):
    """单独创建一个模型实例计算 FLOPs，避免 thop 污染"""
    model = ForensicModel(cfg).to(device)
    model.eval()
    inputs = torch.randn(input_size).to(device)
    flops, _ = profile(model, inputs=(inputs,), verbose=False)
    # 清理模型，释放显存，并移除可能的 hook 残留
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return flops

def measure_inference_time(model, data_loader, device):
    """遍历整个 DataLoader，测量总推理时间，返回平均每张图片耗时(ms)和 FPS"""
    model.eval()
    total_images = 0
    total_time = 0.0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(images, training=False)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            total_time += elapsed
            total_images += images.size(0)
    avg_time_ms = (total_time / total_images) * 1000
    fps = total_images / total_time
    return avg_time_ms, fps

def count_params(model):
    """统计总参数量、可训练参数量（部位检测器 + LoRA + 分类头）及占比"""
    total = sum(p.numel() for p in model.parameters())
    part_params = sum(p.numel() for p in model.part_detector.parameters())
    cls_params = sum(p.numel() for p in model.classifier.parameters())
    lora_params = 0
    for name, param in model.named_parameters():
        if 'lora_experts' in name:
            lora_params += param.numel()
    trainable = part_params + cls_params + lora_params
    ratio = trainable / total * 100 if total > 0 else 0
    return total, trainable, part_params, cls_params, lora_params, ratio

def main():
    # 预先构建数据加载器（所有配置共用同一份图片数据）
    print("Preparing data loader...")
    data_loader = get_data_loader(SAMPLE_RATIO)
    if len(data_loader.dataset) == 0:
        print("Error: No images found. Check your training data paths in config.py")
        return

    results = []
    for rank, alpha in CONFIGS_TO_TEST:
        print(f"\n========== Testing rank={rank}, alpha={alpha} ==========")

        # 创建配置
        cfg = Config()
        cfg.lora_rank = rank
        cfg.lora_alpha = alpha
        cfg.device = device

        # 1. 计算 FLOPs（独立模型实例，避免污染）
        flops = compute_flops_for_config(cfg, (1, 3, cfg.clip_img_size, cfg.clip_img_size), device)

        # 2. 创建新模型实例用于参数统计和时间测试
        model = ForensicModel(cfg).to(device)
        total_params, trainable_params, part_p, cls_p, lora_p, ratio = count_params(model)

        # 3. 测量推理时间
        avg_time_ms, fps = measure_inference_time(model, data_loader, device)

        results.append({
            'rank': rank,
            'alpha': alpha,
            'FLOPs (G)': flops / 1e9,
            'Total Params (M)': total_params / 1e6,
            'Trainable (M)': trainable_params / 1e6,
            'Trainable Ratio (%)': ratio,
            'Inference Time (ms)': avg_time_ms,
            'FPS': fps,
        })

        print(f"FLOPs: {flops/1e9:.2f} G")
        print(f"总参数量: {total_params/1e6:.2f} M")
        print(f"可训练参数量: {trainable_params/1e6:.2f} M ({ratio:.2f}%)")
        print(f"   ├─ 部位检测器: {part_p/1e6:.2f} M")
        print(f"   ├─ 分类头: {cls_p/1e6:.2f} M")
        print(f"   └─ LoRA 专家: {lora_p/1e6:.2f} M")
        print(f"推理时间 (平均每张): {avg_time_ms:.2f} ms")
        print(f"FPS (吞吐量): {fps:.2f} img/s")

        # 释放模型显存
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # 打印汇总表格
    print("\n" + "=" * 120)
    header = f"{'Rank':<6} {'Alpha':<6} {'FLOPs(G)':<10} {'Total(M)':<10} {'Trainable(M)':<12} {'Ratio(%)':<10} {'Time(ms)':<10} {'FPS':<10}"
    print(header)
    print("-" * 120)
    for r in results:
        print(f"{r['rank']:<6} {r['alpha']:<6} {r['FLOPs (G)']:<10.2f} {r['Total Params (M)']:<10.2f} "
              f"{r['Trainable (M)']:<12.2f} {r['Trainable Ratio (%)']:<10.2f} {r['Inference Time (ms)']:<10.2f} {r['FPS']:<10.2f}")

    # 保存 CSV
    import csv
    csv_path = "benchmark_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n结果已保存至 {csv_path}")

if __name__ == "__main__":
    main()