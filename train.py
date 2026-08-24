import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import RandomErasing
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import logging
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath

from config import Config
from model import ForensicModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_images_from_dirs(directories, sample_ratio=1.0, recursive=True):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    all_images = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: {directory} does not exist, skipping")
            continue
        if recursive:
            files = list(Path(directory).rglob("*"))
        else:
            files = list(Path(directory).iterdir())
        valid = [str(f) for f in files if f.is_file() and f.suffix.lower() in image_extensions]
        all_images.extend(valid)
        print(f"Found {len(valid)} images in {directory}")
    if sample_ratio < 1.0:
        sample_size = max(1, int(len(all_images) * sample_ratio))
        all_images = random.sample(all_images, sample_size)
        print(f"Sampled {len(all_images)} images (ratio={sample_ratio})")
    return all_images


def get_npy_path(img_path):
    return img_path.replace('frames', 'landmarks').replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy')


def load_landmarks_from_npy(npy_path):
    try:
        data = np.load(npy_path)
        if data.shape[-1] == 2 and len(data.shape) == 2:
            return data.astype(np.float32)
        else:
            print(f"Unexpected shape {data.shape} in {npy_path}, expected (N,2)")
            return None
    except Exception:
        return None


def generate_patch_labels_from_landmarks(landmarks, img_size=224, patch_size=14):
    """
    生成 5 类标签（轮廓、眉毛、眼睛、鼻子、嘴巴）
    landmarks: 已缩放到 img_size 坐标系，形状 (N,2)
    返回 torch.Tensor [num_patches, 5]
    """
    grid_size = img_size // patch_size  # 16
    num_patches = grid_size ** 2  # 256
    labels = np.zeros((num_patches, 5), dtype=np.float32)
    if landmarks is None or landmarks.shape[0] == 0:
        return torch.from_numpy(labels)

    part_indices = Config.part_landmark_indices
    part_order = ["contour", "eyebrows", "eyes", "nose", "mouth"]  # 共5类

    for part_idx, part_name in enumerate(part_order):
        idx_list = part_indices.get(part_name, [])
        for idx in idx_list:
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


class PartDetectorDataset(Dataset):
    def __init__(self, real_dirs, fake_dirs, transform=None, sample_ratio=1.0, recursive=True):
        self.transform = transform
        self.real_images = collect_images_from_dirs(real_dirs, sample_ratio, recursive)
        self.fake_images = collect_images_from_dirs(fake_dirs, sample_ratio, recursive)
        self.all_images = self.real_images + self.fake_images
        self.valid_indices = []
        self.npy_paths = []
        print("Checking npy files for part detector training...")
        success_count = 0
        for idx, img_path in enumerate(self.all_images):
            npy_path = get_npy_path(img_path)
            if os.path.exists(npy_path):
                self.valid_indices.append(idx)
                self.npy_paths.append(npy_path)
                success_count += 1
        print(f"Found {success_count}/{len(self.all_images)} images with valid npy landmark files.")
        if self.npy_paths:
            sample_npy = np.load(self.npy_paths[0])
            print(f"Sample npy shape: {sample_npy.shape}")
        else:
            raise RuntimeError("No npy files found for part detector training!")
        self.filtered_images = [self.all_images[i] for i in self.valid_indices]

    def __len__(self):
        return len(self.filtered_images)

    def __getitem__(self, idx):
        img_path = self.filtered_images[idx]
        npy_path = self.npy_paths[idx]

        pil_img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        landmarks = load_landmarks_from_npy(npy_path)
        if landmarks is None:
            landmarks = np.zeros((0, 2), dtype=np.float32)

        target_size = (224, 224)
        pil_img_resized = pil_img.resize(target_size, Image.BILINEAR)

        if landmarks.shape[0] > 0:
            if landmarks.max() > 1.0:
                scale_x = target_size[0] / orig_w
                scale_y = target_size[1] / orig_h
                landmarks[:, 0] *= scale_x
                landmarks[:, 1] *= scale_y
            else:
                landmarks[:, 0] *= target_size[0]
                landmarks[:, 1] *= target_size[1]

        patch_labels = generate_patch_labels_from_landmarks(landmarks, img_size=224, patch_size=14)

        if self.transform:
            image = self.transform(pil_img_resized)
        else:
            image = transforms.ToTensor()(pil_img_resized)

        return image, patch_labels


class ForensicDataset(Dataset):
    def __init__(self, real_dirs, fake_dirs, transform=None, sample_ratio=1.0, recursive=True):
        self.transform = transform
        self.real_images = collect_images_from_dirs(real_dirs, sample_ratio, recursive)
        self.fake_images = collect_images_from_dirs(fake_dirs, sample_ratio, recursive)
        self.all_images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        print(f"Real: {len(self.real_images)}, Fake: {len(self.fake_images)}, Total: {len(self.all_images)}")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros(3, 256, 256)
        return image, label


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        set_seed(config.seed)
        self.model = ForensicModel(config).to(self.device)
        self._setup_logging()
        self.part_detector_loader = self._get_part_detector_loader()
        self.train_loader, self.test_loaders = None, None
        self.best_avg_auc = 0.0
        self.start_epoch = 0
    def _setup_logging(self):
        log_file = os.path.join(self.config.save_dir, "training_log.txt")
        os.makedirs(self.config.save_dir, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Training run started at {datetime.now()}\n")
            f.write("Complete Configuration:\n")
            for k, v in self.config.__dict__.items():
                if not k.startswith('__'):
                    f.write(f"  {k}: {v}\n")
            f.write("-" * 80 + "\n")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Random seed set to {}".format(self.config.seed))
        # 记录随机擦除配置
        self.logger.info("Random Erasing enabled: prob={}, scale={}, ratio={}".format(
            self.config.random_erasing_prob,
            self.config.random_erasing_scale,
            self.config.random_erasing_ratio
        ))
    def _get_part_detector_loader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = PartDetectorDataset(
            real_dirs=self.config.train_real_dirs,
            fake_dirs=self.config.train_fake_dirs,
            transform=transform,
            sample_ratio=self.config.train_sample_ratio,
            recursive=self.config.recursive_search
        )
        return DataLoader(dataset, batch_size=self.config.part_detector_batch_size,
                          shuffle=True, num_workers=self.config.num_workers, pin_memory=True)
    def _get_train_test_loaders(self):
        max_size = self.config.clip_img_size
        # 训练集的 transform：加入随机擦除
        train_transform = transforms.Compose([
            transforms.Resize((max_size, max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(
                p=self.config.random_erasing_prob,
                scale=self.config.random_erasing_scale,
                ratio=self.config.random_erasing_ratio,
                value=self.config.random_erasing_value
            )
        ])
        # 测试集不需要随机擦除
        test_transform = transforms.Compose([
            transforms.Resize((max_size, max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = ForensicDataset(
            real_dirs=self.config.train_real_dirs,
            fake_dirs=self.config.train_fake_dirs,
            transform=train_transform,
            sample_ratio=self.config.train_sample_ratio,
            recursive=self.config.recursive_search
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                  num_workers=self.config.num_workers, pin_memory=True)

        test_loaders = []
        for ds in self.config.test_datasets:
            name = ds["name"]
            real_dirs = [d for d in ds["real_dirs"] if os.path.exists(d)]
            fake_dirs = [d for d in ds["fake_dirs"] if os.path.exists(d)]
            if not real_dirs or not fake_dirs:
                self.logger.warning(f"Test dataset {name} has missing dirs, skipped")
                continue
            test_dataset = ForensicDataset(real_dirs, fake_dirs, test_transform,
                                           sample_ratio=self.config.test_sample_ratio,
                                           recursive=self.config.recursive_search)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                                     num_workers=self.config.num_workers, pin_memory=True)
            test_loaders.append((name, test_loader))
        return train_loader, test_loaders
    def visualize_part_detector(self, dataset_name="train", num_samples=100, loader=None):
        """对给定数据集进行部位热力图可视化，每个样本保存5张图"""
        self.model.eval()
        viz_dir = os.path.join(self.config.save_dir, f"part_detector_viz_{dataset_name}")
        os.makedirs(viz_dir, exist_ok=True)

        if loader is None:
            dataset = self.part_detector_loader.dataset
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            samples = []
            for idx in indices:
                img_tensor, _ = dataset[idx]
                samples.append(img_tensor)
        else:
            samples = []
            for images, _ in loader:
                for img in images:
                    samples.append(img)
                    if len(samples) >= num_samples:
                        break
                if len(samples) >= num_samples:
                    break

        patch_size = self.config.patch_size
        grid_size = self.config.clip_img_size // patch_size
        part_names = ['contour', 'eyebrows', 'eyes', 'nose', 'mouth']  # 5个部位

        for i, img_tensor in enumerate(samples[:num_samples]):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = img_tensor * std + mean
            img_display = torch.clamp(img_display, 0, 1).permute(1, 2, 0).cpu().numpy()
            with torch.no_grad():
                patch_tokens = self.model.extract_patch_tokens(img_tensor.unsqueeze(0).to(self.device))
                logits = self.model.part_detector(patch_tokens)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [256,5]
            for p_idx, p_name in enumerate(part_names):
                prob_map = probs[:, p_idx].reshape(grid_size, grid_size)
                prob_map_resized = cv2.resize(prob_map, (224, 224), interpolation=cv2.INTER_LINEAR)
                heatmap = plt.cm.jet(prob_map_resized)[:, :, :3]
                overlay = 0.6 * img_display + 0.4 * heatmap
                overlay = np.clip(overlay, 0, 1)
                plt.figure(figsize=(5, 5))
                plt.imshow(overlay)
                plt.title(f"{p_name} probability")
                plt.axis('off')
                plt.savefig(os.path.join(viz_dir, f"sample_{i:03d}_{p_name}.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
            self.logger.info(f"Visualization saved for sample {i+1}/{min(num_samples, len(samples))} in {dataset_name}")
        self.logger.info(f"All visualizations saved to {viz_dir}")
    def train_part_detector(self):
        self.logger.info("=" * 50)
        self.logger.info("Stage 1: Training Part Detector (5 parts: contour, eyebrows, eyes, nose, mouth)")
        self.logger.info("=" * 50)
        optimizer = optim.Adam(self.model.part_detector.parameters(),
                               lr=self.config.part_detector_lr)
        criterion = nn.BCEWithLogitsLoss()
        self.model.freeze_all_except(train_part_detector=True, train_lora_classifier=False)
        self.model.train()
        for epoch in range(self.config.part_detector_epochs):
            total_loss = 0
            pbar = tqdm(self.part_detector_loader, desc=f"PartDetector Epoch {epoch+1}")
            for images, patch_labels in pbar:
                images = images.to(self.device)
                patch_labels = patch_labels.to(self.device)
                optimizer.zero_grad()
                loss = self.model.train_part_detector_step(images, patch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            avg_loss = total_loss / len(self.part_detector_loader)
            self.logger.info(f"PartDetector Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        torch.save(self.model.part_detector.state_dict(),
                   os.path.join(self.config.save_dir, "part_detector.pth"))
        self.logger.info("Part detector training completed and saved.")
        self.visualize_part_detector(dataset_name="train", num_samples=100)
        self.model.set_part_detector_trainable(False)
    def train_lora_classifier(self):
        self.logger.info("=" * 50)
        self.logger.info("Stage 2: Training LoRA Experts and Classifier")
        self.logger.info("Random Erasing is ACTIVE in this stage.")
        self.logger.info("=" * 50)
        self.train_loader, self.test_loaders = self._get_train_test_loaders()
        self.model.freeze_all_except(train_part_detector=False, train_lora_classifier=True)
        optimizer = optim.Adam([
            {'params': self.model.classifier.parameters()},
            {'params': self.model.clip_with_lora.parameters(), 'lr': self.config.lr}
        ], lr=self.config.lr, weight_decay=self.config.weight_decay)
        # 使用普通交叉熵损失（无标签平滑）
        criterion = nn.CrossEntropyLoss()
        total_epochs = self.config.epochs
        for epoch in range(self.start_epoch, total_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            part_counter = {i: 0 for i in range(5)}
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images, training=True)
                logits = outputs['logits']
                part_labels = outputs['part_labels']
                hard_labels = (part_labels > 0.5).float()
                for b in range(hard_labels.shape[0]):
                    for p in range(5):
                        part_counter[p] += hard_labels[b, :, p].sum().item()
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100 * correct / total:.1f}%'})
            avg_loss = total_loss / len(self.train_loader)
            train_acc = 100 * correct / total
            num_images = len(self.train_loader.dataset)
            num_patches_per_image = 256
            part_ratios = {k: v / (num_images * num_patches_per_image) for k, v in part_counter.items()}
            self.logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} Train Acc: {train_acc:.2f}%")
            self.logger.info(
                f"Part ratios - Contour: {part_ratios[0]:.3f}, Eyebrows: {part_ratios[1]:.3f}, Eyes: {part_ratios[2]:.3f}, Nose: {part_ratios[3]:.3f}, Mouth: {part_ratios[4]:.3f}")
            # 测试
            test_results = {}
            auc_list = []
            for name, loader in self.test_loaders:
                acc, auc, eer = self.evaluate(loader, name)
                test_results[name] = {'acc': acc, 'auc': auc, 'eer': eer}
                auc_list.append(auc)
                self.logger.info(f"Test {name}: Acc={acc:.2f}%, AUC={auc:.4f}, EER={eer:.4f}")
            avg_auc = np.mean(auc_list) if auc_list else 0.0
            self.logger.info(f"Average AUC over {len(auc_list)} datasets: {avg_auc:.4f}")
            is_best = avg_auc > self.best_avg_auc
            if is_best:
                self.best_avg_auc = avg_auc
            self.save_checkpoint(epoch, avg_auc, is_best)
            self.visualize_on_test_datasets(num_samples=10)
        self.logger.info(f"Training completed. Best average AUC = {self.best_avg_auc:.4f}")
    def evaluate(self, loader, name):
        self.model.eval()
        all_labels, all_probs = [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=f"Test {name}", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images, training=False)
                probs = outputs['probs'][:, 1].cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        if len(all_labels) == 0:
            return 0.0, 0.0, 0.0
        acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int)) * 100
        auc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return acc, auc, eer
    def visualize_on_test_datasets(self, num_samples=10):
        for name, loader in self.test_loaders:
            samples = []
            for images, _ in loader:
                for img in images:
                    samples.append(img)
                    if len(samples) >= num_samples:
                        break
                if len(samples) >= num_samples:
                    break
            if len(samples) == 0:
                continue
            self.visualize_part_detector(dataset_name=f"test_{name}", num_samples=num_samples, loader=loader)
    def save_checkpoint(self, epoch, avg_auc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_avg_auc': self.best_avg_auc,
            'config': self.config.__dict__
        }
        latest_path = os.path.join(self.config.save_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = os.path.join(self.config.save_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with AUC = {avg_auc:.4f}")
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            self.best_avg_auc = ckpt.get('best_avg_auc', 0.0)
            self.logger.info(f"Resumed from epoch {self.start_epoch}, best AUC = {self.best_avg_auc:.4f}")
            return True
        return False
    def train(self):
        self.train_part_detector()
        latest_ckpt = os.path.join(self.config.save_dir, "latest_checkpoint.pth")
        self.load_checkpoint(latest_ckpt)
        self.train_lora_classifier()


def main():
    config = Config()
    if not any(os.path.exists(d) for d in config.train_real_dirs) or not any(
            os.path.exists(d) for d in config.train_fake_dirs):
        print("Error: Training data directories missing.")
        return
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()