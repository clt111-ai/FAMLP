import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import clip
import numpy as np

from config import Config


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, dropout):
        super().__init__()
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class ParallelLoRAMLP(nn.Module):
    """原始 MLP + 5个并行的 LoRA 专家（轮廓、眉毛、眼睛、鼻子、嘴巴）"""
    def __init__(self, original_mlp, config, num_experts=5):
        super().__init__()
        self.original_fc1 = original_mlp[0]
        self.original_fc2 = original_mlp[2]
        self.act = original_mlp[1] if len(original_mlp) > 1 else QuickGELU()
        for param in self.original_fc1.parameters():
            param.requires_grad = False
        for param in self.original_fc2.parameters():
            param.requires_grad = False

        dim = original_mlp[0].in_features  # 应为 1024
        self.lora_experts = nn.ModuleList([
            LoRALayer(dim, dim, config.lora_rank, config.lora_alpha, config.lora_dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x, part_labels):
        # x: [B, L, D], part_labels: [B, L, 5] (概率值 0~1)
        out = self.original_fc2(self.act(self.original_fc1(x)))
        lora_sum = torch.zeros_like(x)
        for i in range(5):
            mask = part_labels[..., i].unsqueeze(-1)   # [B,L,1]
            if mask.any():
                lora_sum = lora_sum + self.lora_experts[i](x) * mask
        return out + lora_sum


class PartDetector(nn.Module):
    """部位识别器，输出5类（轮廓、眉毛、眼睛、鼻子、嘴巴）"""
    def __init__(self, in_dim=1024, hidden_dim=256, out_dim=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, patch_tokens):
        return self.mlp(patch_tokens)   # [B, N, 5]


class CLIPWithLoRA(nn.Module):
    def __init__(self, clip_model, config):
        super().__init__()
        self.clip_model = clip_model
        self.config = config
        self._replace_mlps()

    def _replace_mlps(self):
        visual = self.clip_model.visual
        if hasattr(visual, 'transformer'):
            resblocks = visual.transformer.resblocks
        else:
            resblocks = visual.transformer.resblocks
        for block in resblocks:
            original_mlp = block.mlp
            new_mlp = ParallelLoRAMLP(original_mlp, self.config, num_experts=5)
            block.mlp = new_mlp

    def forward(self, x, part_labels):
        # x: [B,3,224,224]
        visual = self.clip_model.visual
        x = visual.conv1(x)                     # [B,1024,16,16]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B,1024,256]
        x = x.permute(0, 2, 1)                  # [B,256,1024]
        class_embedding = visual.class_embedding.to(x.dtype)
        class_embedding = class_embedding.unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([class_embedding, x], dim=1)   # [B,257,1024]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)                  # [257,B,1024]

        # part_labels 扩展 cls token 占位
        B = part_labels.shape[0]
        device = part_labels.device
        cls_pad = torch.zeros(B, 1, 5, device=device, dtype=part_labels.dtype)
        full_labels = torch.cat([cls_pad, part_labels], dim=1)   # [B,257,5]
        full_labels = full_labels.permute(1, 0, 2)               # [257,B,5]

        for block in visual.transformer.resblocks:
            # 自注意力
            attn_out, _ = block.attn(block.ln_1(x), block.ln_1(x), block.ln_1(x), need_weights=False)
            x = x + attn_out
            # MLP with LoRA
            mlp_input = block.ln_2(x)
            mlp_out = block.mlp(mlp_input.permute(1,0,2), full_labels.permute(1,0,2)).permute(1,0,2)
            x = x + mlp_out

        x = x.permute(1, 0, 2)                  # [B,257,1024]
        cls_token = visual.ln_post(x[:, 0, :])  # [B,1024]
        return cls_token


class ForensicModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 加载原始 CLIP 模型并冻结
        self.raw_clip_model, _ = clip.load(config.clip_model_name, device=config.device)
        self.raw_clip_model = self.raw_clip_model.float()
        for param in self.raw_clip_model.parameters():
            param.requires_grad = False

        # 替换 MLP 为带 LoRA 的版本
        self.clip_with_lora = CLIPWithLoRA(self.raw_clip_model, config)

        # 部位识别器（输出5类：轮廓、眉毛、眼睛、鼻子、嘴巴）
        self.part_detector = PartDetector(in_dim=1024, hidden_dim=config.part_detector_hidden_dim, out_dim=5)

        # 分类头（输入维度 1024）
        self.classifier = nn.Sequential(
            nn.Linear(1024, config.classifier_mid_dim),
            nn.GELU(),
            nn.Linear(config.classifier_mid_dim, 2)
        )

    def extract_patch_tokens(self, images):
        """获取 patch embedding（不包含 cls token），形状 [B,256,1024]"""
        visual = self.raw_clip_model.visual
        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        return x

    def get_part_labels(self, images):
        """
        使用训练好的部位识别器生成 part_labels [B,256,5]
        类别顺序: 0:轮廓, 1:眉毛, 2:眼睛, 3:鼻子, 4:嘴巴
        """
        with torch.no_grad():
            patch_tokens = self.extract_patch_tokens(images)
            logits = self.part_detector(patch_tokens)          # [B,256,5]
            probs = torch.sigmoid(logits)                      # [B,256,5]
        return probs

    def forward(self, images, training=True):
        clip_images = F.interpolate(images, size=(self.config.clip_img_size, self.config.clip_img_size))
        part_labels = self.get_part_labels(clip_images)
        cls_token = self.clip_with_lora(clip_images, part_labels)
        logits = self.classifier(cls_token)
        if training:
            return {'logits': logits, 'part_labels': part_labels}
        else:
            return {'logits': logits, 'probs': F.softmax(logits, dim=1)}

    # ---------- 部位识别器训练专用 ----------
    def train_part_detector_step(self, images, target_labels):
        """
        target_labels: [B,256,5] 二值标签（轮廓、眉毛、眼睛、鼻子、嘴巴）
        """
        patch_tokens = self.extract_patch_tokens(images)
        logits = self.part_detector(patch_tokens)
        loss = F.binary_cross_entropy_with_logits(logits, target_labels)
        return loss

    def set_part_detector_trainable(self, trainable):
        for param in self.part_detector.parameters():
            param.requires_grad = trainable

    def set_lora_and_classifier_trainable(self, trainable):
        for name, param in self.clip_with_lora.named_parameters():
            if 'lora_experts' in name:
                param.requires_grad = trainable
        for param in self.classifier.parameters():
            param.requires_grad = trainable

    def freeze_all_except(self, train_part_detector=False, train_lora_classifier=False):
        for param in self.parameters():
            param.requires_grad = False
        if train_part_detector:
            self.set_part_detector_trainable(True)
        if train_lora_classifier:
            self.set_lora_and_classifier_trainable(True)