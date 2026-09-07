#!/usr/bin/env python3
"""
Ablation Study: Train models with different sets of LoRA experts.
Experiments:
  1. no_experts (empty list)
  2. drop_contour  (experts [1,2,3,4])
  3. drop_eyebrows (experts [0,2,3,4])
  4. drop_eyes     (experts [0,1,3,4])
  5. drop_nose     (experts [0,1,2,4])
  6. drop_mouth    (experts [0,1,2,3])
"""

import os
import sys
import json
import copy
import torch
import numpy as np
from datetime import datetime

# 导入原始模块
from train import Trainer, Config, set_seed
from config import Config as BaseConfig


# ==================== 实验配置 ====================
ROOT_DIR = "./ablation_results_5face2"          # 保存所有实验结果的根目录
EPOCHS = 1                               # 第二阶段训练轮数（可调）
PART_DETECTOR_EPOCHS = 10                 # 部位检测器训练轮数
BATCH_SIZE = 16                          # 分类训练批大小
PART_DETECTOR_BATCH_SIZE = 32            # 部位检测器批大小
SEED = 42                                # 随机种子
# =================================================


class AblationConfig(BaseConfig):
    """扩展配置，支持覆盖 save_dir 和 active_experts"""
    def __init__(self, active_experts=None, save_dir=None, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if active_experts is not None:
            self.active_experts = active_experts
        if save_dir is not None:
            self.save_dir = save_dir


def run_single_experiment(exp_name, active_experts, root_dir, config_updates):
    """
    运行单个实验
    exp_name: 实验名称，用于创建子目录
    active_experts: 激活的专家索引列表（如 [0,1,2,3,4] 或 [] 或 [0,1,2,3]）
    root_dir: 总根目录
    config_updates: 更新配置的字典（包含 epochs, batch_size 等）
    """
    exp_dir = os.path.join(root_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 创建配置
    config = AblationConfig(active_experts=active_experts, save_dir=exp_dir)
    for key, value in config_updates.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: config has no attribute {key}")

    # 固定随机种子
    set_seed(config.seed)

    # 创建 Trainer 并训练
    trainer = Trainer(config)
    trainer.train()

    # 收集最佳测试结果
    best_results = trainer.best_test_results if hasattr(trainer, 'best_test_results') else {}
    best_avg_auc = trainer.best_avg_auc
    best_epoch = trainer.best_epoch if hasattr(trainer, 'best_epoch') else -1

    summary = {
        'exp_name': exp_name,
        'active_experts': active_experts,
        'best_avg_auc': best_avg_auc,
        'best_epoch': best_epoch,
        'test_results': best_results
    }
    summary_path = os.path.join(exp_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Experiment {exp_name} finished. Best Avg AUC: {best_avg_auc:.4f}")
    return summary


def format_table_row(cols, widths):
    """根据列宽格式化一行文本"""
    return " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(cols))


def main():
    # 创建根目录
    os.makedirs(ROOT_DIR, exist_ok=True)

    # 实验定义：名称 -> 激活的专家列表
    # 专家索引: 0=contour, 1=eyebrows, 2=eyes, 3=nose, 4=mouth
    experiments = {
        'no_experts':           [],               # 无任何专家
        'all_experts':          [0, 1, 2, 3, 4],
        'drop_contour':         [1, 2, 3, 4],     # 去掉轮廓
        'drop_eyebrows':        [0, 2, 3, 4],     # 去掉眉毛
        'drop_eyes':            [0, 1, 3, 4],     # 去掉眼睛
        'drop_nose':            [0, 1, 2, 4],     # 去掉鼻子
        'drop_mouth':           [0, 1, 2, 3],     # 去掉嘴巴
    }

    # 配置覆盖（可根据需要调整）
    config_updates = {
        'epochs': EPOCHS,
        'part_detector_epochs': PART_DETECTOR_EPOCHS,
        'batch_size': BATCH_SIZE,
        'part_detector_batch_size': PART_DETECTOR_BATCH_SIZE,
        'seed': SEED,
    }

    all_summaries = []

    for exp_name, active_experts in experiments.items():
        print(f"\n{'='*60}\n>>> Running {exp_name} (active_experts={active_experts})\n{'='*60}")
        summary = run_single_experiment(exp_name, active_experts, ROOT_DIR, config_updates)
        all_summaries.append(summary)

    # ==================== 生成总体汇总文本文件 ====================
    summary_path = os.path.join(ROOT_DIR, 'ablation_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ABLATION STUDY FINAL REPORT\n")
        f.write(f"Generated at {datetime.now()}\n")
        f.write(f"Root directory: {ROOT_DIR}\n")
        f.write("Configuration used for all experiments:\n")
        for key, value in config_updates.items():
            f.write(f"  {key} = {value}\n")
        f.write("="*80 + "\n\n")

        # 收集所有测试集名称
        all_test_names = set()
        for s in all_summaries:
            all_test_names.update(s['test_results'].keys())
        test_names = sorted(all_test_names)

        # 准备表格数据
        headers = ["Experiment", "Active Experts", "Best Avg AUC", "Best Epoch"] + [f"{n}_AUC" for n in test_names]
        col_widths = [len(h) for h in headers]
        rows_data = []
        for s in all_summaries:
            row = [
                s['exp_name'],
                str(s['active_experts']),
                f"{s['best_avg_auc']:.4f}",
                str(s['best_epoch'])
            ]
            for name in test_names:
                auc_val = s['test_results'].get(name, {}).get('auc', 0.0)
                row.append(f"{auc_val:.4f}")
            rows_data.append(row)
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        # 写入表头
        f.write(format_table_row(headers, col_widths) + "\n")
        f.write("-" * (sum(col_widths) + 3 * (len(headers)-1)) + "\n")
        # 写入数据行
        for row in rows_data:
            f.write(format_table_row(row, col_widths) + "\n")
        f.write("\n")

        # 计算相对 baseline (无专家) 的性能变化
        baseline = next((s for s in all_summaries if s['exp_name'] == 'no_experts'), None)
        if baseline:
            baseline_auc = baseline['best_avg_auc']
            f.write("Performance difference relative to 'no_experts' (baseline):\n")
            for s in all_summaries:
                if s['exp_name'] != 'no_experts':
                    diff = (s['best_avg_auc'] - baseline_auc) / baseline_auc * 100
                    f.write(f"  {s['exp_name']:15s}: {diff:+6.2f}%  (AUC {s['best_avg_auc']:.4f} vs {baseline_auc:.4f})\n")
            f.write("\n")

        # 额外列出每个测试集的详细 AUC
        f.write("Detailed per-dataset AUC:\n")
        for test_name in test_names:
            f.write(f"\n  {test_name}:\n")
            for s in all_summaries:
                auc_val = s['test_results'].get(test_name, {}).get('auc', 0.0)
                f.write(f"    {s['exp_name']:15s}: {auc_val:.4f}\n")

        # 可选：将完整配置参数也写入（从任意一个实验的 config 中获取，这里从第一个实验的 config 字典）
        if all_summaries:
            f.write("\n" + "="*80 + "\n")
            f.write("Full configuration parameters (example from first experiment):\n")
            # 重新读取第一个实验的 config.json 或直接使用 config_updates + 默认值
            # 更简单：打印 config_updates 和几个关键默认参数
            f.write("Config overrides:\n")
            for k,v in config_updates.items():
                f.write(f"  {k}: {v}\n")
            f.write("Other default settings (from BaseConfig):\n")
            default_cfg = BaseConfig()
            for attr in ['lora_rank', 'lora_alpha', 'lora_dropout', 'clip_img_size', 'patch_size',
                         'part_detector_hidden_dim', 'classifier_mid_dim', 'random_erasing_prob']:
                f.write(f"  {attr}: {getattr(default_cfg, attr)}\n")

    print(f"\nSummary report saved to {summary_path}")
    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()