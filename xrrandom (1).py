"""
实验脚本：针对 Random Erasing 参数进行网格搜索。
每组参数独立训练，保存模型、日志和配置文件副本。
"""

import os
import sys
import copy
import itertools
import subprocess
from datetime import datetime

# 导入原始 Config 和 Trainer
from config import Config
from train import Trainer, set_seed

def run_single_experiment(exp_name, erasing_params, base_save_root):
    """
    运行单组实验
    exp_name: 实验标识符，用于创建子目录
    erasing_params: dict，包含 random_erasing_prob, random_erasing_scale, random_erasing_ratio, random_erasing_value
    base_save_root: 所有实验的根保存目录
    """
    # 创建该实验的保存目录
    exp_save_dir = os.path.join(base_save_root, exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    
    # 复制配置文件到实验目录（便于后续查看）
    config_copy_path = os.path.join(exp_save_dir, "config_used.py")
    with open(config_copy_path, 'w') as f:
        f.write(f"# Experiment: {exp_name}\n")
        f.write(f"# Erasing params: {erasing_params}\n")
        f.write("# Full Config object will be logged in training_log.txt\n")
        f.write("# This is a snapshot of the used parameters.\n\n")
        # 写入原始 Config 类的代码（简化：仅记录关键参数）
        f.write("import torch\n")
        f.write("class Config:\n")
        for attr, value in Config.__dict__.items():
            if not attr.startswith('_') and not callable(value):
                f.write(f"    {attr} = {repr(value)}\n")
        f.write("\n")
        f.write("# Overridden erasing params:\n")
        for k, v in erasing_params.items():
            f.write(f"Config.{k} = {repr(v)}\n")
    
    # 修改配置
    cfg = Config()
    for k, v in erasing_params.items():
        setattr(cfg, k, v)
    # 修改保存目录
    cfg.save_dir = exp_save_dir
    # 可选的：为每个实验设置不同的随机种子以保证可重复性（可选）
    # 这里使用原始种子，保证除了擦除参数外其他一致
    # 注意：随机擦除本身带有随机性，但固定种子后不同实验仍然可比
    
    # 检查数据目录
    if not any(os.path.exists(d) for d in cfg.train_real_dirs) or not any(
            os.path.exists(d) for d in cfg.train_fake_dirs):
        print(f"Error: Training data directories missing for experiment {exp_name}. Skipping.")
        return
    
    print(f"\n{'='*80}")
    print(f"Starting experiment: {exp_name}")
    print(f"Parameters: {erasing_params}")
    print(f"Save dir: {cfg.save_dir}")
    print(f"{'='*80}\n")
    
    # 实例化 Trainer 并开始训练
    trainer = Trainer(cfg)
    trainer.train()
    
    print(f"Finished experiment: {exp_name}\n")

def main():
    # 定义要搜索的随机擦除参数组合
    # 可以根据需要调整网格
    prob_values = [0.1, 0.3, 0.5, 0.7]          # 擦除概率
    scale_values = [(0.01, 0.1), (0.02, 0.2), (0.03, 0.3)]  # 擦除面积比例范围
    ratio_values = [(0.3, 3.3), (0.5, 2.0)]      # 擦除宽高比范围
    value_options = ['random']                   # 填充值，可扩展为 [0, 128, 255]
    
    # 生成所有组合（笛卡尔积）
    all_combinations = list(itertools.product(prob_values, scale_values, ratio_values, value_options))
    
    # 构建实验名称列表
    experiments = []
    for prob, scale, ratio, value in all_combinations:
        # 生成易读的名称
        scale_str = f"{scale[0]}_{scale[1]}".replace('.', '_')
        ratio_str = f"{ratio[0]}_{ratio[1]}".replace('.', '_')
        exp_name = f"prob_{prob}_scale_{scale_str}_ratio_{ratio_str}_value_{value}"
        params = {
            'random_erasing_prob': prob,
            'random_erasing_scale': scale,
            'random_erasing_ratio': ratio,
            'random_erasing_value': value
        }
        experiments.append((exp_name, params))
    
    # 设置实验根目录
    base_save_root = "./random_erasing_experiments2"
    os.makedirs(base_save_root, exist_ok=True)
    
    # 记录总览日志
    overview_log = os.path.join(base_save_root, "experiments_overview.txt")
    with open(overview_log, 'w') as f:
        f.write(f"Random Erasing Parameter Search\n")
        f.write(f"Start time: {datetime.now()}\n")
        f.write(f"Total experiments: {len(experiments)}\n")
        f.write("\nExperiment list:\n")
        for exp_name, params in experiments:
            f.write(f"  {exp_name}: {params}\n")
    
    # 依次运行每个实验
    for idx, (exp_name, params) in enumerate(experiments):
        print(f"\n>>> Running experiment {idx+1}/{len(experiments)}: {exp_name}")
        run_single_experiment(exp_name, params, base_save_root)
    
    print("\nAll experiments completed.")
    print(f"Results saved under {base_save_root}")

if __name__ == "__main__":
    main()