import torch
import os

class Config:
    # 训练数据路径
    train_fake_dirs = [
        "/root/autodl-tmp/newmethods/FF1FAKE/Deepfakes/c23/frames",
        "/root/autodl-tmp/newmethods/FF1FAKE/Face2Face/c23/frames",
        "/root/autodl-tmp/newmethods/FF1FAKE/FaceSwap/c23/frames",
        "/root/autodl-tmp/newmethods/FF1FAKE/NeuralTextures/c23/frames",
    ]
    train_real_dirs = [
        "/root/autodl-tmp/newmethods/FF1REAL/youtube/c23/frames",
    ]

    # 测试数据集（请根据实际路径修改）
    test_datasets = [
        {"name": "CDFv1",
         "real_dirs": ["/root/autodl-tmp/newmethods/Celebv1-real/frames","/root/autodl-tmp/newmethods/YouTube-real/frames"],
         "fake_dirs": ["/root/autodl-tmp/newmethods/Celebv1-synthesis/frames"]},
        {"name": "DFD",
         "real_dirs": ["/root/autodl-tmp/newmethods/FF1REAL/actors/c23/frames"],
         "fake_dirs": ["/root/autodl-tmp/newmethods/FF1FAKE/DeepFakeDetection/c23/frames"]},
        {"name": "CDFv2", "real_dirs": ["/root/autodl-tmp/newmethods/CDFv2total/real"], "fake_dirs": ["/root/autodl-tmp/newmethods/CDFv2total/fake"]},
        {"name": "DFDC", "real_dirs": ["/root/autodl-tmp/newmethods/DFDCtotal/real"], "fake_dirs": ["/root/autodl-tmp/newmethods/DFDCtotal/fake"]},
        {"name": "DFDCP", "real_dirs": ["/root/autodl-tmp/newmethods/DFDCPtotal/real"], "fake_dirs": ["/root/autodl-tmp/newmethods/DFDCPtotal/fake"]},
    ]

    train_sample_ratio = 1.0
    test_sample_ratio = 1.0
    recursive_search = True

    batch_size = 16
    num_workers = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.00005
    weight_decay = 0.0005
    epochs = 1
    save_dir = "./checkpoints_random128"

    clip_img_size = 224
    clip_model_name = "ViT-L/14"
    patch_size = 14
    num_patches = (clip_img_size // patch_size) ** 2   # 256

    lora_rank = 16
    lora_alpha = 16
    lora_dropout = 0.1

    part_detector_hidden_dim = 256
    part_detector_epochs = 10
    part_detector_lr = 0.001
    part_detector_batch_size = 64

    # 81点关键点部位映射（索引从0开始，共81个点）
    # 五个部位：轮廓、眉毛、眼睛、鼻子、嘴巴
    part_landmark_indices = {
        "contour": list(range(0, 17)),                       # 下颌线
        "eyebrows": list(range(17, 27)),                     # 眉毛（17-21左眉，22-26右眉）
        "eyes": list(range(36, 48)),                         # 眼睛（36-41左眼，42-47右眼）
        "nose": list(range(27, 36)),                         # 鼻子
        "mouth": list(range(48, 68)),                        # 嘴巴
    }
    # 将未列出的剩余索引（68-80）归入轮廓，确保所有81个点都有归属
    remaining = list(range(68, 81))
    part_landmark_indices["contour"].extend(remaining)

    classifier_mid_dim = 512
    seed = 42

    # ========== 随机擦除参数 ==========
    random_erasing_prob = 0.5          # 随机擦除的概率
    random_erasing_scale = (0.02, 0.2) # 擦除区域面积占原图的比例范围
    random_erasing_ratio = (0.5, 2.0)  # 擦除区域的宽高比范围
    random_erasing_value = 'random'    # 擦除区域填充值：'random' 或 0-255 整数

    def __init__(self):
        os.makedirs(self.save_dir, exist_ok=True)