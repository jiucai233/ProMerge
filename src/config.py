import os
import sys as _sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
# Headless cloud GPU box (Linux): use EGL for MuJoCo offscreen rendering so eval /
# online-eval works without a display. No effect on macOS (uses its own GL backend).
if _sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
import torch
from enum import Enum

class PolicyVariant(Enum):
    MONOLITHIC_ACT = 1  # 基准线：全量 Token 计算 (100%)
    RANDOM_PRUNE = 2    # 对照组：随机剪枝 (保留 30%)
    TOME_CLUSTERING = 3 # 竞品组：纯视觉双边聚类 Token Merging (压缩到 30%)
    PROMERGE_ONLY = 4   # 消融组：仅利用 Qpos 余弦打分剪枝 (保留 30%)
    PROMERGE_FILM = 5   # 终极形态：VLA低频语义FiLM调制 + Qpos余弦打分剪枝 (30%)
    THINKPROPRIO = 6    # 竞品论文复现：[指令; 本体感知] 引导的投票式硬剪枝 (Gumbel+STE, ViT)

class EvalNoise(Enum):
    NONE = 0
    FLICKER = 1         # 全局高频闪烁（抗光照测试）
    LOCAL_SHADOW = 2    # 局部非线性阴影

# 核心超参数账本
CONFIG = {
    "variant": PolicyVariant.PROMERGE_FILM,
    "eval_noise": EvalNoise.NONE,
    # keep_ratio overridable via env (PROMERGE_KEEP_RATIO) for the
    # token-compression pareto sweep (success vs. latency at 0.1/0.2/0.3/0.5/0.7).
    "keep_ratio": float(os.environ.get("PROMERGE_KEEP_RATIO", "0.3")),  # Token 预算留存率
    "num_cameras": 2,           # 默认多相机：前视 + 手腕
    "chunk_size": 50,           # ACT 的 Action Chunking 长度
    "qpos_dim": 9,              # 9-DOF 关节状态维度 (7个手臂关节 + 2个手指)
    "future_horizon_predict": True, # 开启未来时空感知视界掩码
    "num_episodes": 50,         # 适应数据量的实际值 50
    "num_epochs": 50,           # 快速演示训练 50 epoch
    "image_size": (240, 320),   # 图像输入分辨率 (height, width)
    # === ProMerge v2: ViT Backbone + Cross-Attention Gate ===
    "backbone": os.environ.get("PROMERGE_BACKBONE", "vit_small"),  # 'resnet18' | 'vit_small' | 'clip_vit'
    "vit_pruning_layer": 6,     # 在 ViT 第几层后做剪枝 (0-indexed)
    "gate_num_queries": 8,      # Cross-attention 查询向量数
    "gate_num_heads": 4,        # Cross-attention 头数
    "merge_tokens": True,       # True: Token Merging (ToMe), False: Token Pruning (Hard Selection)
    # === ThinkProprio baseline (Sec 3.3) ===
    "tp_num_bins": 256,         # 本体感知文本分箱数 (论文 256 bins over [-3,3])
    "tp_alpha_start": 1.0,      # Gumbel 噪声初始温度
    "tp_alpha_end": 0.01,       # Gumbel 噪声终止温度 (cosine 退火)
    "tp_anneal_steps": 5000,    # Gumbel 温度退火的训练步数
    # === Early stopping (used by experiments/_trainer.py) ===
    "early_stop_enabled": True,
    "early_stop_patience": 5,   # 连续多少次验证 (每10个epoch一次) 无提升则停
    "early_stop_min_delta": 1e-3,
    # === DataLoader workers (0 on mac/MPS; set DATALOADER_WORKERS=8 on a cloud GPU box) ===
    "dataloader_workers": int(os.environ.get("DATALOADER_WORKERS", "0")),
}


# --- 基础路径配置 ---
DATA_DIR = 'data/'
CHECKPOINT_DIR = 'checkpoints/'

# --- M1 Pro 硬件自适应流 ---
import multiprocessing
is_main = (multiprocessing.current_process().name == 'MainProcess')

if is_main and torch.backends.mps.is_available():
    device = 'mps'
    print("🔥 成功锁定 M1 Pro GPU 加速管线！")
elif is_main and torch.cuda.is_available():
    device = 'cuda'
    print("🔥 成功锁定 CUDA GPU 加速管线！")
else:
    device = 'cpu'
    if is_main:
        print("⚠️ 警告：当前处于 CPU 盲跑状态，请检查 PyTorch 版本！")
os.environ['DEVICE'] = device

# --- 任务配置 (根据我们的仿真沙盘进行了校准) ---
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 400,        # 对齐沙盘 max_steps
    'state_dim': 2,            # 对应沙盘中的 2个真实关节位置 (arm_j1, arm_j2)
    'action_dim': 2,           # 对应输出的目标关节位置
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'],
    'camera_port': 0
}

# --- 核心网络与我们的 ProMerge 路由器配置 ---
POLICY_CONFIG = {
    'policy_class': 'ACT',     # 保持对 ACT 的兼容
    'lr': 1e-5,
    'device': device,
    'num_queries': 100,        # Chunk size：一次性生成 100 步开环动作
    'kl_weight': 10,
    'hidden_dim': 384,         # 对齐 ViT-Small embed_dim (原 512)
    'dim_feedforward': 1536,   # 4× hidden_dim (原 3200)
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',    # 此处仅作 argparse 默认值，实际由 CONFIG["backbone"] 控制
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,               # 384 / 8 = 48 per head
    'camera_names': ['front'],
    'temporal_agg': True,       # (unused by LIBERO eval — see num_open_loop_steps)
    'num_open_loop_steps': 8,   # LIBERO eval: execute 8 chunk steps then re-query
                                # (OpenVLA-OFT protocol; far better than temporal_agg here)

    # =========================================================================
    # 🚀 PROMERGE EXCLUSIVE HYPERPARAMETERS (我们的独家 IP 控制阀门)
    # =========================================================================
    'use_promerge': True,         # 一键切换：True 激活 ProMerge, False 退回原生 ACT
    'promerge_layer_idx': 2,      # 在 ViT / Backbone 的哪一层执行 Token 劫持融合
    'keep_ratio': 0.3,            # Token 压缩压缩比：只保留 30% 最核心的视觉 Token
    'routing_mode': 'state_guided' # 路由模式: 'state_guided' (我们的创新) | 'uniform' (对比组)
}

# --- 挂机训练配置 ---
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 600,            # 扩展训练时长到 600
    'batch_size_val': 8,
    'batch_size_train': 16,       # M1 Pro 32GB 统一内存，直接拉大 Batch 压榨算力
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}