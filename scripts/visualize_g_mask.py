import os
import sys
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

# Setup paths
project_root = "/Users/jiucai/my_codes/ProMerge"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

import config
from config import CONFIG, POLICY_CONFIG
from utils import make_policy, get_norm_stats

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="PROMERGE_ONLY", choices=["PROMERGE_ONLY", "PROMERGE_FILM"])
    args = parser.parse_args()
    variant_name = args.variant

    # Force use of selected variant for setup
    CONFIG["variant"] = config.PolicyVariant[variant_name]
    
    # Set hidden_dim, dim_feedforward and backbone based on variant
    if CONFIG["variant"] == config.PolicyVariant.PROMERGE_FILM:
        POLICY_CONFIG['hidden_dim'] = 384
        POLICY_CONFIG['dim_feedforward'] = 1536
        CONFIG["backbone"] = "vit_small"
    else:
        POLICY_CONFIG['hidden_dim'] = 384
        POLICY_CONFIG['dim_feedforward'] = 1536
        CONFIG["backbone"] = "vit_small"

    POLICY_CONFIG['camera_names'] = ['front', 'wrist']
    POLICY_CONFIG['state_dim'] = 9
    POLICY_CONFIG['action_dim'] = 9

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    print(f"Visualizing variant: {variant_name}")

    # Build policy
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
    sys.argv = original_argv

    # Load weights
    checkpoint_path = f"/Users/jiucai/my_codes/ProMerge/checkpoints/{variant_name}/policy_last.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    # Load dataset sample
    dataset_dir = "/Users/jiucai/my_codes/ProMerge/data"
    dataset_path = os.path.join(dataset_dir, "episodes_500_tuple.hdf5")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading data from {dataset_path}...")
    norm_stats = get_norm_stats(dataset_dir, 200)

    # Inspect episode 0
    with h5py.File(dataset_path, 'r') as root:
        ep_group = root['episode_0']
        qpos_all = ep_group['qpos'][()]
        images_all = ep_group['images'][()] # [L, 2, 3, 480, 640]
        episode_len = qpos_all.shape[0]

    print(f"Episode length: {episode_len}")

    # Timesteps to visualize
    timesteps = [20, 100, 180]
    
    fig, axes = plt.subplots(len(timesteps), 4, figsize=(20, 12))
    plt.suptitle("ProMerge Kinematic-Guided Token Gate Mask Visualization (g)", fontsize=18, fontweight='bold', y=0.98)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    for i, t in enumerate(timesteps):
        if t >= episode_len:
            print(f"Skipping timestep {t} as it exceeds episode length {episode_len}")
            continue

        qpos_raw = qpos_all[t]
        qpos_norm = (qpos_raw - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
        qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(device)

        images_raw = images_all[t] # [2, 3, 480, 640]
        images_tensor = torch.from_numpy(images_raw).float() / 255.0
        
        target_size = CONFIG.get("image_size")
        if target_size is not None and target_size != (480, 640):
            images_tensor = F.interpolate(
                images_tensor,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            
        images_tensor = images_tensor.unsqueeze(0).to(device)

        # Normalize image tensor
        normalized_images = normalize(images_tensor)

        # Generate fixed slow_semantic (zeros or random with seed)
        torch.manual_seed(42)
        slow_semantic = torch.randn(1, 512).to(device)

        # Run forward pass
        with torch.no_grad():
            _ = policy.model(qpos_tensor, normalized_images, None, slow_semantic=slow_semantic)

        # Get gate mask: [B, 600, 1]
        g_mask = policy.model.gatekeeper.last_g_hybrid.squeeze(0).squeeze(-1) # [600]
        
        # Reshape to [15, 40]
        g_grid = g_mask.reshape(15, 40)
        g_front = g_grid[:, :20]
        g_wrist = g_grid[:, 20:]

        # Upscale to [480, 640] for overlay
        g_front_large = F.interpolate(g_front.unsqueeze(0).unsqueeze(0), size=(480, 640), mode='bilinear', align_corners=False).squeeze().numpy()
        g_wrist_large = F.interpolate(g_wrist.unsqueeze(0).unsqueeze(0), size=(480, 640), mode='bilinear', align_corners=False).squeeze().numpy()

        # Transpose raw images for plotting [3, 480, 640] -> [480, 640, 3]
        img_front = images_raw[0].transpose(1, 2, 0)
        img_wrist = images_raw[1].transpose(1, 2, 0)

        # Plot Front Raw
        axes[i, 0].imshow(img_front)
        axes[i, 0].set_title(f"T={t} | Front Camera Raw", fontsize=11, fontweight='bold')
        axes[i, 0].axis('off')

        # Plot Front Mask Overlay
        axes[i, 1].imshow(img_front)
        overlay_f = axes[i, 1].imshow(g_front_large, cmap='jet', alpha=0.55, vmin=0, vmax=1)
        axes[i, 1].set_title(f"T={t} | Front Gate Overlay", fontsize=11, fontweight='bold')
        axes[i, 1].axis('off')
        fig.colorbar(overlay_f, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Plot Wrist Raw
        axes[i, 2].imshow(img_wrist)
        axes[i, 2].set_title(f"T={t} | Wrist Camera Raw", fontsize=11, fontweight='bold')
        axes[i, 2].axis('off')

        # Plot Wrist Mask Overlay
        axes[i, 3].imshow(img_wrist)
        overlay_w = axes[i, 3].imshow(g_wrist_large, cmap='jet', alpha=0.55, vmin=0, vmax=1)
        axes[i, 3].set_title(f"T={t} | Wrist Gate Overlay", fontsize=11, fontweight='bold')
        axes[i, 3].axis('off')
        fig.colorbar(overlay_w, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_filename = "g_mask_visualization.png"
    
    # Save to variant subfolder in results
    result_subfolder = os.path.join(project_root, "result", variant_name)
    os.makedirs(result_subfolder, exist_ok=True)
    result_output_path = os.path.join(result_subfolder, output_filename)
    plt.savefig(result_output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {result_output_path}")
    
    project_output_path = os.path.join(project_root, output_filename)
    plt.savefig(project_output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {project_output_path}")

    artifacts_dir = "/Users/jiucai/.gemini/antigravity-ide/brain/45d1f158-cffc-4176-9c13-283ec5553bfe"
    if os.path.exists(artifacts_dir):
        artifacts_output_path = os.path.join(artifacts_dir, output_filename)
        plt.savefig(artifacts_output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to artifacts directory: {artifacts_output_path}")

        # Also save to subfolder in artifacts directory
        artifacts_subfolder = os.path.join(artifacts_dir, "result", variant_name)
        os.makedirs(artifacts_subfolder, exist_ok=True)
        plt.savefig(os.path.join(artifacts_subfolder, output_filename), dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()
