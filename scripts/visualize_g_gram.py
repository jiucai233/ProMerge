import os
import sys
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
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

    variant = config.PolicyVariant[variant_name]
    checkpoint_path = f"/Users/jiucai/my_codes/ProMerge/checkpoints/{variant_name}/policy_last.ckpt"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    CONFIG["variant"] = variant
    POLICY_CONFIG['camera_names'] = ['front', 'wrist']
    POLICY_CONFIG['state_dim'] = 9
    POLICY_CONFIG['action_dim'] = 9

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    print(f"Visualizing G-Gram for variant: {variant_name}")

    # Build policy
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
    sys.argv = original_argv

    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    # Load dataset sample
    dataset_dir = "/Users/jiucai/my_codes/ProMerge/data"
    dataset_path = os.path.join(dataset_dir, "episodes_500_tuple.hdf5")
    norm_stats = get_norm_stats(dataset_dir, 200)

    # Read episode 0
    with h5py.File(dataset_path, 'r') as root:
        ep_group = root['episode_0']
        qpos_all = ep_group['qpos'][()]
        images_all = ep_group['images'][()] # [L, 2, 3, 480, 640]
        episode_len = qpos_all.shape[0]

    print(f"Episode length: {episode_len}")

    front_ggram = []
    wrist_ggram = []
    timesteps = list(range(0, episode_len, 2)) # Sample every 2 steps

    print("Extracting gate masks across episode...")
    for t in timesteps:
        qpos_raw = qpos_all[t]
        qpos_norm = (qpos_raw - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
        qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(device)

        images_raw = images_all[t]
        images_tensor = torch.from_numpy(images_raw).float() / 255.0
        
        # Resize images to CONFIG["image_size"] if different from 480x640
        target_size = CONFIG.get("image_size")
        if target_size is not None and target_size != (480, 640):
            images_tensor = torch.nn.functional.interpolate(
                images_tensor,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            
        images_tensor = images_tensor.unsqueeze(0).to(device)

        torch.manual_seed(42)
        slow_semantic = torch.randn(1, 512).to(device)

        with torch.no_grad():
            if variant == config.PolicyVariant.PROMERGE_FILM:
                _ = policy.model(qpos_tensor, images_tensor, None, slow_semantic=slow_semantic)
            else:
                _ = policy.model(qpos_tensor, images_tensor, None)

        g_mask = policy.model.gatekeeper.last_g_hybrid.squeeze(0).squeeze(-1).numpy() # [600]
        front_ggram.append(g_mask[:300])
        wrist_ggram.append(g_mask[300:])

    # Transpose to [300, Num_Steps] for standard spectrogram format
    front_ggram = np.stack(front_ggram).T
    wrist_ggram = np.stack(wrist_ggram).T

    print(f"Generating G-Gram plot...")
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    plt.suptitle(f"ProMerge G-Gram Spectrogram ({variant.name})", fontsize=16, fontweight='bold')

    # Front Camera G-Gram
    im0 = axes[0].imshow(front_ggram, cmap='viridis', aspect='auto', origin='lower',
                         extent=[0, episode_len, 0, 300], vmin=0, vmax=1)
    axes[0].set_ylabel("Front Token Index (0-299)", fontsize=12, fontweight='bold')
    axes[0].set_title("Front Camera G-Gram over Time", fontsize=13, fontweight='bold')
    fig.colorbar(im0, ax=axes[0], label="Gate Value (g)")

    # Wrist Camera G-Gram
    im1 = axes[1].imshow(wrist_ggram, cmap='viridis', aspect='auto', origin='lower',
                         extent=[0, episode_len, 0, 300], vmin=0, vmax=1)
    axes[1].set_xlabel("Episode Timestep", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Wrist Token Index (0-299)", fontsize=12, fontweight='bold')
    axes[1].set_title("Wrist Camera G-Gram over Time", fontsize=13, fontweight='bold')
    fig.colorbar(im1, ax=axes[1], label="Gate Value (g)")

    plt.tight_layout()
    output_filename = "promerge_g_gram.png"
    
    # Save to variant subfolder in results
    result_subfolder = os.path.join(project_root, "result", variant_name)
    os.makedirs(result_subfolder, exist_ok=True)
    result_output_path = os.path.join(result_subfolder, output_filename)
    plt.savefig(result_output_path, dpi=150, bbox_inches='tight')
    print(f"Saved G-Gram to {result_output_path}")
    
    project_output_path = os.path.join(project_root, output_filename)
    plt.savefig(project_output_path, dpi=150, bbox_inches='tight')
    print(f"Saved G-Gram to {project_output_path}")

    artifacts_dir = "/Users/jiucai/.gemini/antigravity-ide/brain/45d1f158-cffc-4176-9c13-283ec5553bfe"
    if os.path.exists(artifacts_dir):
        artifacts_output_path = os.path.join(artifacts_dir, output_filename)
        plt.savefig(artifacts_output_path, dpi=150, bbox_inches='tight')
        print(f"Saved G-Gram to artifacts directory: {artifacts_output_path}")

        # Also save to subfolder in artifacts directory
        artifacts_subfolder = os.path.join(artifacts_dir, "result", variant_name)
        os.makedirs(artifacts_subfolder, exist_ok=True)
        plt.savefig(os.path.join(artifacts_subfolder, output_filename), dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()
