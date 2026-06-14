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

def load_policy_and_get_masks(variant, checkpoint_path, timesteps, qpos_all, images_all, norm_stats, device):
    # Setup configuration
    CONFIG["variant"] = variant
    POLICY_CONFIG['camera_names'] = ['front', 'wrist']
    POLICY_CONFIG['state_dim'] = 9
    POLICY_CONFIG['action_dim'] = 9

    # Dynamically set hidden_dim, dim_feedforward and backbone based on variant to support baseline weights (ResNet18, 512) and ProMerge weights (ViT-Small, 384)
    if variant.name in ["MONOLITHIC_ACT", "RANDOM_PRUNE", "TOME_CLUSTERING"]:
        POLICY_CONFIG['hidden_dim'] = 512
        POLICY_CONFIG['dim_feedforward'] = 3200
        CONFIG["backbone"] = "resnet18"
        CONFIG["image_size"] = (480, 640)
    else:
        POLICY_CONFIG['hidden_dim'] = 384
        POLICY_CONFIG['dim_feedforward'] = 1536
        CONFIG["backbone"] = "vit_small"
        CONFIG["image_size"] = (240, 320)

    # Build policy
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
    sys.argv = original_argv

    # Load weights
    print(f"Loading weights for {variant.name} from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    masks_by_t = {}
    
    # Define hooks
    captured_gates = []
    
    hook = None
    if variant == config.PolicyVariant.TOME_CLUSTERING:
        def gatekeeper_hook(module, inputs, outputs):
            # inputs[0] is visual_tokens, shape [B, N, C]
            vt = inputs[0]
            self_attn = torch.bmm(vt, vt.permute(0, 2, 1)) # [B, N, N]
            importance = self_attn.sum(dim=-1).detach().cpu() # [B, N]
            
            # Normalize to [0, 1] for visualization
            imp_min = importance.min(dim=-1, keepdim=True).values
            imp_max = importance.max(dim=-1, keepdim=True).values
            norm_imp = (importance - imp_min) / (imp_max - imp_min + 1e-8)
            captured_gates.append(norm_imp)
        
        hook = policy.model.gatekeeper.register_forward_hook(gatekeeper_hook)

    for t in timesteps:
        qpos_raw = qpos_all[t]
        qpos_norm = (qpos_raw - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
        qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(device)

        images_raw = images_all[t]
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
        normalized_images = normalize(images_tensor)

        torch.manual_seed(42)
        slow_semantic = torch.randn(1, 512).to(device)

        captured_gates.clear()

        with torch.no_grad():
            if variant == config.PolicyVariant.PROMERGE_FILM:
                _ = policy.model(qpos_tensor, normalized_images, None, slow_semantic=slow_semantic)
            else:
                _ = policy.model(qpos_tensor, normalized_images, None)

        if variant == config.PolicyVariant.TOME_CLUSTERING:
            if captured_gates:
                g_mask = captured_gates[0].squeeze(0) # [600]
            else:
                print(f"Warning: No mask captured for {variant.name} at T={t}")
                continue
        else:
            g_mask = policy.model.gatekeeper.last_g_hybrid.squeeze(0).squeeze(-1)

        g_grid = g_mask.reshape(15, 40)
        g_front = g_grid[:, :20]
        g_wrist = g_grid[:, 20:]
        
        # Interpolate to 480x640
        g_front_large = F.interpolate(g_front.unsqueeze(0).unsqueeze(0), size=(480, 640), mode='bilinear', align_corners=False).squeeze().numpy()
        g_wrist_large = F.interpolate(g_wrist.unsqueeze(0).unsqueeze(0), size=(480, 640), mode='bilinear', align_corners=False).squeeze().numpy()
        
        masks_by_t[t] = {
            'front': g_front_large,
            'wrist': g_wrist_large
        }

    if hook is not None:
        hook.remove()
    del policy
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
        
    return masks_by_t

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Load dataset sample
    dataset_dir = "/Users/jiucai/my_codes/ProMerge/data"
    dataset_path = os.path.join(dataset_dir, "episodes_500_tuple.hdf5")
    norm_stats = get_norm_stats(dataset_dir, 200)

    with h5py.File(dataset_path, 'r') as root:
        ep_group = root['episode_0']
        qpos_all = ep_group['qpos'][()]
        images_all = ep_group['images'][()] # [L, 2, 3, 480, 640]
        episode_len = qpos_all.shape[0]

    timesteps = [20, 100, 180]

    # Get masks for each model
    results = {}
    
    # ToMe
    results['TOME_CLUSTERING'] = load_policy_and_get_masks(
        config.PolicyVariant.TOME_CLUSTERING,
        "/Users/jiucai/my_codes/ProMerge/checkpoints/TOME_CLUSTERING/policy_last.ckpt",
        timesteps, qpos_all, images_all, norm_stats, device
    )

    # ProMerge Only
    results['PROMERGE_ONLY'] = load_policy_and_get_masks(
        config.PolicyVariant.PROMERGE_ONLY,
        "/Users/jiucai/my_codes/ProMerge/checkpoints/PROMERGE_ONLY/policy_last.ckpt",
        timesteps, qpos_all, images_all, norm_stats, device
    )

    # ProMerge FiLM
    results['PROMERGE_FILM'] = load_policy_and_get_masks(
        config.PolicyVariant.PROMERGE_FILM,
        "/Users/jiucai/my_codes/ProMerge/checkpoints/PROMERGE_FILM/policy_last.ckpt",
        timesteps, qpos_all, images_all, norm_stats, device
    )

    print("Generating comparison grid...")
    
    # 6 Rows (T=20 Front/Wrist, T=100 Front/Wrist, T=180 Front/Wrist)
    # 4 Columns (Raw, ToMe, ProMerge Only, ProMerge FiLM)
    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    plt.suptitle("Attention / Token Gating Comparison Across Policy Variants", fontsize=20, fontweight='bold', y=0.99)

    row_idx = 0
    for t in timesteps:
        images_raw = images_all[t]
        img_front = images_raw[0].transpose(1, 2, 0)
        img_wrist = images_raw[1].transpose(1, 2, 0)

        # ------------------ FRONT CAMERA ROW ------------------
        # Column 0: Raw
        axes[row_idx, 0].imshow(img_front)
        axes[row_idx, 0].set_title(f"T={t} | Front Raw", fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Column 1: ToMe
        axes[row_idx, 1].imshow(img_front)
        axes[row_idx, 1].imshow(results['TOME_CLUSTERING'][t]['front'], cmap='jet', alpha=0.55, vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f"T={t} | ToMe Attention", fontsize=11, fontweight='bold')
        axes[row_idx, 1].axis('off')

        # Column 2: ProMerge Only
        axes[row_idx, 2].imshow(img_front)
        axes[row_idx, 2].imshow(results['PROMERGE_ONLY'][t]['front'], cmap='jet', alpha=0.55, vmin=0, vmax=1)
        axes[row_idx, 2].set_title(f"T={t} | ProMerge Only Gate", fontsize=11, fontweight='bold')
        axes[row_idx, 2].axis('off')

        # Column 3: ProMerge FiLM
        axes[row_idx, 3].imshow(img_front)
        axes[row_idx, 3].imshow(results['PROMERGE_FILM'][t]['front'], cmap='jet', alpha=0.55, vmin=0, vmax=1)
        axes[row_idx, 3].set_title(f"T={t} | ProMerge FiLM Gate", fontsize=11, fontweight='bold')
        axes[row_idx, 3].axis('off')

        row_idx += 1

        # ------------------ WRIST CAMERA ROW ------------------
        # Column 0: Raw
        axes[row_idx, 0].imshow(img_wrist)
        axes[row_idx, 0].set_title(f"T={t} | Wrist Raw", fontsize=11, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Column 1: ToMe
        axes[row_idx, 1].imshow(img_wrist)
        axes[row_idx, 1].imshow(results['TOME_CLUSTERING'][t]['wrist'], cmap='jet', alpha=0.55, vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f"T={t} | ToMe Attention", fontsize=11, fontweight='bold')
        axes[row_idx, 1].axis('off')

        # Column 2: ProMerge Only
        axes[row_idx, 2].imshow(img_wrist)
        axes[row_idx, 2].imshow(results['PROMERGE_ONLY'][t]['wrist'], cmap='jet', alpha=0.55, vmin=0, vmax=1)
        axes[row_idx, 2].set_title(f"T={t} | ProMerge Only Gate", fontsize=11, fontweight='bold')
        axes[row_idx, 2].axis('off')

        # Column 3: ProMerge FiLM
        axes[row_idx, 3].imshow(img_wrist)
        axes[row_idx, 3].imshow(results['PROMERGE_FILM'][t]['wrist'], cmap='jet', alpha=0.55, vmin=0, vmax=1)
        axes[row_idx, 3].set_title(f"T={t} | ProMerge FiLM Gate", fontsize=11, fontweight='bold')
        axes[row_idx, 3].axis('off')

        row_idx += 1

    plt.tight_layout()
    output_filename = "all_variants_mask_comparison.png"
    
    # Save to result folder
    result_folder = os.path.join(project_root, "result")
    os.makedirs(result_folder, exist_ok=True)
    result_output_path = os.path.join(result_folder, output_filename)
    plt.savefig(result_output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization comparison to {result_output_path}")
    
    project_output_path = os.path.join(project_root, output_filename)
    plt.savefig(project_output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization comparison to {project_output_path}")

    artifacts_dir = "/Users/jiucai/.gemini/antigravity-ide/brain/45d1f158-cffc-4176-9c13-283ec5553bfe"
    if os.path.exists(artifacts_dir):
        artifacts_output_path = os.path.join(artifacts_dir, output_filename)
        plt.savefig(artifacts_output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization comparison to artifacts directory: {artifacts_output_path}")

        # Also save to result folder in artifacts directory
        artifacts_result_folder = os.path.join(artifacts_dir, "result")
        os.makedirs(artifacts_result_folder, exist_ok=True)
        plt.savefig(os.path.join(artifacts_result_folder, output_filename), dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()
