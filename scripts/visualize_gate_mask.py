#!/usr/bin/env python3
"""
Visualize the PerceptualGatekeeper's gate mask (g) from a trained PROMERGE_FILM checkpoint.

Generates spatial heatmaps of g_kin, g_vis, and g_hybrid overlaid on original camera images.
Each gate mask [B, N, 1] is reshaped to [B, H, W] per camera view for 2D visualization.

Output: saves a publication-quality figure to scripts/gate_mask_visualization.png
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

import config
from config import CONFIG, POLICY_CONFIG, PolicyVariant
from utils import make_policy, get_norm_stats
import h5py

def load_sample_data(dataset_dir, norm_stats, sample_idx=0, timestep=50):
    """Load a single sample from the HDF5 dataset."""
    dataset_path = os.path.join(dataset_dir, 'episodes_500_tuple.hdf5')
    with h5py.File(dataset_path, 'r') as root:
        ep_group = root[f'episode_{sample_idx}']
        qpos = ep_group['qpos'][timestep]
        images = ep_group['images'][timestep]  # [num_cameras, C, H, W] uint8
    
    # Convert images to float [0,1]
    images_float = torch.from_numpy(images).float() / 255.0  # [num_cam, C, H, W]
    
    # CPU-side resizing to match configured resolution
    from config import CONFIG
    target_size = CONFIG.get("image_size")
    if target_size is not None and target_size != (480, 640):
        images_float = torch.nn.functional.interpolate(
            images_float,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
    
    # Normalize qpos
    qpos_tensor = torch.from_numpy(qpos).float()
    qpos_tensor = (qpos_tensor - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
    
    return images_float, qpos_tensor, images  # also return raw images for overlay


def main():
    # === Configuration ===
    checkpoint_path = os.path.join(project_root, 'checkpoints', 'PROMERGE_FILM', 'policy_last.ckpt')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Training may still be in progress. Checking for available PROMERGE variants...")
        # Fallback to PROMERGE_ONLY if FILM isn't ready yet
        alt_path = os.path.join(project_root, 'checkpoints', 'PROMERGE_ONLY', 'policy_last.ckpt')
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
            print(f"   ✅ Using fallback: {alt_path}")
            CONFIG["variant"] = PolicyVariant.PROMERGE_ONLY
        else:
            print("   ❌ No PROMERGE checkpoints available yet.")
            sys.exit(1)
    else:
        CONFIG["variant"] = PolicyVariant.PROMERGE_FILM
    
    print(f"🔧 Using variant: {CONFIG['variant'].name}")
    print(f"📂 Checkpoint: {checkpoint_path}")
    
    # === Setup model ===
    num_cameras = CONFIG["num_cameras"]
    camera_names = ['front', 'wrist'][:num_cameras]
    POLICY_CONFIG['camera_names'] = camera_names
    POLICY_CONFIG['state_dim'] = CONFIG['qpos_dim']
    POLICY_CONFIG['action_dim'] = CONFIG['qpos_dim']
    
    device = 'cpu'  # Use CPU for visualization to avoid MPS issues
    
    # Clear sys.argv to avoid nested argparse conflict
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    
    policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
    
    sys.argv = original_argv
    
    # Load checkpoint
    print("📥 Loading checkpoint weights...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    policy.load_state_dict(state_dict)
    policy.eval()
    policy.to(device)
    
    # === Load sample data ===
    dataset_dir = os.path.join(project_root, 'data')
    norm_stats = get_norm_stats(dataset_dir, CONFIG.get("num_episodes", 200))
    
    # Try multiple samples for a good visualization
    sample_indices = [0, 5, 10, 25, 42]
    timesteps = [50, 100, 150, 200, 250]
    
    # Pick the first valid sample
    images_float = None
    for s_idx in sample_indices:
        for t_idx in timesteps:
            try:
                images_float, qpos_tensor, raw_images = load_sample_data(
                    dataset_dir, norm_stats, sample_idx=s_idx, timestep=t_idx
                )
                print(f"✅ Loaded sample: episode={s_idx}, timestep={t_idx}")
                break
            except Exception as e:
                continue
        if images_float is not None:
            break
    
    if images_float is None:
        print("❌ Could not load any sample data")
        sys.exit(1)
    
    # Prepare input tensors: [1, num_cam, C, H, W]
    images_input = images_float.unsqueeze(0).to(device)
    qpos_input = qpos_tensor.unsqueeze(0).to(device)
    
    # Optional slow_semantic for PROMERGE_FILM
    slow_semantic = None
    if CONFIG["variant"] == PolicyVariant.PROMERGE_FILM:
        slow_semantic = torch.randn(1, 384).to(device)
    
    # === Forward pass to extract gate masks ===
    print("🔄 Running forward pass...")
    with torch.no_grad():
        if slow_semantic is not None:
            a_hat, _, (mu, logvar) = policy.model(qpos_input, images_input, None, slow_semantic=slow_semantic)
        else:
            a_hat = policy(qpos_input, images_input)
    
    # Extract gate masks from the gatekeeper
    gatekeeper = policy.model.gatekeeper
    g_kin = gatekeeper.last_g_kin  # [1, N, 1]
    g_vis = gatekeeper.last_g_vis  # [1, N, 1]
    g_hybrid = gatekeeper.last_g_hybrid  # [1, N, 1]
    
    print(f"📐 Gate shapes: g_kin={g_kin.shape}, g_vis={g_vis.shape}, g_hybrid={g_hybrid.shape}")
    
    N = g_kin.shape[1]
    n_per_cam = N // num_cameras
    
    # Determine spatial dimensions from ResNet18 downsampling (÷32)
    # Input: 480×640 → Feature: 15×20
    H_feat = 15
    W_feat = 20
    assert n_per_cam == H_feat * W_feat, f"Token count mismatch: {n_per_cam} != {H_feat}×{W_feat}"
    
    print(f"📐 Spatial dims: {H_feat}×{W_feat} per camera, {num_cameras} cameras, {N} total tokens")
    
    # === Reshape to spatial maps ===
    # g shape: [1, N, 1] → split per camera → reshape to [H, W]
    def to_spatial(g_tensor, cam_idx):
        """Extract and reshape gate values for a specific camera to spatial HxW."""
        start = cam_idx * n_per_cam
        end = (cam_idx + 1) * n_per_cam
        cam_g = g_tensor[0, start:end, 0].numpy()  # [n_per_cam]
        return cam_g.reshape(H_feat, W_feat)
    
    # === Custom colormaps ===
    # Cool blue-to-hot-red for kinematic gate
    cmap_kin = LinearSegmentedColormap.from_list('kin_gate', [
        '#0a0a2e', '#1a1a5e', '#2d4f8e', '#3d8eb9', '#5ec4b6', 
        '#a8e06c', '#f0e442', '#f28c28', '#e64545', '#ff2222'
    ])
    # Purple saliency
    cmap_vis = LinearSegmentedColormap.from_list('vis_gate', [
        '#0a0a1a', '#1a0a3e', '#3d1a6e', '#6b2fa0', '#9b4dca',
        '#c77dff', '#e0a8ff', '#f0c8ff', '#ffe0f0', '#ffffff'
    ])
    # Fire gradient for hybrid
    cmap_hybrid = LinearSegmentedColormap.from_list('hybrid_gate', [
        '#000004', '#1b0c41', '#4a0c6b', '#781c6d', '#a52c60',
        '#cf4446', '#ed6925', '#fb9b06', '#f7d13d', '#fcffa4'
    ])
    
    cam_labels = ['Front Camera', 'Wrist Camera'][:num_cameras]
    gate_labels = ['$g_{kin}$ (Kinematic)', '$g_{vis}$ (Visual Saliency)', '$g_{hybrid}$ (Fused Gate)']
    cmaps = [cmap_kin, cmap_vis, cmap_hybrid]
    gates = [g_kin, g_vis, g_hybrid]
    
    # === Generate visualization ===
    fig = plt.figure(figsize=(18, 12), facecolor='#0d1117')
    gs = gridspec.GridSpec(3, num_cameras * 2 + 1, 
                           width_ratios=[1] * (num_cameras * 2) + [0.05],
                           hspace=0.35, wspace=0.15)
    
    fig.suptitle(
        f'ProMerge Gate Mask Visualization — {CONFIG["variant"].name}',
        fontsize=18, fontweight='bold', color='white', y=0.97
    )
    
    for gate_idx, (gate_tensor, gate_label, cmap) in enumerate(zip(gates, gate_labels, cmaps)):
        for cam_idx in range(num_cameras):
            # Column 0,2: raw camera image
            ax_img = fig.add_subplot(gs[gate_idx, cam_idx * 2])
            
            # Get raw image and display (CHW → HWC)
            raw_img = raw_images[cam_idx].transpose(1, 2, 0)  # [H, W, C]
            ax_img.imshow(raw_img)
            ax_img.set_title(f'{cam_labels[cam_idx]}', fontsize=10, color='#8b949e', pad=4)
            ax_img.axis('off')
            
            # Column 1,3: gate heatmap overlay
            ax_heat = fig.add_subplot(gs[gate_idx, cam_idx * 2 + 1])
            
            spatial_gate = to_spatial(gate_tensor, cam_idx)
            
            # Show camera image dimmed as background
            ax_heat.imshow(raw_img, alpha=0.3)
            
            # Overlay gate heatmap (upscaled to image resolution via interpolation)
            from scipy.ndimage import zoom
            gate_upscaled = zoom(spatial_gate, 
                                (raw_img.shape[0] / H_feat, raw_img.shape[1] / W_feat), 
                                order=1)
            im = ax_heat.imshow(gate_upscaled, cmap=cmap, alpha=0.85, vmin=0, vmax=1)
            
            # Add gate statistics as text overlay
            g_mean = spatial_gate.mean()
            g_max = spatial_gate.max()
            g_min = spatial_gate.min()
            stats_text = f'μ={g_mean:.3f}  max={g_max:.3f}'
            ax_heat.text(0.02, 0.96, stats_text, transform=ax_heat.transAxes,
                        fontsize=7, color='white', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            ax_heat.set_title(f'{gate_label}', fontsize=10, color='white', fontweight='bold', pad=4)
            ax_heat.axis('off')
        
        # Colorbar
        cbar_ax = fig.add_subplot(gs[gate_idx, -1])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(colors='#8b949e', labelsize=7)
        cbar.set_label('Gate Value', color='#8b949e', fontsize=8)
    
    # Add keep_ratio annotation
    fig.text(0.5, 0.01, 
             f'keep_ratio = {CONFIG["keep_ratio"]} → Top {int(N * CONFIG["keep_ratio"])}/{N} tokens retained',
             ha='center', fontsize=10, color='#58a6ff',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22', edgecolor='#30363d'))
    
    output_path = os.path.join(project_root, 'scripts', 'gate_mask_visualization.png')
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    
    print(f"\n✅ Visualization saved to: {output_path}")
    
    # === Print gate statistics summary ===
    print("\n" + "="*60)
    print("📊 Gate Mask Statistics Summary")
    print("="*60)
    for gate_tensor, gate_label in zip(gates, gate_labels):
        g = gate_tensor[0, :, 0].numpy()
        print(f"\n  {gate_label}:")
        print(f"    Mean: {g.mean():.4f}  Std: {g.std():.4f}")
        print(f"    Min:  {g.min():.4f}  Max: {g.max():.4f}")
        print(f"    Median: {np.median(g):.4f}")
        # Print per-camera breakdown
        for cam_idx in range(num_cameras):
            cam_g = to_spatial(gate_tensor, cam_idx)
            print(f"    {cam_labels[cam_idx]}: μ={cam_g.mean():.4f}, σ={cam_g.std():.4f}")


if __name__ == "__main__":
    main()
