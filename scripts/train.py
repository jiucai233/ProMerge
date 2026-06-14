import os
import sys
import time
import argparse
import torch
import numpy as np
import wandb

# Add src and project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Import config and utils
import config
from config import CONFIG, POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from utils import load_data, make_policy, make_optimizer

def run_online_evaluation(policy, device, norm_stats):
    from sim.sandbox import M1LocalSandbox
    import mujoco
    
    sandbox = M1LocalSandbox()
    num_rollouts = 5
    max_steps = 400
    success_count = 0
    min_distances = []
    
    # Configure parameters
    num_cameras = CONFIG["num_cameras"]
    camera_names = ['front', 'wrist'][:num_cameras]
    
    # Save original training mode
    was_training = policy.training
    policy.eval()
    
    home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    
    query_frequency = 1 if POLICY_CONFIG.get('temporal_agg', False) else 100
    
    # Generate constant slow_semantic globally for the evaluation
    slow_semantic = None
    if CONFIG["variant"] == config.PolicyVariant.PROMERGE_FILM:
        state = torch.random.get_rng_state()
        torch.manual_seed(42)
        slow_semantic = torch.randn(1, 512).to(device)
        torch.random.set_rng_state(state)
        
    for r in range(num_rollouts):
        # Reset sandbox with randomized initial ball trajectory (similar to dataset generator)
        sandbox.data.time = 0
        mujoco.mj_resetData(sandbox.model, sandbox.data)
        
        sandbox.data.qpos[:9] = home_qpos
        sandbox.data.ctrl[:9] = home_qpos
        
        sandbox.data.joint('ball_free').qpos[0] = 1.5
        sandbox.data.joint('ball_free').qpos[1] = np.random.uniform(-0.12, 0.12)
        sandbox.data.joint('ball_free').qpos[2] = np.random.uniform(0.38, 0.48)
        
        sandbox.data.joint('ball_free').qvel[0] = np.random.uniform(-4.0, -3.4)
        sandbox.data.joint('ball_free').qvel[1] = np.random.uniform(-0.2, 0.2)
        sandbox.data.joint('ball_free').qvel[2] = np.random.uniform(1.9, 2.3)
        
        mujoco.mj_forward(sandbox.model, sandbox.data)
        
        if hasattr(policy.model, 'gatekeeper') and hasattr(policy.model.gatekeeper, 'reset_history'):
            policy.model.gatekeeper.reset_history()
            
        rollout_min_dist = float('inf')
        action_chunk = None
        if POLICY_CONFIG.get('temporal_agg', False):
            all_time_actions = np.zeros([max_steps, max_steps + 100, CONFIG['qpos_dim']])
        
        for step in range(max_steps):
            arm_qpos, ball_xpos, ball_xvel = sandbox.get_privileged_states()
            
            # Distance between catcher (site) and ball
            disk_pos = sandbox.data.site('catcher_site').xpos
            dist = np.linalg.norm(disk_pos - ball_xpos)
            rollout_min_dist = min(rollout_min_dist, dist)
            
            # Execute policy action query
            if step % query_frequency == 0:
                if not hasattr(sandbox, 'renderer'):
                    import mujoco
                    sandbox.renderer = mujoco.Renderer(sandbox.model, height=480, width=640)
                
                images_list = []
                for cam_name in camera_names:
                    sandbox.renderer.update_scene(sandbox.data, camera=cam_name)
                    rgb_img = sandbox.renderer.render()
                    img_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1) / 255.0
                    images_list.append(img_tensor)
                
                images_tensor = torch.stack(images_list).to(device)
                
                target_size = CONFIG.get("image_size")
                if target_size is not None and target_size != (480, 640):
                    images_tensor = torch.nn.functional.interpolate(
                        images_tensor,
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )
                
                images_tensor = images_tensor.unsqueeze(0)
                # Normalize qpos before policy pass
                qpos_norm = (arm_qpos - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
                qpos_input = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
                
                # Use pre-computed slow_semantic vector
                with torch.no_grad():
                    action_normalized = policy(qpos_input, images_tensor, slow_semantic=slow_semantic)
                
                # Unnormalize action output
                action_unnorm = action_normalized[0].cpu().numpy() * norm_stats["action_std"] + norm_stats["action_mean"]
                
                if POLICY_CONFIG.get('temporal_agg', False):
                    all_time_actions[step, step:step+100] = action_unnorm
                else:
                    action_chunk = action_unnorm # [100, 9]
            
            # Get action for current step
            if POLICY_CONFIG.get('temporal_agg', False):
                s_start = max(0, step - 99)
                s_end = step + 1
                actions_at_step = all_time_actions[s_start:s_end, step]
                ages = np.arange(step - s_start, -1, -1)
                k = 0.01
                weights = np.exp(-k * ages)
                weights = weights / np.sum(weights)
                target_qpos = np.sum(actions_at_step * weights[:, np.newaxis], axis=0)
            else:
                target_qpos = action_chunk[step % 100]
            
            # Apply physically via PD control
            sandbox.data.ctrl[:9] = target_qpos[:9]
            
            mujoco.mj_step(sandbox.model, sandbox.data)
            
        min_distances.append(rollout_min_dist)
        if rollout_min_dist < 0.115:
            success_count += 1
            
    success_rate = success_count / num_rollouts
    avg_min_distance = np.mean(min_distances)
    
    # Restore original training mode
    if was_training:
        policy.train()
        
    return success_rate, avg_min_distance

def run_training():
    parser = argparse.ArgumentParser(description="ACT Policy Training Script")
    parser.add_argument("--epochs", type=int, default=CONFIG.get("num_epochs", 300), help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/default", help="Directory to save checkpoints")
    parser.add_argument("--variant", type=str, default=None, help="Policy variant name")
    args = parser.parse_args()

    # Apply command line overrides
    num_epochs = args.epochs
    batch_size = args.batch_size
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if args.variant is not None:
        from config import PolicyVariant
        try:
            CONFIG["variant"] = PolicyVariant[args.variant]
        except KeyError:
            print(f"Error: Invalid policy variant name '{args.variant}'. Valid variants: {[v.name for v in PolicyVariant]}")
            sys.exit(1)
            
    # Configure policy parameters to align with 5-DOF Sandbox
    num_cameras = CONFIG["num_cameras"]
    camera_names = ['front', 'wrist'][:num_cameras]
    
    POLICY_CONFIG['camera_names'] = camera_names
    POLICY_CONFIG['state_dim'] = CONFIG['qpos_dim']
    POLICY_CONFIG['action_dim'] = CONFIG['qpos_dim']
    
    device = config.device
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Policy Variant: {CONFIG['variant'].name}")
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    # Standard Aloha uses 500 episodes, but we regenerated 50 episodes to contain full visual observations
    dataset_dir = os.path.join(project_root, 'data')
    num_episodes = CONFIG.get("num_episodes", 200)
    
    try:
        train_dataloader, val_dataloader, norm_stats, is_sim = load_data(
            dataset_dir, num_episodes, camera_names, batch_size, batch_size
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}", file=sys.stderr)
        raise e

    # Build policy and optimizer
    # Clear sys.argv to avoid nested argparse conflict in build_ACT_model_and_optimizer
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    
    policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
    policy.to(device)
    optimizer = make_optimizer(POLICY_CONFIG['policy_class'], policy)
    
    sys.argv = original_argv
    
    # Initialize wandb
    try:
        wandb.init(
            project="ProMerge",
            name=f"{CONFIG['variant'].name}_bs{batch_size}",
            group=CONFIG['variant'].name,
            config={
                "variant": CONFIG["variant"].name,
                "batch_size": batch_size,
                "epochs": num_epochs,
                "device": device,
                "keep_ratio": CONFIG["keep_ratio"],
                "num_cameras": CONFIG["num_cameras"],
                "qpos_dim": CONFIG["qpos_dim"]
            },
            anonymous="allow"
        )
        print("🚀 Weights & Biases (wandb) initialized successfully.")
    except Exception as e:
        print(f"⚠️ Failed to initialize wandb: {e}. Running without wandb.")
    
    loss_log_path = os.path.join(checkpoint_dir, "loss_log.txt")
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        policy.train()
        epoch_total_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, qpos, actions, is_pad, slow_semantic) in enumerate(train_dataloader):
            images = images.to(device)
            qpos = qpos.to(device)
            actions = actions.to(device)
            is_pad = is_pad.to(device)
            
            # Optional VLA intention embedding for variant 5
            if CONFIG["variant"] == config.PolicyVariant.PROMERGE_FILM:
                slow_semantic = slow_semantic.to(device)
            else:
                slow_semantic = None
            
            try:
                loss_dict = policy(qpos, images, actions, is_pad, slow_semantic=slow_semantic)
                
                optimizer.zero_grad()
                loss_dict['loss'].backward()
                optimizer.step()
                
                # Accumulate
                epoch_total_loss += loss_dict['loss'].item()
                epoch_l1_loss += loss_dict['l1'].item()
                epoch_kl_loss += loss_dict['kl'].item()
                num_batches += 1
                
                print(f"  Batch {batch_idx + 1}/{len(train_dataloader)} | Loss: {loss_dict['loss'].item():.4f}", flush=True)
                
            except RuntimeError as e:
                # Catch MPS / CUDA Out of Memory errors or other MPS issues
                err_str = str(e).lower()
                if "out of memory" in err_str or "oom" in err_str or "mps" in err_str or "alloc" in err_str:
                    print(f"\n❌ OUT OF MEMORY (OOM) / Device allocation error during epoch {epoch}: {e}", file=sys.stderr)
                    # Exit with specific OOM code 137 to signal OOM to the master coordinator
                    sys.exit(137)
                else:
                    print(f"\n❌ RuntimeError during epoch {epoch}: {e}", file=sys.stderr)
                    raise e
                    
        # Validation loop
        policy.eval()
        val_total_loss = 0.0
        val_l1_loss = 0.0
        val_kl_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_images, val_qpos, val_actions, val_is_pad, val_slow_semantic in val_dataloader:
                val_images = val_images.to(device)
                val_qpos = val_qpos.to(device)
                val_actions = val_actions.to(device)
                val_is_pad = val_is_pad.to(device)
                
                # Optional VLA intention embedding for variant 5
                if CONFIG["variant"] == config.PolicyVariant.PROMERGE_FILM:
                    val_slow_semantic = val_slow_semantic.to(device)
                else:
                    val_slow_semantic = None
                
                val_loss_dict = policy(val_qpos, val_images, val_actions, val_is_pad, slow_semantic=val_slow_semantic)
                
                val_total_loss += val_loss_dict['loss'].item()
                val_l1_loss += val_loss_dict['l1'].item()
                val_kl_loss += val_loss_dict['kl'].item()
                val_batches += 1
                
                print(f"  Val Batch {val_batches}/{len(val_dataloader)} | Loss: {val_loss_dict['loss'].item():.4f}", flush=True)

        avg_val_loss = val_total_loss / val_batches if val_batches > 0 else 0.0
        avg_val_l1 = val_l1_loss / val_batches if val_batches > 0 else 0.0
        avg_val_kl = val_kl_loss / val_batches if val_batches > 0 else 0.0

        # Log epoch level losses
        if num_batches > 0:
            avg_loss = epoch_total_loss / num_batches
            avg_l1 = epoch_l1_loss / num_batches
            avg_kl = epoch_kl_loss / num_batches
            
            log_str = f"Epoch {epoch:03d} | Total_Loss: {avg_loss:.6f} | L1_Loss: {avg_l1:.6f} | KL_Loss: {avg_kl:.6f} | Val_Loss: {avg_val_loss:.6f}"
            print(log_str)
            
            # Append log to loss_log.txt
            with open(loss_log_path, "a") as f:
                f.write(log_str + "\n")
                
            # Log metrics to wandb
            try:
                if wandb.run is not None:
                    log_data = {
                        "epoch": epoch,
                        "loss/total": avg_loss,
                        "loss/l1": avg_l1,
                        "loss/kl": avg_kl,
                        "val_loss/total": avg_val_loss,
                        "val_loss/l1": avg_val_l1,
                        "val_loss/kl": avg_val_kl,
                    }
                    
                    # Periodic Online Sandbox Evaluation
                    # Run every 10 epochs (epoch is 0-indexed, so 9, 19, 29, 39, 49)
                    if (epoch + 1) % 10 == 0:
                        print(f"Running online sandbox evaluation at epoch {epoch}...")
                        success_rate, avg_min_dist = run_online_evaluation(policy, device, norm_stats)
                        print(f"📊 Online Eval | Success Rate: {success_rate:.2f} | Avg Min Distance: {avg_min_dist:.4f}m")
                        log_data["eval/success_rate"] = success_rate
                        log_data["eval/min_distance"] = avg_min_dist
                        
                    wandb.log(log_data)
            except Exception as e:
                print(f"⚠️ Error in logging/evaluation: {e}")
                
        # Save checkpoint at the end of each epoch
        ckpt_path = os.path.join(checkpoint_dir, "policy_last.ckpt")
        torch.save(policy.state_dict(), ckpt_path)
        
        # Save named checkpoint periodically (every 50 epochs and at the final epoch)
        if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            named_ckpt_name = f"{CONFIG['variant'].name}_v3_epoch{epoch:03d}_{timestamp}.ckpt"
            named_ckpt_path = os.path.join(checkpoint_dir, named_ckpt_name)
            torch.save(policy.state_dict(), named_ckpt_path)
            print(f"💾 Saved periodic checkpoint to {named_ckpt_path}")
        
    print("Training Completed Successfully!")
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

if __name__ == "__main__":
    run_training()
