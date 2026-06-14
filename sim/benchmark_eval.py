import os
import sys
import time
import argparse
import numpy as np
import torch

# Ensure src and root are in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Import config first to set up environment variables
import config
from config import CONFIG, PolicyVariant, EvalNoise, POLICY_CONFIG
from utils import make_policy, get_norm_stats
from sim.sandbox import M1LocalSandbox, M1SortingSandbox

def apply_lighting_noise(image_tensor):
    """
    输入图像张量维度: [Num_Cameras, C, H, W]，值区间 [0, 1]
    """
    if CONFIG["eval_noise"] == EvalNoise.NONE:
        return image_tensor
        
    elif CONFIG["eval_noise"] == EvalNoise.FLICKER:
        # 模拟现实流水线日光灯的高频强非线性闪烁
        # 随机施加一个全局亮度滑移增益
        gain = np.random.uniform(0.3, 1.8)
        return torch.clamp(image_tensor * gain, 0.0, 1.0)
        
    elif CONFIG["eval_noise"] == EvalNoise.LOCAL_SHADOW:
        # 模拟机械臂移动时自身投射在衣服/方块上的非线性移动斑驳阴影
        _, _, H, W = image_tensor.shape
        mask = torch.ones_like(image_tensor)
        # 随机在半边画布上砸下一个大面积的线性衰减阴影
        shadow_boundary = np.random.randint(int(H*0.2), int(H*0.8))
        mask[:, :, shadow_boundary:, :] *= np.random.uniform(0.2, 0.5)
        return torch.clamp(image_tensor * mask, 0.0, 1.0)

    return image_tensor

def run_evaluation(selected_variant=None, task_name="static_manipulation", noise_type="NONE", num_rollouts=50, render=False, merge_tokens_override="default"):
    print("====================================================")
    print("🚀 Starting ProMerge Embodied AI Policy Evaluation")
    print(f"Task Name: {task_name}")
    print(f"Noise Type: {noise_type}")
    print(f"Num Rollouts: {num_rollouts}")
    print("====================================================")
    
    if merge_tokens_override != "default":
        CONFIG["merge_tokens"] = (merge_tokens_override == "True")
        print(f"🔧 Overrode CONFIG['merge_tokens'] to: {CONFIG['merge_tokens']}")
        
    device = config.device
    print(f"Using Device: {device}")
    
    # Configure policy dimensions to align with 5-DOF Sandbox
    num_cameras = CONFIG["num_cameras"]
    camera_names = ['front', 'wrist'][:num_cameras]
    
    POLICY_CONFIG['camera_names'] = camera_names
    POLICY_CONFIG['state_dim'] = CONFIG['qpos_dim']
    POLICY_CONFIG['action_dim'] = CONFIG['qpos_dim']
    
    # Set eval noise dynamically in global config
    CONFIG["eval_noise"] = EvalNoise[noise_type]
    
    # Initialize simulator sandbox (headless rollout)
    if task_name == "multi_object_sorting":
        sandbox = M1SortingSandbox()
    else:
        sandbox = M1LocalSandbox()
    results = {}
    
    # Load dataset normalization stats
    dataset_dir = os.path.join(project_root, 'data')
    num_episodes = CONFIG.get("num_episodes", 200)
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    
    viewer = None
    if render:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sandbox.model, sandbox.data)
        print("📺 MuJoCo passive viewer launched. Realtime rendering is enabled.")
    
    # Determine which variants to run
    if selected_variant is not None:
        variants_to_run = [selected_variant]
    else:
        variants_to_run = list(PolicyVariant)
        
    # Iterate through selected Policy Variants
    for variant in variants_to_run:
        CONFIG["variant"] = variant
        
        # Dynamically set hidden_dim, dim_feedforward and backbone based on variant to support baseline weights (ResNet18, 512) and ProMerge weights (ViT-Small, 384)
        if variant.name in ["MONOLITHIC_ACT", "RANDOM_PRUNE", "TOME_CLUSTERING"]:
            POLICY_CONFIG['hidden_dim'] = 512
            POLICY_CONFIG['dim_feedforward'] = 3200
            CONFIG["backbone"] = "resnet18"
        else:
            POLICY_CONFIG['hidden_dim'] = 384
            POLICY_CONFIG['dim_feedforward'] = 1536
            CONFIG["backbone"] = "vit_small"
            
        # Reset sys.argv to clear command line arguments before policy initialization
        # to avoid conflict with the nested argparse parser inside detr/main.py
        original_argv = list(sys.argv)
        sys.argv = [sys.argv[0]]
        
        # Initialize policy architecture
        policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
        policy.to(device)
        
        # Load checkpoint weights if available
        checkpoint_path = f"checkpoints/{variant.name}/policy_last.ckpt"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            policy.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        else:
            print(f"WARNING: Checkpoint {checkpoint_path} not found! Using random weights.")
            
        policy.eval()
        sys.argv = original_argv
        
        # Warmup runs to avoid GPU/MPS trace overhead in metrics
        print(f"Warming up {variant.name} JIT compiler...")
        img_h, img_w = CONFIG.get("image_size", (480, 640))
        dummy_img = torch.randn(num_cameras, 3, img_h, img_w).to(device)
        dummy_qpos = torch.randn(1, CONFIG['qpos_dim']).to(device)
        dummy_slow_semantic = torch.randn(1, 512).to(device) if variant == PolicyVariant.PROMERGE_FILM else None
        for _ in range(5):
            with torch.no_grad():
                if dummy_slow_semantic is not None:
                    _ = policy.model(dummy_qpos, dummy_img.unsqueeze(0), None, slow_semantic=dummy_slow_semantic)
                else:
                    _ = policy(dummy_qpos, dummy_img.unsqueeze(0))
        
        latencies = []
        success_count = 0
        rollout_jitters = []
        
        # Generate constant slow_semantic globally for the evaluation
        slow_semantic = None
        if variant == PolicyVariant.PROMERGE_FILM and task_name != "multi_object_sorting":
            state = torch.random.get_rng_state()
            torch.manual_seed(42)
            slow_semantic = torch.randn(1, 512).to(device)
            torch.random.set_rng_state(state)
            
        query_frequency = 1 if POLICY_CONFIG.get('temporal_agg', False) else 100
        print(f"Evaluating Variant: {variant.name} across {num_rollouts} rollouts...")
        for r in range(num_rollouts):
            if viewer is not None and not viewer.is_running():
                print("Viewer closed by user. Exiting evaluation loop.")
                break
            # Reset simulator for each rollout
            sandbox.data.time = 0
            import mujoco
            mujoco.mj_resetData(sandbox.model, sandbox.data)
            
            home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
            sandbox.data.qpos[:9] = home_qpos
            sandbox.data.ctrl[:9] = home_qpos
            
            # For multi_object_sorting, slow_semantic updates dynamically per rollout based on task_id
            if task_name == "multi_object_sorting" and variant == PolicyVariant.PROMERGE_FILM:
                task_id = r % 3
                seed_val = 42 if task_id == 0 else 1000 + task_id
                state = torch.random.get_rng_state()
                torch.manual_seed(seed_val)
                slow_semantic = torch.randn(1, 512).to(device)
                torch.random.set_rng_state(state)
            
            # Configure task specific sandbox states
            if task_name == "dyn_intercept":
                sandbox.data.joint('ball_free').qpos[0] = 1.5
                sandbox.data.joint('ball_free').qpos[1] = np.random.uniform(-0.24, 0.24)
                sandbox.data.joint('ball_free').qpos[2] = np.random.uniform(0.33, 0.53)
                
                sandbox.data.joint('ball_free').qvel[0] = np.random.uniform(-4.3, -3.1)
                sandbox.data.joint('ball_free').qvel[1] = np.random.uniform(-0.4, 0.4)
                sandbox.data.joint('ball_free').qvel[2] = np.random.uniform(1.7, 2.5)
            elif task_name == "static_manipulation":
                sandbox.data.joint('ball_free').qpos[0] = np.random.uniform(0.38, 0.42)
                sandbox.data.joint('ball_free').qpos[1] = np.random.uniform(-0.05, 0.05)
                sandbox.data.joint('ball_free').qpos[2] = np.random.uniform(0.40, 0.45)
                # Zero velocity
                sandbox.data.joint('ball_free').qvel[:] = 0.0
            elif task_name == "multi_object_sorting":
                task_id = r % 3
                tx = np.random.uniform(0.38, 0.45)
                ty = np.random.uniform(-0.12, 0.12)
                tz = np.random.uniform(0.08, 0.15)
                
                ox1, oy1, oz1 = 0.7, 0.3, 0.04
                ox2, oy2, oz2 = 0.7, -0.3, 0.04
                
                if task_id == 0:
                    sandbox.data.joint('ball_free').qpos[0:3] = [tx, ty, tz]
                    sandbox.data.joint('box_free').qpos[0:3] = [ox1, oy1, oz1]
                    sandbox.data.joint('cylinder_free').qpos[0:3] = [ox2, oy2, oz2]
                elif task_id == 1:
                    sandbox.data.joint('box_free').qpos[0:3] = [tx, ty, tz]
                    sandbox.data.joint('ball_free').qpos[0:3] = [ox1, oy1, oz1]
                    sandbox.data.joint('cylinder_free').qpos[0:3] = [ox2, oy2, oz2]
                else:
                    sandbox.data.joint('cylinder_free').qpos[0:3] = [tx, ty, tz]
                    sandbox.data.joint('ball_free').qpos[0:3] = [ox1, oy1, oz1]
                    sandbox.data.joint('box_free').qpos[0:3] = [ox2, oy2, oz2]
                
                sandbox.data.joint('ball_free').qvel[:] = 0.0
                sandbox.data.joint('box_free').qvel[:] = 0.0
                sandbox.data.joint('cylinder_free').qvel[:] = 0.0
            
            # Compute forward kinematics to populate initial body positions correctly
            mujoco.mj_forward(sandbox.model, sandbox.data)
            
            if viewer is not None:
                viewer.sync()
                
            if hasattr(policy.model, 'gatekeeper') and hasattr(policy.model.gatekeeper, 'reset_history'):
                policy.model.gatekeeper.reset_history()
            
            rollout_min_dist = float('inf')
            action_chunk = None
            occlusion_commands = []
            if POLICY_CONFIG.get('temporal_agg', False):
                all_time_actions = np.zeros([400, 400 + 100, CONFIG['qpos_dim']])
            
            # Run 400 steps physical simulation rollout
            for step in range(400):
                step_start = time.time()
                
                if task_name == "multi_object_sorting":
                    arm_qpos, ball_pos, box_pos, cylinder_pos = sandbox.get_privileged_states()
                    task_id = r % 3
                    if task_id == 0:
                        target_xpos = ball_pos
                    elif task_id == 1:
                        target_xpos = box_pos
                    else:
                        target_xpos = cylinder_pos
                else:
                    arm_qpos, ball_xpos, ball_xvel = sandbox.get_privileged_states()
                    target_xpos = ball_xpos
                
                # Distance between catcher (site) and target
                disk_pos = sandbox.data.site('catcher_site').xpos
                dist = np.linalg.norm(disk_pos - target_xpos)
                rollout_min_dist = min(rollout_min_dist, dist)
                
                # Query policy
                if step % query_frequency == 0:
                    # Simulated environment observation query via MuJoCo Renderer
                    if not hasattr(sandbox, 'renderer'):
                        import mujoco
                        sandbox.renderer = mujoco.Renderer(sandbox.model, height=480, width=640)
                    
                    images_list = []
                    for cam_name in camera_names:
                        sandbox.renderer.update_scene(sandbox.data, camera=cam_name)
                        rgb_img = sandbox.renderer.render() # [480, 640, 3] (uint8)
                        # Normalize and permute to [3, H, W]
                        img_tensor = torch.from_numpy(rgb_img).float().permute(2, 0, 1) / 255.0
                        images_list.append(img_tensor)
                    
                    images_tensor = torch.stack(images_list).to(device)
                    images_polluted = apply_lighting_noise(images_tensor)
                    # Downsample dynamically if target size is not (480, 640)
                    target_size = CONFIG.get("image_size")
                    if target_size is not None and target_size != (480, 640):
                        images_polluted = torch.nn.functional.interpolate(
                            images_polluted,
                            size=target_size,
                            mode='bilinear',
                            align_corners=False
                        )
                    images_input = images_polluted.unsqueeze(0)  # [1, num_cameras, 3, H, W]
                    # Normalize qpos before policy pass
                    qpos_norm = (arm_qpos - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
                    qpos_input = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
                    # Use the pre-computed fixed slow_semantic vector
                    start_time = time.time()
                    with torch.no_grad():
                        action_normalized = policy(qpos_input, images_input, slow_semantic=slow_semantic)
                    
                    latency = (time.time() - start_time) * 1000.0  # in ms
                    latencies.append(latency)
                    
                    # Unnormalize action output
                    action_unnorm = action_normalized[0].cpu().numpy() * norm_stats["action_std"] + norm_stats["action_mean"]
                    
                    if POLICY_CONFIG.get('temporal_agg', False):
                        all_time_actions[step, step:step+100] = action_unnorm
                    else:
                        action_chunk = action_unnorm
                
                # Map control command back to physics simulator step
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
                
                # Record commands during occlusion phase for Jitter Variance calculation
                if dist < 0.05:
                    occlusion_commands.append(target_qpos[:7].copy())
                
                # Use standard actuator control (PD controller in M1LocalSandbox)
                sandbox.data.ctrl[:9] = target_qpos[:9]
                
                # Physical time stepping
                mujoco.mj_step(sandbox.model, sandbox.data)
                
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    # Sleep to match physics simulation timestep (realtime)
                    elapsed = time.time() - step_start
                    sleep_time = sandbox.model.opt.timestep - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
            if len(occlusion_commands) > 1:
                diffs = np.diff(np.array(occlusion_commands), axis=0)
                jitter_val = np.mean(np.var(diffs, axis=0))
                rollout_jitters.append(jitter_val)
                
            if rollout_min_dist < 0.115:
                success_count += 1
                
        success_rate = (success_count / num_rollouts) * 100.0
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        mean_jitter = np.mean(rollout_jitters) if rollout_jitters else 0.0
        
        # Calculate Avg and Max Hz
        hzs = [1000.0 / l for l in latencies if l > 0]
        avg_hz = np.mean(hzs) if hzs else 0.0
        max_hz = np.max(hzs) if hzs else 0.0
        
        suffix = ""
        if variant.name in ["TOME_CLUSTERING", "PROMERGE_ONLY", "PROMERGE_FILM"]:
            suffix = "_TOME" if CONFIG.get("merge_tokens", True) else "_PRUNE"
        results[variant.name + suffix] = (success_rate, avg_latency, std_latency, avg_hz, max_hz, mean_jitter)
        
    if viewer is not None:
        viewer.close()
        
    print("\n" + "="*115)
    print(f"📊 EVALUATION REPORT (Task: {task_name}, Noise: {noise_type})")
    print("="*115)
    print(f"{'Policy Variant':<20} | {'Success Rate (%)':<16} | {'Avg Latency (ms)':<16} | {'Std Dev (ms)':<12} | {'Avg Hz':<10} | {'Max Hz':<10} | {'Jitter (x10^-4)':<16}")
    print("-"*115)
    for name, (sr, avg, std, avg_hz, max_hz, jitter) in results.items():
        print(f"{name:<20} | {sr:<16.2f} | {avg:<16.2f} | {std:<12.2f} | {avg_hz:<10.2f} | {max_hz:<10.2f} | {jitter * 10000.0:<16.4f}")
    print("="*115)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embodied AI Policy Evaluation and Latency Benchmarking")
    parser.add_argument("--variant", type=str, default=None, choices=[v.name for v in PolicyVariant],
                        help="Specific policy variant to run. If not set, evaluates all variants sequentially.")
    parser.add_argument("--task", type=str, default="static_manipulation",
                        help="Task name to run simulation under.")
    parser.add_argument("--noise", type=str, default="NONE", choices=[n.name for n in EvalNoise],
                        help="Evaluation noise category to apply on observations.")
    parser.add_argument("--num_rollouts", type=int, default=50,
                        help="Number of evaluation rollouts to run.")
    parser.add_argument("--render", action="store_true",
                        help="Visualize physical simulation rollouts in MuJoCo viewer.")
    parser.add_argument("--merge_tokens", type=str, default="default", choices=["default", "True", "False"],
                        help="Override merge_tokens configuration.")
    args = parser.parse_args()
    
    selected_var = None
    if args.variant is not None:
        selected_var = PolicyVariant[args.variant]
        
    run_evaluation(
        selected_variant=selected_var, 
        task_name=args.task, 
        noise_type=args.noise, 
        num_rollouts=args.num_rollouts,
        render=args.render,
        merge_tokens_override=args.merge_tokens
    )
