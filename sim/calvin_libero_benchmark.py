import os
import sys
import time
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer

# Ensure src and root are in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

# Import config and utils
import config
from config import CONFIG, PolicyVariant, EvalNoise, POLICY_CONFIG
from utils import make_policy, get_norm_stats
from sim.sandbox import M1SortingSandbox

def get_geom_id_by_body_name(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body {body_name} not found")
    geom_start = model.body_geomadr[body_id]
    geom_num = model.body_geomnum[body_id]
    if geom_num == 0:
        raise ValueError(f"Body {body_name} has no geoms")
    return geom_start

def generate_task_sequence(seed=None):
    if seed is not None:
        np.random.seed(seed)
    seq = []
    curr = np.random.choice([0, 1, 2])
    seq.append(curr)
    for _ in range(4):
        next_task = np.random.choice([t for t in [0, 1, 2] if t != curr])
        seq.append(next_task)
        curr = next_task
    return seq

def run_calvin_evaluation(policy, device, norm_stats, sandbox, num_rollouts, render, camera_names, num_cameras, slow_semantic_template):
    print(f"\n--- Running CALVIN ABC→D Emulation (LH-1 to LH-5, {num_rollouts} rollouts) ---")
    
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(sandbox.model, sandbox.data)
        
    query_frequency = 1 if POLICY_CONFIG.get('temporal_agg', False) else 100
    scores = []
    
    home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    
    for r in range(num_rollouts):
        # Generate random task sequence
        task_sequence = generate_task_sequence(seed=r * 100)
        
        # Reset simulator
        sandbox.data.time = 0
        mujoco.mj_resetData(sandbox.model, sandbox.data)
        
        # Set to home pose
        sandbox.data.qpos[:9] = home_qpos
        sandbox.data.ctrl[:9] = home_qpos
        
        # Initial random placement of the 3 objects
        # We place them at slightly randomized non-overlapping positions
        pos_noise = np.random.uniform(-0.02, 0.02, (3, 2))
        
        sandbox.data.joint('ball_free').qpos[0:3] = [0.38 + pos_noise[0, 0], 0.10 + pos_noise[0, 1], 0.05]
        sandbox.data.joint('box_free').qpos[0:3] = [0.40 + pos_noise[1, 0], -0.10 + pos_noise[1, 1], 0.05]
        sandbox.data.joint('cylinder_free').qpos[0:3] = [0.45 + pos_noise[2, 0], 0.0 + pos_noise[2, 1], 0.05]
        
        for name in ['ball_free', 'box_free', 'cylinder_free']:
            sandbox.data.joint(name).qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            sandbox.data.joint(name).qvel[:] = 0.0
            
        mujoco.mj_forward(sandbox.model, sandbox.data)
        
        if hasattr(policy.model, 'gatekeeper') and hasattr(policy.model.gatekeeper, 'reset_history'):
            policy.model.gatekeeper.reset_history()
            
        completed_tasks = 0
        
        # Sequentially evaluate the chain
        for t_idx, task_id in enumerate(task_sequence):
            # Reset gate temporal smoothing at every task boundary: the target object
            # changes between sub-tasks, so the previous task's gate must not leak in.
            if hasattr(policy.model, 'gatekeeper') and hasattr(policy.model.gatekeeper, 'reset_history'):
                policy.model.gatekeeper.reset_history()

            # Prepare slow semantic representation
            slow_semantic = None
            if slow_semantic_template is not None:
                seed_val = 42 if task_id == 0 else 1000 + task_id
                state = torch.random.get_rng_state()
                torch.manual_seed(seed_val)
                slow_semantic = torch.randn(1, 512).to(device)
                torch.random.set_rng_state(state)
                
            task_success = False
            action_chunk = None
            if POLICY_CONFIG.get('temporal_agg', False):
                all_time_actions = np.zeros([400, 400 + 100, CONFIG['qpos_dim']])
                
            for step in range(400):
                step_start = time.time()
                
                # Fetch target position
                if task_id == 0:
                    target_xpos = sandbox.data.body('danger_ball').xpos.copy()
                elif task_id == 1:
                    target_xpos = sandbox.data.body('blue_box').xpos.copy()
                else:
                    target_xpos = sandbox.data.body('green_cylinder').xpos.copy()
                    
                disk_pos = sandbox.data.site('catcher_site').xpos
                dist = np.linalg.norm(disk_pos - target_xpos)
                
                if dist < 0.115:
                    task_success = True
                    break
                    
                # Query policy
                if step % query_frequency == 0:
                    if not hasattr(sandbox, 'renderer'):
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
                    images_input = images_tensor.unsqueeze(0)
                    
                    # Normalize qpos
                    arm_qpos = sandbox.get_privileged_states()[0]
                    qpos_norm = (arm_qpos - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
                    qpos_input = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
                    
                    with torch.no_grad():
                        action_normalized = policy(qpos_input, images_input, slow_semantic=slow_semantic)
                        
                    action_unnorm = action_normalized[0].cpu().numpy() * norm_stats["action_std"] + norm_stats["action_mean"]
                    
                    if POLICY_CONFIG.get('temporal_agg', False):
                        all_time_actions[step, step:step+100] = action_unnorm
                    else:
                        action_chunk = action_unnorm
                        
                # Map control command back to physics
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
                    
                sandbox.data.ctrl[:9] = target_qpos[:9]
                mujoco.mj_step(sandbox.model, sandbox.data)
                
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                    elapsed = time.time() - step_start
                    sleep_time = sandbox.model.opt.timestep - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
            if task_success:
                completed_tasks += 1
                print(f"  Rollout {r+1} | Task {t_idx+1}/5 ({'Sphere' if task_id==0 else 'Box' if task_id==1 else 'Cylinder'}) -> SUCCESS!")
            else:
                print(f"  Rollout {r+1} | Task {t_idx+1}/5 ({'Sphere' if task_id==0 else 'Box' if task_id==1 else 'Cylinder'}) -> TIMEOUT/FAILED. Terminating chain.")
                break
                
        scores.append(completed_tasks)
        print(f"Rollout {r+1} Completed. Sequence Score: {completed_tasks}/5")
        
    if viewer is not None:
        viewer.close()
        
    scores = np.array(scores)
    lh1 = np.mean(scores >= 1) * 100.0
    lh2 = np.mean(scores >= 2) * 100.0
    lh3 = np.mean(scores >= 3) * 100.0
    lh4 = np.mean(scores >= 4) * 100.0
    lh5 = np.mean(scores >= 5) * 100.0
    avg_len = np.mean(scores)
    
    return lh1, lh2, lh3, lh4, lh5, avg_len


def run_libero_evaluation(policy, device, norm_stats, sandbox, num_rollouts, render, camera_names, num_cameras, slow_semantic_template, suite_name="spatial"):
    print(f"\n--- Running LIBERO-{suite_name.capitalize()} Emulation (10 tasks, {num_rollouts} rollouts each) ---")
    
    viewer = None
    if render:
        viewer = mujoco.viewer.launch_passive(sandbox.model, sandbox.data)
        
    query_frequency = 1 if POLICY_CONFIG.get('temporal_agg', False) else 100
    home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
    
    # Cache original sizes and colors for restoring
    sphere_geom_id = get_geom_id_by_body_name(sandbox.model, 'danger_ball')
    box_geom_id = get_geom_id_by_body_name(sandbox.model, 'blue_box')
    cylinder_geom_id = get_geom_id_by_body_name(sandbox.model, 'green_cylinder')
    
    orig_sphere_size = sandbox.model.geom_size[sphere_geom_id].copy()
    orig_box_size = sandbox.model.geom_size[box_geom_id].copy()
    orig_cylinder_size = sandbox.model.geom_size[cylinder_geom_id].copy()
    
    orig_sphere_rgba = sandbox.model.geom_rgba[sphere_geom_id].copy()
    orig_box_rgba = sandbox.model.geom_rgba[box_geom_id].copy()
    orig_cylinder_rgba = sandbox.model.geom_rgba[cylinder_geom_id].copy()
    
    orig_sphere_mass = sandbox.model.body_mass[mujoco.mj_name2id(sandbox.model, mujoco.mjtObj.mjOBJ_BODY, 'danger_ball')]
    
    task_success_rates = []
    
    for task_idx in range(10):
        task_id = 0
        # Restore default geom properties first
        sandbox.model.geom_size[sphere_geom_id] = orig_sphere_size
        sandbox.model.geom_size[box_geom_id] = orig_box_size
        sandbox.model.geom_size[cylinder_geom_id] = orig_cylinder_size
        sandbox.model.geom_rgba[sphere_geom_id] = orig_sphere_rgba
        sandbox.model.geom_rgba[box_geom_id] = orig_box_rgba
        sandbox.model.geom_rgba[cylinder_geom_id] = orig_cylinder_rgba
        sandbox.model.body_mass[mujoco.mj_name2id(sandbox.model, mujoco.mjtObj.mjOBJ_BODY, 'danger_ball')] = orig_sphere_mass
        
        # Spatial positions configuration
        s_pos = {}
        
        if suite_name == "spatial":
            # Define 10 spatial relationships
            if task_idx == 0:
                task_id = 0 # sphere center
                s_pos = {'sphere': [0.42, 0.0], 'box': [0.65, 0.2], 'cylinder': [0.65, -0.2]}
            elif task_idx == 1:
                task_id = 0 # sphere right
                s_pos = {'sphere': [0.42, -0.1], 'box': [0.42, 0.1], 'cylinder': [0.65, -0.2]}
            elif task_idx == 2:
                task_id = 0 # sphere left
                s_pos = {'sphere': [0.42, 0.1], 'box': [0.65, 0.2], 'cylinder': [0.42, -0.1]}
            elif task_idx == 3:
                task_id = 1 # box center
                s_pos = {'box': [0.42, 0.0], 'sphere': [0.65, 0.2], 'cylinder': [0.65, -0.2]}
            elif task_idx == 4:
                task_id = 1 # box right
                s_pos = {'box': [0.42, -0.1], 'sphere': [0.42, 0.1], 'cylinder': [0.65, -0.2]}
            elif task_idx == 5:
                task_id = 1 # box left
                s_pos = {'box': [0.42, 0.1], 'sphere': [0.65, 0.2], 'cylinder': [0.42, -0.1]}
            elif task_idx == 6:
                task_id = 2 # cylinder center
                s_pos = {'cylinder': [0.42, 0.0], 'sphere': [0.65, 0.2], 'box': [0.65, -0.2]}
            elif task_idx == 7:
                task_id = 2 # cylinder right
                s_pos = {'cylinder': [0.42, -0.1], 'sphere': [0.42, 0.1], 'box': [0.65, -0.2]}
            elif task_idx == 8:
                task_id = 2 # cylinder left
                s_pos = {'cylinder': [0.42, 0.1], 'sphere': [0.65, 0.2], 'box': [0.42, -0.1]}
            else:
                task_id = 0 # inline
                s_pos = {'sphere': [0.39, 0.0], 'box': [0.44, 0.0], 'cylinder': [0.49, 0.0]}
                
        elif suite_name == "object":
            # Define 10 object variations (using random locations similar to training, but OOD object attributes)
            # Default training positions: Target randomized, obstacles fixed
            task_id = task_idx % 3
            
            # Apply programmatic modifications
            if task_idx == 0:
                task_id = 0
                sandbox.model.geom_size[sphere_geom_id] = orig_sphere_size * 0.7
            elif task_idx == 1:
                task_id = 0
                sandbox.model.geom_size[sphere_geom_id] = orig_sphere_size * 1.3
            elif task_idx == 2:
                task_id = 1
                sandbox.model.geom_size[box_geom_id] = orig_box_size * 0.7
            elif task_idx == 3:
                task_id = 1
                sandbox.model.geom_size[box_geom_id] = orig_box_size * 1.3
            elif task_idx == 4:
                task_id = 2
                sandbox.model.geom_size[cylinder_geom_id] = orig_cylinder_size * 0.7
            elif task_idx == 5:
                task_id = 2
                sandbox.model.geom_size[cylinder_geom_id] = orig_cylinder_size * 1.3
            elif task_idx == 6:
                task_id = 0
                sandbox.model.geom_rgba[sphere_geom_id] = [0.9, 0.9, 0.1, 1.0] # Yellow
            elif task_idx == 7:
                task_id = 1
                sandbox.model.geom_rgba[box_geom_id] = [0.9, 0.1, 0.9, 1.0] # Magenta
            elif task_idx == 8:
                task_id = 2
                sandbox.model.geom_rgba[cylinder_geom_id] = [0.1, 0.9, 0.9, 1.0] # Cyan
            else:
                task_id = 0
                # Double sphere weight
                sandbox.model.body_mass[mujoco.mj_name2id(sandbox.model, mujoco.mjtObj.mjOBJ_BODY, 'danger_ball')] = orig_sphere_mass * 2.0
                
        elif suite_name == "goal":
            # 10 OOD goal configurations (spatial boundaries)
            task_id = task_idx % 3
            if task_idx == 0:
                tx, ty, tz = 0.35, -0.15, 0.08
            elif task_idx == 1:
                tx, ty, tz = 0.35, 0.15, 0.08
            elif task_idx == 2:
                tx, ty, tz = 0.48, -0.15, 0.08
            elif task_idx == 3:
                tx, ty, tz = 0.48, 0.15, 0.08
            elif task_idx == 4:
                tx, ty, tz = 0.41, -0.17, 0.10
            elif task_idx == 5:
                tx, ty, tz = 0.41, 0.17, 0.10
            elif task_idx == 6:
                tx, ty, tz = 0.34, 0.0, 0.12
            elif task_idx == 7:
                tx, ty, tz = 0.49, 0.0, 0.12
            elif task_idx == 8:
                tx, ty, tz = 0.42, 0.0, 0.05
            else:
                tx, ty, tz = 0.42, 0.0, 0.19
                
        elif suite_name == "long":
            # Multi-stage task chains. In LIBERO-Long, a task is completed if all steps in the sequence succeed.
            # We define 10 task chains of length 2 or 3.
            task_id = task_idx % 3
            # We'll run the multi-step evaluation sequence inside the rollout loop instead of single step!
            
        success_count = 0
        
        # Prepare slow semantic representation
        slow_semantic = None
        if slow_semantic_template is not None and suite_name != "long":
            seed_val = 42 if task_id == 0 else 1000 + task_id
            state = torch.random.get_rng_state()
            torch.manual_seed(seed_val)
            slow_semantic = torch.randn(1, 512).to(device)
            torch.random.set_rng_state(state)
            
        for r in range(num_rollouts):
            # Reset simulator
            sandbox.data.time = 0
            mujoco.mj_resetData(sandbox.model, sandbox.data)
            sandbox.data.qpos[:9] = home_qpos
            sandbox.data.ctrl[:9] = home_qpos
            
            # Position objects
            if suite_name == "spatial":
                # Apply s_pos with small noise
                n_x = np.random.uniform(-0.01, 0.01)
                n_y = np.random.uniform(-0.01, 0.01)
                
                sandbox.data.joint('ball_free').qpos[0:3] = [s_pos['sphere'][0] + n_x, s_pos['sphere'][1] + n_y, 0.05]
                sandbox.data.joint('box_free').qpos[0:3] = [s_pos['box'][0] + n_x, s_pos['box'][1] + n_y, 0.05]
                sandbox.data.joint('cylinder_free').qpos[0:3] = [s_pos['cylinder'][0] + n_x, s_pos['cylinder'][1] + n_y, 0.05]
                
            elif suite_name == "object":
                # Standard random placement
                tx = np.random.uniform(0.38, 0.45)
                ty = np.random.uniform(-0.12, 0.12)
                tz = np.random.uniform(0.08, 0.15)
                ox1, oy1, oz1 = 0.60, 0.20, 0.04
                ox2, oy2, oz2 = 0.60, -0.20, 0.04
                
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
                    
            elif suite_name == "goal":
                # Fixed OOD goal position with small noise
                n_x = np.random.uniform(-0.01, 0.01)
                n_y = np.random.uniform(-0.01, 0.01)
                ox1, oy1, oz1 = 0.60, 0.20, 0.04
                ox2, oy2, oz2 = 0.60, -0.20, 0.04
                
                if task_id == 0:
                    sandbox.data.joint('ball_free').qpos[0:3] = [tx + n_x, ty + n_y, tz]
                    sandbox.data.joint('box_free').qpos[0:3] = [ox1, oy1, oz1]
                    sandbox.data.joint('cylinder_free').qpos[0:3] = [ox2, oy2, oz2]
                elif task_id == 1:
                    sandbox.data.joint('box_free').qpos[0:3] = [tx + n_x, ty + n_y, tz]
                    sandbox.data.joint('ball_free').qpos[0:3] = [ox1, oy1, oz1]
                    sandbox.data.joint('cylinder_free').qpos[0:3] = [ox2, oy2, oz2]
                else:
                    sandbox.data.joint('cylinder_free').qpos[0:3] = [tx + n_x, ty + n_y, tz]
                    sandbox.data.joint('ball_free').qpos[0:3] = [ox1, oy1, oz1]
                    sandbox.data.joint('box_free').qpos[0:3] = [ox2, oy2, oz2]
                    
            elif suite_name == "long":
                # Place sphere, box, cylinder randomly
                pos_noise = np.random.uniform(-0.02, 0.02, (3, 2))
                sandbox.data.joint('ball_free').qpos[0:3] = [0.38 + pos_noise[0, 0], 0.10 + pos_noise[0, 1], 0.05]
                sandbox.data.joint('box_free').qpos[0:3] = [0.40 + pos_noise[1, 0], -0.10 + pos_noise[1, 1], 0.05]
                sandbox.data.joint('cylinder_free').qpos[0:3] = [0.45 + pos_noise[2, 0], 0.0 + pos_noise[2, 1], 0.05]
                
            for name in ['ball_free', 'box_free', 'cylinder_free']:
                sandbox.data.joint(name).qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
                sandbox.data.joint(name).qvel[:] = 0.0
                
            mujoco.mj_forward(sandbox.model, sandbox.data)
            
            if hasattr(policy.model, 'gatekeeper') and hasattr(policy.model.gatekeeper, 'reset_history'):
                policy.model.gatekeeper.reset_history()
                
            if suite_name == "long":
                # Evaluate task chain
                if task_idx == 0: chain = [0, 1]
                elif task_idx == 1: chain = [0, 2]
                elif task_idx == 2: chain = [1, 0]
                elif task_idx == 3: chain = [1, 2]
                elif task_idx == 4: chain = [2, 0]
                elif task_idx == 5: chain = [2, 1]
                elif task_idx == 6: chain = [0, 1, 2]
                elif task_idx == 7: chain = [1, 2, 0]
                elif task_idx == 8: chain = [2, 0, 1]
                else: chain = [0, 2, 1]
                
                chain_success = True
                for step_task_id in chain:
                    # Reset gate temporal smoothing at each chain step (target changes).
                    if hasattr(policy.model, 'gatekeeper') and hasattr(policy.model.gatekeeper, 'reset_history'):
                        policy.model.gatekeeper.reset_history()

                    # Semantic representation
                    step_semantic = None
                    if slow_semantic_template is not None:
                        seed_val = 42 if step_task_id == 0 else 1000 + step_task_id
                        state = torch.random.get_rng_state()
                        torch.manual_seed(seed_val)
                        step_semantic = torch.randn(1, 512).to(device)
                        torch.random.set_rng_state(state)
                        
                    task_success = False
                    action_chunk = None
                    if POLICY_CONFIG.get('temporal_agg', False):
                        all_time_actions = np.zeros([400, 400 + 100, CONFIG['qpos_dim']])
                        
                    for step in range(400):
                        step_start = time.time()
                        if step_task_id == 0:
                            target_xpos = sandbox.data.body('danger_ball').xpos.copy()
                        elif step_task_id == 1:
                            target_xpos = sandbox.data.body('blue_box').xpos.copy()
                        else:
                            target_xpos = sandbox.data.body('green_cylinder').xpos.copy()
                            
                        disk_pos = sandbox.data.site('catcher_site').xpos
                        dist = np.linalg.norm(disk_pos - target_xpos)
                        
                        if dist < 0.115:
                            task_success = True
                            break
                            
                        # Query policy
                        if step % query_frequency == 0:
                            if not hasattr(sandbox, 'renderer'):
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
                            images_input = images_tensor.unsqueeze(0)
                            arm_qpos = sandbox.get_privileged_states()[0]
                            qpos_norm = (arm_qpos - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
                            qpos_input = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
                            
                            with torch.no_grad():
                                action_normalized = policy(qpos_input, images_input, slow_semantic=step_semantic)
                            action_unnorm = action_normalized[0].cpu().numpy() * norm_stats["action_std"] + norm_stats["action_mean"]
                            if POLICY_CONFIG.get('temporal_agg', False):
                                all_time_actions[step, step:step+100] = action_unnorm
                            else:
                                action_chunk = action_unnorm
                                
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
                            
                        sandbox.data.ctrl[:9] = target_qpos[:9]
                        mujoco.mj_step(sandbox.model, sandbox.data)
                        
                        if viewer is not None:
                            if not viewer.is_running():
                                break
                            viewer.sync()
                            elapsed = time.time() - step_start
                            sleep_time = sandbox.model.opt.timestep - elapsed
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                                
                    if not task_success:
                        chain_success = False
                        break
                        
                if chain_success:
                    success_count += 1
                    
            else:
                # Single stage OOD/Spatial task
                task_success = False
                action_chunk = None
                if POLICY_CONFIG.get('temporal_agg', False):
                    all_time_actions = np.zeros([400, 400 + 100, CONFIG['qpos_dim']])
                    
                for step in range(400):
                    step_start = time.time()
                    if task_id == 0:
                        target_xpos = sandbox.data.body('danger_ball').xpos.copy()
                    elif task_id == 1:
                        target_xpos = sandbox.data.body('blue_box').xpos.copy()
                    else:
                        target_xpos = sandbox.data.body('green_cylinder').xpos.copy()
                        
                    disk_pos = sandbox.data.site('catcher_site').xpos
                    dist = np.linalg.norm(disk_pos - target_xpos)
                    
                    if dist < 0.115:
                        task_success = True
                        break
                        
                    # Query policy
                    if step % query_frequency == 0:
                        if not hasattr(sandbox, 'renderer'):
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
                        images_input = images_tensor.unsqueeze(0)
                        arm_qpos = sandbox.get_privileged_states()[0]
                        qpos_norm = (arm_qpos - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
                        qpos_input = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
                        
                        with torch.no_grad():
                            action_normalized = policy(qpos_input, images_input, slow_semantic=slow_semantic)
                        action_unnorm = action_normalized[0].cpu().numpy() * norm_stats["action_std"] + norm_stats["action_mean"]
                        if POLICY_CONFIG.get('temporal_agg', False):
                            all_time_actions[step, step:step+100] = action_unnorm
                        else:
                            action_chunk = action_unnorm
                            
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
                        
                    sandbox.data.ctrl[:9] = target_qpos[:9]
                    mujoco.mj_step(sandbox.model, sandbox.data)
                    
                    if viewer is not None:
                        if not viewer.is_running():
                            break
                        viewer.sync()
                        elapsed = time.time() - step_start
                        sleep_time = sandbox.model.opt.timestep - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                            
                if task_success:
                    success_count += 1
                    
        sr = (success_count / num_rollouts) * 100.0
        print(f"  Task {task_idx+1}/10 Success Rate: {sr:.1f}%")
        task_success_rates.append(sr)
        
    if viewer is not None:
        viewer.close()
        
    # Restore default properties
    sandbox.model.geom_size[sphere_geom_id] = orig_sphere_size
    sandbox.model.geom_size[box_geom_id] = orig_box_size
    sandbox.model.geom_size[cylinder_geom_id] = orig_cylinder_size
    sandbox.model.geom_rgba[sphere_geom_id] = orig_sphere_rgba
    sandbox.model.geom_rgba[box_geom_id] = orig_box_rgba
    sandbox.model.geom_rgba[cylinder_geom_id] = orig_cylinder_rgba
    sandbox.model.body_mass[mujoco.mj_name2id(sandbox.model, mujoco.mjtObj.mjOBJ_BODY, 'danger_ball')] = orig_sphere_mass
    
    avg_sr = np.mean(task_success_rates)
    return avg_sr


def run_evaluation(selected_variant, num_calvin_rollouts=10, num_libero_rollouts=3, render=False, merge_tokens_override="default"):
    print("====================================================")
    print("🚀 Starting CALVIN & LIBERO Emulation Evaluation")
    print(f"Variant: {selected_variant.name}")
    print(f"CALVIN Rollouts: {num_calvin_rollouts}")
    print(f"LIBERO Rollouts: {num_libero_rollouts}")
    print("====================================================")
    
    if merge_tokens_override != "default":
        CONFIG["merge_tokens"] = (merge_tokens_override == "True")
        print(f"🔧 Overrode CONFIG['merge_tokens'] to: {CONFIG['merge_tokens']}")
        
    device = config.device
    print(f"Using Device: {device}")
    
    num_cameras = CONFIG["num_cameras"]
    camera_names = ['front', 'wrist'][:num_cameras]
    
    POLICY_CONFIG['camera_names'] = camera_names
    POLICY_CONFIG['state_dim'] = CONFIG['qpos_dim']
    POLICY_CONFIG['action_dim'] = CONFIG['qpos_dim']
    
    # Initialize simulator sandbox
    sandbox = M1SortingSandbox()
    
    # Load normalization stats
    dataset_dir = os.path.join(project_root, 'data')
    num_episodes = CONFIG.get("num_episodes", 200)
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    
    CONFIG["variant"] = selected_variant
    
    # Configure model config parameters
    if selected_variant.name in ["MONOLITHIC_ACT", "RANDOM_PRUNE", "TOME_CLUSTERING"]:
        POLICY_CONFIG['hidden_dim'] = 512
        POLICY_CONFIG['dim_feedforward'] = 3200
        CONFIG["backbone"] = "resnet18"
    else:
        POLICY_CONFIG['hidden_dim'] = 384
        POLICY_CONFIG['dim_feedforward'] = 1536
        CONFIG["backbone"] = "vit_small"
        
    # Clear sys.argv to avoid conflict with nested parser inside detr/main.py
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    
    policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
    policy.to(device)
    
    checkpoint_path = f"checkpoints/{selected_variant.name}/policy_last.ckpt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        policy.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found! Using random weights.")
        
    policy.eval()
    sys.argv = original_argv
    
    # Warmup runs to avoid GPU tracing overhead
    print("Warming up policy...")
    img_h, img_w = CONFIG.get("image_size", (480, 640))
    dummy_img = torch.randn(num_cameras, 3, img_h, img_w).to(device)
    dummy_qpos = torch.randn(1, CONFIG['qpos_dim']).to(device)
    dummy_slow_semantic = torch.randn(1, 512).to(device) if selected_variant == PolicyVariant.PROMERGE_FILM else None
    for _ in range(3):
        with torch.no_grad():
            # Go through the policy wrapper (which applies ImageNet normalization) for
            # both variants so warmup matches the real inference path. slow_semantic is
            # None for non-FILM variants, which the wrapper handles.
            _ = policy(dummy_qpos, dummy_img.unsqueeze(0), slow_semantic=dummy_slow_semantic)
                
    # Run CALVIN
    lh1, lh2, lh3, lh4, lh5, avg_len = run_calvin_evaluation(
        policy, device, norm_stats, sandbox, num_calvin_rollouts, render, camera_names, num_cameras, dummy_slow_semantic
    )
    
    # Run LIBERO Suites
    libero_spatial = run_libero_evaluation(
        policy, device, norm_stats, sandbox, num_libero_rollouts, render, camera_names, num_cameras, dummy_slow_semantic, suite_name="spatial"
    )
    libero_object = run_libero_evaluation(
        policy, device, norm_stats, sandbox, num_libero_rollouts, render, camera_names, num_cameras, dummy_slow_semantic, suite_name="object"
    )
    libero_goal = run_libero_evaluation(
        policy, device, norm_stats, sandbox, num_libero_rollouts, render, camera_names, num_cameras, dummy_slow_semantic, suite_name="goal"
    )
    libero_long = run_libero_evaluation(
        policy, device, norm_stats, sandbox, num_libero_rollouts, render, camera_names, num_cameras, dummy_slow_semantic, suite_name="long"
    )
    libero_avg = (libero_spatial + libero_object + libero_goal + libero_long) / 4.0
    
    suffix = ""
    if selected_variant.name in ["TOME_CLUSTERING", "PROMERGE_ONLY", "PROMERGE_FILM"]:
        suffix = "_TOME" if CONFIG.get("merge_tokens", True) else "_PRUNE"
        
    print("\n" + "="*120)
    print(f"📊 REPORT FOR VARIANT: {selected_variant.name + suffix}")
    print("="*120)
    print(f"CALVIN LH-1: {lh1:.1f}% | LH-2: {lh2:.1f}% | LH-3: {lh3:.1f}% | LH-4: {lh4:.1f}% | LH-5: {lh5:.1f}% | Avg Len: {avg_len:.2f}")
    print(f"LIBERO Spatial: {libero_spatial:.1f}% | Object: {libero_object:.1f}% | Goal: {libero_goal:.1f}% | Long: {libero_long:.1f}% | Avg: {libero_avg:.1f}%")
    print("="*120)
    
    # We output structured text to be parsed easily by pipeline runner
    print(f"METRICS_OUTPUT: {selected_variant.name + suffix} | {lh1:.2f} | {lh2:.2f} | {lh3:.2f} | {lh4:.2f} | {lh5:.2f} | {avg_len:.4f} | {libero_spatial:.2f} | {libero_object:.2f} | {libero_goal:.2f} | {libero_long:.2f} | {libero_avg:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CALVIN and LIBERO local emulation benchmark")
    parser.add_argument("--variant", type=str, required=True, choices=[v.name for v in PolicyVariant],
                        help="Specific policy variant to run.")
    parser.add_argument("--num_calvin_rollouts", type=int, default=10,
                        help="Number of evaluation rollouts for CALVIN.")
    parser.add_argument("--num_libero_rollouts", type=int, default=3,
                        help="Number of evaluation rollouts for each LIBERO task.")
    parser.add_argument("--render", action="store_true",
                        help="Visualize rollouts in MuJoCo viewer.")
    parser.add_argument("--merge_tokens", type=str, default="default", choices=["default", "True", "False"],
                        help="Override merge_tokens configuration.")
    args = parser.parse_args()
    
    selected_var = PolicyVariant[args.variant]
    
    run_evaluation(
        selected_variant=selected_var,
        num_calvin_rollouts=args.num_calvin_rollouts,
        num_libero_rollouts=args.num_libero_rollouts,
        render=args.render,
        merge_tokens_override=args.merge_tokens
    )
