import os
import sys
import time
import numpy as np
import argparse
import mujoco
import mujoco.viewer

# Ensure src and root are in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from sim.sandbox import M1LocalSandbox
from data.data_generation import Oracle5DOFInterceptor,Oracle9DOFInterceptor

def visualize_expert_live(num_episodes=10, task_name="dyn_intercept"):
    sandbox = M1LocalSandbox()
    planner = Oracle9DOFInterceptor()
    
    print("====================================================")
    print("📺 Launching MuJoCo Passive Viewer for Expert Policy")
    print(f"Task Name: {task_name}")
    print(f"Episodes to watch: {num_episodes}")
    print("====================================================")
    
    with mujoco.viewer.launch_passive(sandbox.model, sandbox.data) as viewer:
        for ep in range(num_episodes):
            if not viewer.is_running():
                break
                
            print(f"🎬 Episode {ep + 1}/{num_episodes}")
            
            # Reset MuJoCo physics state
            sandbox.data.time = 0
            mujoco.mj_resetData(sandbox.model, sandbox.data)
            
            home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
            sandbox.data.qpos[:9] = home_qpos
            sandbox.data.ctrl[:9] = home_qpos
            
            # Replicate domain randomization from data_generation.py
            if task_name == "dyn_intercept":
                sandbox.data.joint('ball_free').qpos[0] = 1.5
                sandbox.data.joint('ball_free').qpos[1] = np.random.uniform(-0.12, 0.12)
                sandbox.data.joint('ball_free').qpos[2] = np.random.uniform(0.38, 0.48)
                
                sandbox.data.joint('ball_free').qvel[0] = np.random.uniform(-4.0, -3.4)
                sandbox.data.joint('ball_free').qvel[1] = np.random.uniform(-0.2, 0.2)
                sandbox.data.joint('ball_free').qvel[2] = np.random.uniform(1.9, 2.3)
            elif task_name == "static_manipulation":
                sandbox.data.joint('ball_free').qpos[0] = np.random.uniform(0.38, 0.42)
                sandbox.data.joint('ball_free').qpos[1] = np.random.uniform(-0.05, 0.05)
                sandbox.data.joint('ball_free').qpos[2] = np.random.uniform(0.40, 0.45)
                sandbox.data.joint('ball_free').qvel[:] = 0.0
            
            mujoco.mj_forward(sandbox.model, sandbox.data)
            viewer.sync()
            
            # Run 400 steps physical simulation rollout
            for step in range(400):
                step_start = time.time()
                
                if not viewer.is_running():
                    break
                    
                arm_qpos, ball_xpos, ball_xvel = sandbox.get_privileged_states()
                
                # Get expert action
                action = planner.compute_action(arm_qpos, ball_xpos, ball_xvel)
                
                # Set PD controller command target physically
                sandbox.data.ctrl[:9] = action[:9]
                
                mujoco.mj_step(sandbox.model, sandbox.data)
                viewer.sync()
                
                # Sleep to match physics simulation speed (0.002s per step)
                elapsed = time.time() - step_start
                sleep_time = sandbox.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Brief pause between episodes
            if viewer.is_running():
                time.sleep(0.5)
                
    print("✨ Expert visualization completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MuJoCo Expert Trajectories")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize.")
    parser.add_argument("--task", type=str, default="dyn_intercept", choices=["dyn_intercept", "static_manipulation"],
                        help="Task configuration matching dataset generation.")
    args = parser.parse_args()
    
    visualize_expert_live(num_episodes=args.episodes, task_name=args.task)
