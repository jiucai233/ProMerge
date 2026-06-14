import os
import mujoco
import mujoco.viewer
import time
import numpy as np

class M1LocalSandbox:
    def __init__(self):
        # Load the Franka Emika Panda environment from assets
        xml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'robo_assets', 'franka_emika_panda', 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
    def get_privileged_states(self):
        # 9-DOF state vector containing 7 arm joints and 2 finger slide joints
        arm_qpos = np.array([
            self.data.joint('joint1').qpos[0],
            self.data.joint('joint2').qpos[0],
            self.data.joint('joint3').qpos[0],
            self.data.joint('joint4').qpos[0],
            self.data.joint('joint5').qpos[0],
            self.data.joint('joint6').qpos[0],
            self.data.joint('joint7').qpos[0],
            self.data.joint('finger_joint1').qpos[0],
            self.data.joint('finger_joint2').qpos[0]
        ])
        ball_xpos = self.data.body('danger_ball').xpos.copy()
        ball_xvel = self.data.body('danger_ball').cvel[3:6].copy() 
        return arm_qpos, ball_xpos, ball_xvel

    def run_local_generation_loop(self):
        print("🍏 [ProMerge Franka Panda 沙盘] Franka Emika Panda 就位，正在渲染...")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.data.joint('ball_free').qpos[1] = 0.02
            self.data.joint('ball_free').qpos[2] = 0.42
            self.data.joint('ball_free').qvel[0] = -3.8
            self.data.joint('ball_free').qvel[2] = 2.1
            
            step = 0
            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(self.model, self.data)
                
                if step % 20 == 0:
                    qpos, bx, bv = self.get_privileged_states()
                    print(f"🤖 Qpos: {np.round(qpos, 2)} | 🔴 Ball: {np.round(bx, 2)}")
                
                viewer.sync()
                step += 1
                
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

class M1SortingSandbox:
    def __init__(self):
        xml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'robo_assets', 'franka_emika_panda', 'scene_sorting.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
    def get_privileged_states(self):
        arm_qpos = np.array([
            self.data.joint('joint1').qpos[0],
            self.data.joint('joint2').qpos[0],
            self.data.joint('joint3').qpos[0],
            self.data.joint('joint4').qpos[0],
            self.data.joint('joint5').qpos[0],
            self.data.joint('joint6').qpos[0],
            self.data.joint('joint7').qpos[0],
            self.data.joint('finger_joint1').qpos[0],
            self.data.joint('finger_joint2').qpos[0]
        ])
        ball_pos = self.data.body('danger_ball').xpos.copy()
        box_pos = self.data.body('blue_box').xpos.copy()
        cylinder_pos = self.data.body('green_cylinder').xpos.copy()
        return arm_qpos, ball_pos, box_pos, cylinder_pos

if __name__ == "__main__":
    sandbox = M1LocalSandbox()
    sandbox.run_local_generation_loop()
