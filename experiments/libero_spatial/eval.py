"""LIBERO-Spatial official success-rate evaluation (runs on a Linux GPU box).

Requires the `libero` env (robosuite/MuJoCo, EGL). On macOS this won't run — use the
cloud box (see cloud/README_CLOUD.md). Faithful to LIBERO's lifelong/metric.py rollout.

Usage (on the box, after cloud/setup.sh):
    MUJOCO_GL=egl .venv/bin/python experiments/libero_spatial/eval.py \
        --baseline promerge_film --ckpt checkpoints/libero_spatial/promerge_film/policy_best.ckpt \
        --episodes_per_task 20

Reports per-task and suite-average success rate. Uses the SAME normalization stats and
image/qpos/action layout as training (recomputed from the dataset, deterministic).
"""
import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F

from experiments.libero_spatial import config as C


def _build_policy(baseline_name, ckpt, device):
    """Assemble the policy for this baseline + LIBERO-Spatial config and load weights."""
    from experiments._trainer import _apply_baseline
    from experiments.registry import get_experiment
    from config import CONFIG, POLICY_CONFIG
    from utils import make_policy, safe_load_state_dict

    _apply_baseline(baseline_name)                 # model: variant/backbone/dim/ToMe
    get_experiment("libero_spatial").apply(CONFIG, POLICY_CONFIG)  # data dims/cameras/action

    _argv = list(sys.argv); sys.argv = [sys.argv[0]]
    policy = make_policy(POLICY_CONFIG["policy_class"], POLICY_CONFIG).to(device)
    sys.argv = _argv
    safe_load_state_dict(policy, ckpt, device=device)
    policy.eval()
    uses_text = CONFIG["variant"].name in ("PROMERGE_FILM", "THINKPROPRIO")
    return policy, uses_text


def _prep_img(arr):
    """HWC uint8 -> [3,H,W] float in [0,1], flipped/resized exactly like training."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1) / 255.0
    if C.FLIP_IMAGES:                              # opengl -> upright, matches training
        t = torch.flip(t, dims=[1])
    if tuple(t.shape[1:]) != tuple(C.IMAGE_SIZE):
        t = F.interpolate(t.unsqueeze(0), size=C.IMAGE_SIZE, mode="bilinear",
                          align_corners=False).squeeze(0)
    return t


def _obs_key(obs, *cands):
    for k in cands:
        if k in obs:
            return k
    raise KeyError(f"none of {cands} in obs keys {list(obs.keys())}")


def run_libero_spatial_eval(baseline_name, ckpt, episodes_per_task=20, max_steps=520,
                            settle_steps=5, seed=0):
    import config as cfgmod
    from utils import encode_text_instruction
    from experiments.libero_spatial.data import _scan_episodes, _compute_norm_stats
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    device = cfgmod.device
    policy, uses_text = _build_policy(baseline_name, ckpt, device)

    # Same normalization the dataset used (deterministic recompute from the data).
    norm = _compute_norm_stats(_scan_episodes(C.DATA_DIR))
    qm, qs = norm["qpos_mean"], norm["qpos_std"]
    am, as_ = norm["action_mean"], norm["action_std"]

    import os as _os
    _suite_name = _os.environ.get("LIBERO_SUITE", "libero_spatial")
    suite = benchmark.get_benchmark_dict()[_suite_name]()
    n_tasks = suite.get_num_tasks()
    gk = getattr(policy.model, "gatekeeper", None)

    per_task = []
    for i in range(n_tasks):
        task = suite.get_task(i)
        env = OffScreenRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(i),
            camera_heights=128, camera_widths=128,
        )
        env.seed(seed)
        init_states = suite.get_task_init_states(i)
        slow = encode_text_instruction(task.language, device=device).unsqueeze(0) if uses_text else None

        succ = 0
        for ep in range(episodes_per_task):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(settle_steps):                       # settle physics
                obs, _, _, _ = env.step(np.zeros(C.ACTION_DIM))
            if gk is not None and hasattr(gk, "reset_history"):
                gk.reset_history()

            ak = _obs_key(obs, "agentview_image", "agentview_rgb")
            wk = _obs_key(obs, "robot0_eye_in_hand_image", "eye_in_hand_rgb")
            jk = _obs_key(obs, "robot0_joint_pos", "joint_states")
            gkey = _obs_key(obs, "robot0_gripper_qpos", "gripper_states")

            done = False
            # Open-loop action-chunk execution, matching OpenVLA-OFT's LIBERO
            # protocol (num_open_loop_steps): the policy predicts a chunk of
            # future actions, we execute the first `open_loop` of them, then
            # re-query. This MASSIVELY outperforms both naive single-step
            # re-prediction (open_loop=1) and ACT-style temporal aggregation:
            # validated on monolithic_act (3 tasks x 5 ep) -> 20% (single-step),
            # 7% (temporal agg), 53% (open_loop=8). Temporal aggregation is the
            # WRONG choice here: it averages the ±1 gripper signal into ~0 (half-
            # open gripper -> never grasps), and LIBERO is low-freq delta control
            # where averaging successive delta actions cancels real motion.
            from config import POLICY_CONFIG as _PC
            open_loop = int(_PC.get("num_open_loop_steps", 8))
            adim = C.ACTION_DIM
            chunk = None
            ci = 0
            for _t in range(max_steps):
                if chunk is None or ci >= open_loop:
                    images = torch.stack([_prep_img(obs[ak]), _prep_img(obs[wk])], dim=0).unsqueeze(0).to(device)
                    qpos = np.concatenate([obs[jk], obs[gkey]]).astype(np.float32)
                    qpos = ((qpos - qm) / qs)[: C.STATE_DIM]
                    qpos_t = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        a_hat = policy(qpos_t, images, slow_semantic=slow)   # [1, nq, STATE_DIM]
                    chunk = a_hat[0].cpu().numpy()[:, :adim]                 # [nq, adim] normalized
                    ci = 0
                a = chunk[ci] * as_[:adim] + am[:adim]                       # un-normalize
                ci += 1
                obs, reward, done, info = env.step(a[:adim])                 # real 7-D action
                if done:
                    break
            succ += int(done)

        env.close()
        sr = 100.0 * succ / episodes_per_task
        per_task.append(sr)
        print(f"  Task {i+1}/{n_tasks} [{task.language[:50]}] : {sr:.1f}%", flush=True)

    avg = float(np.mean(per_task)) if per_task else 0.0
    print(f"\n📊 LIBERO-Spatial [{baseline_name}] avg success: {avg:.1f}%  | per-task: "
          f"{[round(x,1) for x in per_task]}", flush=True)

    # Log eval results to wandb too (guarded; no-op if wandb unavailable/not logged in).
    try:
        import wandb
        wandb.init(project="ProMerge", name=f"libero_eval_{baseline_name}", group="libero_eval",
                   config={"baseline": baseline_name, "ckpt": ckpt,
                           "episodes_per_task": episodes_per_task})
        wandb.log({"libero_spatial/avg_success": avg,
                   **{f"libero_spatial/task_{i}": s for i, s in enumerate(per_task)}})
        wandb.finish()
    except Exception as e:
        print(f"(wandb eval logging skipped: {e})", flush=True)

    return avg, per_task


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LIBERO-Spatial official success-rate eval")
    ap.add_argument("--baseline", required=True,
                    help="promerge_film | promerge_only | thinkproprio")
    ap.add_argument("--ckpt", required=True, help="path to checkpoint (use policy_best.ckpt)")
    ap.add_argument("--episodes_per_task", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=520)
    args = ap.parse_args()
    run_libero_spatial_eval(args.baseline, args.ckpt,
                            episodes_per_task=args.episodes_per_task, max_steps=args.max_steps)
