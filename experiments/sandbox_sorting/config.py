"""Experiment: local MuJoCo sorting sandbox (the original setup, wrapped as an experiment).

This is the existing pipeline expressed in the experiment x baseline scheme so routing
is uniform. Data: data/episodes_500_tuple.hdf5. Eval: the CALVIN/LIBERO *emulation*
protocols in sim/calvin_libero_benchmark.py (NOT the real benchmarks).
"""
import os

NAME = "Sandbox Sorting (local emulation)"
NAME_SLUG = "sandbox_sorting"
DESCRIPTION = "Local Franka sorting sandbox; CALVIN/LIBERO are emulated eval protocols."
EVAL = "calvin_libero_emulation"

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def apply(CONFIG, POLICY_CONFIG):
    CONFIG["image_size"] = (240, 320)
    CONFIG["num_cameras"] = 2
    CONFIG["qpos_dim"] = 9
    POLICY_CONFIG["camera_names"] = ["front", "wrist"]
    POLICY_CONFIG["state_dim"] = 9
    POLICY_CONFIG["action_dim"] = 9


def build_loaders(CONFIG, POLICY_CONFIG, batch_size):
    from utils import load_data
    dataset_dir = os.path.join(_ROOT, "data")
    num_episodes = CONFIG.get("num_episodes", 50)
    train_loader, val_loader, norm_stats, _is_sim = load_data(
        dataset_dir, num_episodes, POLICY_CONFIG["camera_names"], batch_size, batch_size
    )
    return train_loader, val_loader, norm_stats
