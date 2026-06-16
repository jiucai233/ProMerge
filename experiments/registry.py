"""Experiment registry — the TASK/DATA/EVAL axis (orthogonal to baselines/ = model axis).

An "experiment" decides *what data the policy is trained/evaluated on and how* (dims,
cameras, image size, action space, language instructions, eval protocol). A "baseline"
decides *which model/token-reduction method* is used. Training = experiment x baseline.

Each experiment module must expose:
    NAME            : str   human-readable name
    NAME_SLUG       : str   filesystem-safe id (used for checkpoints/<slug>/<baseline>/)
    DESCRIPTION     : str
    apply(CONFIG, POLICY_CONFIG)            -> None   set dims/img/cameras/action space
    build_loaders(CONFIG, POLICY_CONFIG, batch_size) -> (train_loader, val_loader, norm_stats)
    EVAL            : str   eval-protocol id (for reference)
"""
import importlib

EXPERIMENTS = {
    "sandbox_sorting": "experiments.sandbox_sorting.config",
    "libero_spatial": "experiments.libero_spatial.config",
}


def get_experiment(name):
    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{name}'. Available: {list(EXPERIMENTS)}"
        )
    return importlib.import_module(EXPERIMENTS[name])
