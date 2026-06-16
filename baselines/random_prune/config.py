"""Random Pruning — baseline config (single source of truth for this baseline).

Control baseline: keep a random 30% of visual tokens (no learning, no merging).
Selection logic: RANDOM_PRUNE branch in
src/detr/models/perceptual_gatekeeper.py.
"""
NAME = "Random Pruning (30% kept)"
DESCRIPTION = "Lower-bound control: uniformly random token retention."

VARIANT = "RANDOM_PRUNE"
CHECKPOINT_DIR = "checkpoints/RANDOM_PRUNE"

CONFIG_OVERRIDES = {
    "backbone": "resnet18",
    "keep_ratio": 0.3,
    "merge_tokens": False,
}
POLICY_OVERRIDES = {
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "backbone": "resnet18",
}
