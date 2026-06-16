"""Monolithic ACT — baseline config (single source of truth for this baseline).

Full-token ACT: every visual token is kept (100%). The gatekeeper returns the
tokens unchanged. Selection logic: MONOLITHIC_ACT branch in
src/detr/models/perceptual_gatekeeper.py.
"""
NAME = "Monolithic ACT (no token reduction)"
DESCRIPTION = "Full-token ACT upper bound: 100% of visual tokens retained."

VARIANT = "MONOLITHIC_ACT"
CHECKPOINT_DIR = "checkpoints/MONOLITHIC_ACT"

# Applied on top of src/config.py CONFIG
CONFIG_OVERRIDES = {
    "backbone": "resnet18",
    "keep_ratio": 0.3,      # unused (no pruning) but kept for consistency
    "merge_tokens": True,   # irrelevant for this variant
}
# Applied on top of src/config.py POLICY_CONFIG
POLICY_OVERRIDES = {
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "backbone": "resnet18",
}
