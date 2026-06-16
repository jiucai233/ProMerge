"""ProMerge-Only — OUR method (ablation) config (single source of truth).

Ours without semantic FiLM: static learnable target queries find salient
interaction regions (no qpos coupling — anti-hand-staring), then bipartite soft
ToMe merging keeps the token footprint while preserving occluded evidence.
Selection logic: the PROMERGE branch (variant 4) in
src/detr/models/perceptual_gatekeeper.py.
"""
NAME = "ProMerge-Only (ours, ablation: no FiLM)"
DESCRIPTION = "Ours w/o semantic FiLM: learnable-query gating + ToMe merging."

VARIANT = "PROMERGE_ONLY"
CHECKPOINT_DIR = "checkpoints/PROMERGE_ONLY"

CONFIG_OVERRIDES = {
    "backbone": "vit_small",
    "keep_ratio": 0.3,
    "merge_tokens": True,
}
POLICY_OVERRIDES = {
    "hidden_dim": 384,
    "dim_feedforward": 1536,
    "backbone": "vit_small",
}
