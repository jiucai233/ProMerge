"""ToMe Clustering — baseline config (single source of truth for this baseline).

Competitor: pure-vision bipartite Token Merging (no proprio/intent guidance).
Importance comes from visual self-attention only. Selection logic:
TOME_CLUSTERING branch in src/detr/models/perceptual_gatekeeper.py.

Set merge_tokens=False here (or override at eval) for the hard-selection (top-k)
ablation of the same variant.
"""
NAME = "ToMe Clustering (vision-only token merging)"
DESCRIPTION = "Competitor: bipartite soft merging driven by visual saliency alone."

VARIANT = "TOME_CLUSTERING"
CHECKPOINT_DIR = "checkpoints/TOME_CLUSTERING"

CONFIG_OVERRIDES = {
    "backbone": "resnet18",
    "keep_ratio": 0.3,
    "merge_tokens": True,
}
POLICY_OVERRIDES = {
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "backbone": "resnet18",
}
