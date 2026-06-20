import os
"""ProMerge-FiLM — OUR final method config (single source of truth).

Ours, final form: high-level semantic intent (slow_semantic) modulates visual
tokens via FiLM and generates an intent filter w; tokens are scored by parallel
dot-product (V · w, O(N·D)) — decoupled from qpos (anti-hand-staring) — then
fused with visual saliency and compressed via bipartite soft ToMe merging.
Selection logic: the PROMERGE branch (variant 5) in
src/detr/models/perceptual_gatekeeper.py.

Same ViT-Small backbone and budget as the ThinkProprio reimplementation, so the
only difference is token MERGING (ours) vs HARD pruning (theirs).
"""
NAME = "ProMerge-FiLM (ours, final)"
DESCRIPTION = "Ours: semantic-intent dot-product gating + ToMe merging."

VARIANT = "PROMERGE_FILM"
CHECKPOINT_DIR = "checkpoints/PROMERGE_FILM"

CONFIG_OVERRIDES = {
    "backbone": os.environ.get("PROMERGE_BACKBONE", "vit_small"),
    "keep_ratio": 0.3,
    "merge_tokens": True,
}
POLICY_OVERRIDES = {
    "hidden_dim": 384,
    "dim_feedforward": 1536,
    "backbone": os.environ.get("PROMERGE_BACKBONE", "vit_small"),
}
