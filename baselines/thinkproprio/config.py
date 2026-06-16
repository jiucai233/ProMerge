"""ThinkProprio — competitor paper reimplementation config (single source of truth).

Faithful reimplementation of ThinkProprio (Sec 3.3): guidance H_q = [instruction;
text-tokenized proprioception], per-vision query + score matrix, vote-based
selection with annealed Gumbel noise + straight-through estimator, global context
token, and HARD removal (no merging). Selection logic: THINKPROPRIO branch in
src/detr/models/perceptual_gatekeeper.py (_thinkproprio_select).

Runs on the SAME ViT-Small backbone and token budget as ProMerge, so the only
difference vs PROMERGE_FILM is pruning-vs-merging — the head-to-head comparison.
"""
NAME = "ThinkProprio (paper reimplementation)"
DESCRIPTION = "Competitor: proprio+instruction guided vote-based HARD pruning (Gumbel+STE)."

VARIANT = "THINKPROPRIO"
CHECKPOINT_DIR = "checkpoints/THINKPROPRIO"

CONFIG_OVERRIDES = {
    "backbone": "vit_small",
    "keep_ratio": 0.3,
    "merge_tokens": False,   # ThinkProprio always hard-removes tokens
}
POLICY_OVERRIDES = {
    "hidden_dim": 384,
    "dim_feedforward": 1536,
    "backbone": "vit_small",
}
