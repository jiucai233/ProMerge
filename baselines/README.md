# `baselines/` — per-method folders

Each subfolder is one method in the ProMerge vs. ThinkProprio comparison. The
folders are an **orchestration layer**: the heavy `nn.Module` code (ACT/DETR-VAE,
ViT backbone, transformer, and every token-selection branch) stays shared under
`src/`, so all variants keep identical `state_dict` keys and **existing
checkpoints still load**. Each folder only carries what makes that method distinct.

## Layout (identical in every folder)

| File | Role |
| :--- | :--- |
| `config.py` | **Single source of truth** for this baseline: variant name, checkpoint dir, and `CONFIG`/`POLICY_CONFIG` overrides (backbone, hidden dim, keep ratio, merge mode). The only file that differs between folders. |
| `model.py` | Assembles the ACT policy for this baseline from the shared `src/` components (`build_policy()`). |
| `train.py` | Thin entrypoint over `scripts/train.py:run_training` using this folder's config. |

## Folders

| Folder | Variant | Backbone | Role |
| :--- | :--- | :--- | :--- |
| `monolithic_act/` | `MONOLITHIC_ACT` | ResNet18 | Full-token upper bound |
| `random_prune/` | `RANDOM_PRUNE` | ResNet18 | Random-retention control |
| `tome_clustering/` | `TOME_CLUSTERING` | ResNet18 | Vision-only token merging |
| `thinkproprio/` | `THINKPROPRIO` | ViT-Small | **Competitor** (paper reimpl., hard pruning) |
| `promerge_only/` | `PROMERGE_ONLY` | ViT-Small | **Ours** (ablation, no FiLM) |
| `promerge_film/` | `PROMERGE_FILM` | ViT-Small | **Ours** (final) |

`thinkproprio/` and `promerge_film/` share the same ViT-Small backbone and token
budget, so their only difference is **hard pruning vs. ToMe merging** — the
head-to-head that shows our method wins.

## Usage

```bash
# Train one baseline
.venv/bin/python baselines/promerge_film/train.py --epochs 50 --batch_size 32

# Build its policy in code
python -c "import baselines.promerge_film.model as m; p = m.build_policy()"
```

Evaluation/reporting still runs through `sim/calvin_libero_benchmark.py` and
`scripts/run_calvin_libero_pipeline.py`.
