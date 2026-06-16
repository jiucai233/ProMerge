# `experiments/` — the task/data/eval axis

Two orthogonal axes now drive everything:

- **`baselines/<name>/`** — the **model** (variant, backbone, ToMe vs prune, CLIP guidance).
- **`experiments/<name>/`** — the **task**: dataset, dims, cameras, action space, language
  instructions, eval protocol.

A run is **experiment × baseline**. Routed through `experiments/run.py`.

```
experiments/
├── registry.py          # name -> experiment module
├── run.py               # CLI router (--experiment X --baseline Y --mode train|inspect)
├── _trainer.py          # decoupled training loop (MPS); composes baseline-model + experiment-data
├── sandbox_sorting/     # the original local sandbox, wrapped as an experiment
│   └── config.py
└── libero_spatial/      # REAL LIBERO-Spatial (lightweight single suite)
    ├── config.py        # dims, cameras, paths, obs keys, action space
    ├── data.py          # LIBERO hdf5 adapter + `inspect_hdf5`
    ├── train.py         # thin entry
    └── eval.py          # official-eval interface stub (run on a GPU box)
```

## Acceleration: MPS, not MLX

The model stack is PyTorch (ACT, timm ViT, HF CLIP) and real LIBERO is torch-only, so we
use **MPS (the Apple-Silicon GPU)** — already the project default. MLX is a separate array
framework; porting the whole stack to it would mean reimplementing the models and dropping
timm/CLIP/LIBERO + checkpoint compatibility, so it is intentionally not used. (If you later
want MLX for an isolated standalone demo, add it without touching the shared stack.)

## Usage

```bash
# Local sanity (existing sandbox, any baseline)
python experiments/run.py --experiment sandbox_sorting --baseline promerge_film --mode train

# LIBERO-Spatial (after downloading demos)
python experiments/run.py --experiment libero_spatial --baseline promerge_film --mode train
python experiments/run.py --experiment libero_spatial --baseline thinkproprio  --mode train
python experiments/run.py --experiment libero_spatial --baseline promerge_only --mode train
```

Checkpoints → `checkpoints/<experiment>/<baseline>/policy_last.ckpt`.

## Bringing up LIBERO-Spatial (local dev)

1. Download the LIBERO-Spatial demo `*.hdf5` files; put them in `data/libero_spatial/`
   (or set `LIBERO_SPATIAL_DIR`).
2. Confirm obs key names match `config.py:OBS_KEYS`:
   ```bash
   python experiments/run.py --experiment libero_spatial --mode inspect --path data/libero_spatial/<task>.hdf5
   ```
   Adjust `OBS_KEYS` / `IMAGE_SIZE` / `FLIP_IMAGES` / `ACTION_DIM` if needed.
3. Train on MPS locally. For **official success-rate eval**, finish `eval.py` and run it on a
   Linux GPU box (LIBERO offscreen rendering is painful on macOS).

LIBERO-Spatial dims (9D proprio = 7 joint + 2 gripper) match the sandbox `qpos_dim=9`, so the
proprio path needs no change; the action space differs (7D `rel_actions`), handled via the
experiment's `ACTION_DIM`.
