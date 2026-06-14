# ProMerge: Visual Embodied AI Policy Training & Evaluation

This repository implements the training and evaluation pipeline for ACT (Action Chunking with Transformers) policies combined with the **ProMerge** visual token pruning framework. The system is designed to run on macOS (Apple Silicon MPS hardware acceleration) and is fully integrated with MuJoCo physical sandbox simulation.

---

## 🛠️ Step-by-Step Commands

### 1. Data Generation (Expert Dataset)

To regenerate the 5-DOF dynamic interception expert dataset containing real visual frames from the front and wrist cameras, run:

```bash
python data/data_generation.py
```

- **Dataset output**: `data/episodes_500_tuple.hdf5` (~814 MB, gzip compressed)
- **Configurations**: Default is 50 episodes of 400 steps each.

### 2. Policy Training

To train the ACT models from scratch under MPS acceleration:

#### Option A: Train All 5 Variants Sequentially (Orchestration Coordinator)

This will sequentially train `MONOLITHIC_ACT`, `RANDOM_PRUNE`, `TOME_CLUSTERING`, `PROMERGE_ONLY`, and `PROMERGE_FILM` for 15 epochs each, storing checkpoint weights in `checkpoints/<VARIANT>/`:

```bash
python coordinator_train.py
```

#### Option B: Train a Single Specific Policy Variant

To train a single variant manually, run `train.py` with custom arguments:

```bash
python train.py --epochs 15 --batch_size 32 --checkpoint_dir checkpoints/MONOLITHIC_ACT --variant MONOLITHIC_ACT
```

- **Available Variants**: `MONOLITHIC_ACT`, `RANDOM_PRUNE`, `TOME_CLUSTERING`, `PROMERGE_ONLY`, `PROMERGE_FILM`

### 3. Evaluation & Benchmarking (Inference)

To run evaluations under realistic PD joint actuation, visual renderings, and randomized target conditions:

#### Option A: Run the Complete Evaluation Pipeline (1000 Rollouts)

Runs all 4 physical scenarios (Static, Dynamic, Flicker, Shadow) across all 5 variants ($5 \times 4 \times 50$ rollouts), dynamically computes GFLOPs complexity metrics, and outputs the final markdown results matrix:

```bash
python run_eval_pipeline.py
```

- **Report Output**: `eval_results_matrix.md`

#### Option B: Run a Single Specific Scenario Manually

Run `sim/benchmark_eval.py` to evaluate custom tasks, noise, and rollout lengths:

```bash
python sim/benchmark_eval.py --task dyn_intercept --noise FLICKER --num_rollouts 50 --variant PROMERGE_FILM
```

- **Tasks**: `static_manipulation`, `dyn_intercept`
- **Noise Categories**: `NONE`, `FLICKER`, `LOCAL_SHADOW`
