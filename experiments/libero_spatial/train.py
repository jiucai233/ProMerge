"""Train any baseline on LIBERO-Spatial (thin entry; equivalent to experiments/run.py).

    python experiments/libero_spatial/train.py --baseline promerge_film --epochs 50 --batch_size 16

Requires LIBERO-Spatial demos on disk (see experiments/libero_spatial/config.py:DATA_DIR).
Runs on MPS (Apple GPU). Checkpoints -> checkpoints/libero_spatial/<baseline>/policy_last.ckpt
"""
import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.registry import get_experiment
from experiments._trainer import train_experiment

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a baseline on LIBERO-Spatial")
    ap.add_argument("--baseline", required=True,
                    help="promerge_film | promerge_only | thinkproprio | tome_clustering | ...")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    experiment = get_experiment("libero_spatial")
    train_experiment(experiment, args.baseline, epochs=args.epochs, batch_size=args.batch_size)
