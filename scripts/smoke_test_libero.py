"""
Smoke test for the LIBERO-Spatial training path on this Linux/CUDA box.

Goal: verify data loading -> model build -> forward -> backward runs end to end,
WITHOUT competing for the GPU (VGSR training is using it) and without doing a
full epoch. We force CPU and rely on an external timeout to stop once we've seen
a few training steps print a loss.

Run:
    MUJOCO_GL=egl WANDB_MODE=disabled \
      .venv/bin/python scripts/smoke_test_libero.py --baseline monolithic_act
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from experiments.registry import get_experiment
from experiments._trainer import train_experiment

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="monolithic_act")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    print(f"[smoke] baseline={args.baseline} epochs={args.epochs} bs={args.batch_size} device=cpu", flush=True)

    experiment = get_experiment("libero_spatial")
    # Force CPU so we never touch the GPU that VGSR training is using.
    train_experiment(experiment, args.baseline,
                     epochs=args.epochs, batch_size=args.batch_size, device="cpu")
    print("[smoke] train_experiment returned cleanly", flush=True)
