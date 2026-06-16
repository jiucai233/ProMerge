"""Train this baseline (thin entrypoint over the shared training loop in scripts/train.py)."""
import os
import sys
import argparse
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "scripts"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_spec = importlib.util.spec_from_file_location("_baseline_cfg", os.path.join(_HERE, "config.py"))
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)

from train import run_training

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"Train baseline: {cfg.NAME}")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    print(f"==> Training baseline: {cfg.NAME}  [{cfg.VARIANT}]")
    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=os.path.join(_ROOT, cfg.CHECKPOINT_DIR),
        variant=cfg.VARIANT,
        config_overrides=cfg.CONFIG_OVERRIDES,
        policy_overrides=cfg.POLICY_OVERRIDES,
    )
