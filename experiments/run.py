"""Router CLI: pick which experiment (task/data) x which baseline (model) to run.

Examples
--------
  # Train ProMerge-FiLM on real LIBERO-Spatial
  python experiments/run.py --experiment libero_spatial --baseline promerge_film --mode train

  # Sanity: same model on the local sandbox sorting task
  python experiments/run.py --experiment sandbox_sorting --baseline promerge_film --mode train

  # Inspect a LIBERO hdf5 to confirm obs keys before training
  python experiments/run.py --experiment libero_spatial --mode inspect --path data/libero_spatial/<task>.hdf5
"""
import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.registry import get_experiment, EXPERIMENTS


def main():
    ap = argparse.ArgumentParser(description="ProMerge experiment x baseline router")
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--baseline", default=None,
                    help="baselines/<name> folder (e.g. promerge_film, promerge_only, thinkproprio)")
    ap.add_argument("--mode", default="train", choices=["train", "inspect"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--path", default=None, help="hdf5 path for --mode inspect")
    args = ap.parse_args()

    experiment = get_experiment(args.experiment)

    if args.mode == "inspect":
        # Delegate to the experiment's data module if it provides an inspector.
        import importlib
        data_mod = importlib.import_module(EXPERIMENTS[args.experiment].rsplit(".", 1)[0] + ".data")
        data_mod.inspect_hdf5(args.path)
        return

    if args.baseline is None:
        ap.error("--baseline is required for --mode train")

    from experiments._trainer import train_experiment
    train_experiment(experiment, args.baseline, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
