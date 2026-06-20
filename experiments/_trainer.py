"""Decoupled trainer: trains (experiment x baseline) on the local Apple-Silicon GPU (MPS).

This is intentionally separate from scripts/train.py so the original sandbox pipeline
stays untouched. It composes:
  * the MODEL from baselines/<baseline>/config.py (variant, backbone, hidden dim, ToMe, ...)
  * the DATA/dims from experiments/<experiment>/config.py (dataset, cameras, action space)

NOTE (acceleration): the model stack is PyTorch, so we use MPS (the Apple GPU). MLX is a
separate array framework and is NOT used here — porting ACT/ViT(timm)/CLIP/LIBERO to MLX
would mean reimplementing all of them and dropping checkpoint/ecosystem compatibility.
"""
import os
import sys
import importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import signal

# Set by SIGTERM/SIGINT (AWS spot interruption sends SIGTERM) -> checkpoint & exit cleanly.
_INTERRUPTED = False


def _install_signal_handler():
    def _handler(signum, frame):
        global _INTERRUPTED
        _INTERRUPTED = True
        print(f"\n⚠️ Caught signal {signum} (spot interruption?) — will checkpoint at next batch.", flush=True)
    for _s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_s, _handler)
        except Exception:
            pass


def _atomic_save(obj, path):
    """Atomic torch.save: write .tmp then os.replace — never leaves a half-written ckpt
    if the spot instance is reclaimed mid-write."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_py(path, name="_m"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _apply_baseline(baseline_name):
    """Apply baselines/<name>/config.py overrides onto the global CONFIG / POLICY_CONFIG."""
    from config import CONFIG, POLICY_CONFIG, PolicyVariant
    path = os.path.join(_ROOT, "baselines", baseline_name, "config.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No baseline folder for '{baseline_name}' ({path})")
    bcfg = _load_py(path, "_bcfg")
    CONFIG.update(bcfg.CONFIG_OVERRIDES)
    CONFIG["variant"] = PolicyVariant[bcfg.VARIANT]
    POLICY_CONFIG.update(bcfg.POLICY_OVERRIDES)
    return bcfg


def train_experiment(experiment, baseline_name, epochs=50, batch_size=16,
                     checkpoint_dir=None, device=None):
    import config as cfgmod
    from config import CONFIG, POLICY_CONFIG
    from utils import make_policy, make_optimizer, set_seed

    # Seed everything for reproducible / valid multi-seed runs (PROMERGE_SEED).
    _seed = int(os.environ.get("PROMERGE_SEED", "0"))
    set_seed(_seed)

    device = device or cfgmod.device

    # 1) model config (baseline)  2) data/dims config (experiment overrides dims)
    _apply_baseline(baseline_name)
    experiment.apply(CONFIG, POLICY_CONFIG)

    # Only language variants consume slow_semantic.
    uses_text = CONFIG["variant"].name in ("PROMERGE_FILM", "THINKPROPRIO")

    train_loader, val_loader, norm_stats = experiment.build_loaders(CONFIG, POLICY_CONFIG, batch_size)

    # Build policy (clear argv to dodge nested argparse inside detr/main.py)
    _argv = list(sys.argv); sys.argv = [sys.argv[0]]
    policy = make_policy(POLICY_CONFIG["policy_class"], POLICY_CONFIG).to(device)
    optimizer = make_optimizer(POLICY_CONFIG["policy_class"], policy)
    sys.argv = _argv

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(_ROOT, "checkpoints", experiment.NAME_SLUG, baseline_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    _install_signal_handler()
    state_path = os.path.join(checkpoint_dir, "train_state.pt")   # full resumable state
    gk = getattr(policy.model, "gatekeeper", None)

    print(f"==> [{experiment.NAME} x {baseline_name}] device={device} "
          f"variant={CONFIG['variant'].name} text={uses_text} -> {checkpoint_dir}", flush=True)

    # --- resume (spot-safe): restore model, optimizer, epoch, early-stop & wandb run id ---
    start_epoch = 0
    best_val = float("inf")
    no_improve = 0
    wandb_id = None
    if os.path.exists(state_path):
        try:
            st = torch.load(state_path, map_location=device)
            policy.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            start_epoch = int(st.get("epoch", 0))
            best_val = float(st.get("best_val", float("inf")))
            no_improve = int(st.get("no_improve", 0))
            wandb_id = st.get("wandb_id")
            if gk is not None and hasattr(gk, "tp_step") and "tp_step" in st:
                gk.tp_step.fill_(int(st["tp_step"]))
            print(f"↻ Resuming from epoch {start_epoch} (best_val {best_val:.4f})", flush=True)
        except Exception as e:
            print(f"⚠️ Could not load resume state ({e}); starting fresh.", flush=True)

    # --- wandb: resume the SAME run id across spot restarts (never let it kill training) ---
    run = None
    try:
        import wandb
        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
        run = wandb.init(
            project="ProMerge", id=wandb_id, resume="allow",
            name=f"{experiment.NAME_SLUG}_{baseline_name}_bs{batch_size}",
            group=experiment.NAME_SLUG,
            config={
                "experiment": experiment.NAME_SLUG, "baseline": baseline_name,
                "variant": CONFIG["variant"].name, "epochs": epochs, "batch_size": batch_size,
                "keep_ratio": CONFIG.get("keep_ratio"), "merge_tokens": CONFIG.get("merge_tokens"),
                "uses_text": uses_text,
            },
        )
        print(f"🚀 wandb run: {run.name} (id={wandb_id}, resumed={start_epoch > 0})", flush=True)
    except Exception as e:
        print(f"⚠️ wandb init failed ({e}); continuing without wandb.", flush=True)

    # --- training-time image augmentation (per camera; RandomShift from utils) ---
    aug_cfg = getattr(experiment, "AUGMENTATION", {"enabled": False})
    augs = None
    if aug_cfg.get("enabled"):
        from utils import RandomShiftsAug
        ncam = CONFIG["num_cameras"]
        pads = aug_cfg.get("pad_per_camera") or [aug_cfg.get("pad", 4)] * ncam
        augs = [RandomShiftsAug(pads[i] if i < len(pads) else pads[-1]) for i in range(ncam)]
        print(f"🎲 Augmentation: RandomShift pads={[a.pad for a in augs]}", flush=True)

    def _run_loss(images, qpos, actions, is_pad, slow, augment=False):
        images, qpos = images.to(device), qpos.to(device)
        actions, is_pad = actions.to(device), is_pad.to(device)
        slow = slow.to(device) if uses_text else None
        if augment and augs is not None:
            for ci in range(images.shape[1]):  # [B, ncam, C, H, W]
                images[:, ci] = augs[ci](images[:, ci])
        return policy(qpos, images, actions, is_pad, slow_semantic=slow)

    es_enabled = CONFIG.get("early_stop_enabled", False)
    es_patience = CONFIG.get("early_stop_patience", 5)
    es_delta = CONFIG.get("early_stop_min_delta", 1e-3)

    def _save_state(next_epoch):
        _atomic_save({
            "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": next_epoch, "best_val": best_val, "no_improve": no_improve,
            "wandb_id": wandb_id,
            "tp_step": int(gk.tp_step.item()) if (gk is not None and hasattr(gk, "tp_step")) else 0,
        }, state_path)

    for epoch in range(start_epoch, epochs):
        policy.train()
        tot = n = 0
        for images, qpos, actions, is_pad, slow in train_loader:
            loss_dict = _run_loss(images, qpos, actions, is_pad, slow, augment=True)
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
            tot += loss_dict["loss"].item(); n += 1
            if _INTERRUPTED:
                # spot reclaim: persist (resume re-runs this epoch from scratch) and exit cleanly
                _save_state(epoch)
                _atomic_save(policy.state_dict(), os.path.join(checkpoint_dir, "policy_last.ckpt"))
                print(f"💾 Saved at epoch {epoch} on interruption — exiting for spot reclaim.", flush=True)
                if run is not None:
                    try: wandb.finish()
                    except Exception: pass
                sys.exit(0)
        train_loss = tot / max(n, 1)

        # Validate every epoch. (Was every 10 epochs, which made sense when an
        # "epoch" was ~14 batches; now an epoch is ~1756 batches over all frames,
        # so per-epoch validation is both affordable and needed for early
        # stopping / best-checkpoint selection to react in time.)
        val_loss = None
        val_every = int(CONFIG.get("val_every_epochs", 1))
        if (epoch + 1) % val_every == 0:
            policy.eval()
            vt = vn = 0
            with torch.no_grad():
                for images, qpos, actions, is_pad, slow in val_loader:
                    vd = _run_loss(images, qpos, actions, is_pad, slow)
                    vt += vd["loss"].item(); vn += 1
            val_loss = vt / max(vn, 1)

        msg = f"[{experiment.NAME_SLUG}|{baseline_name}] epoch {epoch:03d} | train {train_loss:.4f}"
        if val_loss is not None:
            msg += f" | val {val_loss:.4f}"
        print(msg, flush=True)

        # --- best-val checkpoint + early stopping ---
        stop = False
        if val_loss is not None:
            if val_loss < best_val - es_delta:
                best_val = val_loss
                no_improve = 0
                _atomic_save(policy.state_dict(), os.path.join(checkpoint_dir, "policy_best.ckpt"))
                print(f"  ↳ new best val {best_val:.4f} → policy_best.ckpt", flush=True)
            else:
                no_improve += 1
                if es_enabled and no_improve >= es_patience:
                    stop = True

        if run is not None:
            try:
                log = {"epoch": epoch, "loss/train": train_loss, "loss/best_val": best_val}
                if val_loss is not None:
                    log["loss/val"] = val_loss
                wandb.log(log)
            except Exception:
                pass

        # atomic checkpoints + resumable state (epoch+1 = next epoch to run on restart)
        _atomic_save(policy.state_dict(), os.path.join(checkpoint_dir, "policy_last.ckpt"))
        _save_state(epoch + 1)
        if (epoch + 1) % 100 == 0:
            _atomic_save(policy.state_dict(),
                         os.path.join(checkpoint_dir, f"policy_epoch{epoch + 1:04d}.ckpt"))

        if stop:
            print(f"⏹️ Early stop at epoch {epoch} (no val improvement for "
                  f"{es_patience} checks; best val {best_val:.4f})", flush=True)
            break

    if run is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    # training complete — drop resume state so a fresh re-run doesn't think it's mid-flight
    if os.path.exists(state_path):
        try: os.remove(state_path)
        except Exception: pass

    print(f"✅ Training done: {experiment.NAME_SLUG} x {baseline_name}", flush=True)
    return checkpoint_dir
