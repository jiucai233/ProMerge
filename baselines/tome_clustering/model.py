"""Model assembly for this baseline.

The actual token-selection logic and all nn.Modules live in the SHARED code under
src/ (src/detr/models/perceptual_gatekeeper.py selects the right branch from
CONFIG["variant"]). Keeping the module tree shared is what preserves checkpoint
state_dict compatibility across every baseline. This file just wires this folder's
config.py onto the shared builder. See config.py for what this baseline IS.
"""
import os
import sys
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_config():
    """Load THIS folder's config.py by file path (avoids clashing with src/config.py)."""
    spec = importlib.util.spec_from_file_location("_baseline_cfg", os.path.join(_HERE, "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_config():
    """Apply this baseline's overrides onto the global CONFIG / POLICY_CONFIG."""
    cfg = load_config()
    from config import CONFIG, POLICY_CONFIG, PolicyVariant
    CONFIG.update(cfg.CONFIG_OVERRIDES)
    CONFIG["variant"] = PolicyVariant[cfg.VARIANT]
    POLICY_CONFIG.update(cfg.POLICY_OVERRIDES)
    num_cameras = CONFIG["num_cameras"]
    POLICY_CONFIG["camera_names"] = ["front", "wrist"][:num_cameras]
    POLICY_CONFIG["state_dim"] = CONFIG["qpos_dim"]
    POLICY_CONFIG["action_dim"] = CONFIG["qpos_dim"]
    return cfg


def build_policy():
    """Build the ACT policy configured for this baseline."""
    apply_config()
    from config import POLICY_CONFIG
    from utils import make_policy
    return make_policy(POLICY_CONFIG["policy_class"], POLICY_CONFIG)
