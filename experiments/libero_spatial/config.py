"""Experiment: LIBERO-Spatial (real benchmark, lightweight single suite).

LIBERO-Spatial: 10 tasks x 50 demos, two RGB views (agentview static + eye_in_hand wrist),
9D proprio (7 joint + 2 gripper), 7D relative actions, native 128x128 images, each task
carries a natural-language instruction — which is exactly what makes the CLIP/FiLM and
ThinkProprio "[instruction; proprio]" guidance meaningful here.

Data is NOT bundled. Download the LIBERO-Spatial demo hdf5s and point LIBERO_SPATIAL_DIR
(env var) or DATA_DIR below at the folder containing the per-task *.hdf5 files. Run
`python experiments/run.py --experiment libero_spatial --mode inspect --path <file>` to
verify the obs key names against OBS_KEYS before training.
"""
import os

NAME = "LIBERO-Spatial"
# Suite is switchable via LIBERO_SUITE env (libero_spatial|libero_object|
# libero_goal|libero_10). Lets us test whether the g_vis fix generalizes beyond
# Spatial (guard against overfitting to one suite).
SUITE = os.environ.get("LIBERO_SUITE", "libero_spatial")
NAME_SLUG = "libero_spatial"  # checkpoint/experiment dir stays stable
DESCRIPTION = f"Real LIBERO suite ({SUITE}), ViT-Small + CLIP-instruction guidance."
EVAL = "libero_official"  # real success-rate eval; needs the libero env (run on a GPU box)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Where the per-task LIBERO *.hdf5 demo files live (follows LIBERO_SUITE).
DATA_DIR = os.environ.get("LIBERO_SPATIAL_DIR", os.path.join(_ROOT, "data", SUITE))

# Observation/action layout (LIBERO-Spatial defaults; adjust after `--mode inspect`).
IMAGE_SIZE = (128, 128)        # native LIBERO resolution; (112,112) matches the paper
NUM_CAMERAS = 2
CAMERA_NAMES = ["agentview", "eye_in_hand"]
STATE_DIM = 9                  # 7 joint + 2 gripper  (matches sandbox qpos_dim, convenient)
ACTION_DIM = 7                 # 7D relative actions (rel_actions) — the REAL action dim
# ACT ties the action head width to state_dim, so the model emits STATE_DIM (9) action dims.
# The data adapter pads the real 7-d actions into a 9-d vector (zeros in the last 2); at eval,
# only the first ACTION_DIM (7) dims are used to drive the robot.
MODEL_ACTION_DIM = STATE_DIM
FLIP_IMAGES = True             # confirmed: this dump uses macros_image_convention='opengl'
                               # (bottom-left origin) -> flip vertically to upright, matching the
                               # ImageNet-pretrained ViT. Eval rendering must be made upright too.

# hdf5 obs keys (robomimic/LIBERO convention). Override here if `--mode inspect` shows others.
OBS_KEYS = {
    "agentview": "agentview_rgb",
    "eye_in_hand": "eye_in_hand_rgb",
    "joint": "joint_states",
    "gripper": "gripper_states",
}


# Training-time image augmentation (RandomShiftsAug, implemented in src/utils.py).
# Per-camera pad in pixels; the paper uses 10 for the static view and 4 for the wrist.
AUGMENTATION = {
    "enabled": True,
    "pad_per_camera": [10, 4],   # [agentview (static), eye_in_hand (wrist)]
}


def apply(CONFIG, POLICY_CONFIG):
    CONFIG["image_size"] = IMAGE_SIZE
    CONFIG["num_cameras"] = NUM_CAMERAS
    CONFIG["qpos_dim"] = STATE_DIM
    POLICY_CONFIG["camera_names"] = list(CAMERA_NAMES)
    POLICY_CONFIG["state_dim"] = STATE_DIM
    POLICY_CONFIG["action_dim"] = MODEL_ACTION_DIM  # = STATE_DIM (ACT couples action head to state)


def build_loaders(CONFIG, POLICY_CONFIG, batch_size):
    from experiments.libero_spatial.data import build_libero_loaders
    return build_libero_loaders(CONFIG, POLICY_CONFIG, batch_size)
