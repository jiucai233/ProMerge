"""LIBERO-Spatial dataset adapter.

Reads LIBERO (robomimic-style) demo hdf5 files and yields the same 5-tuple contract the
rest of the pipeline expects:
    (images[ncam,3,H,W], qpos[state_dim], actions[chunk,action_dim], is_pad[chunk], slow_semantic[384])

Per-task natural-language instructions are encoded with the shared CLIP encoder
(utils.encode_text_instruction -> 384-d), matching PROMERGE_FILM / THINKPROPRIO.

LIBERO is not bundled. Point experiments/libero_spatial/config.py:DATA_DIR (or the
LIBERO_SPATIAL_DIR env var) at a folder of per-task *.hdf5 files, then verify obs keys:
    python experiments/run.py --experiment libero_spatial --mode inspect --path <file.hdf5>
"""
import os
import glob
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from experiments.libero_spatial import config as C


# --------------------------------------------------------------------------------------
# Inspection helper — run this against a downloaded file to confirm the obs key names.
# --------------------------------------------------------------------------------------
def inspect_hdf5(path):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"--path not found: {path}")
    with h5py.File(path, "r") as f:
        print(f"== {path} ==")
        print("top-level keys:", list(f.keys()))
        if "data" in f:
            data = f["data"]
            print("data attrs:", {k: (str(v)[:120]) for k, v in data.attrs.items()})
            demos = [k for k in data.keys() if k.startswith("demo")]
            print(f"num demos: {len(demos)}")
            if demos:
                d0 = data[demos[0]]
                print(f"demo0 keys: {list(d0.keys())}")
                if "obs" in d0:
                    for k in d0["obs"].keys():
                        print(f"  obs/{k}: shape={d0['obs'][k].shape} dtype={d0['obs'][k].dtype}")
                if "actions" in d0:
                    print(f"  actions: shape={d0['actions'].shape}")


def _extract_instruction(h5file, fallback_name):
    """Best-effort language instruction: data.attrs['problem_info'] JSON, else file name."""
    try:
        info = h5file["data"].attrs.get("problem_info", None)
        if info is not None:
            obj = json.loads(info) if isinstance(info, (str, bytes)) else info
            lang = obj.get("language_instruction") or obj.get("language") or obj.get("instruction")
            if lang:
                return lang if isinstance(lang, str) else lang[0]
    except Exception:
        pass
    # Fall back to a readable form of the task filename.
    name = os.path.splitext(os.path.basename(fallback_name))[0]
    name = name.replace("_demo", "").replace("SCENE", "").replace("_", " ").strip()
    return name or "complete the task"


def _resolve_obs_key(obs_group, wanted):
    """Tolerate naming differences; return the actual key present in this file."""
    if wanted in obs_group:
        return wanted
    for k in obs_group.keys():
        if wanted.replace("_rgb", "") in k or wanted in k:
            return k
    raise KeyError(f"obs key '{wanted}' not found; available: {list(obs_group.keys())}. "
                   f"Update OBS_KEYS in experiments/libero_spatial/config.py.")


class LiberoSpatialDataset(torch.utils.data.Dataset):
    def __init__(self, episode_index, norm_stats, chunk, instruction_embeds):
        self.episode_index = episode_index    # list of dicts: file, demo, length, instruction
        self.norm_stats = norm_stats
        self.chunk = chunk
        self.instruction_embeds = instruction_embeds  # {instruction_str: tensor[384]}
        self._handles = {}

    def __len__(self):
        return len(self.episode_index)

    def _file(self, path):
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, idx):
        ep = self.episode_index[idx]
        f = self._file(ep["file"])
        demo = f["data"][ep["demo"]]
        obs = demo["obs"]
        T = ep["length"]
        start = np.random.randint(0, T)

        ak = _resolve_obs_key(obs, C.OBS_KEYS["agentview"])
        wk = _resolve_obs_key(obs, C.OBS_KEYS["eye_in_hand"])
        jk = _resolve_obs_key(obs, C.OBS_KEYS["joint"])
        gk = _resolve_obs_key(obs, C.OBS_KEYS["gripper"])

        def _img(key):
            arr = obs[key][start]                      # [H,W,3] uint8
            t = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1) / 255.0
            if C.FLIP_IMAGES:
                t = torch.flip(t, dims=[1])
            if tuple(t.shape[1:]) != tuple(C.IMAGE_SIZE):
                t = F.interpolate(t.unsqueeze(0), size=C.IMAGE_SIZE, mode="bilinear",
                                  align_corners=False).squeeze(0)
            return t

        images = torch.stack([_img(ak), _img(wk)], dim=0)  # [2,3,H,W]

        qpos = np.concatenate([obs[jk][start], obs[gk][start]]).astype(np.float32)  # [9]

        actions = demo["actions"][start:start + self.chunk].astype(np.float32)      # [<=chunk, 7]
        alen = actions.shape[0]
        # ACT's action head emits STATE_DIM dims, so pad the real 7-d action into a 9-d vector.
        padded = np.zeros((self.chunk, C.MODEL_ACTION_DIM), dtype=np.float32)
        padded[:alen, :C.ACTION_DIM] = actions[:, :C.ACTION_DIM]
        is_pad = np.zeros(self.chunk, dtype=bool)
        is_pad[alen:] = True

        # normalize
        qpos = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        padded = (padded - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        slow = self.instruction_embeds[ep["instruction"]]

        return (images,
                torch.from_numpy(qpos).float(),
                torch.from_numpy(padded).float(),
                torch.from_numpy(is_pad),
                slow.clone())


def _scan_episodes(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(
            f"No LIBERO-Spatial *.hdf5 found in {data_dir}.\n"
            f"Download the LIBERO-Spatial demos and set LIBERO_SPATIAL_DIR or DATA_DIR.\n"
            f"Then verify keys: python experiments/run.py --experiment libero_spatial "
            f"--mode inspect --path <file.hdf5>"
        )
    index = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            instruction = _extract_instruction(f, fp)
            for demo in [k for k in f["data"].keys() if k.startswith("demo")]:
                n = f["data"][demo].attrs.get("num_samples", f["data"][demo]["actions"].shape[0])
                index.append({"file": fp, "demo": demo, "length": int(n), "instruction": instruction})
    return index


def _compute_norm_stats(index, max_demos=80):
    qpos_all, act_all = [], []
    seen = {}
    for ep in index:
        if seen.get(ep["file"], 0) >= max_demos:
            continue
        seen[ep["file"]] = seen.get(ep["file"], 0) + 1
        with h5py.File(ep["file"], "r") as f:
            demo = f["data"][ep["demo"]]
            obs = demo["obs"]
            jk = _resolve_obs_key(obs, C.OBS_KEYS["joint"])
            gk = _resolve_obs_key(obs, C.OBS_KEYS["gripper"])
            q = np.concatenate([obs[jk][()], obs[gk][()]], axis=1)
            qpos_all.append(q)
            act_all.append(demo["actions"][()])
    qpos_all = np.concatenate(qpos_all, axis=0)
    act_all = np.concatenate(act_all, axis=0)            # [M, ACTION_DIM=7]
    eps = 1e-6
    # Pad action stats to MODEL_ACTION_DIM (mean 0 / std 1 on the padded dims so the
    # zero-filled action columns normalize to 0 and don't affect the loss).
    n_pad = C.MODEL_ACTION_DIM - C.ACTION_DIM
    action_mean = np.concatenate([act_all.mean(0), np.zeros(n_pad)]).astype(np.float32)
    action_std = np.concatenate([act_all.std(0) + eps, np.ones(n_pad)]).astype(np.float32)
    return {
        "qpos_mean": qpos_all.mean(0).astype(np.float32),
        "qpos_std": (qpos_all.std(0) + eps).astype(np.float32),
        "action_mean": action_mean,
        "action_std": action_std,
    }


def build_libero_loaders(CONFIG, POLICY_CONFIG, batch_size, val_frac=0.1, seed=42):
    from torch.utils.data import DataLoader

    index = _scan_episodes(C.DATA_DIR)
    norm_stats = _compute_norm_stats(index)
    chunk = POLICY_CONFIG.get("num_queries", 100)

    # Pre-encode the (few) unique instructions once with CLIP -> 384-d.
    from utils import encode_text_instruction
    unique = sorted({ep["instruction"] for ep in index})
    instruction_embeds = {s: encode_text_instruction(s).cpu() for s in unique}
    print(f"🚀 LIBERO-Spatial: {len(index)} demos, {len(unique)} unique instructions encoded via CLIP.")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(index))
    n_val = max(1, int(len(index) * val_frac))
    val_idx, train_idx = set(perm[:n_val].tolist()), set(perm[n_val:].tolist())
    train_ep = [index[i] for i in sorted(train_idx)]
    val_ep = [index[i] for i in sorted(val_idx)]

    train_ds = LiberoSpatialDataset(train_ep, norm_stats, chunk, instruction_embeds)
    val_ds = LiberoSpatialDataset(val_ep, norm_stats, chunk, instruction_embeds)

    from config import CONFIG, device as _dev
    nw = CONFIG.get("dataloader_workers", 0)          # 0 on mac; set DATALOADER_WORKERS on cloud
    pin = (_dev == "cuda")
    common = dict(num_workers=nw, pin_memory=pin)
    if nw > 0:
        common["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, norm_stats
