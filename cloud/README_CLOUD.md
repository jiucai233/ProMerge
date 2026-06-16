# Running ProMerge on a Cloud GPU (GCP / AWS)

Goal: fine-tune the LIBERO-Spatial variants on a real GPU and run the **official LIBERO
success-rate eval** (which can't run on the Mac). The repo already auto-selects `cuda` and
sets `MUJOCO_GL=egl` on Linux — this guide covers provisioning, setup, data, and runs.

## TL;DR (scripts)

```bash
git clone <repo> ProMerge && cd ProMerge
bash cloud/setup.sh                                  # deps + venv + LIBERO env
source .venv/bin/activate && source cloud/env.sh
export WANDB_API_KEY=...                             # wandb (training + eval log here)
bash cloud/get_data.sh                               # download LIBERO-Spatial -> data/

# train + eval + commit results + upload checkpoints + auto-shutdown, unattended:
TRAIN=1 EPISODES=20 BUCKET=gs://your-bucket/promerge SHUTDOWN=1 \
  nohup bash cloud/run_all.sh > logs/run_all.log 2>&1 &
```
`run_all.sh` does: [optional train] → eval each variant (`policy_best`) → write+commit
`RESULTS_libero_spatial.md` (pushed) → upload checkpoints to `BUCKET` → **poweroff**.
Safety: it will **not** shut down unless results were persisted (push OK *or* bucket upload OK),
so you never lose the gitignored checkpoints. Manual steps below if you prefer step-by-step.

---

## 1. Provision a GPU box

Pick one. A single 24 GB GPU (A10G / L4 / A100) is plenty for ViT-Small + ACT.

**GCP**
- Image: *Deep Learning VM* (PyTorch CUDA) or a clean Ubuntu 22.04.
- Machine: `g2-standard-8` (+ 1× **L4**) or `a2-highgpu-1g` (1× **A100**).
- Disk: ≥ 100 GB (data ~6 GB + checkpoints + cache).
```bash
gcloud compute instances create promerge-gpu \
  --zone=us-central1-a --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 --maintenance-policy=TERMINATE \
  --image-family=common-cu121-ubuntu-2204 --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB
gcloud compute ssh promerge-gpu --zone=us-central1-a
```

**AWS**
- AMI: *Deep Learning AMI (Ubuntu 22.04)*.
- Instance: `g5.xlarge` (1× **A10G**, 24 GB) or `p3.2xlarge` (V100).
- Storage: ≥ 100 GB EBS.
```bash
# via console or CLI; then:
ssh -i key.pem ubuntu@<public-ip>
```

---

## 2. Get the code + run setup

```bash
git clone <your-fork-or-repo-url> ProMerge && cd ProMerge
bash cloud/setup.sh                       # apt + venv + deps + LIBERO env (~10-15 min)
source .venv/bin/activate && source cloud/env.sh
# put your wandb key:
export WANDB_API_KEY=xxximshen19xxx        # or: wandb login
```

---

## 2b. Persistent storage (recommended — don't keep data on the boot disk)

The boot disk is ephemeral: delete the instance and the 5.8 GB data + checkpoints are gone.
Put them on a **separate persistent disk** (GCP PD / AWS EBS) that you provision once and
re-attach to future instances. hdf5 random reads also want a real block disk, not a
fuse-mounted bucket.

**GCP** — create + attach a PD, then mount:
```bash
gcloud compute disks create promerge-data --size=100GB --zone=us-central1-a
gcloud compute instances attach-disk promerge-gpu --disk=promerge-data --zone=us-central1-a
# on the box:
lsblk                                   # find the device (e.g. /dev/sdb or /dev/nvme1n1)
sudo bash cloud/mount_disk.sh /dev/sdb /mnt/promerge
```
**AWS** — create + attach an EBS volume (console/CLI), then on the box:
```bash
lsblk
sudo bash cloud/mount_disk.sh /dev/nvme1n1 /mnt/promerge
```
`mount_disk.sh` symlinks `data/` and `checkpoints/` onto the disk (only formats a *blank*
disk). Next time, just re-attach the disk and re-run it (it keeps existing data).

Alternative (no managed disk): keep a **bucket as the durable home** — download data per run
and push checkpoints back via `BUCKET=` in `cloud/run_all.sh`. Simpler, but re-downloads each run.

## 2c. Spot instances — auto-resume (storage/compute separation)

Training is **spot-safe**: `experiments/_trainer.py` saves a full `train_state.pt`
(model + optimizer + epoch + early-stop state + wandb run id) **atomically every epoch and on
SIGTERM** (the signal AWS sends on a spot reclaim). Re-running the *same* command continues
from the saved epoch and **resumes the same wandb run** (no new curve). Plain `policy_last.ckpt`
/ `policy_best.ckpt` are also written atomically (no more half-written files).

For this to survive a reclaim, `train_state.pt` must live on **storage that outlives the
instance** — i.e. the separate disk from §2b:

1. Create the EBS volume **`delete-on-termination = false`** and **pin the spot request to one
   AZ** (EBS is AZ-locked, so the replacement spot must come up in the same AZ to reattach).
2. On reclaim → launch a new spot in that AZ → attach the volume → on the box:
   ```bash
   cd ProMerge && sudo bash cloud/mount_disk.sh /dev/nvme1n1 /mnt/promerge   # reattach (keeps data)
   source .venv/bin/activate && source cloud/env.sh
   python -u experiments/run.py --experiment libero_spatial --baseline promerge_film \
     --mode train --epochs 250 --batch_size 32         # ← prints "↻ Resuming from epoch N"
   ```
3. **Unattended**: put that train command in EC2 **user-data** or a **systemd** service so a
   replacement spot resumes by itself. `cloud/run_all.sh` is also re-run-safe (each variant
   resumes from its own `train_state.pt`).

Cross-AZ alternative (if you can't pin the AZ): use **S3** as the durable home and
`aws s3 sync checkpoints/ s3://bucket/ckpts/` periodically; on boot, `aws s3 sync` it back
before launching training. EBS-reattach in one AZ is simpler — prefer it unless you need AZ flexibility.

## 3. Get the data

**LIBERO-Spatial (5.8 GB)** — download on the box (same official source as local):
```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='yifengzhu-hf/LIBERO-datasets', repo_type='dataset', \
local_dir='data', allow_patterns='libero_spatial/*')"
```

Sandbox data (`episodes_500_tuple.hdf5`) is **not needed** for LIBERO train+eval. Only upload
it if you also want the sandbox emulation here:
```bash
# GCP:  gsutil cp gs://<bucket>/episodes_500_tuple.hdf5 data/
# AWS:  aws s3 cp s3://<bucket>/episodes_500_tuple.hdf5 data/
# or:   scp episodes_500_tuple.hdf5 <box>:~/ProMerge/data/
```

Verify rendering + LIBERO import:
```bash
python -c "import os;os.environ['MUJOCO_GL']='egl';from libero.libero import benchmark;print('OK',list(benchmark.get_benchmark_dict()))"
```

---

## 4. Train (use tmux so it survives disconnect)

```bash
tmux new -s train
source .venv/bin/activate && source cloud/env.sh
for b in promerge_film promerge_only thinkproprio; do
  python -u experiments/run.py --experiment libero_spatial --baseline "$b" \
    --mode train --epochs 250 --batch_size 32
done
# Ctrl-b d to detach;  tmux attach -t train to return.  Watch on wandb.
```
- Aug (RandomShift) + early-stop are on by default; best-val weights → `policy_best.ckpt`.
- On a 24 GB GPU you can raise `--batch_size` (32–64) and `DATALOADER_WORKERS`.

---

## 5. Official LIBERO-Spatial eval (success rate)

Use the **val-selected** `policy_best.ckpt` (not `policy_last`):
```bash
for b in promerge_film promerge_only thinkproprio; do
  python -u experiments/libero_spatial/eval.py --baseline "$b" \
    --ckpt checkpoints/libero_spatial/$b/policy_best.ckpt --episodes_per_task 20
done
```
Prints per-task + suite-average success rate. (First eval run JIT-compiles MuJoCo; the first
task is slow.) If images look wrong / success is ~0, re-check `FLIP_IMAGES` in
`experiments/libero_spatial/config.py` against the env's image convention.

---

## 6. Pull results back

```bash
# GCP:  gcloud compute scp --recurse promerge-gpu:~/ProMerge/checkpoints/libero_spatial ./ --zone=us-central1-a
# AWS:  scp -i key.pem -r ubuntu@<ip>:~/ProMerge/checkpoints/libero_spatial ./
```

**Remember to stop/delete the instance when done** (GPU boxes bill by the hour).

---

## Notes / gotchas
- `data/`, `checkpoints/` are git-ignored — they live only on the box; copy back what you need.
- If `pip install -e external/LIBERO` downgraded torch to a CPU build, re-run
  `pip install -r requirements.txt` to restore the CUDA wheel.
- `eval.py` is written against LIBERO's `lifelong/metric.py` rollout but **could not be tested
  locally** (no GPU/libero on the Mac) — sanity-check the first task's success before trusting
  a full sweep.
