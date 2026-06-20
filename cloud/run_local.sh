#!/usr/bin/env bash
# Local long-run orchestration for ProMerge on this RTX-4090 box.
# Adapted from cloud/run_all.sh but WITHOUT auto-shutdown and WITHOUT bucket
# upload (this is a persistent local machine, not an ephemeral cloud VM).
#
# Trains all 6 baselines on LIBERO-Spatial, then evals each (LIBERO official
# success rate) into RESULTS_libero_spatial.md. wandb logs to project "ProMerge".
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
source cloud/env.sh 2>/dev/null || true
export MUJOCO_GL=egl

BASELINES="monolithic_act random_prune tome_clustering promerge_only thinkproprio promerge_film"
# NOTE: dataset is now per-frame (~56k train samples, 1756 batches/epoch) after
# the data.py fix, vs ~450 samples before. So far fewer epochs are needed; 50 is
# plenty and early stopping (patience=5) will cut it shorter if it plateaus.
EPOCHS="${EPOCHS:-50}"; BATCH="${BATCH:-32}"; EPISODES="${EPISODES:-20}"

echo "######## ProMerge LOCAL run | epochs=$EPOCHS batch=$BATCH | $(date) ########"

for b in $BASELINES; do
  echo "==================== TRAIN $b $(date) ===================="
  python -u experiments/run.py --experiment libero_spatial --baseline "$b" \
    --mode train --epochs "$EPOCHS" --batch_size "$BATCH"
done

SUMMARY="RESULTS_libero_spatial.md"
{ echo "# LIBERO-Spatial local results — $(date)"; echo; } > "$SUMMARY"
for b in $BASELINES; do
  CK="checkpoints/libero_spatial/$b/policy_best.ckpt"
  [ -f "$CK" ] || CK="checkpoints/libero_spatial/$b/policy_last.ckpt"
  if [ ! -f "$CK" ]; then echo "## $b — NO CHECKPOINT, skipped" | tee -a "$SUMMARY"; continue; fi
  { echo "## $b (\`$CK\`)"; echo '```'; } >> "$SUMMARY"
  echo "==================== EVAL $b $(date) ===================="
  python -u experiments/libero_spatial/eval.py --baseline "$b" --ckpt "$CK" \
    --episodes_per_task "$EPISODES" 2>&1 | tee -a "$SUMMARY"
  { echo '```'; echo; } >> "$SUMMARY"
done

echo "######## DONE $(date) — results in $SUMMARY (NO shutdown, NO upload) ########"
