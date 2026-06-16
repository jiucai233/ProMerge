#!/usr/bin/env bash
# End-to-end on a cloud GPU box:  [train] -> eval all -> results -> commit/push -> [upload ckpts] -> [shutdown]
#
# Env flags:
#   TRAIN=1                 train the 3 variants first (default 0: assume already trained)
#   EPOCHS=250 BATCH=32 EPISODES=20
#   BUCKET=gs://b/p | s3://b/p   upload checkpoints before shutdown (recommended for safety)
#   SHUTDOWN=1              poweroff at end (default 1). SHUTDOWN=0 disables. SHUTDOWN=force ignores the safety guard.
#
# Safety: shuts down ONLY if results were persisted (git push OK or BUCKET upload OK),
# so a misconfigured box won't power off and lose the (gitignored) checkpoints.
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
source cloud/env.sh 2>/dev/null || true

BASELINES="promerge_film promerge_only thinkproprio"
EPOCHS="${EPOCHS:-250}"; BATCH="${BATCH:-32}"; EPISODES="${EPISODES:-20}"; SHUTDOWN="${SHUTDOWN:-1}"

if [ "${TRAIN:-0}" = "1" ]; then
  for b in $BASELINES; do
    echo "==================== TRAIN $b $(date) ===================="
    python -u experiments/run.py --experiment libero_spatial --baseline "$b" \
      --mode train --epochs "$EPOCHS" --batch_size "$BATCH"
  done
fi

SUMMARY="RESULTS_libero_spatial.md"
{ echo "# LIBERO-Spatial cloud results — $(date)"; echo; } > "$SUMMARY"
for b in $BASELINES; do
  CK="checkpoints/libero_spatial/$b/policy_best.ckpt"
  [ -f "$CK" ] || CK="checkpoints/libero_spatial/$b/policy_last.ckpt"
  if [ ! -f "$CK" ]; then echo "## $b — NO CHECKPOINT, skipped" | tee -a "$SUMMARY"; continue; fi
  { echo "## $b (\`$CK\`)"; echo '```'; } >> "$SUMMARY"
  python -u experiments/libero_spatial/eval.py --baseline "$b" --ckpt "$CK" \
    --episodes_per_task "$EPISODES" 2>&1 | tee -a "$SUMMARY"
  { echo '```'; echo; } >> "$SUMMARY"
done

# --- persist results: commit the summary (data/ & checkpoints/ are gitignored, not committed) ---
persisted=0
git config user.email >/dev/null 2>&1 || git config user.email "cloud@promerge"
git config user.name  >/dev/null 2>&1 || git config user.name  "promerge-cloud"
git add "$SUMMARY" 2>/dev/null && git commit -m "cloud: LIBERO-Spatial eval $(date +%F_%H%M)" 2>/dev/null || true
if git push 2>/dev/null; then echo "✅ results pushed"; persisted=1; else
  echo "⚠️ git push failed (no remote/auth) — $SUMMARY committed locally only"; fi

# --- optional: upload checkpoints to a bucket (recommended; they're gitignored) ---
if [ -n "${BUCKET:-}" ]; then
  case "$BUCKET" in
    gs://*) gsutil -m cp -r checkpoints/libero_spatial "$BUCKET/" && persisted=1 ;;
    s3://*) aws s3 cp --recursive checkpoints/libero_spatial "${BUCKET%/}/libero_spatial" && persisted=1 ;;
    *) echo "⚠️ unknown BUCKET scheme: $BUCKET" ;;
  esac
fi

# --- shutdown (guarded) ---
if [ "$SHUTDOWN" = "1" ] || [ "$SHUTDOWN" = "force" ]; then
  if [ "$persisted" = "1" ] || [ "$SHUTDOWN" = "force" ]; then
    echo "Shutting down in 30s (Ctrl-C to cancel)..."; sleep 30; sudo shutdown -h now
  else
    echo "⛔ NOT shutting down: results not persisted (no git push + no BUCKET upload)."
    echo "   Set BUCKET=gs://... (or fix git remote/auth), or re-run with SHUTDOWN=force."
  fi
fi
