#!/usr/bin/env bash
# Mount a persistent disk (GCP Persistent Disk / AWS EBS) and point the repo's data/ and
# checkpoints/ at it via symlinks — so they SURVIVE instance delete/recreate and aren't
# stuck on the ephemeral boot disk. Run ONCE per fresh disk; reattach the disk next time.
#
#   # find the device: lsblk
#   sudo bash cloud/mount_disk.sh /dev/nvme1n1 /mnt/promerge
#
# Safety: formats ONLY if the device has no filesystem yet (never wipes an existing disk).
set -euo pipefail
DEV="${1:?usage: mount_disk.sh <device, e.g. /dev/nvme1n1 or /dev/sdb> [mountpoint]}"
MNT="${2:-/mnt/promerge}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

if ! sudo blkid "$DEV" >/dev/null 2>&1; then
  echo "==> No filesystem on $DEV — formatting ext4 (treating as a NEW disk)."
  sudo mkfs.ext4 -m 0 -F "$DEV"
else
  echo "==> $DEV already has a filesystem — NOT formatting, just mounting (data kept)."
fi

sudo mkdir -p "$MNT"
sudo mount -o discard,defaults "$DEV" "$MNT"
sudo chown -R "$USER:$USER" "$MNT"
mkdir -p "$MNT/data" "$MNT/checkpoints"

# Symlink repo data/ + checkpoints/ to the persistent disk (no code change needed).
for d in data checkpoints; do
  if [ -e "$REPO/$d" ] && [ ! -L "$REPO/$d" ]; then
    echo "==> Migrating existing $REPO/$d -> $MNT/$d"
    rsync -a "$REPO/$d/" "$MNT/$d/" && rm -rf "$REPO/$d"
  fi
  ln -sfn "$MNT/$d" "$REPO/$d"
done

echo "✅ data/ and checkpoints/ now live on $DEV ($MNT)."
echo "   Persist across reboots (optional):"
echo "     echo \"UUID=\$(sudo blkid -s UUID -o value $DEV) $MNT ext4 discard,defaults,nofail 0 2\" | sudo tee -a /etc/fstab"
