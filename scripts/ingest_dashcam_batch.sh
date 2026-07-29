#!/usr/bin/env bash
# Batch-ingest the real dashcam videos. Resumable + idempotent: each video's basename is recorded in a
# done-list once ingested, so a re-run skips it (a fresh ingest would create a duplicate session otherwise).
set -u
cd /home/jo/Desktop/LabeloxAV
VID_DIR="/home/jo/Documents/dashcam-videos"
WORK=".perception_work"
mkdir -p "$WORK"
LOG="$WORK/ingest_batch.log"
DONE="$WORK/ingested_videos.txt"
touch "$DONE"

mapfile -t vids < <(ls "$VID_DIR"/*.MP4 2>/dev/null | sort)
total=${#vids[@]}
n=0
for v in "${vids[@]}"; do
  n=$((n + 1))
  base="$(basename "$v")"
  if grep -qxF "$base" "$DONE"; then
    echo "[$n/$total] SKIP (done) $base" >> "$LOG"
    continue
  fi
  echo "[$n/$total] ingesting $base ..." >> "$LOG"
  if .venv/bin/python -m services.ingest.run --video "$v" --vehicle DASHCAM-01 --city BLR --cam cam_f \
        >> "$LOG" 2>&1; then
    echo "$base" >> "$DONE"
    echo "[$n/$total] OK $base" >> "$LOG"
  else
    echo "[$n/$total] FAILED $base" >> "$LOG"
  fi
done
echo "BATCH DONE: $(wc -l < "$DONE")/$total videos ingested" >> "$LOG"
