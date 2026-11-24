#!/usr/bin/env bash
# The backup a timer runs: take one, keep the last few, and optionally copy it off the machine.
#
# `make backup` was the whole story, and it only ran when somebody remembered. On a single-box product the
# box is the only copy of every label a team has made, so a backup that depends on memory is no backup.
# This is what `labeloxav backup-schedule on` (or `make backup-schedule`) points a daily timer at.
#
#   LBX_BACKUP_DIR      where backups are kept             default: <install>/.scratch/backups
#   LBX_BACKUP_KEEP     how many to keep                   default: 7
#   LBX_BACKUP_REMOTE   an rsync destination for a copy off the machine, e.g. backup@nas:/srv/labelox
#                       or /mnt/nas/labelox. Unset: nothing leaves the box, and the run says so.
#
# Each backup is a complete directory that restore.sh accepts on its own. Objects that have not changed
# since the previous backup are hard links to it, so keeping seven costs one copy of the corpus plus what
# changed. Only one run at a time; a failed run leaves the earlier backups alone and exits non-zero, which
# is what makes the timer's failure visible in `systemctl status` and `labeloxav backup-schedule status`.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  while IFS='=' read -r k v; do
    case "$k" in LBX_BACKUP_*) ;; *) continue ;; esac
    v="${v%\"}"; v="${v#\"}"
    [ -z "${!k+x}" ] && export "$k=$v"
  done < <(grep -E '^LBX_BACKUP_[A-Z_]*=' .env || true)
fi

DIR="${LBX_BACKUP_DIR:-$ROOT/.scratch/backups}"
KEEP="${LBX_BACKUP_KEEP:-7}"
REMOTE="${LBX_BACKUP_REMOTE:-}"
case "$KEEP" in ''|*[!0-9]*) echo "LBX_BACKUP_KEEP must be a whole number, got '$KEEP'" >&2; exit 2 ;; esac
[ "$KEEP" -ge 1 ] || { echo "LBX_BACKUP_KEEP must be at least 1" >&2; exit 2; }

mkdir -p "$DIR"
DIR="$(cd "$DIR" && pwd)"
STATUS="$DIR/last_run.json"

exec 9>"$DIR/.lock"
if ! flock -n 9; then
  echo "another backup is still running in $DIR; not starting a second" >&2
  exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$DIR/$STAMP"
# Two runs inside one second would otherwise write into the same directory, the second over the first.
[ -e "$DEST" ] && DEST="${DEST}_$$"
STAMP="$(basename "$DEST")"

# A complete backup is a directory holding a MANIFEST; anything else here is a run that did not finish.
complete() { find "$DIR" -mindepth 2 -maxdepth 2 -name MANIFEST -printf '%h\n' 2>/dev/null | sort; }

fail() {
  local why="$1"
  printf '{"ok": false, "at": "%s", "error": "%s"}\n' "$(date -Iseconds)" "$why" > "$STATUS"
  # A half-written directory must not be mistaken for a backup, by a person or by the next run.
  [ -d "$DEST" ] && [ ! -f "$DEST/MANIFEST" ] && rm -rf "$DEST"
  echo "backup FAILED: $why" >&2
  exit 1
}
trap 'fail "the backup command exited with an error; see the log above"' ERR

PREV="$(complete | tail -1 || true)"
echo "==> scheduled backup to $DEST (keeping $KEEP)"
LBX_BACKUP_LINK_DEST="$PREV" ./scripts/backup.sh "$DEST"
[ -f "$DEST/MANIFEST" ] || fail "backup.sh finished without writing a MANIFEST"

if [ -n "$REMOTE" ]; then
  command -v rsync >/dev/null 2>&1 || fail "LBX_BACKUP_REMOTE is set but rsync is not installed"
  echo "==> copying to $REMOTE"
  # -H keeps the hard links between snapshots as links on the other side when the earlier snapshots are
  # there too, so the remote copy costs what the local one does. Sent from the backup root so the
  # relative layout (and therefore the links) survives.
  ( cd "$DIR" && rsync -aH --relative "./$STAMP" "$REMOTE/" ) || fail "the copy to $REMOTE failed"
fi

# Prune after the new backup exists and has been copied, never before: a failed run must not cost an old one.
mapfile -t ALL < <(complete)
if [ "${#ALL[@]}" -gt "$KEEP" ]; then
  for old in "${ALL[@]:0:$(( ${#ALL[@]} - KEEP ))}"; do
    case "$old" in "$DIR"/*) echo "==> removing $old"; rm -rf "$old" ;; esac
  done
fi

trap - ERR
SIZE="$(du -sh "$DEST" 2>/dev/null | cut -f1)"
printf '{"ok": true, "at": "%s", "dir": "%s", "size": "%s", "kept": %s, "remote": "%s"}\n' \
  "$(date -Iseconds)" "$DEST" "$SIZE" "$(complete | wc -l)" "$REMOTE" > "$STATUS"
echo "==> done: $DEST ($SIZE)"
[ -n "$REMOTE" ] || echo "    note: LBX_BACKUP_REMOTE is not set, so this backup is on the same disk as the data it protects"
