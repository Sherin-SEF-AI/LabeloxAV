#!/usr/bin/env bash
# Turn the daily backup on or off, and say whether the last one worked.
#
#   ./scripts/backup_schedule.sh on [HH:MM]     default 02:30, local time
#   ./scripts/backup_schedule.sh off
#   ./scripts/backup_schedule.sh status
#
# A systemd timer, not a container: the backup drives `docker compose` from the host, and a timer is the
# one scheduler every supported host already has, survives a reboot, catches up on a run it missed while
# the machine was off (Persistent=true), and reports a failure where an operator already looks. As root it
# installs a system timer; as an ordinary user, a user timer, which needs lingering enabled to run while
# that user is logged out (the command says so when it is not).
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT="labeloxav-backup"
cmd="${1:-status}"
when="${2:-02:30}"

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemd is not available here. Run this from cron instead:" >&2
  echo "  30 2 * * *  $ROOT/scripts/backup_scheduled.sh >> $ROOT/.scratch/backups/backup.log 2>&1" >&2
  exit 1
fi

if [ "$(id -u)" -eq 0 ]; then
  SCTL=(systemctl); UNIT_DIR=/etc/systemd/system; WANTED=multi-user.target
else
  SCTL=(systemctl --user); UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"; WANTED=default.target
fi

last_run() {
  local dir status
  dir="$(sed -n 's/^LBX_BACKUP_DIR=\(.*\)$/\1/p' "$ROOT/.env" 2>/dev/null | tail -1)"
  status="${LBX_BACKUP_DIR:-${dir:-$ROOT/.scratch/backups}}/last_run.json"
  if [ -f "$status" ]; then echo "last run: $(cat "$status")"; else echo "last run: none recorded yet"; fi
}

case "$cmd" in
  on)
    case "$when" in [0-2][0-9]:[0-5][0-9]) ;; *) echo "time must be HH:MM, got '$when'" >&2; exit 2 ;; esac
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/$UNIT.service" <<SERVICE
[Unit]
Description=LabeloxAV backup (database and stored files, together)
After=docker.service

[Service]
Type=oneshot
WorkingDirectory=$ROOT
ExecStart=$ROOT/scripts/backup_scheduled.sh
# A backup competes with labelling for the disk; let the product win.
Nice=10
IOSchedulingClass=idle
SERVICE
    cat > "$UNIT_DIR/$UNIT.timer" <<TIMER
[Unit]
Description=Daily LabeloxAV backup

[Timer]
OnCalendar=*-*-* $when:00
# Run at the next start if the machine was off at the scheduled time.
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
TIMER
    "${SCTL[@]}" daemon-reload
    "${SCTL[@]}" enable --now "$UNIT.timer"
    echo "daily backup on, at $when. Next run:"
    "${SCTL[@]}" list-timers "$UNIT.timer" --no-pager 2>/dev/null | sed -n '1,2p' || true
    if [ "$(id -u)" -ne 0 ] && command -v loginctl >/dev/null 2>&1 \
       && [ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null)" != "yes" ]; then
      echo "note: this is a user timer and lingering is off, so it only runs while you are logged in."
      echo "      sudo loginctl enable-linger $(id -un)    makes it run regardless."
    fi
    grep -qE '^LBX_BACKUP_REMOTE=.+' "$ROOT/.env" 2>/dev/null \
      || echo "note: LBX_BACKUP_REMOTE is not set in .env, so backups stay on this machine's disk."
    ;;
  off)
    "${SCTL[@]}" disable --now "$UNIT.timer" 2>/dev/null || true
    rm -f "$UNIT_DIR/$UNIT.timer" "$UNIT_DIR/$UNIT.service"
    "${SCTL[@]}" daemon-reload
    echo "daily backup off. Existing backups were left where they are."
    ;;
  status)
    if "${SCTL[@]}" is-enabled "$UNIT.timer" >/dev/null 2>&1; then
      echo "daily backup: on"
      "${SCTL[@]}" list-timers "$UNIT.timer" --no-pager 2>/dev/null | sed -n '1,2p' || true
    else
      echo "daily backup: off   (turn it on with: $0 on)"
    fi
    last_run
    ;;
  *)
    echo "usage: $0 on [HH:MM] | off | status" >&2
    exit 2
    ;;
esac
