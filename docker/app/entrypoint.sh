#!/bin/sh
# Run the application as an unprivileged user.
#
# The image used to run everything as root: the API, the workers, and every model file they load. Nothing
# in the product needs that, and it turns any code-execution bug (an archive, a model file, a dependency)
# into root inside the container and root-owned files on every mounted volume.
#
# The container still starts as root, for one reason: volumes created by an earlier version are owned by
# root, and the process could not write its own models or scratch directory without fixing that first. So
# this hands the writable directories to the application user, then gives up root for good with setpriv
# and execs the real command. The application never runs as root. Started with an explicit --user (the
# backup script does), it skips all of this and runs the command as that user.
set -eu

APP_USER=labelox

if [ "$(id -u)" = "0" ]; then
  for d in /app/models /app/.scratch /app/runs /home/"$APP_USER"; do
    mkdir -p "$d"
    # Only what is not already ours: a recursive chown of a models volume holding tens of gigabytes on
    # every start would make each restart as slow as the disk is large.
    find "$d" \( ! -user "$APP_USER" -o ! -group "$APP_USER" \) -exec chown -h "$APP_USER":"$APP_USER" {} + 2>/dev/null || true
  done
  exec setpriv --reuid "$APP_USER" --regid "$APP_USER" --init-groups --inh-caps=-all "$@"
fi

exec "$@"
