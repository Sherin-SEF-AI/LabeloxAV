#!/usr/bin/env bash
# Back up both halves of the corpus, together, into one timestamped directory.
#
# The old `make backup` had three faults and every one of them was silent.
#
# It piped pg_dump into gzip, so the recipe's exit status was gzip's. A dump that failed - wrong password,
# container not running, disk full mid-stream - produced a small, valid .gz and a green target. The first
# time you would learn otherwise is the restore.
#
# It hardcoded the database name `labeloxav`, ignoring POSTGRES_DB, so a deployment that renamed its
# database backed up nothing (or, worse, something else).
#
# And it printed the MinIO mirror command instead of running it. docs/DEPLOY.md is blunt that a
# Postgres-only restore leaves a corpus of dangling references and is not a restore, and the tool that
# would have made it one was a line of prose.
#
#   ./scripts/backup.sh [destination]      default: .scratch/backups/<timestamp>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load .env as DEFAULTS, not overrides: an explicitly exported POSTGRES_DB (or a caller running against a
# second deployment) must win over the file, and `set -a; . ./.env` silently clobbers it the other way.
if [ -f .env ]; then
  while IFS='=' read -r k v; do
    case "$k" in ''|\#*) continue ;; esac
    v="${v%\"}"; v="${v#\"}"
    [ -z "${!k+x}" ] && export "$k=$v"
  done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' .env)
fi

PG_USER="${POSTGRES_USER:-labelox}"
PG_DB="${POSTGRES_DB:-labeloxav}"
BUCKET="${MINIO_BUCKET:-labeloxav}"
MINIO_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_PASS="${MINIO_ROOT_PASSWORD:-minioadmin}"

DEST="${1:-.scratch/backups/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$DEST"

echo "==> Postgres: ${PG_DB} as ${PG_USER}"
# No pipe into gzip: pg_dump's own -Z writes the compression, so a failure here is this command's failure
# rather than gzip's success at compressing an error message.
docker compose exec -T postgres pg_dump -U "$PG_USER" -d "$PG_DB" -Z 6 -f /tmp/pg_dump.sql.gz
docker compose cp postgres:/tmp/pg_dump.sql.gz "$DEST/postgres.sql.gz"
docker compose exec -T postgres rm -f /tmp/pg_dump.sql.gz

if [ ! -s "$DEST/postgres.sql.gz" ]; then
  echo "ERROR: the Postgres dump is empty; refusing to call this a backup" >&2
  exit 1
fi

echo "==> MinIO: bucket ${BUCKET}"
NET="$(docker compose ps --format '{{.Name}}' postgres | head -1 | sed 's/-postgres-1$//')_default"
# The bucket is mirrored by scripts/bucket_mirror.py, which needs boto3 alone. It used to run MinIO's `mc`
# image, which MinIO no longer publishes (Docker Hub withdrew it, quay.io refuses anonymous pulls), so this
# failed on any machine without a cached copy. In a deployment it runs in the app image on the compose
# network, offline; on a development checkout, from the virtualenv against the published MinIO port.
# Credentials travel in the environment, never on a command line where the process list would show them.
# --user: the files belong to whoever runs this, so the operator can move or delete them without sudo.
export MINIO_ROOT_USER="$MINIO_USER" MINIO_ROOT_PASSWORD="$MINIO_PASS"
APP_IMAGE="labeloxav/app:${LBX_VERSION:-latest}"
DIR_ABS="$(cd "$DEST" && pwd)"
if docker image inspect "$APP_IMAGE" >/dev/null 2>&1; then
  docker run --rm --user "$(id -u):$(id -g)" --network "$NET" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
    -e MINIO_ROOT_USER -e MINIO_ROOT_PASSWORD -v "$DIR_ABS:/backup" --entrypoint python "$APP_IMAGE" \
    /app/scripts/bucket_mirror.py download --endpoint http://minio:9000 --bucket "$BUCKET" --dir /backup/minio
elif [ -x .venv/bin/python ]; then
  .venv/bin/python scripts/bucket_mirror.py download --endpoint "http://localhost:${MINIO_PORT:-9000}" \
    --bucket "$BUCKET" --dir "$DIR_ABS/minio"
else
  echo "ERROR: neither the app image ($APP_IMAGE) nor a .venv is here to run scripts/bucket_mirror.py" >&2
  exit 1
fi

OBJECTS=$(find "$DEST/minio" -type f 2>/dev/null | wc -l)
echo "==> wrote $DEST"
echo "    postgres.sql.gz  $(du -h "$DEST/postgres.sql.gz" | cut -f1)"
echo "    minio/           ${OBJECTS} objects"

# The pair is the unit. Recording what went into it together is what lets restore refuse a half set.
cat > "$DEST/MANIFEST" <<MANIFEST
taken_at=$(date -Iseconds)
postgres_db=${PG_DB}
minio_bucket=${BUCKET}
minio_objects=${OBJECTS}
MANIFEST

echo
echo "Restore with: ./scripts/restore.sh $DEST"
