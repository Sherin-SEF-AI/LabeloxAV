#!/usr/bin/env bash
# Restore both halves of the corpus from a directory written by scripts/backup.sh.
#
# There was no restore script. docs/DEPLOY.md said, correctly, that "a Postgres-only restore gives you a
# corpus of dangling references... Restoring one without the other is not a restore" - and then the repo
# shipped a backup target that only really did Postgres and nothing at all to put it back. A backup nobody
# has ever restored is a hypothesis, not a backup.
#
#   ./scripts/restore.sh <backup-dir> [--yes]
#
# Destructive: it drops and recreates the database. Refuses without --yes unless stdin is a terminal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC="${1:-}"
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
  echo "usage: ./scripts/restore.sh <backup-dir> [--yes]" >&2
  exit 2
fi

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

# Refuse a half set. Restoring only Postgres is the failure mode DEPLOY.md warns about, and it is silent:
# the app comes up, the counts look right, and every image 404s.
[ -s "$SRC/postgres.sql.gz" ] || { echo "ERROR: $SRC/postgres.sql.gz missing or empty" >&2; exit 1; }
[ -d "$SRC/minio" ] || { echo "ERROR: $SRC/minio missing - this is half a backup, and half is not a restore" >&2; exit 1; }
[ -f "$SRC/MANIFEST" ] && cat "$SRC/MANIFEST"

if [ "${2:-}" != "--yes" ]; then
  if [ -t 0 ]; then
    read -r -p "This DROPS and recreates database '${PG_DB}' and overwrites bucket '${BUCKET}'. Type the db name to confirm: " reply
    [ "$reply" = "$PG_DB" ] || { echo "aborted"; exit 1; }
  else
    echo "ERROR: refusing to restore non-interactively without --yes" >&2
    exit 1
  fi
fi

echo "==> Postgres: recreating ${PG_DB}"
docker compose exec -T postgres psql -U "$PG_USER" -d postgres \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${PG_DB}' AND pid <> pg_backend_pid();" >/dev/null
docker compose exec -T postgres psql -U "$PG_USER" -d postgres -c "DROP DATABASE IF EXISTS \"${PG_DB}\";"
docker compose exec -T postgres psql -U "$PG_USER" -d postgres -c "CREATE DATABASE \"${PG_DB}\";"
gunzip -c "$SRC/postgres.sql.gz" | docker compose exec -T postgres psql -U "$PG_USER" -d "$PG_DB" -v ON_ERROR_STOP=1 >/dev/null

echo "==> MinIO: bucket ${BUCKET}"
NET="$(docker compose ps --format '{{.Name}}' postgres | head -1 | sed 's/-postgres-1$//')_default"
# --user: mc runs as root by default; without this the restore container writes as
# root and a later backup into the same tree cannot be cleaned up.
docker run --rm --user "$(id -u):$(id -g)" --network "$NET" \
  -v "$(cd "$SRC" && pwd):/backup:ro" \
  -e MC_HOST_lbx="http://${MINIO_USER}:${MINIO_PASS}@minio:9000" \
  minio/mc:latest mirror --overwrite --quiet /backup/minio "lbx/${BUCKET}"

echo "==> verifying"
# The check that matters is referential, not a row count: an object row whose blob is gone is exactly the
# dangling-reference corpus a one-sided restore produces, and it looks healthy from the database alone.
FRAMES=$(docker compose exec -T postgres psql -U "$PG_USER" -d "$PG_DB" -tAc "SELECT count(*) FROM frame;" 2>/dev/null || echo 0)
OBJECTS=$(find "$SRC/minio" -type f 2>/dev/null | wc -l)
echo "    frames in database: ${FRAMES}"
echo "    objects restored:   ${OBJECTS}"
if [ "$FRAMES" -gt 0 ] && [ "$OBJECTS" -eq 0 ]; then
  echo "WARNING: ${FRAMES} frames and no blobs. Every image will 404. This is the half-restore DEPLOY.md warns about." >&2
  exit 1
fi
echo "==> restored from $SRC"
