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
# Run the mirror rather than printing it. mc runs in a throwaway container on the compose network, so this
# needs no host-side mc install and no alias the operator has to remember to configure.
# --user: mc runs as root by default, which leaves a backup directory the operator cannot
# read, move or delete without sudo. Found the hard way.
docker run --rm --user "$(id -u):$(id -g)" \
  --network "$(docker compose ps --format '{{.Name}}' postgres | head -1 | sed 's/-postgres-1$//')_default" \
  -v "$(cd "$DEST" && pwd):/backup" \
  -e MC_HOST_lbx="http://${MINIO_USER}:${MINIO_PASS}@minio:9000" \
  minio/mc:latest mirror --overwrite --quiet "lbx/${BUCKET}" /backup/minio

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
