#!/usr/bin/env bash
#
# The annotation editor, driven in a real browser against a throwaway database.
#
#   ./scripts/e2e_editor.sh          (or: make e2e-editor)
#
# The unit tiers test the editor's reducer and its key handling; nothing tested the editor: that an arrow
# key moves a box and the move is saved, that a paste on another frame lands, that a second person opening
# a frame is view-only and the server refuses their save. Those need the browser, the API and the database
# to agree, and they were checked by hand once. This is that check, kept.
#
# It needs the infrastructure up (make up) and nothing else: it creates its own database and bucket,
# migrates, seeds six synthetic frames and three users (scripts/e2e_seed.py), starts the API and the web
# app on free ports, runs web/e2e/editor.spec.ts and presence.spec.ts, and removes all of it afterwards,
# pass or fail. No GPU, no network, no model weights. Set LBX_E2E_KEEP=1 to leave it running.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
WORK="$(mktemp -d)"
DB="labeloxav_e2e_$$"
BUCKET="labeloxav-e2e-$$"

free_port() { "$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
API_PORT="$(free_port)"
WEB_PORT="$(free_port)"

export LBX_POSTGRES__DB="$DB" LBX_MINIO__BUCKET="$BUCKET" LBX_REDIS__DB="${LBX_E2E_REDIS_DB:-9}"
export LBX_AUTH__ENABLED=true LBX_PII__ENABLED=false LBX_JOBS__AUTO_RESUME=false
[ -n "${MINIO_PORT:-}" ] && export LBX_MINIO__ENDPOINT="${LBX_MINIO__ENDPOINT:-http://localhost:${MINIO_PORT}}"

API_PID="" WEB_PID=""
cleanup() {
  status=$?
  if [ "${LBX_E2E_KEEP:-0}" = "1" ]; then
    echo "LBX_E2E_KEEP=1: left running. web http://127.0.0.1:${WEB_PORT}  api http://127.0.0.1:${API_PORT}  db ${DB}  dir ${WORK}"
    exit "$status"
  fi
  [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null || true
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
  sleep 1
  "$PY" - <<PY || true
import boto3, psycopg
from botocore.config import Config
from core.config import get_settings
s = get_settings()
pg, m = s.postgres, s.minio
with psycopg.connect(host=pg.host, port=pg.port, user=pg.user, password=pg.password, dbname="postgres",
                     autocommit=True) as c:
    c.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", ("$DB",))
    c.execute('DROP DATABASE IF EXISTS "$DB"')
s3 = boto3.resource("s3", endpoint_url=m.endpoint, aws_access_key_id=m.access_key,
                    aws_secret_access_key=m.secret_key, region_name="us-east-1",
                    config=Config(s3={"addressing_style": "path"}))
b = s3.Bucket("$BUCKET")
try:
    # One at a time: the batch delete needs a Content-MD5 header this client no longer sends, and the
    # object store refuses it, which left every run's bucket behind without saying so.
    for obj in b.objects.all():
        obj.delete()
    b.delete()
except Exception as exc:
    print(f"note: could not remove the bucket $BUCKET: {exc}")
PY
  if [ "$status" -ne 0 ]; then
    echo "--- last lines of the API log" >&2; tail -20 "${WORK}/api.log" >&2 || true
    echo "--- last lines of the web log" >&2; tail -20 "${WORK}/web.log" >&2 || true
  fi
  rm -rf "$WORK"
  exit "$status"
}
trap cleanup EXIT

echo "==> database ${DB}"
"$PY" - <<PY
import psycopg
from core.config import get_settings
pg = get_settings().postgres
with psycopg.connect(host=pg.host, port=pg.port, user=pg.user, password=pg.password, dbname="postgres",
                     autocommit=True) as c:
    c.execute('CREATE DATABASE "$DB"')
PY
"${ROOT}/.venv/bin/alembic" upgrade head > "${WORK}/migrate.log" 2>&1 || { tail -30 "${WORK}/migrate.log" >&2; exit 1; }
"$PY" scripts/seed_ontology.py > /dev/null 2>&1
"$PY" -m scripts.ensure_bucket > /dev/null 2>&1
"$PY" -m scripts.e2e_seed --out "$WORK" 2> "${WORK}/seed.log" || { tail -30 "${WORK}/seed.log" >&2; exit 1; }

echo "==> API on ${API_PORT}, web on ${WEB_PORT}"
"${ROOT}/.venv/bin/uvicorn" services.api.main:app --host 127.0.0.1 --port "$API_PORT" > "${WORK}/api.log" 2>&1 &
API_PID=$!
( cd web && LBX_API_URL="http://127.0.0.1:${API_PORT}" exec npx next dev -p "$WEB_PORT" ) > "${WORK}/web.log" 2>&1 &
WEB_PID=$!

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${API_PORT}/api/readyz" >/dev/null 2>&1 \
     && curl -fsS -o /dev/null "http://127.0.0.1:${WEB_PORT}/login" 2>/dev/null; then ready=1; break; fi
  sleep 2
done
[ "${ready:-0}" = "1" ] || { echo "the app did not come up" >&2; exit 1; }

cd web
LBX_E2E_BASE_URL="http://localhost:${WEB_PORT}" LBX_E2E_API_URL="http://127.0.0.1:${API_PORT}" \
LBX_E2E_FIXTURE="${WORK}/fixture.json" \
  npx playwright test e2e/editor.spec.ts e2e/presence.spec.ts --reporter=line
