#!/usr/bin/env bash
#
# LabeloxAV one-command installer.
#
#   ./scripts/install.sh
#
# Brings up the whole product with Docker: infrastructure, schema, ontology, API, and web app, then creates
# the first admin and prints how to sign in. Safe to re-run: it never regenerates secrets that already exist
# and never re-creates an admin that is already there.
#
# Why it generates secrets rather than asking: core/config.py refuses the built-in dev defaults on any
# non-local deployment and will not boot, which is correct, but it means a first run otherwise fails listing
# seven variables the operator has never seen. Making a human invent seven high-entropy strings is also how
# you end up with seven weak ones.

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_FILE="${REPO_ROOT}/.env"
# Plain `docker compose`: the files and optional services come from COMPOSE_FILE and COMPOSE_PROFILES, which
# this script writes into .env below and Compose reads from there on every later command.
COMPOSE=(docker compose)
# The same command as a string, for the messages that tell an operator what to run next.
COMPOSE_HINT="docker compose"

# Colours only when attached to a terminal, so piping to a file or a log stays readable.
if [ -t 1 ]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; N=""
fi

say()  { printf '%s\n' "$*"; }
step() { printf '\n%s==>%s %s\n' "$B" "$N" "$*"; }
ok()   { printf '%s  ok%s  %s\n' "$G" "$N" "$*"; }
warn() { printf '%s  !!%s  %s\n' "$Y" "$N" "$*"; }
die()  { printf '\n%serror%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

trap 'die "failed on line $LINENO. Nothing was destroyed; fix the cause and re-run."' ERR

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
step "Checking prerequisites"

command -v docker >/dev/null 2>&1 || die "docker is not installed. See https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || die "the docker compose plugin is missing (you may have the old docker-compose; v2 is required)."
docker info >/dev/null 2>&1 || die "the docker daemon is not running, or this user cannot reach it."
ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo present)"

# openssl is used for the secrets. Fall back to /dev/urandom rather than failing, so a minimal container
# image without openssl can still install.
if command -v openssl >/dev/null 2>&1; then
  gen_secret() { openssl rand -hex 32; }
else
  warn "openssl not found; using /dev/urandom (equivalent entropy, different tool)"
  gen_secret() { head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
fi

# Disk: the corpus, model weights and image layers add up, and running out mid-import corrupts an ingest.
avail_gb=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print int($4/1048576)}')
if [ "${avail_gb:-0}" -lt 20 ]; then
  warn "only ${avail_gb}GB free here. 20GB+ is recommended; images and model weights alone are several GB."
fi

# ---------------------------------------------------------------------------
# 2. Secrets
# ---------------------------------------------------------------------------
step "Preparing configuration"

# Set a variable in .env only if it is absent or empty, so re-running never rotates a live secret (which
# would invalidate every issued token and orphan the object store behind a changed password).
set_if_missing() {
  local key="$1" val="$2"
  if grep -qE "^${key}=.+" "$ENV_FILE" 2>/dev/null; then
    return 0
  fi
  # An intentionally empty setting (COMPOSE_PROFILES) is already present; do not append it again.
  if [ -z "$val" ] && grep -qE "^${key}=$" "$ENV_FILE" 2>/dev/null; then
    return 0
  fi
  # drop an empty-valued line if present, then append
  if [ -f "$ENV_FILE" ]; then
    sed -i.bak "/^${key}=\$/d" "$ENV_FILE" && rm -f "${ENV_FILE}.bak"
  fi
  printf '%s=%s\n' "$key" "$val" >> "$ENV_FILE"
}

# Who holds a TCP port on this host: a container (by compose project, or name), another process, or nobody.
# Docker is asked first because a container's published port shows up in `ss` as docker-proxy, which says
# nothing about whose it is.
port_owner() {
  local port="$1" c
  c="$(docker ps --filter "publish=${port}" --format '{{.Label "com.docker.compose.project"}}|{{.Names}}' | head -1)"
  if [ -n "$c" ]; then
    printf '%s\n' "$c"
    return 0
  fi
  if command -v ss >/dev/null 2>&1 && [ -n "$(ss -ltnH "sport = :${port}" 2>/dev/null)" ]; then
    printf '|another program on this host\n'
    return 0
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
    printf '|another program on this host\n'
  fi
}

# The compose project whose containers count as this install's own when a port is checked. Compose takes it
# from the environment or from .env, so this does too.
if [ -z "${COMPOSE_PROJECT_NAME:-}" ] && [ -f "$ENV_FILE" ]; then
  COMPOSE_PROJECT_NAME="$(sed -n 's/^COMPOSE_PROJECT_NAME=\(.*\)$/\1/p' "$ENV_FILE" | tail -1)"
fi

# Ports already handed out in this run, so two services never get the same free port.
CLAIMED_PORTS=" "

# Pick a port for KEY: keep a configured one (and refuse if somebody else now holds it), otherwise take the
# default if it is free or the next free port above it. Another stack on this box holding 5432, 6379, 8000 or
# 9000 is common, and before this the install failed half way with a bind error naming neither the port's
# owner nor the setting that moves it.
pick_port() {
  local key="$1" default="$2" current owner p
  current="$(sed -n "s/^${key}=\(.*\)$/\1/p" "$ENV_FILE" 2>/dev/null | tail -1)"
  if [ -n "$current" ]; then
    owner="$(port_owner "$current")"
    if [ -n "$owner" ] && [ "${owner%%|*}" != "${COMPOSE_PROJECT_NAME:-labeloxav}" ]; then
      die "port ${current} (${key} in .env) is in use by ${owner#*|}.
       Stop that, or give LabeloxAV a different port by editing ${key} in .env, then re-run."
    fi
    CLAIMED_PORTS="${CLAIMED_PORTS}${current} "
    return 0
  fi
  p="$default"
  while :; do
    owner="$(port_owner "$p")"
    # Already handed to another setting in this run. Checked over the owner, because the owner may be this
    # deployment's own container, which would otherwise make the port look free for a second service too.
    case "$CLAIMED_PORTS" in *" ${p} "*) owner="|${key%_PORT} sharing a port already given out above";; esac
    if [ -z "$owner" ] || [ "${owner%%|*}" = "${COMPOSE_PROJECT_NAME:-labeloxav}" ]; then
      break
    fi
    [ "$p" = "$default" ] && warn "port ${default} is in use by ${owner#*|}; looking for a free one for ${key}"
    p=$((p + 1))
    [ "$p" -gt $((default + 100)) ] && die "no free port near ${default} for ${key}. Set ${key} in .env and re-run."
  done
  [ "$p" != "$default" ] && ok "${key}=${p}"
  CLAIMED_PORTS="${CLAIMED_PORTS}${p} "
  set_if_missing "$key" "$p"
}

if [ -f "$ENV_FILE" ]; then
  # Refuse to convert a local-dev .env into a half-configured deployment. The dev file carries the
  # well-known credentials on purpose (config allows them on a local box), and appending LBX_ENV=production
  # to it would leave a machine that neither runs as a dev box nor boots as a deployment, with the reason
  # buried in a config validator. Fail loudly and let the operator decide.
  if grep -qE '^(POSTGRES_PASSWORD=labelox|MINIO_ROOT_PASSWORD=labelox123)$' "$ENV_FILE" \
     && ! grep -qE '^LBX_AUTH__SIGNING_KEY=.+' "$ENV_FILE"; then
    die "$ENV_FILE looks like the local-development file (it has the well-known dev credentials).
       Deploying would need to overwrite it, and that would break \`make up\` for host-run development.
       Move it aside first, then re-run:

         mv .env .env.dev
         ./scripts/install.sh"
  fi
  ok "using the existing .env (secrets already there are left untouched)"
else
  say "  writing a new .env with generated secrets"
  cat > "$ENV_FILE" <<'HEADER'
# Generated by scripts/install.sh. Contains secrets: do not commit, do not share.
# Re-running the installer will not change anything already set here.
HEADER
  chmod 600 "$ENV_FILE"
fi

set_if_missing LBX_ENV                        production
set_if_missing LBX_LOG_LEVEL                  INFO
pick_port WEB_PORT                            "${WEB_PORT:-3000}"
pick_port API_PORT                            "${API_PORT:-8000}"

set_if_missing POSTGRES_USER                  labelox
set_if_missing POSTGRES_PASSWORD              "$(gen_secret)"
set_if_missing POSTGRES_DB                    labeloxav
pick_port POSTGRES_PORT                       5432

# The MinIO access key is a username, not a secret, but a generated one avoids the well-known default.
set_if_missing MINIO_ROOT_USER                "lbx$(gen_secret | cut -c1-13)"
set_if_missing MINIO_ROOT_PASSWORD            "$(gen_secret)"
set_if_missing MINIO_BUCKET                   labeloxav
pick_port MINIO_PORT                          9000
pick_port MINIO_CONSOLE_PORT                  9001

pick_port REDIS_PORT                          6379
pick_port REDPANDA_PORT                       19092
pick_port REDPANDA_ADMIN_PORT                 9644

set_if_missing LAKEFS_USER                    admin
set_if_missing LAKEFS_ACCESS_KEY              AKIALABELOXAVKEY01
set_if_missing LAKEFS_SECRET_KEY              "$(gen_secret)"
set_if_missing LAKEFS_ENCRYPT_SECRET          "$(gen_secret)"
set_if_missing LAKEFS_PORT                    8001
set_if_missing LICHTBLICK_PORT                8090
set_if_missing MLFLOW_PORT                    5500

set_if_missing LBX_AUTH__SIGNING_KEY                "$(gen_secret)"
set_if_missing LBX_FORGYX__DEPLOY_SIGNING_KEY       "$(gen_secret)"
set_if_missing LBX_PHASE4__GOVERN__ATTESTATION_KEY  "$(gen_secret)"

# Which compose files and optional services this install runs. Docker Compose reads COMPOSE_FILE and
# COMPOSE_PROFILES from .env itself, so a plain `docker compose up -d`, `make app-up` and the packaged
# `labeloxav` command all start exactly what was installed, without anyone repeating -f flags.
#
# The GPU overlay is added only when Docker's NVIDIA runtime is present. Without the runtime a device
# reservation stops compose from starting anything, so a CPU-only host gets the same images running on the
# CPU instead: slower auto-labelling, everything else unchanged.
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
  set_if_missing COMPOSE_FILE "docker-compose.yml:docker-compose.app.yml:docker-compose.gpu.yml"
  ok "NVIDIA container runtime found; models will run on the GPU"
else
  set_if_missing COMPOSE_FILE "docker-compose.yml:docker-compose.app.yml"
  if command -v nvidia-smi >/dev/null 2>&1; then
    warn "this machine has an NVIDIA GPU but Docker cannot use it (no nvidia runtime). Models will run on
       the CPU. Install the NVIDIA Container Toolkit, add docker-compose.gpu.yml to COMPOSE_FILE in .env,
       and run: docker compose up -d"
  else
    warn "no GPU found; models will run on the CPU. Labelling works, auto-labelling is much slower."
  fi
fi
# Optional services, empty by default: lakefs (dataset branches), viewer (MCAP viewer), mlflow (experiment
# tracking), or extras for all three. Edit this line in .env and run `docker compose up -d` to add them.
set_if_missing COMPOSE_PROFILES ""

# Read the ports back so the messages at the end match what was actually configured.
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

# Presigned download URLs go straight from the browser to MinIO, so this has to be an address the browser
# can reach. `minio:9000` (the in-network name) would 404 in a way that looks like missing data.
set_if_missing LBX_MINIO__PUBLIC_ENDPOINT     "http://localhost:${MINIO_PORT}"
set -a; . "$ENV_FILE"; set +a

ok "configuration at .env (mode 600)"

# ---------------------------------------------------------------------------
# 3. Build and start
# ---------------------------------------------------------------------------
step "Building images (first run takes a few minutes)"
"${COMPOSE[@]}" build --quiet
ok "images built"

step "Starting infrastructure"
"${COMPOSE[@]}" up -d postgres minio redis redpanda
ok "infrastructure starting"

step "Checking the database accepts the configured credentials"
# Postgres applies POSTGRES_PASSWORD only when it initialises an empty data directory. A volume created by
# an earlier run (a dev `make up`, or a previous install with different secrets) keeps its original
# password, so the generated one silently does not match and the first thing the operator sees is an
# asyncpg traceback from a migration. Detect it here and say exactly what happened and what the options are.
for i in $(seq 1 30); do
  # Connect over the service hostname, not the socket and not loopback. The postgres image trusts both of
  # those, so either would accept a wrong password and push the failure into the migration, where an
  # authentication problem reads as a code bug. Only a non-local address exercises the real auth rule, which
  # is what every other container does.
  if docker compose exec -T -e PGPASSWORD="${POSTGRES_PASSWORD}" postgres \
       psql -h postgres -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c 'SELECT 1' >/dev/null 2>&1; then
    ok "database credentials accepted"
    break
  fi
  if [ "$i" -eq 30 ]; then
    die "the database rejected the credentials in .env.

       This almost always means the Postgres volume already exists and was initialised with a different
       password. Postgres sets POSTGRES_PASSWORD only when it first creates its data directory, so changing
       it in .env afterwards has no effect on an existing volume.

       Two options.

         1. Keep the data and adopt the generated password. Postgres trusts a connection over its own
            socket, so the password can be changed from inside the container without knowing the old one:

              docker compose exec -T postgres \\
                psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} \\
                -c \"ALTER USER ${POSTGRES_USER} WITH PASSWORD '\$(grep '^POSTGRES_PASSWORD=' .env | cut -d= -f2-)';\"

            then re-run this installer. Note that a dev checkout pointing at the same volume will stop
            working until its .env carries the new password too.

         2. Start clean. This DESTROYS every session, frame, and annotation in that volume:

              ${COMPOSE_HINT} down -v
              ./scripts/install.sh

       Putting the old dev password back in .env is NOT an option here: it is a known-weak credential, and
       the app refuses to boot on it outside a local dev box."
  fi
  sleep 2
done

step "Applying the schema and seeding the ontology"
# The migrate service is one-shot and depends on a healthy Postgres; compose waits for that.
"${COMPOSE[@]}" up migrate --exit-code-from migrate >/dev/null 2>&1 \
  || die "migration failed. Run: ${COMPOSE_HINT} logs migrate"
ok "schema at head, ontology seeded"

step "Starting the application"
# Everything else in the configured files and profiles: the API, the web app, the GPU worker, the governance
# daemon, and any optional services named in COMPOSE_PROFILES.
"${COMPOSE[@]}" up -d
ok "api, web and GPU worker starting"

# ---------------------------------------------------------------------------
# 4. Wait for readiness
# ---------------------------------------------------------------------------
step "Waiting for the API to become ready"
# /api/readyz is 503 until every dependency answers, unlike /api/health which is 200 with a degraded body.
# Waiting on the wrong one would declare success while Postgres was still starting.
api_url="http://localhost:${API_PORT}"
for i in $(seq 1 90); do
  if curl -fsS "${api_url}/api/readyz" >/dev/null 2>&1; then
    ok "API ready at ${api_url}"
    break
  fi
  if [ "$i" -eq 90 ]; then
    die "the API did not become ready within 3 minutes. Run: ${COMPOSE_HINT} logs api"
  fi
  sleep 2
done

# ---------------------------------------------------------------------------
# 5. First admin
# ---------------------------------------------------------------------------
step "Creating the first administrator"
# POST /api/users is allowed without a token only while the user table is empty. That bootstrap closes the
# moment this succeeds, so a second run finds a user and skips instead of creating a second admin.
admin_name="${LBX_ADMIN_NAME:-admin}"

# Migration 0011 seeds a default `admin` row so the corpus always has a usable identity, which means the
# user table is never empty and the API's unauthenticated-bootstrap path (POST /api/users while no user
# exists) is closed before the API ever serves a request. Minting from inside the container is therefore the
# normal path here, not a fallback: possession of the signing key is the authority, the same authority the
# API itself has. The HTTP route is still tried first, for a deployment whose table really is empty.
token=""
admin_body_file="$(mktemp)"
admin_status="$(curl -s -o "$admin_body_file" -w '%{http_code}' \
  -X POST "${api_url}/api/users" \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"${admin_name}\",\"role\":\"admin\"}" || echo 000)"
admin_body="$(cat "$admin_body_file")"
rm -f "$admin_body_file"

case "$admin_status" in
  200|201)
    token="$(printf '%s' "$admin_body" | sed -n 's/.*"token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
    [ -n "$token" ] && ok "administrator '${admin_name}' created"
    ;;
  401|403|409)
    # Expected: an administrator already exists (seeded by the migration, or left by an earlier install).
    ;;
  *)
    warn "unexpected response creating the administrator (HTTP ${admin_status}): ${admin_body}"
    ;;
esac

if [ -z "$token" ]; then
  minted="$("${COMPOSE[@]}" exec -T api python -m scripts.mint_token --name "${admin_name}" 2>/dev/null || true)"
  token="$(printf '%s' "$minted" | grep -oE '^lbx2\.[A-Za-z0-9_.-]+' | head -1)"
  [ -n "$token" ] && ok "minted a token for the existing administrator '${admin_name}'"
fi

if [ -n "$token" ]; then
  printf '%s\n' "$token" > "${REPO_ROOT}/.admin-token"
  chmod 600 "${REPO_ROOT}/.admin-token"
else
  warn "could not obtain an administrator token. Mint one with:
       ${COMPOSE_HINT} exec api python -m scripts.mint_token --name ${admin_name}"
fi

# ---------------------------------------------------------------------------
# 6. Done
# ---------------------------------------------------------------------------
web_url="http://localhost:${WEB_PORT}"
cat <<EOF

${G}${B}LabeloxAV is running.${N}

  Web app     ${B}${web_url}${N}   (open this from any machine on your network: http://<this-host>:${WEB_PORT})
  API         ${api_url}   (this machine only; other machines use ${web_url}/api)
  Storage     http://localhost:${MINIO_CONSOLE_PORT}   (MinIO console, this machine only)
  Health      ./scripts/doctor.sh

EOF

if [ -n "$token" ]; then
  cat <<EOF
  Sign in with this token (also saved to .admin-token, mode 600):

    ${B}${token}${N}

  Open ${web_url}/login and paste it. Tokens expire; the app refreshes yours silently while you use it.

EOF
else
  cat <<EOF
  To sign in, use an existing token, or mint one for a user with:

    ${COMPOSE_HINT} exec api python -m scripts.mint_token --name <user>

EOF
fi

cat <<EOF
  Next steps
    Upload footage      ${web_url}/annotate/new
    Read the runbook    docs/RUNBOOK.md
    Stop everything     ${COMPOSE_HINT} down
    Stop and wipe data  ${COMPOSE_HINT} down -v   ${Y}(destroys the corpus)${N}

EOF
