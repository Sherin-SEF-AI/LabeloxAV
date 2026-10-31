#!/usr/bin/env bash
#
# Check a LabeloxAV installation and say what to fix.
#
#   ./scripts/doctor.sh          (or: labeloxav doctor)
#
# Two halves. This script checks what only the host can see: Docker, the GPU runtime, disk, memory, the
# configured ports and the state of each container. Then it runs `python -m services.doctor` inside the API
# container, which checks what only the application can see: the database schema, storage, the event bus,
# torch and the GPU from inside the container, the model weights, and whether the GPU worker is running.
#
# Exits 1 if anything failed, so it can gate a script or a cron job.

set -uo pipefail

REPO_ROOT="${LABELOXAV_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT" || exit 1

if [ -t 1 ]; then
  G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; B=$'\033[1m'; N=$'\033[0m'
else
  G=""; Y=""; R=""; B=""; N=""
fi
failed=0
pass() { printf '  %sok%s    %s\n' "$G" "$N" "$*"; }
warn() { printf '  %swarn%s  %s\n' "$Y" "$N" "$1"; [ -n "${2:-}" ] && printf '        fix: %s\n' "$2"; return 0; }
fail() { printf '  %sFAIL%s  %s\n' "$R" "$N" "$1"; [ -n "${2:-}" ] && printf '        fix: %s\n' "$2"; failed=1; }

env_get() { sed -n "s/^$1=\(.*\)$/\1/p" .env 2>/dev/null | tail -1; }

printf '\n%sHost%s\n' "$B" "$N"

if ! command -v docker >/dev/null 2>&1; then
  fail "docker is not installed" "https://docs.docker.com/get-docker/"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  fail "cannot reach the Docker daemon" "start Docker, or run this as a user in the docker group"
  exit 1
fi
pass "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"

if [ ! -f .env ]; then
  fail "no .env here: this checkout has not been installed" "./scripts/install.sh"
  exit 1
fi
compose_file="$(env_get COMPOSE_FILE)"
# Containers are matched to this install by their compose project, which .env may name.
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-$(env_get COMPOSE_PROJECT_NAME)}"
if [ -z "$compose_file" ]; then
  warn ".env has no COMPOSE_FILE, so plain 'docker compose' starts only the infrastructure" \
       "re-run ./scripts/install.sh, which adds it without touching existing secrets"
fi

has_runtime=0
docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia && has_runtime=1
case "$compose_file" in
  *docker-compose.gpu.yml*)
    if [ "$has_runtime" = 1 ]; then
      pass "GPU: Docker's NVIDIA runtime is present and the GPU overlay is enabled"
    else
      fail "the GPU overlay is enabled but Docker has no NVIDIA runtime; containers will not start" \
           "install the NVIDIA Container Toolkit, or remove docker-compose.gpu.yml from COMPOSE_FILE in .env"
    fi
    ;;
  *)
    if [ "$has_runtime" = 1 ]; then
      warn "Docker can use the GPU but this install does not ask for it; models run on the CPU" \
           "add :docker-compose.gpu.yml to COMPOSE_FILE in .env, then docker compose up -d"
    else
      warn "no GPU for containers; models run on the CPU, auto-label is much slower"
    fi
    ;;
esac

avail_gb=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print int($4/1048576)}')
docker_gb=$(df -Pk "$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /)" 2>/dev/null \
            | awk 'NR==2 {print int($4/1048576)}')
for pair in "this folder:${avail_gb}" "Docker's data directory:${docker_gb:-$avail_gb}"; do
  where="${pair%%:*}"; gb="${pair##*:}"
  if [ "${gb:-0}" -lt 10 ]; then
    fail "${gb} GB free in ${where}; imports and frame storage will fail when it runs out" \
         "free space, or move Docker's data-root to a larger disk"
  elif [ "${gb:-0}" -lt 30 ]; then
    warn "${gb} GB free in ${where}; footage fills this quickly"
  else
    pass "${gb} GB free in ${where}"
  fi
done

mem_gb=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 0)
avail_mem_gb=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo 2>/dev/null || echo 0)
if [ "$mem_gb" -lt 16 ]; then
  warn "${mem_gb} GB RAM; the API and GPU worker together want about 16 GB or more"
else
  pass "${mem_gb} GB RAM, ${avail_mem_gb} GB available now"
fi

printf '\n%sPorts%s\n' "$B" "$N"
for key in WEB_PORT API_PORT POSTGRES_PORT MINIO_PORT MINIO_CONSOLE_PORT REDIS_PORT REDPANDA_PORT; do
  port="$(env_get "$key")"
  [ -n "$port" ] || continue
  owner="$(docker ps --filter "publish=${port}" --format '{{.Label "com.docker.compose.project"}}|{{.Names}}' | head -1)"
  if [ -z "$owner" ]; then
    if command -v ss >/dev/null 2>&1 && [ -n "$(ss -ltnH "sport = :${port}" 2>/dev/null)" ]; then
      fail "${key}=${port} is held by a program that is not LabeloxAV" "stop it, or change ${key} in .env"
    else
      warn "${key}=${port} is not listening: its service is not running" "docker compose up -d"
    fi
  elif [ "${owner%%|*}" = "${COMPOSE_PROJECT_NAME:-labeloxav}" ]; then
    pass "${key}=${port} (${owner#*|})"
  else
    fail "${key}=${port} is held by ${owner#*|}, not LabeloxAV" "stop it, or change ${key} in .env"
  fi
done

printf '\n%sContainers%s\n' "$B" "$N"
states="$(docker compose ps -a --format '{{.Service}}|{{.State}}|{{.Health}}|{{.ExitCode}}' 2>/dev/null)"
if [ -z "$states" ]; then
  fail "no LabeloxAV containers exist" "docker compose up -d"
else
  while IFS='|' read -r svc state health code; do
    case "$svc" in
      migrate|pii-models|minio-init)
        if [ "$state" = "exited" ] && [ "$code" = "0" ]; then
          pass "${svc}: finished"
        elif [ "$state" = "exited" ]; then
          fail "${svc}: exited with code ${code}" "docker compose logs ${svc}"
        else
          pass "${svc}: ${state}"
        fi
        ;;
      *)
        if [ "$state" != "running" ]; then
          fail "${svc}: ${state}" "docker compose up -d ${svc}; docker compose logs ${svc}"
        elif [ "$health" = "unhealthy" ]; then
          fail "${svc}: running but unhealthy" "docker compose logs ${svc}"
        elif [ "$health" = "starting" ]; then
          warn "${svc}: starting"
        else
          pass "${svc}: running${health:+ (${health})}"
        fi
        ;;
    esac
  done <<< "$states"
fi

printf '\n%sApplication%s\n' "$B" "$N"
if docker compose ps --status running --services 2>/dev/null | grep -qx api; then
  docker compose exec -T api python -m services.doctor 2>/dev/null
  rc=$?
  [ "$rc" -ne 0 ] && failed=1
else
  fail "the API container is not running, so the application checks cannot run" \
       "docker compose up -d api; docker compose logs api"
fi

echo
if [ "$failed" -ne 0 ]; then
  printf '%sSome checks failed.%s Fix them in order: a failure near the top often causes the ones below.\n' "$R" "$N"
  exit 1
fi
printf '%sAll checks passed.%s\n' "$G" "$N"
