#!/usr/bin/env bash
#
# A new team's first hour, end to end, on a throwaway install.
#
#   ./scripts/e2e_first_run.sh
#
# Checks out the committed tree into a temporary git worktree, installs it with the real installer under its
# own Compose project (its own ports, volumes and secrets, nothing shared with a dev setup), takes the sign-in
# link the installer prints, and runs web/e2e/first-run.spec.ts against it: sign in, load a sample drive,
# auto-label it on the GPU worker, accept a label, export a dataset, with the home checklist ticking along.
# Then it removes the install, its volumes and the worktree, whether the run passed or not.
#
# It is slow (the sample is downloaded and imported, auto-labelling runs on the GPU) and needs Docker, a GPU
# for the worker, and network access for the sample. Set LBX_E2E_KEEP=1 to leave the install up afterwards.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PROJECT="${LBX_E2E_PROJECT:-lbxfirstrun}"
WORK="$(mktemp -d)"
TREE="${WORK}/labeloxav"

cleanup() {
  status=$?
  if [ "${LBX_E2E_KEEP:-0}" = "1" ]; then
    echo "LBX_E2E_KEEP=1: leaving the install up in ${TREE} (project ${PROJECT})"
    exit "$status"
  fi
  echo "removing the throwaway install"
  (cd "$TREE" 2>/dev/null && docker compose down -v --remove-orphans >/dev/null 2>&1) || true
  git -C "$ROOT" worktree remove --force "$TREE" >/dev/null 2>&1 || true
  rm -rf "$WORK"
  exit "$status"
}
trap cleanup EXIT

if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]; then
  echo "note: uncommitted changes are not part of this run; it installs HEAD ($(git -C "$ROOT" rev-parse --short HEAD))"
fi

git -C "$ROOT" worktree add --detach "$TREE" HEAD >/dev/null
cd "$TREE"
echo "COMPOSE_PROJECT_NAME=${PROJECT}" > .env

./scripts/install.sh 2>&1 | tee "${WORK}/install.log"
link="$(grep -oE 'https?://[^ ]+/login\?invite=[A-Za-z0-9._~%-]+' "${WORK}/install.log" | tail -1)"
if [ -z "$link" ]; then
  echo "the installer printed no sign-in link; see ${WORK}/install.log" >&2
  exit 1
fi
web_url="${link%%/login*}"
echo "fresh install at ${web_url}"

cd "$ROOT/web"
LBX_E2E_BASE_URL="$web_url" LBX_E2E_SIGNIN_URL="$link" npx playwright test e2e/first-run.spec.ts --reporter=line
