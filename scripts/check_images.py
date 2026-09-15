"""Check that every image the compose files pull still resolves in its registry.

A service with no `build:` section has to be pulled on a fresh machine, and a registry is an external
dependency that can change under a pinned tag. MinIO withdrew its Docker Hub repositories, so
`minio/minio` stopped resolving anywhere; the development machine kept working off a copy cached almost
two years earlier, and every fresh install failed at "Starting infrastructure". Nothing caught it,
because nothing started the stack on a machine without a cache.

Reads the compose YAML directly rather than `docker compose config --images`, which refuses to render
until the generated secrets exist, so this runs in CI with no configuration at all.

    python scripts/check_images.py          exit 1 and name every image that does not resolve
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ("docker-compose.yml", "docker-compose.app.yml")


def pulled_images() -> list[tuple[str, str]]:
    """(service, image) for every service that is pulled rather than built, across both files."""
    out: dict[str, str] = {}
    built: set[str] = set()
    for name in COMPOSE:
        services = (yaml.safe_load((ROOT / name).read_text()) or {}).get("services", {}) or {}
        for svc, spec in services.items():
            spec = spec or {}
            if "build" in spec:
                built.add(svc)
            if spec.get("image"):
                out[svc] = spec["image"]
    return sorted((svc, img) for svc, img in out.items() if svc not in built and "${" not in img)


def resolves(image: str) -> tuple[bool, str]:
    try:
        r = subprocess.run(["docker", "manifest", "inspect", image], capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    # Docker's first line is often a bare "errors:" heading; the reason is on the line after it.
    lines = [ln.strip() for ln in (r.stderr or r.stdout).splitlines() if ln.strip() and ln.strip() != "errors:"]
    return r.returncode == 0, "" if r.returncode == 0 else (lines[0][:160] if lines else f"exit {r.returncode}")


def main() -> int:
    images = pulled_images()
    if not images:
        print("no pulled images found in the compose files; that is itself worth looking at")
        return 1
    bad = 0
    for svc, img in images:
        ok, why = resolves(img)
        print(f"  {'ok ' if ok else 'BAD'}  {svc:14} {img}{'' if ok else '   ' + why}")
        bad += 0 if ok else 1
    print(f"{len(images)} pulled images, {bad} do not resolve")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
