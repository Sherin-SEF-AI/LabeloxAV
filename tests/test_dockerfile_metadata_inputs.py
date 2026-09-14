"""Every file the package metadata reads must be present when the Dockerfile builds it.

The Dockerfile installs the project's dependencies before copying the tree, so that a source edit does
not invalidate the dependency layer. That means the metadata build runs against only the files named on
the `COPY` line that precedes it. Hatchling opens the readme and the licence file while building the
metadata, so a `pyproject.toml` that names a file the `COPY` line does not carry fails with "License
file does not exist" at that step.

That is exactly what shipped in v0.1.0: the commit that added the Apache licence pointed the metadata
at `LICENSE`, the `COPY` line still said `pyproject.toml README.md`, and the published package could not
complete its first install. The test suite passed, because nothing in it built the image. This test
reads both files and refuses the mismatch before it can be built again.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _metadata_files() -> set[str]:
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    out: set[str] = set()
    readme = proj.get("readme")
    if isinstance(readme, str):
        out.add(readme)
    elif isinstance(readme, dict) and "file" in readme:
        out.add(readme["file"])
    lic = proj.get("license")
    if isinstance(lic, dict) and "file" in lic:
        out.add(lic["file"])
    for f in proj.get("license-files", []) or []:
        out.add(f)
    return out


def _early_copy_lines() -> list[tuple[int, set[str]]]:
    """Each `COPY a b ./` that precedes a `uv pip install -e` in its stage, with the files it carries."""
    lines = (ROOT / "Dockerfile").read_text().splitlines()
    out: list[tuple[int, set[str]]] = []
    pending: tuple[int, set[str]] | None = None
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("FROM "):
            pending = None
        m = re.match(r"COPY\s+(?!--from)(.+?)\s+\./?$", stripped)
        if m:
            files = set(m.group(1).split())
            pending = (i, files) if files != {"."} else None
        if "pip install" in stripped and "-e" in stripped and pending is not None:
            out.append(pending)
            pending = None
    return out


def test_the_dockerfile_has_early_installs_to_check():
    assert _early_copy_lines(), "no COPY-then-install step found; the Dockerfile shape changed"


def test_every_file_the_metadata_reads_is_copied_before_the_first_install():
    needed = _metadata_files()
    assert needed, "pyproject names no readme or licence file; that is a change worth looking at"
    for lineno, copied in _early_copy_lines():
        missing = sorted(needed - copied)
        assert not missing, (
            f"Dockerfile line {lineno} copies {sorted(copied)} before installing, but pyproject's "
            f"metadata reads {missing}. The build will fail there with 'file does not exist'."
        )


def test_the_named_files_exist_in_the_repository():
    for f in sorted(_metadata_files()):
        assert (ROOT / f).is_file(), f"pyproject names {f} but it is not in the repository"
