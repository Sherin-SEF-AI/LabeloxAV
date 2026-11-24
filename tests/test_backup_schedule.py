"""The scheduled backup keeps what it should, removes only what it may, and fails loudly.

Run against a stand-in for backup.sh, because the property under test is the scheduling around it:
retention, never pruning before the new backup exists, cleaning up a half-written directory, the copy off
the machine, and one run at a time.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock(1)")

GOOD = """#!/usr/bin/env bash
set -e
mkdir -p "$1/minio"
echo "dump" > "$1/postgres.sql.gz"
echo "{}" > "$1/minio.manifest.json"
echo "link_dest=${LBX_BACKUP_LINK_DEST:-}" > "$1/SAW"
echo "taken_at=now" > "$1/MANIFEST"
"""
BAD = """#!/usr/bin/env bash
mkdir -p "$1"
echo partial > "$1/postgres.sql.gz"
exit 3
"""


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "install"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "backup_scheduled.sh", root / "scripts" / "backup_scheduled.sh")
    return root


def _run(root: Path, body: str, **env) -> subprocess.CompletedProcess:
    script = root / "scripts" / "backup.sh"
    script.write_text(body)
    script.chmod(0o755)
    full = {**os.environ, "LBX_BACKUP_DIR": str(root / "bk"), **{k: str(v) for k, v in env.items()}}
    full.pop("LBX_BACKUP_REMOTE", None) if "LBX_BACKUP_REMOTE" not in env else None
    return subprocess.run([str(root / "scripts" / "backup_scheduled.sh")], env=full, capture_output=True,
                          text=True, timeout=60)


def _backups(root: Path) -> list[Path]:
    return sorted(p.parent for p in (root / "bk").glob("*/MANIFEST"))


def _seed(root: Path, names: list[str]) -> None:
    for n in names:
        d = root / "bk" / n
        (d / "minio").mkdir(parents=True)
        (d / "MANIFEST").write_text("taken_at=then\n")
        (d / "minio.manifest.json").write_text("{}")


def test_a_run_writes_a_backup_and_records_success(tree):
    r = _run(tree, GOOD)
    assert r.returncode == 0, r.stderr
    assert len(_backups(tree)) == 1
    status = json.loads((tree / "bk" / "last_run.json").read_text())
    assert status["ok"] is True and status["kept"] == 1
    assert "same disk" in r.stdout, "a backup that never leaves the box must say so"


def test_only_the_newest_are_kept(tree):
    _seed(tree, ["20260101_000000", "20260102_000000", "20260103_000000"])
    r = _run(tree, GOOD, LBX_BACKUP_KEEP=2)
    assert r.returncode == 0, r.stderr
    names = [p.name for p in _backups(tree)]
    assert len(names) == 2 and "20260101_000000" not in names and "20260102_000000" not in names
    assert "20260103_000000" in names


def test_the_previous_backup_is_offered_for_reuse(tree):
    _seed(tree, ["20260101_000000"])
    assert _run(tree, GOOD).returncode == 0
    newest = _backups(tree)[-1]
    assert (newest / "SAW").read_text().strip().endswith("20260101_000000")


def test_a_failed_run_removes_nothing_and_leaves_no_half_backup(tree):
    _seed(tree, ["20260101_000000", "20260102_000000"])
    r = _run(tree, BAD, LBX_BACKUP_KEEP=1)
    assert r.returncode != 0
    assert [p.name for p in _backups(tree)] == ["20260101_000000", "20260102_000000"], \
        "a failed backup must not cost an earlier one"
    leftovers = [p for p in (tree / "bk").iterdir() if p.is_dir() and not (p / "MANIFEST").exists()]
    assert leftovers == [], "a half-written directory would be mistaken for a backup"
    assert json.loads((tree / "bk" / "last_run.json").read_text())["ok"] is False


@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync(1)")
def test_a_remote_gets_a_copy(tree, tmp_path):
    remote = tmp_path / "nas"
    remote.mkdir()
    r = _run(tree, GOOD, LBX_BACKUP_REMOTE=remote)
    assert r.returncode == 0, r.stderr
    copied = list(remote.glob("*/MANIFEST"))
    assert len(copied) == 1 and (copied[0].parent / "postgres.sql.gz").exists()
    assert "same disk" not in r.stdout


@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync(1)")
def test_a_failed_copy_fails_the_run_and_prunes_nothing(tree, tmp_path):
    _seed(tree, ["20260101_000000"])
    r = _run(tree, GOOD, LBX_BACKUP_KEEP=1, LBX_BACKUP_REMOTE=tmp_path / "does" / "not" / "exist" / "x")
    assert r.returncode != 0
    assert "20260101_000000" in [p.name for p in _backups(tree)]


@pytest.mark.parametrize("keep", ["0", "seven", "-1"])
def test_a_nonsense_retention_is_refused(tree, keep):
    r = _run(tree, GOOD, LBX_BACKUP_KEEP=keep)
    assert r.returncode == 2 and _backups(tree) == []
