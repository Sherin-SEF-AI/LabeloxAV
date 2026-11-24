"""An import request may not name an arbitrary server path.

`_acquire_source` did `Path(source_uri)` on a value that arrives from a reviewer (POST /api/imports/start,
/api/imports/dry-run) or an outside workforce (the return upload). Either could point an import at any
directory or archive the API process can read. These pin the allowlist, the symlink case that defeats a
string-prefix check, and the archive size cap.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from core.config import get_settings
from services.imports.run import SourceNotAllowed, _acquire_source, check_request_source


@pytest.fixture
def roots(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(get_settings().paths, "import_roots", [str(allowed)])
    return allowed


def test_object_storage_is_always_allowed(monkeypatch):
    monkeypatch.setattr(get_settings().paths, "import_roots", [])
    check_request_source("s3://uploads/a.zip")


def test_local_path_refused_when_no_roots_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings().paths, "import_roots", [])
    with pytest.raises(SourceNotAllowed, match="LBX_PATHS__IMPORT_ROOTS"):
        check_request_source(str(tmp_path))


def test_path_inside_a_root_is_allowed(roots):
    (roots / "ds").mkdir()
    check_request_source(str(roots / "ds"))
    check_request_source(str(roots))


def test_path_outside_every_root_is_refused(roots):
    with pytest.raises(SourceNotAllowed):
        check_request_source("/etc")
    with pytest.raises(SourceNotAllowed):
        check_request_source(str(roots.parent))


def test_dotdot_escape_is_refused(roots):
    with pytest.raises(SourceNotAllowed):
        check_request_source(str(roots / ".." / ".." / "etc"))


def test_symlink_inside_a_root_pointing_out_is_refused(roots, tmp_path):
    outside = tmp_path / "secret"
    outside.mkdir()
    (roots / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(SourceNotAllowed):
        check_request_source(str(roots / "link"))


@pytest.mark.parametrize("uri", ["", "   ", "relative/dir", "file:///etc", "http://example.com/a.zip",
                                 "gopher://x/y"])
def test_malformed_and_foreign_schemes_are_refused(roots, uri):
    with pytest.raises(SourceNotAllowed):
        check_request_source(uri)


def test_archive_over_the_extract_cap_is_refused_before_extraction(tmp_path, monkeypatch):
    z = tmp_path / "big.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.bin", b"\0" * 200_000)
    monkeypatch.setattr(get_settings().paths, "import_max_extract_bytes", 1000)
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(ValueError, match="limit"):
        _acquire_source(str(z), work)
    assert not any((work / "extracted").iterdir())


def test_archive_under_the_cap_extracts(tmp_path):
    z = tmp_path / "ok.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("d/a.txt", "x")
    work = tmp_path / "work"
    work.mkdir()
    out = _acquire_source(str(z), work)
    assert (Path(out) / "d" / "a.txt").read_text() == "x"


def test_every_route_that_acquires_a_request_source_checks_it_first():
    """A new route that calls _acquire_source on a payload without the guard reopens the hole."""
    root = Path(__file__).resolve().parents[1] / "services" / "api" / "routers"
    offenders = []
    for f in root.glob("*.py"):
        src = f.read_text()
        if "_acquire_source(" in src and "check_request_source(" not in src:
            offenders.append(f.name)
    assert not offenders, f"routers acquiring a caller-supplied source without the guard: {offenders}"
