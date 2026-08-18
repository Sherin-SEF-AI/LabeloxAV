"""Every test that touches Postgres has to say so.

Two things depend on the `db` marker, and both were broken by the same drift.

`make test-unit` is `pytest -m "not db and not gpu and not infra"` and is documented as the fast tier that
needs no Postgres. It was selecting 117 files that open a database session, so on a machine without a
Postgres it failed rather than skipping, and nobody ran it: the fast tier did not exist.

And the production guard was reachable but skipped. `_provision_test_db` refuses to run against a database
whose name does not contain "test" - but that refusal used to sit after an early return taken when no test
in the run carried the `db` marker. A run selecting only unmarked tests skipped the guard and still
committed rows, against whatever LBX_POSTGRES__DB happened to name. That is the mechanism behind the 1,730
fixture sessions purged from the real corpus on 2026-07-30. The guard now runs first regardless; this file
is what stops the marker set drifting back to the state that made it worth bypassing.

Shaped after tests/test_route_auth.py and web/lib/nofetch.test.ts: scan the real tree, assert the offender
list is empty, and floor the scan size so a detector that quietly stops matching cannot report green.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# What counts as touching the database, by what the file's own source says. Deliberately syntactic rather
# than import-graph-based: importing services.api.main transitively reaches db.models from almost anywhere,
# so an import-graph rule would mark the entire suite and mean nothing.
#
#   get_sessionmaker / get_engine  - opens its own session or engine
#   create_engine / *_dsn          - builds its own SQLAlchemy engine, as the migration tests do
#   db_session                     - the FastAPI dependency; a route test driving it hits the real DB
#   TestClient(                    - drives the mounted app, whose routes open sessions
#   from db.models import          - hand-built ORM queries that do not name the sessionmaker
#
# This is a floor, not a ceiling. It cannot see a test that calls a service function which opens its own
# session (services.calibration.resolve and services.autolabel.grounding both do), because nothing in the
# test's own source says "database". Those are marked by hand and listed in SERVICE_LEVEL_DB below. The
# ground truth is empirical and cheap: run `make test-unit` with Postgres stopped, and anything that errors
# needs the marker.
_TOUCHES_DB = re.compile(
    r"get_sessionmaker|get_engine\s*\(|create_engine\s*\(|_dsn\b|\bdb_session\b|TestClient\s*\("
    r"|from db\.models import")

# Files with no database symbol of their own that nonetheless reach Postgres through a service call.
# Discovered by running the unit tier with Postgres unreachable; kept here so the marker is not mistaken
# for an accident and removed by someone tidying up.
SERVICE_LEVEL_DB = {
    "test_calibration_resolve.py",      # services.calibration.resolve reads stored extrinsics
    "test_mq0_ontology_pullback.py",    # services.autolabel.grounding resolves the ontology from the DB
}

_MARKED = re.compile(r"pytest\.mark\.db")

# Files matching the detector that genuinely need no database, each with the reason. Keep this list short:
# every entry is a place the detector is wrong, and a long list means the detector is the thing to fix.
ALLOWED = {
    # Builds an Object in memory to exercise the snapshot/restore round trip. Never opens a session.
    "test_cleanup_sweep.py",
    # Wraps RateLimitMiddleware around its own bare FastAPI(), so its TestClient never reaches a route
    # that opens a session.
    "test_ratelimit_middleware.py",
    # This guard: its own detection pattern is a literal match for what it looks for.
    "test_db_markers.py",
    # Asserts on the shape of the DSN string that settings builds. Never opens a connection with it.
    "test_m0_foundation.py",
}

# A run that scans almost nothing and finds no offenders is indistinguishable from a green run. These
# floors are the difference. Both sit below today's figures (337 files, 162 db-touching) so ordinary growth
# does not trip them, and far above zero so a detector that stops matching does.
_MIN_FILES_SCANNED = 250
_MIN_DB_TOUCHING = 100


def _test_files() -> list[Path]:
    return sorted(TESTS_DIR.glob("test_*.py"))


def _db_touching() -> list[Path]:
    return [p for p in _test_files()
            if p.name in SERVICE_LEVEL_DB or _TOUCHES_DB.search(p.read_text())]


def test_the_scan_is_not_trivially_green():
    # The non-triviality floor, copied from test_route_auth.py's `assert checked > 100`. A rename, a moved
    # directory or a regex that stops matching would otherwise turn this whole file into an assertion
    # about the empty set.
    files = _test_files()
    assert len(files) >= _MIN_FILES_SCANNED, f"only {len(files)} test files found; the glob is wrong"
    touching = _db_touching()
    assert len(touching) >= _MIN_DB_TOUCHING, (
        f"only {len(touching)} files detected as touching the database; the detector has stopped matching")


def test_every_db_touching_file_carries_the_marker():
    offenders = [p.name for p in _db_touching()
                 if p.name not in ALLOWED and not _MARKED.search(p.read_text())]
    assert offenders == [], (
        f"{len(offenders)} test files open a database session but carry no `db` marker, so `make test-unit` "
        "selects them despite being documented as needing no Postgres. Add `pytestmark = pytest.mark.db` at "
        f"module level, or add the file to ALLOWED with a reason: {offenders}")


def test_the_allowlist_has_not_gone_stale():
    # An allowlist entry for a file that no longer exists, or no longer matches the detector, is a
    # permission nobody is using that every later reader still has to reason about. Same failure mode as a
    # stale KNOWN_FAILURES.md row, and the same fix: check it mechanically.
    stale = []
    for name in sorted(ALLOWED):
        path = TESTS_DIR / name
        if not path.exists():
            stale.append(f"{name} (no such file)")
        elif not _TOUCHES_DB.search(path.read_text()):
            stale.append(f"{name} (no longer matches the detector)")
    for name in sorted(SERVICE_LEVEL_DB):
        if not (TESTS_DIR / name).exists():
            stale.append(f"{name} (no such file)")
    assert stale == [], f"stale ALLOWED/SERVICE_LEVEL_DB entries: {stale}"


def test_the_unit_tier_still_deselects_db():
    # The marker only means something because `make test-unit` deselects it. If that expression is edited
    # away, every assertion above keeps passing while the tier silently goes back to needing a Postgres.
    makefile = (REPO_ROOT / "Makefile").read_text()
    deselecting = [ln for ln in makefile.splitlines() if "pytest" in ln and "not db" in ln]
    assert deselecting, "no pytest invocation in the Makefile deselects `not db`; the unit tier is gone"


def test_the_markers_are_declared():
    # --strict-markers turns a typo'd marker into a collection error rather than a test that is silently
    # never selected, and that only works while the marker is declared.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    for marker in ("db:", "gpu:", "infra:"):
        assert marker in pyproject, f"marker `{marker}` is no longer declared in pyproject.toml"
