import os
import sys
from pathlib import Path

import pytest

# Allow running the suite without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Auth is ON by default so the suite exercises the production posture: a test that drives the HTTP surface
# must carry a signed token, exactly as a real client does. The handful of TestClient tests use the helpers
# in tests/_authutil.py to seed a user and attach a Bearer header. Service-level tests (the bulk of the
# suite) never touch the middleware, so this does not affect them. A file that needs the gate off asks for
# it explicitly (LBX_AUTH__ENABLED override) rather than relying on a global default that never ships.
os.environ.setdefault("LBX_AUTH__ENABLED", "true")

# The test corpus is face-only (no license plates) and ships no plate model, so the mandatory-plate gate
# is relaxed here. test_pii_gate.py constructs PiiSettings(plate_mandatory=True) explicitly to verify it.
os.environ.setdefault("LBX_PII__PLATE_MANDATORY", "false")

# ISOLATION: the suite seeds synthetic sessions/frames/objects and commits them. Point it at a dedicated
# test database so it never writes to the production corpus (which had accumulated ~1600 synthetic noise
# sessions from prior runs against the live DB). Override LBX_POSTGRES__DB in CI to change the name.
os.environ.setdefault("LBX_POSTGRES__DB", "labeloxav_test")


@pytest.fixture(scope="session", autouse=True)
def _provision_test_db(request):
    """Create the isolated test database (if missing) and bring it to the head schema once per session, so
    tests run against a real-schema DB that is never the production corpus. Refuses unless the target db
    name looks like a test db: a guard against accidentally pointing the suite at production.

    Skipped when the selected run has no db-marked test (e.g. make test-unit): the pure-unit tier must run
    without a Postgres. Tests that truly need the DB carry the `db` marker; the marker both selects them out
    of the unit tier and triggers provisioning here."""
    if not any(item.get_closest_marker("db") for item in request.session.items):
        yield
        return

    import subprocess

    import psycopg

    from core.config import get_settings
    get_settings.cache_clear()
    pg = get_settings().postgres
    if "test" not in pg.db.lower():
        raise RuntimeError(f"refusing to run the suite against non-test database '{pg.db}'. "
                           "Set LBX_POSTGRES__DB to a *_test database.")
    admin = psycopg.connect(host=pg.host, port=pg.port, user=pg.user, password=pg.password,
                            dbname="postgres", autocommit=True)
    with admin.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (pg.db,))
        if cur.fetchone() is None:
            cur.execute(f'CREATE DATABASE "{pg.db}"')
    admin.close()
    subprocess.run([".venv/bin/alembic", "upgrade", "head"], check=True, cwd=REPO_ROOT, env={**os.environ})
    yield


@pytest.fixture(autouse=True)
def _reset_db_caches():
    """Each async test runs in its own event loop. The app caches one engine per process (correct,
    since the CLI uses a single asyncio.run loop), so clear that cache around every test to avoid
    reusing an engine bound to a closed loop. Also reset the settings cache so per-test env overrides
    (e.g. auth) take effect."""
    from core.config import get_settings
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    get_settings.cache_clear()
    yield
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    get_settings.cache_clear()
