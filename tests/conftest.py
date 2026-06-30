import os
import sys
from pathlib import Path

import pytest

# Allow running the suite without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The bulk of the suite predates API auth and drives endpoints without a user header. Default auth off
# for tests; test_auth.py re-enables it explicitly to exercise the gate.
os.environ.setdefault("LBX_AUTH__ENABLED", "false")

# The test corpus is face-only (no license plates) and ships no plate model, so the mandatory-plate gate
# is relaxed here. test_pii_gate.py constructs PiiSettings(plate_mandatory=True) explicitly to verify it.
os.environ.setdefault("LBX_PII__PLATE_MANDATORY", "false")

# ISOLATION: the suite seeds synthetic sessions/frames/objects and commits them. Point it at a dedicated
# test database so it never writes to the production corpus (which had accumulated ~1600 synthetic noise
# sessions from prior runs against the live DB). Override LBX_POSTGRES__DB in CI to change the name.
os.environ.setdefault("LBX_POSTGRES__DB", "labeloxav_test")


@pytest.fixture(scope="session", autouse=True)
def _provision_test_db():
    """Create the isolated test database (if missing) and bring it to the head schema once per session, so
    tests run against a real-schema DB that is never the production corpus. Refuses unless the target db
    name looks like a test db: a guard against accidentally pointing the suite at production."""
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
