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

# Webhook receivers in the suite are on localhost, which the SSRF guard refuses by default (a webhook URL is
# attacker-controlled input the server fetches with its own network position). Opt in for tests only.
os.environ.setdefault("LBX_INTEGRATIONS__ALLOW_PRIVATE_WEBHOOK_TARGETS", "true")

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
    _reset_corpus(pg)
    yield


# Tables whose contents are not test residue, so emptying them would break the schema rather than clean it.
# Determined by migrating an empty database and listing what is non-empty: PostGIS reference data, the
# alembic revision, and the governance singleton. Everything outside this set is either seeded by tests or
# re-seeded below.
#
# app_user is migration-seeded but deliberately absent: the suite creates users constantly and they were
# accumulating into the thousands, which is the same rot this reset exists to stop. Its one bootstrap row is
# re-inserted instead.
_MIGRATION_SEEDED = frozenset({"alembic_version", "spatial_ref_sys", "governance_state"})


def _reset_corpus(pg) -> None:
    """Empty the corpus tables so a run starts from the schema's own reference data and nothing else.

    The suite seeds sessions, frames and objects and commits them, and nothing ever cleaned up, so this
    database had accumulated 6,865 sessions and 12,262 frames across runs. Tests that assert on a
    corpus-wide figure (an availability count, a search hit, the largest cluster) were therefore written
    against whatever had piled up that month and expired quietly later. That is the recorded
    order-dependent failure category and the cause of the intermittent ones that replaced it, and it got
    worse with every run.

    Once per session rather than once per test, for two reasons. Per-test isolation through a rolled back
    transaction is the textbook fix and is not available here: these tests call asyncio.run more than once
    (seed, then exercise), and an asyncpg connection cannot outlive the loop that opened it, so no
    transaction can span a test. And a per-test truncate would force every test to re-seed the 182 ontology
    rows it currently guards on, for no benefit, because within one run each affected test seeds once. What
    actually broke these tests was accumulation across runs.
    """
    import psycopg

    conn = psycopg.connect(host=pg.host, port=pg.port, user=pg.user, password=pg.password,
                           dbname=pg.db, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            targets = sorted(t for (t,) in cur.fetchall() if t not in _MIGRATION_SEEDED)
            if targets:
                cur.execute("TRUNCATE TABLE {} RESTART IDENTITY CASCADE".format(
                    ", ".join(f'"{t}"' for t in targets)))
            # The bootstrap admin, exactly as db/migrations/versions/0011_users.py creates it. Re-inserted
            # rather than preserved so per-run test users do not pile up behind it.
            cur.execute("INSERT INTO app_user (user_id, name, role) "
                        "VALUES (gen_random_uuid(), 'admin', 'admin')")
            # CASCADE reaches any table holding a foreign key into one being truncated, which would silently
            # take the preserved rows with it. Checked rather than assumed, because an empty
            # governance_state would surface far from here as unrelated governance failures.
            for t in ("governance_state", "alembic_version"):
                cur.execute(f'SELECT count(*) FROM "{t}"')  # noqa: S608
                if cur.fetchone()[0] == 0:
                    raise RuntimeError(
                        f"resetting the test corpus emptied '{t}', which a migration seeds. A foreign key "
                        f"into a truncated table pulled it in through CASCADE; either add the referencing "
                        f"table to _MIGRATION_SEEDED or re-seed '{t}' here.")
    finally:
        conn.close()

    # The ontology is reference data the application seeds (scripts/seed_ontology.py), not something tests
    # own, and roughly half of them assume it is present rather than seeding it themselves. It survived
    # before only because it was residue nobody had cleared, which is precisely the dependency worth making
    # explicit: the suite needs a seeded ontology, so seed one.
    import asyncio

    from scripts.seed_ontology import seed

    asyncio.run(seed())


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
