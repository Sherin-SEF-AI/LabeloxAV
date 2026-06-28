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
