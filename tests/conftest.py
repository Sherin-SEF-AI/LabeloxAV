import sys
from pathlib import Path

import pytest

# Allow running the suite without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _reset_db_caches():
    """Each async test runs in its own event loop. The app caches one engine per process (correct,
    since the CLI uses a single asyncio.run loop), so clear that cache around every test to avoid
    reusing an engine bound to a closed loop."""
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    yield
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
