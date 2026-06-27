"""Configure the MinIO bucket CORS so the browser can PUT multipart parts directly and read ETag.

One-time setup for direct-to-storage uploads (Deliverable 3). Without ExposeHeaders=ETag the browser
multipart completion silently fails.

    uv run python scripts/setup_minio_cors.py
"""

from __future__ import annotations

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.storage import get_object_store

log = get_logger("minio_cors")

ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def main() -> None:
    setup_logging(get_settings().log_level)
    store = get_object_store()
    store.ensure_bucket()
    store.set_cors(ORIGINS)
    log.info("minio_cors.done", origins=ORIGINS)


if __name__ == "__main__":
    main()
