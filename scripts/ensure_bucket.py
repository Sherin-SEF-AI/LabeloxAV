"""Create the object-store bucket if it does not exist. Idempotent; run by the deployment's migrate step.

    python -m scripts.ensure_bucket
"""

from __future__ import annotations

from core.config import get_settings
from core.storage import get_object_store


def main() -> None:
    get_object_store().ensure_bucket()
    print(f"bucket '{get_settings().minio.bucket}' ready")


if __name__ == "__main__":
    main()
