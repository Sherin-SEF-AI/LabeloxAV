"""Seed a throwaway database for the editor end-to-end suite, and write the fixture the specs read.

    python -m scripts.e2e_seed --out <dir>

Run by scripts/e2e_editor.sh against a database named for the purpose (it refuses any other). Creates
three users, one per role, and one session of ten synthetic frames with a few boxes each, imported through
the real importer so the rows are what the product itself would have written. Writes <dir>/fixture.json:
the frame ids and, per user, the stored sign-in state the web app keeps in localStorage.

The fixture holds bearer tokens for the throwaway database. It is written into the run's temporary
directory, never the repository, and the database it opens is dropped when the run ends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select

USERS = (("e2e-admin", "admin"), ("e2e-reviewer", "reviewer"), ("e2e-annotator", "annotator"))


def synthetic_dataset(root: Path, n_frames: int = 10) -> Path:
    """A COCO folder of road-ish frames: a horizon, a few coloured rectangles, a box around each."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    images, anns = [], []
    aid = 1
    for i in range(n_frames):
        img = np.full((720, 1280, 3), (70, 70, 70), np.uint8)
        img[:300] = (150, 120, 90)
        boxes = []
        for _ in range(4):
            w, h = int(rng.integers(60, 220)), int(rng.integers(50, 160))
            x, y = int(rng.integers(20, 1280 - w - 20)), int(rng.integers(260, 720 - h - 20))
            cv2.rectangle(img, (x, y), (x + w, y + h), tuple(int(c) for c in rng.integers(40, 230, 3)), -1)
            boxes.append((x, y, w, h))
        name = f"frame_{i:02d}.jpg"
        cv2.imwrite(str(root / name), img)
        images.append({"id": i + 1, "file_name": name, "width": 1280, "height": 720})
        for x, y, w, h in boxes:
            anns.append({"id": aid, "image_id": i + 1, "category_id": 1, "bbox": [x, y, w, h],
                         "labelox": {"class_name": "sedan", "conf": 0.9}})
            aid += 1
    (root / "annotations.json").write_text(json.dumps(
        {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "car"}]}))
    return root


async def seed(out: Path) -> dict:
    from core.config import get_settings
    from db.models import Frame, User
    from db.session import get_sessionmaker
    from services.api.auth_token import mint_token
    from services.imports.records import ImportSpec
    from services.imports.run import import_dataset

    settings = get_settings()
    if "e2e" not in settings.postgres.db.lower():
        raise SystemExit(f"refusing to seed '{settings.postgres.db}': this writes users and tokens, and only a "
                         "database named for end-to-end runs (LBX_POSTGRES__DB=..._e2e) is fair game")

    res = await import_dataset(ImportSpec(format="coco", source_uri=str(synthetic_dataset(out / "coco")),
                                          target_vehicle="E2E-01", city="BLR"))
    users = {}
    async with get_sessionmaker()() as db:
        for name, role in USERS:
            user = (await db.execute(select(User).where(User.name == name))).scalars().first()
            if user is None:
                user = User(name=name, role=role)
                db.add(user)
                await db.flush()
                await db.refresh(user)
            users[role] = {"user_id": str(user.user_id), "name": name, "role": role,
                           "token": mint_token(str(user.user_id), settings.auth.signing_key,
                                               ttl_seconds=6 * 3600,
                                               token_version=int(user.token_version or 1))}
        await db.commit()
        frames = [str(f) for (f,) in (await db.execute(
            select(Frame.frame_id).where(Frame.session_id == res["session_id"]).order_by(Frame.ts_ns))).all()]
    fixture = {"session_id": res["session_id"], "frames": frames, "users": users}
    (out / "fixture.json").write_text(json.dumps(fixture))
    return fixture


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fx = asyncio.run(seed(args.out))
    print(f"seeded {len(fx['frames'])} frames and {len(fx['users'])} users")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
