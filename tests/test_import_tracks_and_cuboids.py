"""An import keeps the tracks, the 3D boxes and the point clouds its source carried.

Nine adapters read a source's track identity and the importer ignored it, so a tracked export arrived as
unrelated boxes. A 3D box rode in provenance and was never written to the column that holds one. A
nuScenes import brought camera frames and left the LiDAR on disk. Checked here through the real
importer against a synthetic nuScenes release built by tests/test_nuscenes_standard_import.py.
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import func, select

from core.config import get_settings
from db.models import Frame, Object, PointCloud, Track
from db.session import get_sessionmaker
from services.imports.records import ImportSpec

pytestmark = pytest.mark.db

_SPEC = importlib.util.spec_from_file_location(
    "nuscenes_fixture", Path(__file__).with_name("test_nuscenes_standard_import.py"))
_fixture = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_fixture)


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _release(root: Path) -> Path:
    """The fixture's scene, with real images and real sweeps where it writes placeholders."""
    _fixture._build_scene().write(root)
    rng = np.random.default_rng(5)
    for row in json.loads((root / "v1.0-mini" / "sample_data.json").read_text()):
        path = root / row["filename"]
        if path.suffix == ".jpg":
            cv2.imwrite(str(path), np.full((row.get("height") or 900, row.get("width") or 1600, 3), 80, np.uint8))
        elif path.name.endswith(".pcd.bin"):
            pts = np.c_[rng.uniform(-30, 30, size=(64, 3)), rng.uniform(0, 255, size=64), np.arange(64) % 32]
            path.write_bytes(pts.astype("<f4").tobytes())
    return root


@pytest.fixture
def no_pii(monkeypatch):
    monkeypatch.setattr(get_settings().pii, "enabled", False)


@requires_infra
@pytest.mark.asyncio
async def test_a_nuscenes_release_arrives_with_classes_cuboids_tracks_and_lidar(tmp_path, no_pii):
    from services.imports.run import import_dataset

    root = _release(tmp_path / "nuscenes")
    res = await import_dataset(ImportSpec(format="nuscenes", source_uri=str(root), target_vehicle="NUSC-T"))
    sid = uuid.UUID(res["session_id"])
    assert res["counts"]["frames"] == 3 and res["counts"]["objects"] == 5 and res["counts"]["unmapped"] == 0

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Object, Frame.cam_id).join(Frame, Frame.frame_id == Object.frame_id)
            .where(Frame.session_id == sid))).all()
        by_source = {}
        for obj, cam in rows:
            by_source.setdefault(obj.provenance["original_name"], []).append((obj, cam))
        assert set(by_source) == {"vehicle.car", "vehicle.truck", "vehicle.bus.rigid", "movable_object.barrier"}

        for obj, _cam in rows:
            assert obj.cuboid_3d and set(obj.cuboid_3d) >= {"center", "size", "yaw"}, \
                "the 3D box was left in provenance instead of the column that holds one"
            assert obj.track_id is not None, "the source's instance id did not become a track"

        # The car is one physical object seen in two samples by the same camera: one track, two boxes.
        car_tracks = {obj.track_id for obj, _cam in by_source["vehicle.car"]}
        assert len(by_source["vehicle.car"]) == 2 and len(car_tracks) == 1
        track = await db.get(Track, car_tracks.pop())
        assert track.first_ts_ns < track.last_ts_ns and track.tracker_version == "imported"
        assert (await db.execute(select(func.count()).select_from(Track)
                                 .where(Track.session_id == sid))).scalar_one() == 4

        clouds = (await db.execute(select(PointCloud).where(PointCloud.session_id == sid))).scalars().all()
        assert len(clouds) == 1, "only the annotated keyframe sweep is stored, and it is stored"
        assert clouds[0].point_count == 64 and clouds[0].source == "dataset"
        same_ts = (await db.execute(select(func.count()).select_from(Frame).where(
            Frame.session_id == sid, Frame.ts_ns == clouds[0].ts_ns))).scalar_one()
        assert same_ts >= 1, "the cloud is not at the timestamp of the camera frames it belongs with"


@requires_infra
@pytest.mark.asyncio
async def test_a_track_id_in_any_format_becomes_a_track(tmp_path, no_pii):
    from services.imports.run import import_dataset

    root = tmp_path / "coco"
    root.mkdir()
    images, anns = [], []
    for i in range(3):
        name = f"f{i}.jpg"
        cv2.imwrite(str(root / name), np.full((120, 160, 3), 60 + i, np.uint8))
        images.append({"id": i + 1, "file_name": name, "width": 160, "height": 120, "ts_ns": 1000 + i})
        anns.append({"id": 2 * i + 1, "image_id": i + 1, "category_id": 1, "bbox": [10 + i, 10, 30, 20],
                     "labelox": {"class_name": "sedan", "track_id": "veh-7"}})
        anns.append({"id": 2 * i + 2, "image_id": i + 1, "category_id": 1, "bbox": [80, 40, 20, 20],
                     "labelox": {"class_name": "sedan"}})                       # untracked
    (root / "annotations.json").write_text(json.dumps(
        {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "car"}]}))

    res = await import_dataset(ImportSpec(format="coco", source_uri=str(root), target_vehicle="TRK-T"))
    sid = uuid.UUID(res["session_id"])
    assert res["counts"]["tracks"] == 1
    async with get_sessionmaker()() as db:
        objs = (await db.execute(select(Object).join(Frame, Frame.frame_id == Object.frame_id)
                                 .where(Frame.session_id == sid))).scalars().all()
        tracked = [o for o in objs if o.track_id is not None]
        assert len(tracked) == 3 and len({o.track_id for o in tracked}) == 1
        assert all(o.provenance.get("track_ref") == "veh-7" for o in tracked)
        assert sum(1 for o in objs if o.track_id is None) == 3, "an untracked box must stay untracked"
        track = await db.get(Track, tracked[0].track_id)
        assert (track.first_ts_ns, track.last_ts_ns) == (1000, 1002)
