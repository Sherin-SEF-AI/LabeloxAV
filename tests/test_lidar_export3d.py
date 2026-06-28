"""M-L3.4: a 3D slice seals to a pinned commit and exports to OpenLABEL, nuScenes, KITTI, and Waymo with a
provenance sidecar, raw clouds export to LAS and PCD, the analytics show 3D coverage, and a natural-language
query returns matching clouds.

Format writers + seal need no infra; export_3d_dataset + metrics + search need DB + MinIO."""

from __future__ import annotations

import asyncio
import json
import uuid

import numpy as np
import pytest

from core.config import get_settings
from services.lidar.export import (
    Slice3D,
    seal_3d_commit_id,
    write_kitti_3d,
    write_las,
    write_nuscenes_3d,
    write_openlabel_3d,
    write_pcd,
    write_waymo_3d,
)
from services.lidar.ingest import Cloud


def _rec(name, cloud_id, conf=0.9):
    return {"object_3d_id": str(uuid.uuid4()), "cloud_id": cloud_id, "ts_ns": 1, "cloud_uri": "s3://x",
            "cloud_source": "pseudo", "vehicle_id": "V", "city": "BLR", "class_id": 11, "class_name": name,
            "center": [10.0, 1.0, 0.75], "dims": [4.0, 1.8, 1.5], "yaw": 0.2, "pitch": 0.0, "roll": 0.0,
            "conf": conf, "state": "accepted", "box_source": "lifted", "track_3d_id": None,
            "object_id": None, "provenance": {"n_points": 120}}


def test_format_writers(tmp_path):
    cid = str(uuid.uuid4())
    records = [_rec("sedan", cid), _rec("pedestrian", cid)]

    ol = json.loads(write_openlabel_3d(records, tmp_path).read_text())
    assert len(ol["openlabel"]["objects"]) == 2
    assert len(next(iter(ol["openlabel"]["objects"].values()))["object_data"]["cuboid"][0]["val"]) == 9

    nu = json.loads(write_nuscenes_3d(records, tmp_path).read_text())
    assert len(nu["sample_annotation"]) == 2 and len(nu["sample_annotation"][0]["rotation"]) == 4

    kdir = write_kitti_3d(records, tmp_path)
    line = (kdir / f"{cid}.txt").read_text().splitlines()[0].split()
    assert line[0] == "sedan" and len(line) == 16    # type trunc occl alpha x1 y1 x2 y2 h w l x y z ry conf

    wa = json.loads(write_waymo_3d(records, tmp_path).read_text())
    assert wa["frames"][0]["laser_labels"][0]["box"]["heading"] == 0.2


def test_cloud_writers_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    cloud = Cloud(xyz=rng.uniform(-10, 30, (500, 3)).astype(np.float32),
                  intensity=rng.uniform(0, 1, 500).astype(np.float32), ts_ns=1, source="pseudo")
    import laspy
    write_las(cloud, tmp_path / "c.las")
    las = laspy.read(str(tmp_path / "c.las"))
    assert np.allclose(np.stack([las.x, las.y, las.z], 1), cloud.xyz, atol=1e-2)
    from pypcd4 import PointCloud as PCD
    write_pcd(cloud, tmp_path / "c.pcd")
    pcd = PCD.from_path(str(tmp_path / "c.pcd"))
    assert pcd.numpy(("x", "y", "z")).shape == (500, 3)


def test_seal_is_deterministic():
    cid = str(uuid.uuid4())
    a = [_rec("sedan", cid)]
    s1 = seal_3d_commit_id(Slice3D(), a, "onto-1")
    s2 = seal_3d_commit_id(Slice3D(), a, "onto-1")
    assert s1 == s2 and s1.startswith("lbx3d-")
    assert seal_3d_commit_id(Slice3D(), a + [_rec("bus", cid)], "onto-1") != s1


def _infra_up() -> bool:
    try:
        import redis as redis_lib
        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up")


def _clear():
    from db.session import get_engine, get_sessionmaker
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@requires_infra
def test_export_metrics_and_search(tmp_path):
    async def run():
        from core.storage import get_object_store
        from core.timebase import now_ns
        from db.models import DatasetCommit, Object3D
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.lidar.export import export_3d_dataset, metrics_3d, search_clouds_3d
        from services.lidar.ingest import store_cloud

        get_object_store().ensure_bucket()
        onto = get_ontology()
        sedan, ped = onto.by_name("sedan").id, onto.by_name("pedestrian").id
        sid, ts = uuid.uuid4(), now_ns()
        cloud = Cloud(xyz=np.random.default_rng(0).uniform(-10, 30, (2000, 3)).astype(np.float32),
                      intensity=np.ones(2000, np.float32), ts_ns=ts, source="pseudo")
        async with get_sessionmaker()() as db:
            db.add(DbSession(session_id=sid, vehicle_id="EXP-3D", start_ts_ns=ts, end_ts_ns=ts + 1,
                             city="BLR", sensors={}, ontology_version=onto.version))
            await db.commit()
        stored = await store_cloud(cloud, sid, source="pseudo")
        cloud_id = uuid.UUID(stored["cloud_id"])
        async with get_sessionmaker()() as db:
            for cls in (sedan, ped):
                db.add(Object3D(cloud_id=cloud_id, class_id=cls, center=[10, 0, 0.75], dims=[4, 1.8, 1.5],
                                yaw=0.1, conf=0.9, box_source="lifted", source="fused", state="accepted"))
            await db.commit()

        res = await export_3d_dataset(Slice3D(session_ids=[sid]),
                                      formats=["openlabel", "nuscenes", "kitti", "waymo"], out_root=tmp_path)
        assert res["object_3d_count"] == 2 and res["cloud_count"] == 1
        assert {"openlabel", "nuscenes", "kitti", "waymo", "provenance"} <= set(res["formats"])
        async with get_sessionmaker()() as db:
            commit = await db.get(DatasetCommit, res["commit_id"])
            assert commit.object_3d_count == 2 and commit.cloud_count == 1

        m = await metrics_3d()
        assert m["object_3d_count"] >= 2 and "sedan" in m["objects_by_class"]

        found = await search_clouds_3d("clouds with pedestrians near a sedan")
        assert set(found["classes"]) == {"sedan", "pedestrian"} and str(cloud_id) in found["clouds"]

    _clear()
    try:
        asyncio.run(run())
    finally:
        _clear()
