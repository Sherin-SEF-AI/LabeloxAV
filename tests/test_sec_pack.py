"""SEC-M3: the LabeloxSec pack - ontology, safety, and end-to-end static-camera ingestion.

Proves the Sec pack registers and conforms, its ontology/safety are genuinely Sec (person + weapon, not the AV
VRU/animal), and a fixed-camera clip ingests to real Session/Frame rows with a null vehicle and a Sec ontology
binding. AV and Sec are shown to coexist without leaking into each other.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from packs.base import DomainPack, IngestionAdapter, IngestSource, SceneModelFactory
from packs.registry import get_pack
from services.autolabel.ontology import get_ontology

pytestmark = pytest.mark.db


def _write_clip(path: Path, n: int = 8, fps: int = 8, size: int = 32) -> bool:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    if not writer.isOpened():
        return False
    for i in range(n):
        f = np.full((size, size, 3), 80, dtype=np.uint8)
        f[i:i + 4, i:i + 4] = 240
        writer.write(f)
    writer.release()
    return path.exists() and path.stat().st_size > 0


def test_sec_pack_registers_and_conforms():
    sec = get_pack("sec")
    assert isinstance(sec, DomainPack)
    assert sec.manifest.id == "sec"
    assert sec.manifest.default_scene_model == "static_camera"
    assert isinstance(sec.scene_model, SceneModelFactory)
    assert all(isinstance(a, IngestionAdapter) for a in sec.ingestion_adapters)
    assert len(sec.ingestion_adapters) == 1


def test_sec_ontology_loads_via_registry():
    o = get_ontology("sec")
    assert o.version == "labelox-sec-0.1.0"
    assert o.has_name("person") and o.has_name("weapon") and o.has_name("abandoned_object")
    # it is a different taxonomy from AV
    assert o.version != get_ontology("av").version


def test_sec_safety_is_person_and_weapon_not_vru_animal():
    sec = get_pack("sec")
    sp = sec.safety_policy
    assert sp.is_safety_class("person") and sp.is_safety_class("weapon") and sp.is_safety_class("firearm")
    assert not sp.is_safety_class("car")
    # the AV safety words are not Sec safety classes (and are not even in the Sec ontology)
    assert not sp.is_safety_class("pedestrian")
    assert set(sp.critical_class_names()) == {"person", "weapon", "firearm", "knife", "abandoned_object"}


def test_sec_affinity_is_safety_aware_for_its_own_superclasses():
    sec = get_pack("sec")
    o = get_ontology("sec")
    person, car = o.by_name("person").id, o.by_name("car").id
    knife, weapon = o.by_name("knife").id, o.by_name("weapon").id
    assert sec.safety_policy.affinity_cost(person, car) == 1.0     # safety vs non-safety = max cost
    assert sec.safety_policy.affinity_cost(knife, weapon) == 0.2   # same superclass
    assert sec.safety_policy.affinity_cost(person, person) == 0.0


def test_sec_autolabel_profile_is_cctv():
    prof = get_pack("sec").autolabel_profile
    assert "CCTV" in prof.vlm_prompt_template or "security camera" in prof.vlm_prompt_template
    assert prof.disable_ego_hood_mask is True
    assert prof.gate_policy.safety_l1 == frozenset({"person", "weapon"})


def test_av_and_sec_coexist_without_leakage():
    av, sec = get_pack("av"), get_pack("sec")
    assert av.manifest.default_scene_model == "moving_camera"
    assert sec.manifest.default_scene_model == "static_camera"
    # AV keeps its VRU/animal safety; Sec keeps person/weapon; neither answers for the other's words.
    assert av.safety_policy.is_safety_class("pedestrian") and not av.safety_policy.is_safety_class("weapon")
    assert sec.safety_policy.is_safety_class("weapon") and not sec.safety_policy.is_safety_class("pedestrian")


async def test_static_camera_clip_ingests_end_to_end(tmp_path):
    from sqlalchemy import select

    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from packs.sec.ingest import persist

    clip = tmp_path / "gate_cam.avi"
    if not _write_clip(clip):
        pytest.skip("no OpenCV VideoWriter codec available")

    result = await persist(IngestSource("video", str(clip), "cam_gate", meta={"fps": 4, "city": "BLR"}))
    assert result["pack_id"] == "sec"
    assert result["n_frames"] >= 2

    async with get_sessionmaker()() as db:
        sess = await db.get(DbSession, __import__("uuid").UUID(result["session_id"]))
        assert sess.pack_id == "sec"
        assert sess.vehicle_id is None
        assert sess.ontology_version == "labelox-sec-0.1.0"
        frames = (await db.execute(select(Frame).where(Frame.session_id == sess.session_id))).scalars().all()
        assert len(frames) == result["n_frames"]
        assert all(f.cam_id == "cam_gate" for f in frames)
        assert all(f.ego_speed is None for f in frames)      # fixed camera: no CAN speed
