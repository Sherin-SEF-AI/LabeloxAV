"""SEC-M6: the privacy plane is pack-driven.

Redaction targets and the legal regime now come from pack.privacy (AV: face+plate, DPDPA), so the anonymizer
runs the pack's image targets in order and the DPDPA gate records the pack's regime. AV stays byte-identical:
face then plate, same PiiResult. A pack can subset (face only) or extend; an unknown target fails loud; an
audio target (speech) is skipped by the image anonymizer.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import PiiSettings
from packs.base import RedactionTarget
from services import domain
from services.anonymize.anonymizer import PiiAnonymizer
from services.anonymize.compliance import evaluate_dpdpa


class _StubFace:
    available = True

    def __init__(self, regions):
        self._regions = regions

    def detect(self, img):
        return list(self._regions)


class _StubPlate:
    available = True

    def __init__(self, regions=()):
        self._regions = regions

    def detect(self, img):
        return list(self._regions)


def _img():
    return np.full((200, 200, 3), 200, dtype=np.uint8)


# ---- pack-supplied targets + regime ---------------------------------------------------------------------

def test_av_and_sec_targets_and_regime():
    assert tuple(t.name for t in domain.redaction_targets("av")) == ("face", "plate")
    assert tuple(t.name for t in domain.redaction_targets("sec")) == ("face", "plate")
    assert domain.legal_regime("av") == "DPDPA"
    assert domain.legal_regime("sec") == "DPDPA"


def test_anonymize_is_byte_identical_for_av_face_then_plate():
    anon = PiiAnonymizer(PiiSettings(plate_mandatory=False),
                         face_detector=_StubFace([(10.0, 10.0, 40.0, 40.0, 0.9)]),
                         plate_detector=_StubPlate([(50.0, 50.0, 90.0, 70.0, 0.8)]))
    res = anon.anonymize(_img())
    assert res.n_faces == 1 and res.n_plates == 1
    # order preserved: face region first, then plate
    assert [r["type"] for r in res.regions] == ["face", "plate"]


def test_face_only_pack_skips_plate(monkeypatch):
    monkeypatch.setattr(domain, "redaction_targets",
                        lambda *a, **k: (RedactionTarget(name="face", detector="face"),))
    anon = PiiAnonymizer(PiiSettings(plate_mandatory=False),
                         face_detector=_StubFace([(10.0, 10.0, 40.0, 40.0, 0.9)]),
                         plate_detector=_StubPlate([(50.0, 50.0, 90.0, 70.0, 0.8)]))  # would fire, but not a target
    res = anon.anonymize(_img())
    assert res.n_faces == 1 and res.n_plates == 0
    assert [r["type"] for r in res.regions] == ["face"]


def test_unknown_target_fails_loud(monkeypatch):
    monkeypatch.setattr(domain, "redaction_targets",
                        lambda *a, **k: (RedactionTarget(name="iris", detector="iris"),))
    anon = PiiAnonymizer(PiiSettings(plate_mandatory=False),
                         face_detector=_StubFace([]), plate_detector=_StubPlate())
    with pytest.raises(RuntimeError, match="no detector for redaction target 'iris'"):
        anon.anonymize(_img())


def test_audio_speech_target_is_skipped_by_the_image_anonymizer(monkeypatch):
    monkeypatch.setattr(domain, "redaction_targets", lambda *a, **k: (
        RedactionTarget(name="face", detector="face"),
        RedactionTarget(name="speech", detector="speech"),   # audio: not the anonymizer's job
    ))
    anon = PiiAnonymizer(PiiSettings(plate_mandatory=False),
                         face_detector=_StubFace([(10.0, 10.0, 40.0, 40.0, 0.9)]),
                         plate_detector=_StubPlate())
    res = anon.anonymize(_img())   # must not raise on the speech target
    assert res.n_faces == 1
    assert [r["type"] for r in res.regions] == ["face"]


# ---- the DPDPA verdict carries the regime, behaviour unchanged ------------------------------------------

def test_dpdpa_verdict_carries_regime_and_keeps_blocking():
    ok = evaluate_dpdpa({"f1"}, {"f1"}, [], regime="DPDPA")
    assert ok["pass"] and ok["regime"] == "DPDPA"
    blocked = evaluate_dpdpa({"f1", "f2"}, {"f1"}, [{"is_personal": True, "redacted": False}])
    assert not blocked["pass"]
    assert {b["kind"] for b in blocked["blockers"]} == {"unredacted_visual_pii", "unredacted_speech"}
    assert blocked["regime"] == "DPDPA"   # default preserved for the 3-arg callers
