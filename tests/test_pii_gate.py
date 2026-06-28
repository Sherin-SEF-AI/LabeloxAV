"""R1.2: the PII gate is fail-loud when a required detector is missing, so license plates can never
silently reach the object store un-blurred (DPDPA). No infra required (detectors are injected)."""

from __future__ import annotations

import numpy as np
import pytest

from core.config import PiiSettings
from services.anonymize.anonymizer import PiiAnonymizer


class _Det:
    def __init__(self, available: bool, regions=None):
        self._a = available
        self._r = regions or []

    @property
    def available(self) -> bool:
        return self._a

    def detect(self, image_bgr):
        return self._r


def test_gate_requires_plate_by_default():
    with pytest.raises(RuntimeError, match="license-plate"):
        PiiAnonymizer(PiiSettings(enabled=True, plate_mandatory=True),
                      face_detector=_Det(True), plate_detector=_Det(False))


def test_gate_requires_face():
    with pytest.raises(RuntimeError, match="face"):
        PiiAnonymizer(PiiSettings(enabled=True),
                      face_detector=_Det(False), plate_detector=_Det(True))


def test_opt_out_allows_face_only():
    a = PiiAnonymizer(PiiSettings(enabled=True, plate_mandatory=False),
                      face_detector=_Det(True), plate_detector=_Det(False))
    assert a is not None


def test_blurs_plate_when_available():
    plate = _Det(True, regions=[(10, 10, 40, 30, 0.9)])
    a = PiiAnonymizer(PiiSettings(enabled=True), face_detector=_Det(True), plate_detector=plate)
    res = a.anonymize(np.full((100, 100, 3), 200, dtype=np.uint8))
    assert res.n_plates == 1


def test_disabled_gate_allows_missing_detectors():
    a = PiiAnonymizer(PiiSettings(enabled=False), face_detector=_Det(False), plate_detector=_Det(False))
    assert a is not None
