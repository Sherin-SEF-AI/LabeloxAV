"""M-IMU.2 (audio): the RMS envelope that time-locks dashcam audio energy to the inertial timeline, plus a
real-dashcam decode when the sample is present."""

from __future__ import annotations

import os

import numpy as np
import pytest

from services.ingest.audio import audio_envelope, rms_envelope

_DASHCAM = "/home/jo/Documents/Dashcam_2026-06-25/20260624093914_000001F.MP4"


def test_rms_envelope_tracks_energy():
    env = rms_envelope(np.concatenate([np.zeros(2000, np.float32), np.full(2000, 0.8, np.float32)]), buckets=4)
    assert env[0] < 0.1 and env[-1] > 0.5


def test_rms_envelope_scales_int16_range():
    env = rms_envelope(np.full(1000, 16384.0, np.float32), buckets=2)   # half of int16 max -> ~0.5
    assert all(0.4 < v < 0.6 for v in env)


def test_rms_envelope_empty():
    assert rms_envelope([]) == []


@pytest.mark.skipif(not os.path.exists(_DASHCAM), reason="dashcam sample not present")
def test_audio_envelope_on_real_dashcam():
    res = audio_envelope(_DASHCAM, buckets=200)
    assert res["found"] is True
    assert res["sample_rate"] == 16000
    assert res["buckets"] > 0 and any(v > 0 for v in res["envelope"])
