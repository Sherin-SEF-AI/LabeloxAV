"""MCAP indexer (M-I.1): measured per-topic rates, time range, and gap detection over a real generated MCAP."""

from __future__ import annotations

import pytest

pytest.importorskip("mcap")
pytest.importorskip("mcap_protobuf")

from scripts.make_inspector_fixture import build_mcap
from services.inspector.indexer import build_index_from_bytes


def test_measured_rates_and_topics():
    data, lo, hi, _ = build_mcap(seconds=6.0, imu_rate=200.0, gap_topic=None)
    idx = build_index_from_bytes(data, gap_min_factor=5.0)
    topics = idx["topics"]
    assert set(topics) == {"/camera/cam_f", "/gnss", "/imu", "/can/speed"}
    # the IMU topic must show its TRUE rate, not a nominal one
    assert abs(topics["/imu"]["rate"] - 200.0) < 5.0
    assert abs(topics["/camera/cam_f"]["rate"] - 10.0) < 1.0
    assert abs(topics["/can/speed"]["rate"] - 100.0) < 2.0
    assert topics["/imu"]["schema"] == "sensor.Imu"
    assert idx["time_range"][0] == lo and idx["gaps"] == {}
    span_s = (idx["time_range"][1] - idx["time_range"][0]) / 1e9
    assert 5.5 < span_s < 6.5


def test_gap_detection():
    data, _lo, _hi, _ = build_mcap(seconds=6.0, imu_rate=200.0, gap_topic="/imu")
    idx = build_index_from_bytes(data, gap_min_factor=5.0)
    assert "/imu" in idx["gaps"]
    win = idx["gaps"]["/imu"][0]
    assert 0.9 < (win[1] - win[0]) / 1e9 < 1.2      # the seeded ~1s dropout, with its evidence window


def test_wrong_rate_is_measured():
    data, _lo, _hi, _ = build_mcap(seconds=6.0, imu_rate=247.0, gap_topic=None)
    idx = build_index_from_bytes(data, gap_min_factor=5.0)
    assert abs(idx["topics"]["/imu"]["rate"] - 247.0) < 6.0   # the real 247Hz, which M-I.2 flags vs a 200 target
