"""Milestone A: the canonical timeline alignment math (frame -> IMU -> audio index mapping), pure, no infra.
Covers nearest-sample lookup, the audio sample index, a GNSS dropout reported (never a false fix), and the
sync-method + accumulated-error report."""

from __future__ import annotations

from services.intelligence.timeline import (
    align_frame,
    audio_sample_index,
    nearest_index,
    sync_report,
)

SEC = 1_000_000_000


def test_nearest_index_picks_closest():
    ts = [0, 100, 200, 300]
    assert nearest_index(ts, 140) == 1      # 100 is closer than 200
    assert nearest_index(ts, 160) == 2
    assert nearest_index(ts, -50) == 0
    assert nearest_index(ts, 9999) == 3
    assert nearest_index([], 5) is None


def test_audio_sample_index_at_rate():
    # 0.5 s into a 16 kHz stream that starts at 1e9 ns -> sample 8000
    assert audio_sample_index(int(1.5 * SEC), SEC, 16000) == 8000
    assert audio_sample_index(SEC // 2, SEC, 16000) is None   # before the stream start
    assert audio_sample_index(SEC, SEC, 0) is None            # no audio


def test_align_frame_maps_all_modalities():
    imu = [i * (SEC // 200) for i in range(200)]   # 200 Hz IMU over ~1 s
    gnss = [0, SEC]                                  # GNSS fixes 1 s apart
    a = align_frame(SEC // 2, imu, gnss, audio_start_ns=0, audio_sr=16000, max_gap_ns=SEC)
    assert a["imu_index"] is not None and abs(a["imu_dt_ns"]) <= SEC // 200
    assert a["gnss_index"] is not None and a["gnss_dropout"] is False
    assert a["audio_sample"] == 8000


def test_gnss_dropout_is_flagged_not_interpolated():
    # the nearest GNSS fix is 5 s away, well beyond a 1 s max gap -> dropout, no index
    a = align_frame(5 * SEC, imu_ts=[], gnss_ts=[0, 10 * SEC], audio_start_ns=None, audio_sr=None,
                    max_gap_ns=SEC)
    assert a["gnss_dropout"] is True and a["gnss_index"] is None


def test_sync_report_hardware_vs_interpolated():
    assert sync_report("hardware", [0, 1, 2], [0, 1, 2]) == {"sync_method": "hardware", "accumulated_error_ns": 0}
    rep = sync_report("interpolated", [10, 110, 210], [0, 100, 200])
    assert rep["sync_method"] == "interpolated" and rep["accumulated_error_ns"] == 10
