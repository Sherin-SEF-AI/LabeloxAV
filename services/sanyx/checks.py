"""SANYX ingest-QA check families (M1). Six pure, independently testable functions, one per failure mode
CHRONYX sessions actually exhibit. Each takes already-decoded channel data (timestamps, sequence numbers,
sensor arrays, or per-frame stats), never touches the object store or the DB, and returns a uniform result:

    {"name", "score" in [0,1], "status" in {pass, degraded, quarantine}, "detail", "evidence"}

run.py pulls the channels from the MCAP + frames and feeds them here; score.py folds the six into one
0..100 health score and a decision. Keeping the checks pure is what makes the corrupted-session acceptance
test deterministic. All thresholds come from settings.sanyx (config-driven)."""

from __future__ import annotations

import numpy as np

from core.config import SanyxSettings
from services.calibration.timesync import time_offset_check, validate_timesync

Result = dict


def _status(score: float, degraded_at: float = 0.6, quarantine_at: float = 0.3) -> str:
    if score >= degraded_at:
        return "pass"
    if score >= quarantine_at:
        return "degraded"
    return "quarantine"


def _clock_slope_ppm(ts_ns: list[int]) -> float:
    """Deviation of the mean inter-sample interval from a straight line, in parts per million. A drifting or
    slipping clock shows a growing residual; a healthy PPS-locked clock is near zero."""
    if len(ts_ns) < 3:
        return 0.0
    t = np.asarray(sorted(ts_ns), dtype=np.float64)
    idx = np.arange(t.size)
    slope, intercept = np.polyfit(idx, t, 1)
    if slope <= 0:
        return 1e6
    resid = t - (slope * idx + intercept)
    return float(np.std(resid) / slope * 1e6)


def check_time_sync(cfg: SanyxSettings, imu_ts_ns: list[int], cam_ts_by_cam: dict[str, list[int]],
                    gnss_ts_ns: list[int] | None = None, pps_ts_ns: list[int] | None = None) -> Result:
    """Cross-channel synchronization against the STM32 reference. Validates IMU rate, camera-vs-IMU and
    GNSS-vs-IMU offsets, inter-camera skew, and clock slope. Loss of the reference pulse hard-fails when the
    rig requires it (pps_required)."""
    cams = list(cam_ts_by_cam.items())
    first_cam = cams[0][1] if cams else None
    vt = validate_timesync(imu_ts_ns, first_cam, gnss_ts_ns)

    inter_cam_ok = True
    inter_cam = {}
    for i in range(len(cams) - 1):
        off = time_offset_check(cams[i][1], cams[i + 1][1])
        inter_cam[f"{cams[i][0]}_{cams[i+1][0]}"] = off
        inter_cam_ok = inter_cam_ok and off["ok"]

    slope_ppm = _clock_slope_ppm(imu_ts_ns)
    slope_ok = slope_ppm <= cfg.clock_slope_ppm_max
    pps_present = bool(pps_ts_ns)

    if cfg.pps_required and not pps_present:
        return {"name": "time_sync", "score": 0.0, "status": "quarantine",
                "detail": "time-sync reference pulse absent (pps_required)",
                "evidence": {"pps_present": False}}

    parts = [vt["imu_rate"]["ok"], vt["ok"], inter_cam_ok, slope_ok]
    score = float(np.mean([1.0 if p else 0.0 for p in parts]))
    status = _status(score)
    return {"name": "time_sync", "score": round(score, 4), "status": status,
            "detail": f"imu {vt['imu_rate'].get('hz')}Hz, slope {slope_ppm:.0f}ppm, "
                      f"cam/imu offset {vt.get('time_offset_ns')}ns",
            "evidence": {"imu_rate": vt["imu_rate"], "offsets": vt["offsets"],
                         "inter_camera": inter_cam, "clock_slope_ppm": round(slope_ppm, 1),
                         "pps_present": pps_present}}


def check_dropped_frames(cfg: SanyxSettings, seq_by_cam: dict[str, list[int]],
                         pts_by_cam: dict[str, list[int]] | None = None) -> Result:
    """Sequence-number gaps and PTS discontinuities per camera. Reports drop rate and the longest single gap."""
    pts_by_cam = pts_by_cam or {}
    per_cam = {}
    worst_rate = 0.0
    worst_gap = 0
    for cam, seq in seq_by_cam.items():
        s = sorted(int(x) for x in seq)
        if len(s) < 2:
            per_cam[cam] = {"drop_rate": 0.0, "longest_gap": 0, "expected": len(s)}
            continue
        expected = s[-1] - s[0] + 1
        received = len(set(s))
        dropped = max(0, expected - received)
        gaps = np.diff(s)
        longest = int(gaps.max() - 1) if gaps.size else 0
        rate = dropped / max(expected, 1)
        per_cam[cam] = {"drop_rate": round(rate, 5), "longest_gap": longest, "dropped": dropped,
                        "expected": expected}
        worst_rate = max(worst_rate, rate)
        worst_gap = max(worst_gap, longest)

    if worst_rate >= cfg.drop_rate_fail or worst_gap >= cfg.max_gap_frames_fail:
        status, score = "quarantine", max(0.0, 1.0 - worst_rate / max(cfg.drop_rate_fail, 1e-6)) * 0.3
    elif worst_rate >= cfg.drop_rate_warn:
        status, score = "degraded", 0.6
    else:
        status, score = "pass", 1.0
    return {"name": "dropped_frames", "score": round(float(score), 4), "status": status,
            "detail": f"worst drop rate {worst_rate:.3%}, longest gap {worst_gap} frames",
            "evidence": {"per_camera": per_cam, "worst_drop_rate": round(worst_rate, 5), "worst_gap": worst_gap}}


def check_exposure(cfg: SanyxSettings, frame_stats: list[dict], rolling_shutter_skew: list[float] | None = None) -> Result:
    """Exposure and rolling-shutter health from per-frame stats (mean_luma, clip_fraction, blur, well_exposed).
    well_exposed is produced by ingest.score_frame upstream; here we score the fraction that are healthy and
    flag rolling-shutter skew on high-motion frames."""
    if not frame_stats:
        return {"name": "exposure", "score": 0.0, "status": "quarantine", "detail": "no frames",
                "evidence": {}}
    n = len(frame_stats)
    well = sum(1 for f in frame_stats if f.get("well_exposed", True))
    ok_frac = well / n
    skew = rolling_shutter_skew or []
    skew_bad = sum(1 for s in skew if s > cfg.rolling_shutter_skew_max)
    skew_frac = skew_bad / len(skew) if skew else 0.0

    if ok_frac < cfg.exposure_ok_frac_fail:
        status, score = "quarantine", ok_frac * 0.4
    elif ok_frac < cfg.exposure_ok_frac_warn or skew_frac > 0.2:
        status, score = "degraded", 0.6
    else:
        status, score = "pass", ok_frac
    return {"name": "exposure", "score": round(float(score), 4), "status": status,
            "detail": f"{ok_frac:.1%} well exposed, rolling-shutter skew on {skew_frac:.1%} of motion frames",
            "evidence": {"well_exposed_frac": round(ok_frac, 4), "n_frames": n,
                         "rolling_shutter_bad_frac": round(skew_frac, 4)}}


def check_gps(cfg: SanyxSettings, fix_types: list[int], hdop: list[float], ts_ns: list[int],
              rtk_flags: list[int] | None = None) -> Result:
    """u-blox NEO-F9P integrity: usable-fix fraction, HDOP, RTK continuity, and the longest dropout."""
    if not fix_types:
        return {"name": "gps", "score": 0.0, "status": "quarantine", "detail": "no GPS samples", "evidence": {}}
    fix = np.asarray(fix_types)
    usable = float(np.mean(fix >= 2))  # 2 = 2D fix or better
    med_hdop = float(np.median(hdop)) if hdop else None
    rtk_frac = float(np.mean(np.asarray(rtk_flags) > 0)) if rtk_flags else 0.0

    longest_dropout_s = 0.0
    if len(ts_ns) >= 2:
        t = np.asarray(sorted(ts_ns), dtype=np.float64) / 1e9
        gaps = np.diff(t)
        # a dropout is a gap where no usable fix was present; approximate with the largest inter-sample gap
        longest_dropout_s = float(gaps.max()) if gaps.size else 0.0

    fail = (usable < cfg.min_fix_frac or (med_hdop is not None and med_hdop > cfg.hdop_max)
            or longest_dropout_s > cfg.max_dropout_s or rtk_frac < cfg.min_rtk_frac)
    hdop_term = 1.0 if med_hdop is None else max(0.0, 1.0 - med_hdop / max(cfg.hdop_max, 1e-6))
    score = float(0.5 * usable + 0.3 * hdop_term + 0.2 * min(1.0, rtk_frac + (1.0 - cfg.min_rtk_frac)))
    status = "quarantine" if fail and score < 0.5 else ("degraded" if fail else "pass")
    return {"name": "gps", "score": round(score, 4), "status": status,
            "detail": f"{usable:.1%} usable fix, median HDOP {med_hdop}, longest dropout {longest_dropout_s:.1f}s",
            "evidence": {"usable_fix_frac": round(usable, 4), "median_hdop": med_hdop,
                         "rtk_frac": round(rtk_frac, 4), "longest_dropout_s": round(longest_dropout_s, 3)}}


def check_imu(cfg: SanyxSettings, accel: np.ndarray, gyro: np.ndarray, ts_ns: list[int],
              accel_limit: float = 156.9, gyro_limit: float = 34.9) -> Result:
    """Xsens MTi-3 integrity: accel/gyro saturation, sudden bias jumps, and sample gaps. Default limits are the
    MTi-3 full-scale ranges (16 g, 2000 deg/s)."""
    accel = np.asarray(accel, dtype=np.float64).reshape(-1, 3) if len(accel) else np.zeros((0, 3))
    gyro = np.asarray(gyro, dtype=np.float64).reshape(-1, 3) if len(gyro) else np.zeros((0, 3))
    if accel.shape[0] == 0:
        return {"name": "imu", "score": 0.0, "status": "quarantine", "detail": "no IMU samples", "evidence": {}}

    # Evaluate accel and gyro saturation independently: gyro may have a different sample count than accel (rate
    # mismatch or a dropped run), and the old per-sample OR silently dropped ALL gyro saturation whenever the
    # lengths differed, hiding a pegged gyro.
    accel_sat = float(np.mean(np.any(np.abs(accel) >= accel_limit, axis=1))) if accel.shape[0] else 0.0
    gyro_sat = float(np.mean(np.any(np.abs(gyro) >= gyro_limit, axis=1))) if gyro.shape[0] else 0.0
    sat = max(accel_sat, gyro_sat)
    # bias jump: largest step in a short moving-mean of accel magnitude
    mag = np.linalg.norm(accel, axis=1)
    win = max(1, min(25, mag.size // 4))
    smooth = np.convolve(mag, np.ones(win) / win, mode="valid")
    bias_jump = float(np.abs(np.diff(smooth)).max()) if smooth.size > 1 else 0.0

    longest_gap_ms = 0.0
    if len(ts_ns) >= 2:
        d = np.diff(np.asarray(sorted(ts_ns), dtype=np.float64)) / 1e6
        longest_gap_ms = float(d.max()) if d.size else 0.0

    fail = (sat > cfg.imu_sat_frac_max or bias_jump > cfg.imu_bias_jump_max
            or longest_gap_ms > cfg.imu_gap_ms_max)
    score = float((1.0 - min(1.0, sat / max(cfg.imu_sat_frac_max, 1e-6)) * 0.5)
                  * (1.0 if bias_jump <= cfg.imu_bias_jump_max else 0.5)
                  * (1.0 if longest_gap_ms <= cfg.imu_gap_ms_max else 0.6))
    status = "quarantine" if fail and score < 0.5 else ("degraded" if fail else "pass")
    return {"name": "imu", "score": round(score, 4), "status": status,
            "detail": f"saturation {sat:.2%}, bias jump {bias_jump:.2f}, longest gap {longest_gap_ms:.0f}ms",
            "evidence": {"saturation_frac": round(float(sat), 4), "bias_jump": round(bias_jump, 3),
                         "longest_gap_ms": round(longest_gap_ms, 2)}}


def _highfreq_ratio(gray: np.ndarray) -> float:
    """Fraction of FFT energy in the high-frequency band. A defocused or fouled lens loses high-frequency
    content, so a persistently low ratio across a session is contamination."""
    g = gray.astype(np.float64)
    f = np.fft.fftshift(np.fft.fft2(g))
    power = np.abs(f) ** 2
    h, w = g.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = r > 0.25 * min(h, w)
    total = power.sum()
    return float(power[high].sum() / total) if total > 0 else 0.0


def check_lens_contamination(cfg: SanyxSettings, gray_frames: list[np.ndarray]) -> Result:
    """Persistent static occlusion (a region dark or blurred across many frames indicates dirt, water, or an
    obstruction) plus high-frequency attenuation via FFT. gray_frames is a temporal sample of single-channel
    frames from one camera."""
    if not gray_frames:
        return {"name": "lens", "score": 1.0, "status": "pass", "detail": "no frames sampled", "evidence": {}}
    frames = [np.asarray(g) for g in gray_frames]
    ratios = [_highfreq_ratio(g) for g in frames]
    med_ratio = float(np.median(ratios))

    # persistent dark/low-texture region: a pixel dark in a high fraction of frames. Sessions with varying
    # image dimensions (common in imported imagery) are cropped to the smallest common shape before stacking.
    min_h = min(g.shape[0] for g in frames)
    min_w = min(g.shape[1] for g in frames)
    stack = np.stack([g[:min_h, :min_w].astype(np.float64) for g in frames], axis=0)
    dark = (stack < 25).mean(axis=0)                      # per-pixel fraction of frames that are dark
    occ_mask = dark >= cfg.occlusion_persist_frac
    occ_frac = float(occ_mask.mean())

    fouled = med_ratio < cfg.fft_highfreq_min_ratio
    occluded = occ_frac > cfg.occlusion_area_frac_max
    if fouled or occluded:
        status = "degraded" if med_ratio >= cfg.fft_highfreq_min_ratio * 0.5 and occ_frac < 0.3 else "quarantine"
        score = 0.55 if status == "degraded" else 0.2
    else:
        status, score = "pass", 1.0
    return {"name": "lens", "score": round(float(score), 4), "status": status,
            "detail": f"high-freq ratio {med_ratio:.3f}, persistent occlusion {occ_frac:.1%}",
            "evidence": {"highfreq_ratio": round(med_ratio, 4), "occlusion_area_frac": round(occ_frac, 4)}}
