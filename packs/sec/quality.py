"""Sec quality profile: SANYX ingest QA for a fixed CCTV camera.

The image checks (dropped frames, exposure, lens contamination) are the reused domain-neutral SANYX checks and
carry over unchanged. What differs from AV: there are no inertial checks (a fixed camera has no IMU/GNSS/CAN),
so sensor_checks is empty, and the fault vocabulary is CCTV faults (stream dropout, IR-cut oscillation,
tampering, defocus) rather than the AV rig faults (loose GMSL2, urban-canyon multipath, IMU thermal drift).
"""

from __future__ import annotations

from packs.base import QualityProfile, RootCauseSignature

# Standard fixed-camera fault signatures and their operator remedies.
CCTV_REMEDIATION = {
    "stream_dropout": "check the NVR link and camera PoE; the RTSP stream stalled or dropped frames",
    "ir_cut_oscillation": "lock the day/night mode; the IR-cut filter is flipping and strobing exposure",
    "camera_tampering": "inspect the camera; the view is occluded, defocused, or moved off its scene",
    "defocus": "refocus the lens; the scene is persistently blurred",
    "sensor_banding": "reseat the ribbon or replace the camera; a sensor/cable fault is banding the image",
    "timestamp_drift": "sync the camera clock over NTP; frame timestamps are drifting from wall time",
}


def build_quality_profile() -> QualityProfile:
    return QualityProfile(
        image_checks=("dropped_frames", "exposure", "lens_contamination"),
        sensor_checks=(),  # a fixed camera has no inertial channels
        rootcause_signatures=tuple(
            RootCauseSignature(name=k, remedy=v) for k, v in sorted(CCTV_REMEDIATION.items())
        ),
    )
