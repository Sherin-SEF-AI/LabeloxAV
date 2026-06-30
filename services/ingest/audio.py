"""M-IMU.2 (audio): extract the dashcam audio track and a downsampled energy envelope for the inertial
timeline. The dashcam embeds a 16kHz AAC audio track (PyAV decodes it); the per-bucket RMS envelope
time-locks audio energy to the video and the ego-state, so an impact felt (a jerk spike) can be
cross-referenced with one heard. The envelope is what the timeline renders; the raw audio can be stored
alongside.

The freeGPS embedded GPS + G-sensor parser is a deliberate seam, NOT built here: the current footage's
freeGPS telemetry blocks are present but empty (no GPS/G-sensor lock), so there is no populated sample to
define the firmware-specific byte layout. When GPS-locked footage arrives, the parser slots in beside this
extractor and writes the same measured-IMU store the resolver prefers over the derived ego-state.
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger

log = get_logger("ingest_audio")


def rms_envelope(signal, buckets: int = 600) -> list[float]:
    """Per-bucket RMS of a mono signal, normalized to roughly [0, 1]. int16-range input is scaled down."""
    sig = np.asarray(signal, dtype=np.float32)
    n = len(sig)
    if n == 0:
        return []
    if np.abs(sig).max() > 1.5:
        sig = sig / 32768.0
    step = max(1, n // max(1, buckets))
    return [round(float(np.sqrt(np.mean(sig[i:i + step] ** 2))), 4) for i in range(0, n, step)]


def audio_envelope(mp4_path, buckets: int = 600) -> dict:
    """Decode the first audio track of an MP4 and return its RMS envelope + metadata, or {found: False}."""
    import av
    container = av.open(str(mp4_path))
    try:
        astreams = [s for s in container.streams if s.type == "audio"]
        if not astreams:
            return {"found": False}
        a = astreams[0]
        chunks = []
        for frame in container.decode(a):
            arr = frame.to_ndarray()
            chunks.append((arr.mean(axis=0) if arr.ndim > 1 else arr).astype(np.float32))
    finally:
        container.close()
    if not chunks:
        return {"found": False}
    sig = np.concatenate(chunks)
    env = rms_envelope(sig, buckets)
    return {"found": True, "sample_rate": int(a.sample_rate), "n_samples": int(len(sig)),
            "duration_s": round(len(sig) / float(a.sample_rate), 2), "buckets": len(env), "envelope": env}
