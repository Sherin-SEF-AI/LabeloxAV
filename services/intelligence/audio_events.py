"""Milestone B (audio layer): segment audio events from the RMS envelope. A high-energy region is a
candidate audio event (an impact, a horn blast, a screech) with a start and end on the timeline. The
acoustic SOURCE classification (emergency-vehicle siren vs two-wheeler horn vs heavy-vehicle air horn) needs
a trained audio model, so it is a runtime seam, not faked here: the segmenter marks the region as a generic
audio_transient candidate and the classifier fills in the kind when wired.
"""

from __future__ import annotations

import statistics

from core.logging import get_logger

log = get_logger("audio_events")


def segment_audio_events(envelope: list[float], t_start_ns: int, t_step_ns: int, z: float = 2.5) -> list[dict]:
    """Contiguous high-energy regions of the RMS envelope as candidate audio events. The threshold is robust
    (median + z * 1.4826 * MAD) so steady road noise does not trip it. Each region carries its peak energy;
    the acoustic kind is left to the classifier seam."""
    if len(envelope) < 8:
        return []
    med = statistics.median(envelope)
    mad = statistics.median([abs(v - med) for v in envelope]) or 1e-6
    thr = med + z * 1.4826 * mad
    events: list[dict] = []
    run: dict | None = None
    for i, v in enumerate(envelope):
        if v >= thr:
            if run is None:
                run = {"start": i, "end": i, "peak": v}
            run["end"], run["peak"] = i, max(run["peak"], v)
        elif run is not None:
            events.append({"kind": "audio_transient", "t_start_ns": t_start_ns + run["start"] * t_step_ns,
                           "t_end_ns": t_start_ns + run["end"] * t_step_ns, "peak": round(run["peak"], 4)})
            run = None
    if run is not None:
        events.append({"kind": "audio_transient", "t_start_ns": t_start_ns + run["start"] * t_step_ns,
                       "t_end_ns": t_start_ns + run["end"] * t_step_ns, "peak": round(run["peak"], 4)})
    return events


def classify_audio_source(audio_region) -> str:
    """The acoustic source of an audio region (siren type, horn type, screech, impact). Needs a trained audio
    classifier; until wired this returns the unclassified marker rather than a fabricated label."""
    # WIRE: services/autolabel adapter for an audio source classifier (e.g. a panns/yamnet head). No model
    # runtime is invented in this logic; the segmenter still produces honest candidate regions without it.
    return "unclassified"
