"""Traffic signal phases and the transitions that cannot have happened.

10,076 objects in the corpus carry a `signal_state` attribute and nothing has ever read them as a sequence.
Per frame, "this light is red" is a property of a box. Across a track it is a phase, and phases are what a
planner is actually trained against: how long the amber was, whether the ego entered on green, when the
crossing signal changed relative to the vehicle one.

Reading them as a sequence also audits them for free. A signal follows a phase graph, and a transition the
graph does not permit is almost never a broken signal. It is a mislabelled frame, and it is invisible frame
by frame because each individual label looks perfectly reasonable on its own crop. This is the same argument
the reasoning layer makes about physics and context, applied to time: a claim that is fine in isolation and
impossible in sequence is still wrong, and the sequence is the only place to catch it.

`segment_phases` and `find_invalid_transitions` are pure over a list of (ts, state), so both rules can be
tested against constructed sequences.
"""

from __future__ import annotations

from core.logging import get_logger
from services.intelligence.event_taxonomy import signal_min_phase_ns, signal_phase_graph

log = get_logger("signal_events")

SIGNAL_CLASSES = ("traffic_signal",)
ATTRIBUTE = "signal_state"


def segment_phases(samples: list[tuple[int, str | None]]) -> list[dict]:
    """Contiguous runs of one signal state.

    A sample with no state breaks the run rather than extending it. An unlabelled frame between two reds is
    not evidence the light stayed red, and bridging it would manufacture a phase duration nobody observed,
    which is the number this whole function exists to produce.
    """
    out: list[dict] = []
    cur: dict | None = None
    # Set when the previous run ended at an unlabelled frame, so the next phase knows it did not follow the
    # one before it in the list. Without this the two phases either side of a gap look consecutive, and the
    # transition check would flag a transition that was never observed.
    after_gap = False
    for ts, state in sorted(samples, key=lambda s: s[0]):
        if state is None:
            if cur is not None:
                out.append(cur)
                cur = None
            after_gap = True
            continue
        if cur is None or cur["state"] != state:
            if cur is not None:
                out.append(cur)
            cur = {"state": str(state), "t_start_ns": int(ts), "t_end_ns": int(ts), "frames": 1,
                   "after_gap": after_gap}
            after_gap = False
        else:
            cur["t_end_ns"] = int(ts)
            cur["frames"] += 1
    if cur is not None:
        out.append(cur)
    for p in out:
        p["duration_ns"] = p["t_end_ns"] - p["t_start_ns"]
    return out


def find_invalid_transitions(phases: list[dict], graph: dict[str, list[str]] | None = None) -> list[dict]:
    """Transitions between consecutive phases that the phase graph does not permit.

    Only consecutive phases are compared. A gap in annotation already split the run, and two phases either
    side of a gap could legitimately have any relationship because the transition between them was never
    observed.
    """
    g = graph if graph is not None else signal_phase_graph()
    out: list[dict] = []
    for a, b in zip(phases, phases[1:], strict=False):
        if b.get("after_gap"):
            continue
        legal = g.get(a["state"])
        if legal is None:
            # A state the graph says nothing about. Unknown is not the same as impossible, and reporting it
            # as a violation would flag every ontology addition as a labelling error.
            continue
        if b["state"] not in legal:
            out.append({"t_ns": b["t_start_ns"], "from_state": a["state"], "to_state": b["state"],
                        "legal_next": list(legal)})
    return out


def find_flicker(phases: list[dict], min_phase_ns: int | None = None) -> list[dict]:
    """Phases too short to be real that reverted to the state before them.

    The reversion is the load-bearing part. A genuinely short amber is short and then goes red; a mislabelled
    frame goes red, green for one frame, red again. Requiring the revert keeps a fast-cycling signal from
    being reported as noise.
    """
    limit = signal_min_phase_ns() if min_phase_ns is None else min_phase_ns
    out: list[dict] = []
    for prev, mid, nxt in zip(phases, phases[1:], phases[2:], strict=False):
        if mid.get("after_gap") or nxt.get("after_gap"):
            continue
        if mid["duration_ns"] < limit and prev["state"] == nxt["state"] != mid["state"]:
            out.append({"t_start_ns": mid["t_start_ns"], "t_end_ns": mid["t_end_ns"],
                        "state": mid["state"], "reverted_to": prev["state"],
                        "duration_ns": mid["duration_ns"]})
    return out


def derive_for_track(track_id: str, samples: list[tuple[int, str | None, str]]) -> list[dict]:
    """Every signal event for one signal's track. samples are (ts_ns, state, frame_id)."""
    by_ts = {ts: (state, frame_id) for ts, state, frame_id in samples}
    phases = segment_phases([(ts, state) for ts, state, _ in samples])
    if not phases:
        return []

    def frame_at(ts: int) -> str | None:
        got = by_ts.get(ts)
        return got[1] if got else None

    events: list[dict] = []
    for p in phases:
        events.append({
            "kind": "signal_phase", "track_id": track_id, "frame_id": frame_at(p["t_start_ns"]),
            "t_start_ns": p["t_start_ns"], "t_end_ns": p["t_end_ns"],
            # A one-frame phase is a weak claim about a phase even when it is not flicker, and confidence is
            # how the review queue sorts. Saturates at six frames, past which more frames add nothing.
            "conf": round(min(1.0, 0.4 + 0.1 * p["frames"]), 3),
            "payload": {"state": p["state"], "duration_ns": p["duration_ns"], "frames": p["frames"]},
        })

    for t in find_invalid_transitions(phases):
        events.append({
            "kind": "signal_transition_invalid", "track_id": track_id, "frame_id": frame_at(t["t_ns"]),
            "t_start_ns": t["t_ns"], "t_end_ns": None, "conf": 0.8,
            "payload": {"from_state": t["from_state"], "to_state": t["to_state"],
                        "legal_next": t["legal_next"]},
        })

    for f in find_flicker(phases):
        events.append({
            "kind": "signal_flicker", "track_id": track_id, "frame_id": frame_at(f["t_start_ns"]),
            "t_start_ns": f["t_start_ns"], "t_end_ns": f["t_end_ns"], "conf": 0.7,
            "payload": {"state": f["state"], "reverted_to": f["reverted_to"],
                        "duration_ns": f["duration_ns"]},
        })
    return events


async def detect_signal_events(db, session_id) -> list[dict]:
    """Read one session's signal tracks and derive their phases and anomalies."""
    import uuid as _uuid

    from sqlalchemy import select

    from db.models import Frame, Object, OntologyClass

    sid = session_id if isinstance(session_id, _uuid.UUID) else _uuid.UUID(str(session_id))

    signal_ids = (await db.execute(
        select(OntologyClass.id).where(OntologyClass.name.in_(SIGNAL_CLASSES)))).scalars().all()
    if not signal_ids:
        log.warning("signal_events.no_signal_class", classes=list(SIGNAL_CLASSES))
        return []

    rows = (await db.execute(
        select(Object.track_id, Object.attrs, Frame.ts_ns, Frame.frame_id)
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Frame.session_id == sid, Object.track_id.isnot(None),
               Object.class_id.in_(signal_ids))
        .order_by(Frame.ts_ns))).all()

    by_track: dict[str, list[tuple[int, str | None, str]]] = {}
    for track_id, attrs, ts_ns, frame_id in rows:
        state = (attrs or {}).get(ATTRIBUTE)
        by_track.setdefault(str(track_id), []).append(
            (int(ts_ns or 0), None if state is None else str(state), str(frame_id)))

    events: list[dict] = []
    for track_id, samples in by_track.items():
        events.extend(derive_for_track(track_id, samples))

    log.info("signal_events.derived", session=str(session_id), tracks=len(by_track),
             events=len(events))
    return events
