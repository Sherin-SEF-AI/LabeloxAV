"""Event detectors: rule-first, geometry-second (Plane 3). Turn tracks + trajectories + ego-motion
into behaviourally-defined scenarios. India-tuned via the ontology superclasses (autorickshaw,
cattle, water tanker fall out of the class set, not special-cased here).

Each detector is a pure function over the per-track trajectories and the ego series, returning
ScenarioRecord candidates with a criticality score. Thresholds are config-driven.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.config import EventSettings, get_settings
from services.autolabel.ontology import Ontology
from services.intelligence.tracking import TrackResult
from services.intelligence.trajectory import FrameCtx, Trajectory

NS_PER_S = 1_000_000_000

VEHICLE_L1 = {"two_wheeler", "three_wheeler", "four_wheeler", "heavy"}


@dataclass
class ScenarioRecord:
    type: str
    t_in_ns: int
    t_out_ns: int
    actors: list[str] = field(default_factory=list)
    criticality: float = 0.0
    tags: list[str] = field(default_factory=list)
    lat: float | None = None
    lon: float | None = None
    meta: dict = field(default_factory=dict)


def _clamp(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _is_vehicle(class_id: int, onto: Ontology) -> bool:
    return onto.by_id(class_id).l1 in VEHICLE_L1


def _is_animal(class_id: int, onto: Ontology) -> bool:
    return onto.by_id(class_id).l1 == "animal"


def _is_vru(class_id: int, onto: Ontology) -> bool:
    return onto.by_id(class_id).l1 == "vru"


def _geo(track: TrackResult, frame_ctx: dict) -> tuple[float | None, float | None]:
    ctx: FrameCtx = frame_ctx[track.members[0].frame_id]
    return ctx.lat, ctx.lon


def detect_hard_brake(ego_series: list[tuple[int, float]], cfg: EventSettings) -> list[ScenarioRecord]:
    out: list[ScenarioRecord] = []
    for (t0, s0), (t1, s1) in zip(ego_series, ego_series[1:], strict=False):
        dt = (t1 - t0) / NS_PER_S
        if dt <= 0 or s0 is None or s1 is None:
            continue
        accel = (s1 - s0) / dt
        if accel <= cfg.hard_brake_decel:
            out.append(
                ScenarioRecord(
                    type="hard_brake", t_in_ns=t0, t_out_ns=t1,
                    criticality=_clamp(abs(accel) / 8.0), tags=["ego_event"],
                    meta={"decel_mps2": round(accel, 2), "v_from": s0, "v_to": s1},
                )
            )
    return out


def detect_per_track(
    tracks: list[TrackResult], trajs: dict[str, Trajectory], frame_ctx: dict, onto: Ontology, cfg: EventSettings
) -> list[ScenarioRecord]:
    out: list[ScenarioRecord] = []
    # Majority lateral flow among vehicles, for wrong-side detection.
    drifts = [trajs[str(t.track_id)].summary.get("x_drift_frac", 0.0) for t in tracks if _is_vehicle(t.class_id, onto)]
    flow_sign = 0.0
    sig = [d for d in drifts if abs(d) > 0.02]
    if sig:
        flow_sign = 1.0 if sum(1 for d in sig if d > 0) >= len(sig) / 2 else -1.0

    for t in tracks:
        tj = trajs[str(t.track_id)]
        s = tj.summary
        if not s:
            continue
        cls = onto.by_id(t.class_id)
        lat, lon = _geo(t, frame_ctx)
        ctx0: FrameCtx = frame_ctx[t.members[0].frame_id]
        w, h = max(1, ctx0.width), max(1, ctx0.height)
        last_cx = tj.points[-1]["cx"] / w
        last_by = tj.points[-1]["by"] / h
        center_dist = abs(last_cx - 0.5)
        base = dict(actors=[str(t.track_id)], lat=lat, lon=lon)

        # animal-on-road: animal in the lower carriageway region
        if _is_animal(t.class_id, onto) and last_by > 0.45:
            out.append(ScenarioRecord(
                type="animal_on_road", t_in_ns=t.first_ts_ns, t_out_ns=t.last_ts_ns,
                criticality=_clamp(0.5 + (last_by - 0.45)), tags=["animal", cls.name],
                meta={"class": cls.name, "in_path": center_dist < cfg.cut_in_center_frac}, **base,
            ))

        # cut-in: closing while moving into the ego column
        if _is_vehicle(t.class_id, onto) and s["area_growth"] >= cfg.cut_in_area_growth and center_dist < cfg.cut_in_center_frac:
            out.append(ScenarioRecord(
                type="cut_in", t_in_ns=t.first_ts_ns, t_out_ns=t.last_ts_ns,
                criticality=_clamp(0.4 + (s["area_growth"] - 1.0)), tags=["cut_in", cls.name],
                meta={"class": cls.name, "area_growth": s["area_growth"]}, **base,
            ))

        # near-miss: approaching in ego path with short estimated TTC
        if (_is_vehicle(t.class_id, onto) or _is_vru(t.class_id, onto)) and s["approaching"] and center_dist < cfg.cut_in_center_frac:
            growth = s["area_growth"]
            dur_s = max(1e-3, s["duration_ns"] / NS_PER_S)
            ttc = dur_s / max(1e-3, growth - 1.0)  # frames-to-collision proxy
            if ttc < cfg.near_miss_ttc_s:
                out.append(ScenarioRecord(
                    type="near_miss", t_in_ns=t.first_ts_ns, t_out_ns=t.last_ts_ns,
                    criticality=_clamp(1.0 - ttc / cfg.near_miss_ttc_s), tags=["near_miss", cls.name],
                    meta={"class": cls.name, "ttc_s": round(ttc, 2)}, **base,
                ))

        # illegal-park: static vehicle hugging the shoulder
        static = s["net_disp_px"] / w < cfg.static_disp_frac and s["n"] >= cfg.static_min_frames
        on_shoulder = last_cx < cfg.shoulder_margin_frac or last_cx > 1 - cfg.shoulder_margin_frac
        if _is_vehicle(t.class_id, onto) and static and on_shoulder:
            out.append(ScenarioRecord(
                type="illegal_park", t_in_ns=t.first_ts_ns, t_out_ns=t.last_ts_ns,
                criticality=0.4, tags=["illegal_park", cls.name],
                meta={"class": cls.name}, **base,
            ))

        # wrong-side: vehicle lateral drift opposing the majority flow
        d = s["x_drift_frac"]
        if _is_vehicle(t.class_id, onto) and flow_sign != 0 and abs(d) > 0.05 and (1.0 if d > 0 else -1.0) == -flow_sign and s["n"] >= cfg.wrong_side_frames:
            out.append(ScenarioRecord(
                type="wrong_side", t_in_ns=t.first_ts_ns, t_out_ns=t.last_ts_ns,
                criticality=_clamp(0.5 + abs(d)), tags=["wrong_side", cls.name],
                meta={"class": cls.name, "drift": d}, **base,
            ))
    return out


def detect_congestion(
    tracks: list[TrackResult], trajs: dict[str, Trajectory], frame_ctx: dict, cfg: EventSettings
) -> list[ScenarioRecord]:
    # Count active tracks per frame ts and their mean speed; flag dense + slow windows.
    by_ts: dict[int, list[str]] = {}
    for t in tracks:
        for d in t.members:
            by_ts.setdefault(d.ts_ns, []).append(str(t.track_id))
    if not by_ts:
        return []
    ts_sorted = sorted(by_ts)
    any_ctx: FrameCtx = next(iter(frame_ctx.values()))
    w = max(1, any_ctx.width)
    dense = [ts for ts in ts_sorted if len(set(by_ts[ts])) >= cfg.congestion_min_objects]
    if not dense:
        return []
    mean_speed_frac = (
        sum(trajs[str(t.track_id)].summary.get("mean_speed_px", 0.0) for t in tracks) / max(1, len(tracks)) / w
    )
    if mean_speed_frac > cfg.congestion_max_speed_frac:
        return []
    return [ScenarioRecord(
        type="congestion", t_in_ns=dense[0], t_out_ns=dense[-1],
        criticality=0.5, tags=["congestion"],
        meta={"peak_objects": max(len(set(by_ts[ts])) for ts in dense), "mean_speed_frac": round(mean_speed_frac, 4)},
    )]


def detect_events(
    tracks: list[TrackResult],
    trajs: dict[str, Trajectory],
    frame_ctx: dict,
    ego_series: list[tuple[int, float]],
    onto: Ontology,
) -> list[ScenarioRecord]:
    cfg = get_settings().intelligence.events
    out: list[ScenarioRecord] = []
    out += detect_hard_brake(ego_series, cfg)
    out += detect_per_track(tracks, trajs, frame_ctx, onto, cfg)
    out += detect_congestion(tracks, trajs, frame_ctx, cfg)
    return out
