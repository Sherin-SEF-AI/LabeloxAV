"""Auto-label the whole real corpus with every pass the product has, then seal a dataset from it.

    .venv/bin/python scripts/autolabel_corpus.py --weights <local .pt | registry model_version> [--dry-run]

Stage by stage, each over every real session before the next starts, because the later passes read what the
earlier ones wrote: lanes are checked against the drivable mask detection writes, tracks link the boxes,
gap filling needs the tracks, and the dataset is cut last.

    detect      boxes over the full class set, masks, drivable surface, the PII re-check (re-detect: a frame's
                previous machine output is replaced; human work and decided rows are kept, persist.py)
    lanes       lane proposals per frame, typed from the paint, then linked into lanes across frames
    tracks      tracks, scenarios and tube coherence per session (intelligence/run.mine_session)
    gapfill     the frames a tracked object blinks out of, corpus wide
    attributes  a vision-language pass that fills typed attributes and flags likely wrong classes, sampled
                per session, because it costs about two seconds an object
    3d          ego pose, cuboids per frame against the point cloud, tracks lifted to 3D, occupancy; only on
                sessions that have point clouds
    embeddings  object embeddings for search and explore, for objects that do not have one yet
    export      reviewed and auto-accepted labels, every format, 80/10/10 split by session, sealed

Synthetic composites are left alone: they were generated with exact labels, and re-detecting them would
replace those with guesses.

It shares a 30 GB desktop with everything else on it, so it is built to stop rather than to crash it: it
waits on the resource watcher's pause flag (scripts/resource_watch.py) before every session, stops cleanly if
the flag stays up, and records each finished (stage, session) so a rerun resumes exactly where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from resource_watch import wait_while_paused  # noqa: E402

STAGES = ["detect", "lanes", "tracks", "gapfill", "attributes", "3d", "embeddings", "export"]
PER_SESSION = {"detect", "lanes", "tracks", "attributes", "3d", "embeddings"}


QUIET_EXIT = 3        # the child's exit code for "stopped at the start of quiet hours; resume after"
QUIET_MARGIN_S = 300  # a unit must be expected to finish this long before quiet hours begin


def _quiet_window(spec: str) -> tuple[int, int] | None:
    """"10:00-18:00" as minutes after midnight, local time; empty means no quiet hours."""
    if not spec:
        return None
    a, b = spec.split("-")
    ha, ma = (int(x) for x in a.split(":"))
    hb, mb = (int(x) for x in b.split(":"))
    return ha * 60 + ma, hb * 60 + mb


def _seconds_until_quiet(window: tuple[int, int] | None, now: float | None = None) -> float:
    """Seconds from now to the next start of quiet hours (0 inside them); infinity with none set."""
    if window is None:
        return float("inf")
    t = time.localtime(now if now is not None else time.time())
    mins = t.tm_hour * 60 + t.tm_min + t.tm_sec / 60.0
    start, end = window
    if start <= mins < end:
        return 0.0
    return ((start - mins) % (24 * 60)) * 60.0


def _seconds_until_allowed(window: tuple[int, int] | None, now: float | None = None) -> float:
    """Seconds until quiet hours end (0 outside them)."""
    if window is None:
        return 0.0
    t = time.localtime(now if now is not None else time.time())
    mins = t.tm_hour * 60 + t.tm_min + t.tm_sec / 60.0
    start, end = window
    if start <= mins < end:
        return (end - mins) * 60.0
    return 0.0


def _done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("ok"):
            out.add((r["stage"], r.get("session") or "*"))
    return out


def _expected_secs(path: Path, stage: str, default: float) -> float:
    """The longest this stage has taken per unit so far, so a unit is not started that cannot finish in time."""
    worst = 0.0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("stage") == stage and r.get("secs"):
                worst = max(worst, float(r["secs"]))
    return worst or default


def _unload_ollama() -> None:
    """Free the vision-language model's memory now rather than in five minutes, so quiet hours start clean."""
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=5) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        for m in models:
            req = urllib.request.Request("http://localhost:11434/api/generate", method="POST",
                                         data=json.dumps({"model": m, "keep_alive": 0}).encode())
            urllib.request.urlopen(req, timeout=30).read()
    except Exception:  # noqa: BLE001 - Ollama not running is fine, there is nothing to unload
        pass


def _record(path: Path, **row) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


async def real_sessions() -> list[tuple[str, int, bool]]:
    """(session_id, frames, has_point_clouds) for every real session with frames, largest first."""
    from sqlalchemy import text

    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        rows = (await db.execute(text("""
            select s.session_id::text, count(f.frame_id),
                   exists (select 1 from point_cloud p where p.session_id = s.session_id)
            from session s join frame f on f.session_id = s.session_id
            where s.origin = 'real'
            group by s.session_id order by count(f.frame_id) desc"""))).all()
    return [(r[0], int(r[1]), bool(r[2])) for r in rows]


async def resolve_weights(arg: str) -> str:
    """A local .pt as-is; a registry model_version fetched through the verified champion cache."""
    p = Path(arg)
    if p.exists():
        return str(p)
    from db.models import ModelRegistry
    from db.session import get_sessionmaker
    from services.autolabel.runner import _local_champion_weights

    async with get_sessionmaker()() as db:
        reg = await db.get(ModelRegistry, arg)
    if reg is None or not reg.weights_uri:
        raise SystemExit(f"--weights {arg!r} is neither a file nor a registered model with weights")
    local = await asyncio.to_thread(_local_champion_weights, reg.weights_uri)
    if not local:
        raise SystemExit(f"could not fetch weights for {arg} from {reg.weights_uri}")
    return local


# ------------------------------------------------------------------ stages

async def detect(sid: str, weights: str) -> dict:
    from services.autolabel.runner import autolabel_session

    r = await autolabel_session(uuid.UUID(sid), None, detector_weights=weights)
    return {k: r.get(k) for k in ("frames", "objects", "by_state", "vlm_calls", "drivable") if k in r}


async def lanes(sid: str) -> dict:
    import cv2
    import numpy as np
    from sqlalchemy import select

    from core.storage import get_object_store
    from db.models import Frame
    from db.session import get_sessionmaker
    from services.autolabel.lane.propose import propose_frame_lanes
    from services.intelligence.lane_linking import link_lanes_for_session

    store = get_object_store()
    proposed = frames = unreadable = 0
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(Frame).where(Frame.session_id == uuid.UUID(sid))
                                 .order_by(Frame.cam_id, Frame.ts_ns))).scalars().all()
        for i, frame in enumerate(rows, 1):
            try:
                img = cv2.imdecode(np.frombuffer(store.get_bytes(frame.img_uri), np.uint8), cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001 - one unreadable image is a frame without lanes, not a failed run
                img = None
            if img is None:
                unreadable += 1
                continue
            created, _rej = await propose_frame_lanes(db, frame, img)
            proposed += len(created)
            frames += 1
            if i % 25 == 0:
                await db.commit()
        await db.commit()
        linked = await link_lanes_for_session(db, uuid.UUID(sid), apply=True)
        await db.commit()
    return {"frames": frames, "lanes": proposed, "unreadable": unreadable,
            "linked": {k: v for k, v in (linked or {}).items() if isinstance(v, int | float | str)}}


async def tracks(sid: str) -> dict:
    from services.intelligence.run import mine_session

    r = await mine_session(uuid.UUID(sid))
    return {k: v for k, v in (r or {}).items() if isinstance(v, int | float | str)}


async def gapfill() -> dict:
    from db.session import get_sessionmaker
    from services.quality.track_gap_backfill import run_gap_fill, start_gap_fill

    async with get_sessionmaker()() as db:
        run = await start_gap_fill(db, created_by="autolabel_corpus")
    await run_gap_fill(uuid.UUID(run["run_id"]))
    return run


async def attributes(sid: str, limit: int) -> dict:
    from services.intelligence.vlm_qa import vlm_qa_session

    r = await vlm_qa_session(uuid.UUID(sid), limit=limit)
    return {k: v for k, v in (r or {}).items() if isinstance(v, int | float | str)}


async def three_d(sid: str) -> dict:
    sys.path.insert(0, str(ROOT / "scripts" / "demo_tour"))
    from postprocess import lift_frames  # the batched per-frame lift, with a yield between batches

    from services.intelligence.ego_pose import build_ego_pose
    from services.lidar.occupancy4d import build_occupancy_window
    from services.lidar.track3d.from2d import lift_tracks_for_session

    s = uuid.UUID(sid)
    out = {"ego_pose": (await build_ego_pose(s)).get("poses")}
    out["per_frame"] = await lift_frames(s)
    out["tracks"] = await lift_tracks_for_session(s, limit=50)
    out["occupancy"] = await build_occupancy_window(s)
    return json.loads(json.dumps(out, default=str))


async def embeddings(sid: str) -> dict:
    from services.intelligence.embed.service import embed_objects

    r = await embed_objects(session_id=uuid.UUID(sid), only_missing=True)
    return {k: v for k, v in (r or {}).items() if isinstance(v, int | float | str)}


async def export(name: str) -> dict:
    from services.export.dataset import SliceSpec, export_dataset

    spec = SliceSpec(name=name, states=["accepted", "auto_accept"],
                     formats=["coco", "yolo", "openlabel", "nuscenes", "parquet"],
                     val_frac=0.1, test_frac=0.1, split_group_by="session")
    r = await export_dataset(spec)
    return {k: r.get(k) for k in ("commit_id", "object_count", "formats", "splits") if k in r}


# ------------------------------------------------------------------ driver

async def main_async(a) -> int:
    from core.logging import setup_logging

    setup_logging("warning")
    flag = Path(a.flag)
    window = _quiet_window(a.quiet_hours)
    ckpt = Path(a.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done = _done(ckpt)
    sessions = await real_sessions()
    if a.sessions:
        keep = set(a.sessions.split(","))
        sessions = [s for s in sessions if s[0] in keep or s[0][:8] in keep]
    stages = [s for s in STAGES if s in a.stages.split(",")]
    weights = await resolve_weights(a.weights) if ("detect" in stages and a.weights) else None
    print(f"{len(sessions)} real sessions, {sum(s[1] for s in sessions):,} frames; stages {', '.join(stages)}; "
          f"weights {weights}; {len(done)} units already done", flush=True)
    if a.dry_run:
        for st in stages:
            todo = [s for s in sessions if (st, s[0]) not in done and (st != "3d" or s[2])] \
                if st in PER_SESSION else ([] if (st, "*") in done else ["*"])
            print(f"  {st:11s} {len(todo)} to do")
        return 0

    for st in stages:
        units = [s for s in sessions if st != "3d" or s[2]] if st in PER_SESSION else [None]
        n = len(units)
        for i, unit in enumerate(units, 1):
            key = unit[0] if unit else "*"
            if (st, key) in done:
                continue
            # Quiet hours: the machine is someone's during the working day. A unit that could not finish
            # before they begin is not started; the parent waits them out and relaunches this.
            need = _expected_secs(ckpt, st, 900.0) * 1.3 + QUIET_MARGIN_S
            if _seconds_until_quiet(window) < need:
                print(f"{st} {key}: quiet hours {a.quiet_hours} begin before this could finish; stopping "
                      f"cleanly, resuming after them", flush=True)
                _unload_ollama()
                return QUIET_EXIT
            # The one place this waits on the machine: between sessions, where stopping is safe.
            if not wait_while_paused(flag, timeout_s=a.pause_timeout):
                print(f"{st} {key}: resources stayed over their limits for {a.pause_timeout:.0f}s, stopping "
                      f"cleanly; rerun to resume. {flag.read_text() if flag.exists() else ''}", flush=True)
                return 2
            t0 = time.time()
            try:
                if st == "detect":
                    res = await detect(key, weights)
                elif st == "lanes":
                    res = await lanes(key)
                elif st == "tracks":
                    res = await tracks(key)
                elif st == "gapfill":
                    res = await gapfill()
                elif st == "attributes":
                    res = await attributes(key, a.attr_limit)
                elif st == "3d":
                    res = await three_d(key)
                elif st == "embeddings":
                    res = await embeddings(key)
                else:
                    res = await export(a.dataset_name)
                ok, err = True, None
            except Exception as exc:  # noqa: BLE001 - one session's failure is recorded, not fatal
                res, ok, err = None, False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
            secs = round(time.time() - t0, 1)
            _record(ckpt, stage=st, session=key, ok=ok, secs=secs, result=res, error=err, at=time.time())
            print(f"[{st} {i}/{n}] {key[:8]} {'ok' if ok else 'FAIL'} {secs}s {res if ok else err}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="", help="local .pt or a registered model_version (needed for detect)")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--sessions", default="", help="comma separated session ids or 8-char prefixes")
    ap.add_argument("--attr-limit", type=int, default=24, help="objects per session for the attribute pass")
    ap.add_argument("--dataset-name", default=f"corpus-autolabel-{time.strftime('%Y-%m-%d')}")
    ap.add_argument("--checkpoint", default=".scratch/run/autolabel_corpus.jsonl")
    ap.add_argument("--flag", default=".scratch/watch/PAUSE.json")
    ap.add_argument("--pause-timeout", type=float, default=1800.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet-hours", default="", help='local "HH:MM-HH:MM" when nothing may run, e.g. 10:00-18:00')
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if "detect" in a.stages.split(",") and not a.weights and not a.dry_run:
        ap.error("--weights is required when the detect stage runs")
    if a.child or a.dry_run or not a.quiet_hours:
        rc = asyncio.run(main_async(a))
        if rc == 0:
            _unload_ollama()
        return rc
    # The parent: imports nothing heavy and holds no GPU, so waiting out the working day costs nothing. Each
    # run is a fresh child, which also hands every byte of GPU and host memory back when it stops.
    import subprocess

    window = _quiet_window(a.quiet_hours)
    while True:
        wait = _seconds_until_allowed(window)
        if wait > 0:
            print(f"quiet hours {a.quiet_hours}: waiting {wait / 3600:.1f} h until they end", flush=True)
            time.sleep(wait + 30)
            continue
        rc = subprocess.run([sys.executable, __file__, *sys.argv[1:], "--child"]).returncode
        if rc != QUIET_EXIT:
            return rc


if __name__ == "__main__":
    raise SystemExit(main())
