"""Staged GPU runner. A single resource manager owns model lifecycle and enforces the 16 GB
ceiling (Constraint 4).

Stage 1: YOLO26 (FP16) + SAM 3.1 (FP16) co-resident (~8 GB), detect + segment + (M3) fuse + gate.
Stage 2: Qwen3-VL-4B (Q4, ~3.3 GB) over the uncertain subset (M4).

Run modes: sequential (Stage 1 fully unloads before Stage 2 loads; safest on 16 GB) and
concurrent (both resident, ~11 GB; only if measured free VRAM allows). Every model load is
guarded by a free-VRAM check and fails loudly instead of OOM-crashing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import UUID

import click
import cv2
import numpy as np
from sqlalchemy import select

from core.bus import EventBus
from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.schemas import FrameMeta
from core.storage import get_object_store
from db.models import Frame, Object
from db.session import get_sessionmaker
from services.autolabel.paths.base import RawDetection
from services.autolabel.paths.path_a_yolo26 import YoloPath
from services.autolabel.paths.path_b_sam3 import Sam3Path

log = get_logger("runner")


class GpuCapacityError(RuntimeError):
    pass


@dataclass
class FrameDetections:
    frame: FrameMeta
    image_bgr: np.ndarray
    dets_a: list[RawDetection] = field(default_factory=list)
    dets_b: list[RawDetection] = field(default_factory=list)


async def _segment_frame_drivable(db, store, fd, totals: dict) -> None:
    """Segment and store this frame's drivable surface, counting what it could not do.

    An empty result is written like any other: a frame with no road has been checked, and recording that
    is what lets coverage reach 100% without claiming more than was found. A frame the model could not
    reach gets NO row, so it stays visible as work rather than being buried under a zero.
    """
    import json as _json

    from db.models import DrivableMask
    from services.autolabel.drivable import DrivableUnavailable, segment_drivable

    try:
        res = segment_drivable(fd.image_bgr)
    except DrivableUnavailable as exc:
        # Usually the guard refusing because another model holds the card. The frame keeps its objects.
        totals["drivable_skipped"] = totals.get("drivable_skipped", 0) + 1
        log.info("autolabel.drivable_skipped", frame=str(fd.frame.frame_id), reason=str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        totals["drivable_failed"] = totals.get("drivable_failed", 0) + 1
        log.warning("autolabel.drivable_failed", frame=str(fd.frame.frame_id), error=str(exc))
        return

    key = f"masks/drivable/{fd.frame.session_id}/{fd.frame.frame_id}.json"
    uri = store.put_bytes(key, _json.dumps(
        {"classes": res["classes"], "width": res["width"], "height": res["height"]}).encode(),
        "application/json")
    existing = await db.get(DrivableMask, fd.frame.frame_id)
    if existing is None:
        db.add(DrivableMask(frame_id=fd.frame.frame_id, mask_uri=uri, coverage=res["coverage"],
                            source="proposed", model_version=res["model"]))
    elif existing.source != "human":
        # A human refinement is never overwritten by a machine pass.
        existing.mask_uri = uri
        existing.coverage = res["coverage"]
        existing.model_version = res["model"]
    await db.commit()
    totals["drivable"] = totals.get("drivable", 0) + 1


class VramGuard:
    """Reads real free/total VRAM from the driver and refuses loads that would breach headroom."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = self.settings.gpu.device
        self._index = int(self.device.split(":")[1]) if ":" in self.device else 0
        import torch

        self.torch = torch
        if not torch.cuda.is_available():
            raise GpuCapacityError("CUDA not available; the autolabel plane requires a GPU")
        # Force the CUDA context to exist before any memory-stats call (those raise
        # "Invalid device argument" if invoked before the context is initialized).
        torch.cuda.set_device(self._index)
        torch.cuda.init()
        self._warm = torch.zeros(1, device=self.device)

    def free_mb(self) -> float:
        free, _total = self.torch.cuda.mem_get_info(self._index)
        return free / (1024 * 1024)

    def reset_peak(self) -> None:
        self.torch.cuda.reset_peak_memory_stats(self._index)
        self.torch.cuda.empty_cache()

    def peak_mb(self) -> float:
        return self.torch.cuda.max_memory_reserved(self._index) / (1024 * 1024)

    def empty_cache(self) -> None:
        self.torch.cuda.empty_cache()

    def require(self, need_mb: float, name: str) -> None:
        free = self.free_mb()
        head = self.settings.gpu.vram_headroom_mb
        if free - need_mb < head:
            raise GpuCapacityError(
                f"refusing to load {name}: need ~{need_mb:.0f} MB, free {free:.0f} MB, "
                f"headroom {head} MB. Use gpu.mode=sequential or a smaller model."
            )
        log.info("vram.check", model=name, need_mb=round(need_mb), free_mb=round(free))


# Rough resident-set estimates for the guard. The guard also reads actual free VRAM, so these are
# advisory; they exist to fail before a load that obviously will not fit. Measured peak for the
# realized stack (YOLO11 + YOLO-World + SAM) is ~3.5 GB; estimates are deliberately conservative.
EST_YOLO_MB = 1800
EST_PATHB_MB = 3500


class StagedRunner:
    def __init__(self, yolo_weights: str | None = None, supported_ids: set[int] | None = None) -> None:
        self.settings = get_settings()
        self.guard = VramGuard()
        self.yolo: YoloPath | None = None
        self.sam: Sam3Path | None = None
        # When set, Path A loads the governance champion's weights instead of the config default.
        self.yolo_weights = yolo_weights
        # M-Q.0: the grounded supported set restricts Path B's open-vocab concept prompts.
        self.supported_ids = supported_ids

    def open_stage1(self) -> None:
        self.guard.reset_peak()
        self.guard.require(EST_YOLO_MB, "path_a_detector")
        self.yolo = YoloPath(self.yolo_weights)
        self.yolo.load()
        self.guard.require(EST_PATHB_MB, "path_b_openvocab")
        self.sam = Sam3Path(self.supported_ids)
        self.sam.load()
        log.info("stage1.open", free_mb=round(self.guard.free_mb()))

    def close_stage1(self) -> None:
        if self.yolo:
            self.yolo.unload()
        if self.sam:
            self.sam.unload()
        self.yolo = None
        self.sam = None
        self.guard.empty_cache()
        log.info("stage1.close", peak_mb=round(self.guard.peak_mb()), free_mb=round(self.guard.free_mb()))

    def run_stage1_frame(self, image_bgr: np.ndarray) -> tuple[list[RawDetection], list[RawDetection]]:
        if self.yolo is None or self.sam is None:
            raise RuntimeError("stage1 not open")
        dets_a = self.yolo.infer(image_bgr)
        dets_b = self.sam.infer(image_bgr)
        return dets_a, dets_b


def load_image(img_uri: str) -> np.ndarray:
    store = get_object_store()
    buf = np.frombuffer(store.get_bytes(img_uri), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to decode image {img_uri}")
    return img


async def fetch_frames(session_id: UUID, limit: int | None,
                       only_unlabelled: bool = False) -> list[FrameMeta]:
    """The frames a run should cover.

    `only_unlabelled` is what makes a bounded run resumable. `limit` alone takes the first N frames by
    timestamp and nothing skips frames that already carry objects, so asking for 200 frames twice does the
    same 200 twice and a drive can never be worked through in batches: the second batch looks like it ran
    and moves nothing. With it, each batch is the next N frames nobody has labelled, which is what
    "continue this drive" means and is also idempotent if a batch is fired twice by accident.

    Off by default, because a re-detect pass over a session deliberately redoes frames that already have
    objects and would otherwise silently become a no-op.
    """
    maker = get_sessionmaker()
    async with maker() as db:
        stmt = select(Frame).where(Frame.session_id == session_id).order_by(Frame.ts_ns)
        if only_unlabelled:
            stmt = stmt.where(~select(Object.object_id)
                              .where(Object.frame_id == Frame.frame_id).exists())
        if limit:
            stmt = stmt.limit(limit)
        rows = (await db.execute(stmt)).scalars().all()
    return [
        FrameMeta(
            frame_id=f.frame_id,
            session_id=f.session_id,
            ts_ns=f.ts_ns,
            cam_id=f.cam_id,
            img_uri=f.img_uri,
            width=f.width,
            height=f.height,
            ego_speed=f.ego_speed,
            quality=f.quality,
        )
        for f in rows
    ]


def _local_champion_weights(weights_uri: str) -> str | None:
    """Resolve a champion's weights_uri to a local file path Path A can load: a local path is used
    as-is, otherwise the blob is pulled from the object store into a scratch cache.

    The cache is keyed by the sha256 of the blob's *uri* plus the digest of its bytes, not by the basename.
    Keying on the basename meant two different champions both exported as "best.pt" shared one cache entry,
    so a promotion could silently keep serving the previous model's weights. The digest is also written
    alongside and re-verified on a cache hit, so a truncated or corrupted download is detected rather than
    loaded as a model.
    """
    import hashlib
    from pathlib import Path

    p = Path(weights_uri)
    if p.exists() and p.is_file():
        return str(p)

    cache = get_settings().scratch_path() / "models" / "champion"
    cache.mkdir(parents=True, exist_ok=True)
    # Content-addressed by uri: distinct champions never collide even with identical basenames.
    key = hashlib.sha256(weights_uri.encode()).hexdigest()[:16]
    dest = cache / f"{key}.pt"
    digest_file = dest.with_suffix(".sha256")

    if dest.exists() and digest_file.exists():
        want = digest_file.read_text().strip()
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got == want:
            return str(dest)
        log.warning("champion.weights_cache_corrupt", uri=weights_uri, expected=want[:12], got=got[:12])
        dest.unlink(missing_ok=True)
        digest_file.unlink(missing_ok=True)

    try:
        data = get_object_store().get_bytes(weights_uri)
    except Exception:  # noqa: BLE001
        log.warning("champion.weights_unreadable", uri=weights_uri)
        return None
    if not data:
        log.warning("champion.weights_empty", uri=weights_uri)
        return None
    dest.write_bytes(data)
    digest_file.write_text(hashlib.sha256(data).hexdigest())
    log.info("champion.weights_cached", uri=weights_uri, bytes=len(data), key=key)
    return str(dest)


async def resolve_detector_weights(db) -> str | None:
    """The governance champion's detector weights (local path), or None to fall back to config."""
    try:
        from services.govern.registry import get_champion

        champ = await get_champion(db, "detection")
    except Exception:  # noqa: BLE001
        return None
    if not champ or not champ.weights_uri:
        return None
    return _local_champion_weights(champ.weights_uri)


async def process_session(
    session_id: UUID,
    limit: int | None,
    on_frame: Callable[[FrameDetections], Awaitable[None]],
    yolo_weights: str | None = None,
    supported_ids: set[int] | None = None,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    only_unlabelled: bool = False,
) -> dict:
    """Stream a session's frames through Stage 1, invoking on_frame per frame. The callback is the
    plug point for fusion + gate + persistence (M3) and the VLM pass (M4).

    on_progress is told (frames done, frames total) after each frame. This loop is the only place that knows
    both numbers, and a job that reports nothing between its first frame and its last leaves every watcher
    guessing whether it is working or wedged.
    """
    frames = await fetch_frames(session_id, limit, only_unlabelled=only_unlabelled)
    if not frames:
        raise RuntimeError(f"no frames for session {session_id}")

    runner = StagedRunner(yolo_weights, supported_ids)
    runner.open_stage1()
    n_a = 0
    n_b = 0
    try:
        for i, fm in enumerate(frames, start=1):
            img = load_image(fm.img_uri)
            dets_a, dets_b = runner.run_stage1_frame(img)
            n_a += len(dets_a)
            n_b += len(dets_b)
            await on_frame(FrameDetections(frame=fm, image_bgr=img, dets_a=dets_a, dets_b=dets_b))
            if on_progress is not None:
                await on_progress(i, len(frames))
    finally:
        runner.close_stage1()

    summary = {
        "session_id": str(session_id),
        "frames": len(frames),
        "path_a_detections": n_a,
        "path_b_detections": n_b,
        "peak_vram_mb": round(runner.guard.peak_mb()),
        "vram_ceiling_mb": runner.settings.gpu.vram_total_mb,
    }
    log.info("stage1.summary", **summary)
    if summary["peak_vram_mb"] > summary["vram_ceiling_mb"]:
        log.error("vram.ceiling_exceeded", **summary)
    return summary


async def autolabel_session(session_id: UUID, limit: int | None, vlm_client=None,
                            on_progress: Callable[[int, int], Awaitable[None]] | None = None,
                            only_unlabelled: bool = False) -> dict:
    """Full pipeline: detect + segment -> fuse -> calibrate -> gate -> (Path C VLM on the uncertain
    subset) -> persist objects.

    on_progress is passed straight through to the frame loop, which is where the counts are. The caller that
    owns the job row decides what to do with them; this function does not know it has a row.

    Path C is duty-cycled (Principle 08): the VLM runs only on objects the gate would not
    auto-accept, capped by a per-session budget. The VLM call rate is tracked as a first-class
    metric. vlm_client is injectable for tests; otherwise built from config when the VLM is enabled.

    Holds the GPU slot for the session. It belongs here rather than in the router because the router is one
    of six callers: `redetect.py`, `ops_agent.py`, `scripts/autolabel_fleet.py`, `scripts/batch_process.py`
    and this module's own CLI all reach the pipeline directly, and the router's gates - training-holds-GPU
    and one-running-job - never applied to any of them. Nor did they cover the other things on the card: a
    corpus relabel, a judge sweep and a forgyx export could each start beside an autolabel pass, and two GPU
    jobs on one card is not a clean failure but an out-of-memory part way through a batch, which this loop
    counts as a failed frame.

    Per session rather than per fleet, so a multi-session redetect releases the card between drives and a
    training job waiting behind it waits one session instead of the whole corpus.
    """
    from core.gpu_slot import gpu_slot

    async with gpu_slot(f"autolabel:{session_id}", timeout_s=None):
        return await _autolabel_session_locked(session_id, limit, vlm_client, on_progress, only_unlabelled)


async def _autolabel_session_locked(session_id: UUID, limit: int | None, vlm_client=None,
                                    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
                                    only_unlabelled: bool = False) -> dict:
    """The pipeline itself, with the card already held. Never call this directly."""
    from services.autolabel.fusion import FusionEngine
    from services.autolabel.gate import gate_object, needs_vlm
    from services.autolabel.grounding import supported_concept_ids
    from services.autolabel.ontology import get_ontology
    from services.autolabel.paths.path_c_qwen3vl import VlmVerifier, apply_vlm, make_vlm_client
    from services.autolabel.persist import persist_frame_objects
    from services.autolabel.quality_reviewer import review_object_quality

    settings = get_settings()
    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    engine = FusionEngine(settings, onto)
    # M-Q.0: only prompt the open-vocab models (Path B concepts, Path C shortlist) with the grounded
    # supported set, so they cannot invent ungrounded classes.
    supported_ids = await supported_concept_ids()
    maker = get_sessionmaker()
    bus = EventBus()
    await bus.start()

    # Ego-hood exclusion: look up this session's vehicle so on_frame can drop detections that fall inside
    # the camera's pre-estimated hood mask (the ego vehicle labeling itself). No mask cached -> a safe no-op.
    from db.models import Session as DbSession
    from services.autolabel.ego_mask import get_ego_mask
    async with maker() as _s:
        vehicle_id = (await _s.execute(
            select(DbSession.vehicle_id).where(DbSession.session_id == session_id))).scalar_one_or_none()

    verifier = None
    if settings.models.vlm.enabled:
        client = vlm_client or make_vlm_client(settings)
        verifier = VlmVerifier(client, onto, settings, supported_ids=supported_ids)

    totals = {"objects": 0, "by_state": {}, "vlm_calls": 0, "vlm_eligible": 0}
    budget = settings.models.vlm.max_calls_per_session

    async with maker() as db:
        per_frame_cap = settings.models.vlm.max_calls_per_frame

        # Governance wiring: the kill switch gates auto-accept, and the registered champion's weights
        # serve in Path A. Both fail safe (auto-accept on, config weights) if governance is unavailable.
        try:
            from services.govern.killswitch import get_state

            auto_accept_enabled = (await get_state(db)).auto_accept_enabled
        except Exception:  # noqa: BLE001
            auto_accept_enabled = True
        champion_weights = await resolve_detector_weights(db)
        if champion_weights:
            log.info("autolabel.champion_serving", weights=champion_weights)

        # The measured auto-accept thresholds for the model actually serving, resolved once here and passed
        # down. The gate is a pure function on a hot path and must not reach the database per object, and
        # the fit is a property of the run rather than of any one detection.
        #
        # Empty is the ordinary case today: no fit has been activated, so every class falls back to the
        # configured constant and the gate says so once per class. Fitting is deliberately not activating
        # (services/oraclyx/threshold_fit.py), so this only becomes non-empty when a person switches a fit on.
        fitted_thresholds: dict[int, float] = {}
        try:
            from services.govern.registry import get_champion
            from services.oraclyx.threshold_fit import active_thresholds

            champ = await get_champion(db, "detection")
            if champ is not None:
                act = await active_thresholds(db, champ.model_version)
                fitted_thresholds = act["by_class"]
                if fitted_thresholds:
                    log.info("autolabel.thresholds_fitted", model=champ.model_version,
                             fit=act["fit_id"], classes=len(fitted_thresholds),
                             score_field=act["score_field"])
        except Exception as exc:  # noqa: BLE001
            # Fail to the constants rather than to no gate at all, and say so: silently gating on config
            # while believing the thresholds were measured is the exact confusion this work removes.
            log.warning("autolabel.threshold_fit_unavailable", error=str(exc))

        # Per-track observations from the frames just seen, for the reasoner's temporal check. Held here
        # rather than queried per object: the check needs the immediate neighbours, and a database round
        # trip per detection would cost more than the whole reasoning pass.
        track_history: dict[str, list] = {}

        async def on_frame(fd: FrameDetections) -> None:
            fused = engine.fuse_frame(fd.frame.frame_id, fd.dets_a, fd.dets_b)
            # Thing/stuff (panoptic): drop stuff detections (sky, road, vegetation, barriers, walls) before
            # spending any quality-review or VLM budget on them. They belong to semantic seg, not instances.
            # persist enforces the same rule as the hard chokepoint; this just avoids wasted work upstream.
            n_stuff = sum(1 for g in fused if not onto.is_thing(g.obj.class_id))
            if n_stuff:
                fused = [g for g in fused if onto.is_thing(g.obj.class_id)]
                totals["stuff_dropped"] = totals.get("stuff_dropped", 0) + n_stuff
            # Ego-hood: drop any detection that is mostly inside this camera's hood mask (the car labeling
            # its own bonnet). Cached mask per camera; absent -> no-op.
            ego = get_ego_mask(vehicle_id, fd.frame.cam_id) if vehicle_id else None
            if ego is not None:
                def _in_ego(g):
                    return ego.contains_bbox(tuple(g.obj.bbox.as_list()), fd.frame.width, fd.frame.height)
                n_ego = sum(1 for g in fused if _in_ego(g))
                if n_ego:
                    fused = [g for g in fused if not _in_ego(g)]
                    totals["ego_dropped"] = totals.get("ego_dropped", 0) + n_ego
            # M-Q.4: quality-review each object against the rest of the frame (geometric/contextual nonsense),
            # record the reasons in provenance, and carry the verdict into the gate so a flagged object cannot
            # auto-accept.
            objs = [g.obj for g in fused]
            quality_ok: dict[int, bool] = {}
            for fo in fused:
                others = [o for o in objs if o is not fo.obj]
                qv = review_object_quality(fo.obj, others, onto, fd.frame.width, fd.frame.height, settings.quality)
                fo.obj.provenance.quality_flags = qv.reasons
                quality_ok[id(fo.obj)] = qv.ok
                if not qv.ok:
                    totals["quality_demoted"] = totals.get("quality_demoted", 0) + 1

            # The reasoning pass. The quality reviewer above answers "is this box nonsense"; this answers
            # "is this the right class, and does anything the frame knows contradict it". It runs before
            # the gate because the gate's only other input is the detector's own confidence, which is the
            # model grading its own homework and cannot catch a confident wrong label.
            reasoner_ok: dict[int, bool] = {}
            if settings.reasoner.enabled:
                from services.autolabel.reasoner.pass_ import (
                    FrameContext,
                    apply_to_objects,
                    escalate,
                    reason_frame,
                    summarise,
                )

                fctx = FrameContext(width=fd.frame.width, height=fd.frame.height,
                                    scene=getattr(fd.frame, "scene", None) or {},
                                    track_history=track_history)
                verdicts = reason_frame(objs, onto, fctx,
                                        checks=settings.reasoner.checks or None)
                if settings.reasoner.adjudicate and verifier:
                    # Tier 2 draws on the same VLM budget as the old verification pass, and is capped
                    # separately: a bad prior that suddenly conflicts everywhere must not be able to
                    # consume the whole GPU allowance on its own.
                    room = min(budget - totals["vlm_calls"], per_frame_cap,
                               settings.reasoner.max_adjudications_per_session
                               - totals.get("adjudications", 0))
                    used = escalate(objs, verdicts, onto, fd.image_bgr, verifier, budget=max(0, room))
                    totals["adjudications"] = totals.get("adjudications", 0) + used
                    totals["vlm_calls"] += used
                reasoner_ok = apply_to_objects(objs, verdicts,
                                               record_trace=settings.reasoner.record_trace)
                for decision, n in summarise(verdicts).items():
                    key = f"reasoner_{decision}"
                    totals[key] = totals.get(key, 0) + n

            # Spend the per-frame VLM budget on the most uncertain objects first (lowest conf).
            order = sorted(range(len(fused)), key=lambda i: fused[i].obj.conf)
            frame_vlm = 0
            for i in order:
                fo = fused[i]
                # Both signals gate auto-accept, and either one can withhold it. The reasoner never
                # promotes: a verdict of accept only permits the gate to do what its own calibrated
                # thresholds already allow, so a 0.3 detection cannot be talked into being a label.
                qok = quality_ok[id(fo.obj)] and reasoner_ok.get(id(fo.obj), True)
                fo.obj.state = gate_object(fo.obj, onto, settings.gate,
                                           auto_accept_enabled=auto_accept_enabled, quality_ok=qok,
                                           fitted=fitted_thresholds)
                if verifier and needs_vlm(fo.obj, onto, settings.gate, quality_ok=qok,
                                          fitted=fitted_thresholds):
                    totals["vlm_eligible"] += 1
                    if totals["vlm_calls"] < budget and frame_vlm < per_frame_cap:
                        res = verifier.verify_object(fd.image_bgr, tuple(fo.obj.bbox.as_list()), fo.obj.class_id)
                        apply_vlm(fo.obj, res, onto, settings.models.vlm.ollama_tag)
                        fo.obj.state = gate_object(fo.obj, onto, settings.gate,
                                                   auto_accept_enabled=auto_accept_enabled, quality_ok=qok,
                                                   fitted=fitted_thresholds)  # re-gate
                        totals["vlm_calls"] += 1
                        frame_vlm += 1

            # Carry this frame's observations forward so the next frame's temporal check has something to
            # compare against. Bounded to a short window: a track's agreement with itself is a local
            # property, and holding a whole session would grow without limit for no extra signal.
            for fo in fused:
                if fo.obj.track_id:
                    tid = str(fo.obj.track_id)
                    hist = track_history.setdefault(tid, [])
                    hist.append((fd.frame.ts_ns, fo.obj.bbox, fo.obj.class_id))
                    if len(hist) > 6:
                        del hist[:-6]
            by_state = await persist_frame_objects(db, store, bus, fd.frame, fused)
            totals["objects"] += len(fused)
            for k, v in by_state.items():
                totals["by_state"][k] = totals["by_state"].get(k, 0) + v

            # Drivable surface, per frame. This is why the gap existed: segment_drivable had exactly one
            # caller in the tree, the editor's button, so a mask existed only where somebody had opened a
            # frame and clicked. The lane plausibility gate reads this mask and treats an ABSENT one as
            # "plausible", so on a frame without it the gate is wired up and does nothing.
            #
            # Failure costs this frame its mask, never the frame: the objects are already persisted above.
            if settings.models.drivable.autolabel:
                await _segment_frame_drivable(db, store, fd, totals)

        try:
            summary = await process_session(session_id, limit, on_frame, yolo_weights=champion_weights,
                                            supported_ids=supported_ids, on_progress=on_progress,
                                            only_unlabelled=only_unlabelled)
            await db.commit()
        finally:
            await bus.stop()

    objects = max(totals["objects"], 1)
    totals["vlm_call_rate"] = round(totals["vlm_calls"] / objects, 4)
    summary.update(totals)
    store.put_bytes(
        f"autolabel/{session_id}/summary.json",
        json.dumps(summary, indent=2).encode(),
        "application/json",
    )
    log.info(
        "autolabel.done",
        **{k: summary[k] for k in ("objects", "by_state", "vlm_calls", "vlm_call_rate", "peak_vram_mb")},
    )
    return summary


@click.command()
@click.option("--session", "session_id", required=True, type=str)
@click.option("--limit", type=int, default=None, help="cap frames processed (smoke runs)")
def main(session_id: str, limit: int | None) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    summary = asyncio.run(autolabel_session(UUID(session_id), limit))
    click.echo(summary)


if __name__ == "__main__":
    main()
