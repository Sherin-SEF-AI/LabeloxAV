"""SAM 2 video mask tracking: click an object once and its mask follows it through the clip.

`sam_propagate.py` beside this says of itself that it is flow tracking plus per-frame SAM refinement, "not the
SAM2 memory-bank video model", and that the real video predictor was a documented follow-on. This is that
follow-on. It runs the memory-bank path of the same `sam2_b.pt` checkpoint the click-to-segment tool already
loads: every frame's image features are cross-attended against a memory of the masks predicted on earlier
frames (and on every prompted frame), so the mask is carried by what the object looks like, not by optical
flow, and it survives the scale change and partial occlusion that make the flow tracker drift.

**Why not `ultralytics.SAM2VideoPredictor`.** The installed Ultralytics (8.3.0) predates that class. Its
`SAM2Model` does carry the whole video machinery (`track_step`, the memory encoder, the memory attention and
object pointers, ported from the reference implementation), so `Sam2VideoTracker` below drives those
directly in the same order the reference `propagate_in_video` does: prompted frames first as conditioning
frames, then forward from the earliest prompt, then backward from it. No new dependency, the same weights.

**Re-prompting.** A correction is another prompt on another frame. Every prompted frame becomes a
conditioning frame that all other frames attend to, exactly as in the reference predictor, so a run with the
anchor click plus one corrective click is the run the corrected track should have been. The prompts are kept
on the written objects' provenance so a re-prompt replays the earlier ones and adds the new one.

**What is written.** One object per frame with a mask, all on one track: `source='propagated'` (as every
propagation path writes), `interp_source='sam2_video'`, `state='annotate'`, never auto-accepted. A re-run
replaces the masks of objects this tool wrote and nobody has ruled on yet, and leaves every other object on
the track alone (a person's label, or one already reviewed, is never overwritten by a tracker).

**Fallback.** If the video model cannot load or fails mid-run, the job falls back to `sam_propagate.py`
(flow plus per-frame SAM) and says so in its result; it does not pretend the fallback was the video model.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import cast
from uuid import UUID

import numpy as np

from core.config import get_settings
from core.logging import get_logger

log = get_logger("sam2_video")

# SAM 2 runs at a fixed 1024 square; the frame is letterboxed into it (top-left, as the image predictor does).
IMGSZ = 1024
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
# A mask smaller than this many pixels is SAM saying the object is not there (it is occluded or has left
# the frame); the frame gets no object rather than a speck.
MIN_MASK_PX = 24
# The largest window one job may track. A person is waiting on it, and 240 frames at ~15 fps through the
# image encoder is already ~20 s on this card.
MAX_WINDOW = 240
# Interp source written on every object this tool creates (Object.interp_source is String(16)).
INTERP_SOURCE = "sam2_video"


@dataclass
class Prompt:
    """One prompt on one frame of the window: positive/negative clicks, a box, or both (image pixels)."""

    index: int
    points: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    box: list[float] | None = None

    def is_empty(self) -> bool:
        return not self.points and self.box is None


@dataclass
class FrameMask:
    index: int
    mask: np.ndarray | None  # bool HxW at the frame's own size; None when the object is not visible
    score: float             # mean foreground probability inside the mask (0 when there is no mask)

    @property
    def visible(self) -> bool:
        return self.mask is not None


def merge_prompts(prompts: list[Prompt]) -> dict[int, Prompt]:
    """Prompts per frame index, several prompts on one frame combined into one (later boxes win)."""
    out: dict[int, Prompt] = {}
    for p in prompts:
        if p.is_empty():
            continue
        cur = out.setdefault(p.index, Prompt(index=p.index))
        if p.points:
            labels = list(p.labels) if len(p.labels) == len(p.points) else [1] * len(p.points)
            cur.points.extend([float(x), float(y)] for x, y in p.points)
            cur.labels.extend(int(v) for v in labels)
        if p.box is not None:
            cur.box = [float(v) for v in p.box]
    return out


def propagation_order(n_frames: int, cond: list[int]) -> list[tuple[int, bool]]:
    """The order the reference predictor visits frames: forward from the earliest prompt, then backward.

    Returns (index, reverse) for every non-prompted frame. Prompted frames are not in the list; they are
    conditioning frames, run before this order starts.
    """
    if not cond:
        return []
    start = min(cond)
    cset = set(cond)
    fwd = [(i, False) for i in range(start + 1, n_frames) if i not in cset]
    bwd = [(i, True) for i in range(start - 1, -1, -1) if i not in cset]
    return fwd + bwd


def letterbox(img_bgr: np.ndarray, size: int = IMGSZ) -> tuple[np.ndarray, float]:
    """The frame scaled to fit `size` and padded bottom/right, as RGB float CHW normalised; and the scale."""
    import cv2

    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)  # LetterBox's pad value
    canvas[:nh, :nw] = resized
    rgb = canvas[..., ::-1].astype(np.float32)
    rgb = (rgb - _MEAN) / _STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)), r


def prompt_arrays(p: Prompt, r: float) -> tuple[np.ndarray, np.ndarray]:
    """SAM 2's merged prompt: a box is two corner points labelled 2 and 3, ahead of the clicks."""
    coords: list[list[float]] = []
    labels: list[int] = []
    if p.box is not None:
        x1, y1, x2, y2 = p.box
        coords += [[x1 * r, y1 * r], [x2 * r, y2 * r]]
        labels += [2, 3]
    for (x, y), lab in zip(p.points, p.labels, strict=False):
        coords.append([x * r, y * r])
        labels.append(1 if int(lab) else 0)
    return np.asarray(coords, dtype=np.float32)[None], np.asarray(labels, dtype=np.int32)[None]


class Sam2VideoTracker:
    """Drives `SAM2Model.track_step` over an ordered frame list. One object per run."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    def _features(self, img_bgr: np.ndarray):
        import torch

        arr, r = letterbox(img_bgr)
        im = torch.from_numpy(arr)[None].to(self.device)
        backbone_out = self.model.forward_image(im)
        _, vision_feats, vision_pos, feat_sizes = self.model._prepare_backbone_features(backbone_out)
        return vision_feats, vision_pos, feat_sizes, r

    def _to_mask(self, high_res_logits, r: float, h: int, w: int) -> FrameMask:
        import torch
        import torch.nn.functional as F  # noqa: N812

        nh, nw = int(round(h * r)), int(round(w * r))
        logits = high_res_logits[:, :, :nh, :nw].float()
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        m = logits > 0
        n = int(m.sum().item())
        if n < MIN_MASK_PX:
            return FrameMask(index=-1, mask=None, score=0.0)
        score = float(torch.sigmoid(logits[m]).mean().item())
        return FrameMask(index=-1, mask=m.cpu().numpy(), score=score)

    @staticmethod
    def _compact(out: dict) -> dict:
        # Only what later frames read from memory; the high-res logits (4 MB a frame) are dropped.
        return {"maskmem_features": out["maskmem_features"], "maskmem_pos_enc": out["maskmem_pos_enc"],
                "obj_ptr": out["obj_ptr"], "pred_masks": out["pred_masks"]}

    def track(self, load_image: Callable[[int], np.ndarray], n_frames: int,
              prompts: list[Prompt]) -> Iterator[FrameMask]:
        """Yield one FrameMask per frame of the window: prompted frames first, then the propagation order."""
        import torch

        cond = merge_prompts(prompts)
        if not cond:
            raise ValueError("at least one click or box is needed to start tracking")
        if any(i < 0 or i >= n_frames for i in cond):
            raise ValueError("a prompt is outside the tracked window")
        self.model.set_imgsz((IMGSZ, IMGSZ))
        output_dict: dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

        with torch.inference_mode():
            for idx in sorted(cond):
                img = load_image(idx)
                h, w = img.shape[:2]
                feats, pos, sizes, r = self._features(img)
                coords, labels = prompt_arrays(cond[idx], r)
                point_inputs = {"point_coords": torch.as_tensor(coords, device=self.device),
                                "point_labels": torch.as_tensor(labels, device=self.device)}
                out = self.model.track_step(idx, True, feats, pos, sizes, point_inputs, None, output_dict,
                                            n_frames, track_in_reverse=False, run_mem_encoder=True)
                fm = self._to_mask(out["pred_masks_high_res"], r, h, w)
                output_dict["cond_frame_outputs"][idx] = self._compact(out)
                fm.index = idx
                yield fm
            for idx, reverse in propagation_order(n_frames, list(cond)):
                img = load_image(idx)
                h, w = img.shape[:2]
                feats, pos, sizes, r = self._features(img)
                out = self.model.track_step(idx, False, feats, pos, sizes, None, None, output_dict, n_frames,
                                            track_in_reverse=reverse, run_mem_encoder=True)
                fm = self._to_mask(out["pred_masks_high_res"], r, h, w)
                output_dict["non_cond_frame_outputs"][idx] = self._compact(out)
                fm.index = idx
                yield fm


def _model_key(weights: str) -> str:
    return f"sam2video:{weights}"


def load_tracker() -> Sam2VideoTracker:
    """The video tracker on the shared registry, so an idle one can be evicted for a bigger job."""
    from core.model_manager import manager

    settings = get_settings()
    weights = settings.models.openvocab.seg_weights
    dev = settings.gpu.device
    if "sam2" not in str(weights):
        raise RuntimeError(f"video tracking needs a SAM 2 checkpoint; configured seg_weights is {weights!r}")

    def _load():
        from ultralytics.models.sam.build import build_sam

        m = build_sam(weights)
        return m.to(dev).eval()

    model = manager().get(_model_key(weights), _load, est_mb=900, device=dev)
    return Sam2VideoTracker(model, dev)


def unload() -> None:
    from core.model_manager import manager

    manager().release(_model_key(get_settings().models.openvocab.seg_weights))


# ---- persistence ----------------------------------------------------------------------------------------

def mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def prompts_to_json(prompts: list[Prompt], frame_ids: list[UUID]) -> list[dict]:
    """Prompts keyed by frame id, so a re-prompt with a different window can replay them."""
    return [{"frame_id": str(frame_ids[p.index]), "points": p.points, "labels": p.labels, "box": p.box}
            for p in prompts]


def is_tool_owned(obj) -> bool:
    """An object this tool wrote that nobody has ruled on: the only kind a re-run may rewrite or remove."""
    return (obj.interp_source == INTERP_SOURCE and obj.source == "propagated"
            and obj.state == "annotate")


async def load_window(db, anchor, before: int, after: int) -> list:
    """The anchor's camera, `before` frames earlier and `after` later, in time order."""
    from sqlalchemy import select

    from db.models import Frame

    earlier = (await db.execute(select(Frame).where(
        Frame.session_id == anchor.session_id, Frame.cam_id == anchor.cam_id, Frame.ts_ns < anchor.ts_ns)
        .order_by(Frame.ts_ns.desc()).limit(max(0, before)))).scalars().all()
    later = (await db.execute(select(Frame).where(
        Frame.session_id == anchor.session_id, Frame.cam_id == anchor.cam_id, Frame.ts_ns > anchor.ts_ns)
        .order_by(Frame.ts_ns.asc()).limit(max(0, after)))).scalars().all()
    return list(reversed(earlier)) + [anchor] + list(later)


def _decode(store, uri: str) -> np.ndarray:
    import cv2

    img = cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not decode frame image {uri}")
    return img


async def run_track(params: dict, *, job_id: UUID | None = None, user_id: UUID | None = None,
                    progress: Callable | None = None,
                    tracker_factory: Callable[[], Sam2VideoTracker] | None = None) -> dict:
    """Track one object over a window of its session and write the masks. The `sam2_video` job body.

    params: frame_id (the anchor frame), before, after, prompts [{frame_id, points, labels, box}], and one
    of object_id (track an existing label), track_id (re-prompt a track this tool made) or class_name.
    """
    import asyncio

    from sqlalchemy import select

    from core.storage import get_object_store
    from core.timebase import now_ns
    from db.models import Frame, Object, Review, Track
    from db.session import get_sessionmaker
    from services.api.routers.objects import _write_mask
    from services.autolabel.ontology import get_ontology
    from services.autolabel.paths.path_b_openvocab import polygons_from_mask

    store = get_object_store()
    onto = get_ontology()
    before = max(0, int(params.get("before", 10)))
    after = max(0, int(params.get("after", 10)))
    if before + after + 1 > MAX_WINDOW:
        raise ValueError(f"window of {before + after + 1} frames is over the limit of {MAX_WINDOW}")

    async with get_sessionmaker()() as db:
        anchor = await db.get(Frame, UUID(str(params["frame_id"])))
        if anchor is None:
            raise ValueError("anchor frame not found")
        window = await load_window(db, anchor, before, after)
        frame_ids = [f.frame_id for f in window]
        index_of = {f.frame_id: i for i, f in enumerate(window)}

        anchor_obj = None
        track = None
        prior_prompts: list[dict] = []
        if params.get("object_id"):
            anchor_obj = await db.get(Object, UUID(str(params["object_id"])))
            if anchor_obj is None:
                raise ValueError("object not found")
            class_id = anchor_obj.class_id
            if anchor_obj.track_id is not None:
                track = await db.get(Track, anchor_obj.track_id)
        elif params.get("track_id"):
            track = await db.get(Track, UUID(str(params["track_id"])))
            if track is None:
                raise ValueError("track not found")
            class_id = track.class_id
        else:
            name = params.get("class_name")
            if not name or not onto.has_name(name):
                raise ValueError(f"unknown class {name!r}")
            class_id = onto.by_name(name).id

        # Objects already on the track inside the window, and the prompts the last run of this tool used.
        existing: dict[UUID, list] = {}
        if track is not None:
            rows = (await db.execute(select(Object).where(
                Object.track_id == track.track_id, Object.frame_id.in_(frame_ids)))).scalars().all()
            for o in rows:
                existing.setdefault(o.frame_id, []).append(o)
                if is_tool_owned(o) and not prior_prompts:
                    prior_prompts = list((o.provenance or {}).get("sam2_prompts") or [])

        prompts: list[Prompt] = []
        for raw in prior_prompts + list(params.get("prompts") or []):
            fid = UUID(str(raw["frame_id"]))
            if fid not in index_of:
                continue  # an earlier prompt outside this window does not steer it
            prompts.append(Prompt(index=index_of[fid], points=[list(map(float, p)) for p in raw.get("points") or []],
                                  labels=[int(v) for v in raw.get("labels") or []], box=raw.get("box")))
        if anchor_obj is not None and anchor_obj.frame_id in index_of:
            idx = index_of[anchor_obj.frame_id]
            if not any(p.index == idx for p in prompts):
                prompts.append(Prompt(index=idx, box=[float(v) for v in anchor_obj.bbox]))
        if not merge_prompts(prompts):
            raise ValueError("at least one click or box is needed to start tracking")

    # ---- the video model, frame by frame (each step in a worker thread so the loop stays responsive) ----
    method = "sam2_video"
    fallback_reason = None
    masks: dict[int, FrameMask] = {}
    t0 = time.monotonic()
    try:
        tracker = (tracker_factory or load_tracker)()
        cache: dict[int, np.ndarray] = {}

        def load_image(i: int) -> np.ndarray:
            if i not in cache:
                cache.clear()  # one frame at a time is all the tracker reads
                cache[i] = _decode(store, window[i].img_uri)
            return cache[i]

        gen = tracker.track(load_image, len(window), prompts)
        done = object()
        while True:
            item = await asyncio.to_thread(next, gen, done)
            if item is done:
                break
            fm = cast(FrameMask, item)
            masks[fm.index] = fm
            if progress is not None:
                await progress(0.05 + 0.85 * len(masks) / len(window), f"tracking {len(masks)}/{len(window)}")
    except ValueError:
        raise  # a bad request (no prompt, prompt outside the window) is not a model failure
    except Exception as exc:  # noqa: BLE001  the video model failing must still leave a usable result
        log.warning("sam2_video.failed_falling_back", error=f"{type(exc).__name__}: {exc}"[:300])
        method = "sam_propagate_fallback"
        fallback_reason = f"{type(exc).__name__}: {exc}"[:300]

    if method != "sam2_video":
        return await _fallback(params, window, prompts, anchor_obj, class_id, track, fallback_reason,
                               user_id=user_id)

    # ---- write ------------------------------------------------------------------------------------------
    created: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    kept: list[str] = []
    empty: list[str] = []
    prompts_json = prompts_to_json(prompts, frame_ids)
    async with get_sessionmaker()() as db:
        if track is None:
            track = Track(session_id=window[0].session_id, class_id=class_id,
                          first_ts_ns=anchor.ts_ns, last_ts_ns=anchor.ts_ns,
                          tracker_version="sam2_video:" + str(get_settings().models.openvocab.seg_weights))
            db.add(track)
            await db.flush()
            if anchor_obj is not None:
                ao = await db.get(Object, anchor_obj.object_id)
                if ao is not None:
                    ao.track_id = track.track_id
        else:
            track = await db.get(Track, track.track_id)
        if track is None:  # deleted while the model ran
            raise ValueError("the track was deleted while it was being tracked")
        reviewer = "sam2_video"
        span: list[int] = []
        for i, fr in enumerate(window):
            got = masks.get(i)
            mine = [o for o in existing.get(fr.frame_id, []) if is_tool_owned(o)]
            others = [o for o in existing.get(fr.frame_id, []) if not is_tool_owned(o)]
            if anchor_obj is not None and fr.frame_id == anchor_obj.frame_id:
                kept.append(str(anchor_obj.object_id))
                span.append(fr.ts_ns)
                continue
            mask = got.mask if got is not None else None
            if got is None or mask is None:
                empty.append(str(fr.frame_id))
                for o in mine:  # the corrected run no longer sees it here: drop this tool's own guess
                    await db.delete(await db.get(Object, o.object_id))
                    removed.append(str(o.object_id))
                continue
            if others:
                kept.extend(str(o.object_id) for o in others)  # a person's or a reviewed label stays
                span.append(fr.ts_ns)
                continue
            polys = polygons_from_mask(mask)
            if not polys:
                empty.append(str(fr.frame_id))
                continue
            box = mask_bbox(mask)
            conf = round(float(got.score), 4)
            prov = {"method": INTERP_SOURCE, "job_id": str(job_id) if job_id else None,
                    "anchor_frame_id": str(anchor.frame_id), "sam2_prompts": prompts_json,
                    "propagated_from": str(anchor_obj.object_id) if anchor_obj is not None else None,
                    "prompted": i in merge_prompts(prompts)}
            if mine:
                row = cast(Object, await db.get(Object, mine[0].object_id))
                before_state = {"bbox": list(row.bbox), "mask_uri": row.mask_uri, "conf": row.conf,
                                "version": row.version}
                row.mask_uri = _write_mask(store, fr.session_id, fr.frame_id, row.object_id, polys, fr.width,
                                           fr.height)
                row.mask_encoding = "polygon"
                row.bbox, row.conf, row.provenance = box, conf, prov
                row.version = (row.version or 0) + 1
                db.add(Review(object_id=row.object_id, reviewer=reviewer, user_id=user_id, action="edit_mask",
                              before=before_state, after={"bbox": box, "mask_uri": row.mask_uri,
                                                          "version": row.version, "via": INTERP_SOURCE},
                              time_spent_ms=0, ts_ns=now_ns()))
                updated.append(str(row.object_id))
                for extra in mine[1:]:
                    await db.delete(await db.get(Object, extra.object_id))
                    removed.append(str(extra.object_id))
            else:
                oid = uuid.uuid4()
                mask_uri = _write_mask(store, fr.session_id, fr.frame_id, oid, polys, fr.width, fr.height)
                db.add(Object(object_id=oid, frame_id=fr.frame_id, track_id=track.track_id, class_id=class_id,
                              bbox=box, conf=conf, source="propagated", state="annotate",
                              interp_source=INTERP_SOURCE, mask_uri=mask_uri, mask_encoding="polygon",
                              provenance=prov))
                created.append(str(oid))
            span.append(fr.ts_ns)
        if span:
            track.first_ts_ns = min([track.first_ts_ns or span[0], *span])
            track.last_ts_ns = max([track.last_ts_ns or span[0], *span])
        await db.commit()
        track_id = str(track.track_id)

    result = {"method": method, "track_id": track_id, "frames": len(window),
              "anchor_frame_id": str(anchor.frame_id), "window_frame_ids": [str(f) for f in frame_ids],
              "created": created, "updated": updated, "removed": removed, "kept": kept,
              "empty_frame_ids": empty, "prompts": len(merge_prompts(prompts)),
              "seconds": round(time.monotonic() - t0, 2)}
    log.info("sam2_video.done", **{k: (len(v) if isinstance(v, list) else v) for k, v in result.items()})
    return result


async def _fallback(params: dict, window: list, prompts: list[Prompt], anchor_obj, class_id: int, track,
                    reason: str | None, *, user_id: UUID | None) -> dict:
    """Flow plus per-frame SAM from `sam_propagate.py`, from the anchor object (made from the prompt if needed)."""
    import asyncio

    from core.storage import get_object_store
    from db.models import Object, Track
    from db.session import get_sessionmaker
    from services.api.routers.objects import _write_mask
    from services.temporal.sam_propagate import sam_propagate_object

    before = int(params.get("before", 10))
    after = int(params.get("after", 10))
    anchor_id = anchor_obj.object_id if anchor_obj is not None else None
    if anchor_id is None:
        # No label to propagate from yet: segment the first prompted frame with the image SAM to make one.
        from services.api.sam_service import segment

        store = get_object_store()
        p = merge_prompts(prompts)[min(merge_prompts(prompts))]
        fr = window[p.index]
        img = _decode(store, fr.img_uri)
        res = await asyncio.to_thread(segment, img, p.points or None, p.labels or None, p.box)
        if not res.get("polygons") or not res.get("bbox"):
            return {"method": "sam_propagate_fallback", "fallback_reason": reason, "created": [],
                    "note": "the image segmenter found nothing at the prompt either"}
        async with get_sessionmaker()() as db:
            if track is None:
                track = Track(session_id=fr.session_id, class_id=class_id, first_ts_ns=fr.ts_ns, last_ts_ns=fr.ts_ns)
                db.add(track)
                await db.flush()
            oid = uuid.uuid4()
            db.add(Object(object_id=oid, frame_id=fr.frame_id, track_id=track.track_id, class_id=class_id,
                          bbox=[float(v) for v in res["bbox"]], conf=0.5, source="propagated", state="annotate",
                          interp_source="sam_propagated",
                          mask_uri=_write_mask(store, fr.session_id, fr.frame_id, oid, res["polygons"],
                                               fr.width, fr.height),
                          mask_encoding="polygon", provenance={"method": "sam_image", "fallback_for": INTERP_SOURCE}))
            await db.commit()
        anchor_id = oid
    direction = "both" if before and after else ("forward" if after else "backward")
    res = await sam_propagate_object(anchor_id, max(before, after, 1), direction, bool(params.get("refine", True)))
    made = [o for v in (res.get("by_direction") or {}).values() for o in v]
    if anchor_obj is None:
        made = [str(anchor_id), *made]
    return {"method": "sam_propagate_fallback", "fallback_reason": reason, "track_id": res.get("track_id"),
            "frames": len(window), "created": made, "updated": [], "removed": [], "kept": [],
            "empty_frame_ids": [],
            "note": "the SAM 2 video model was unavailable; boxes were carried by optical flow and refined "
                    "per frame with the image SAM"}


def json_safe(d: dict) -> dict:
    return json.loads(json.dumps(d, default=str))
