"""Copy-paste synthesis: real human-masked instances pasted onto real, already-labelled road frames.

Why this and not a generative model: a starved class (cattle, rider) needs positives the detector can learn
from, and the cheapest positives with proven value are real pixels of the class placed at a plausible spot and
size in a real scene. The whole generator is CPU work in cv2; it takes no GPU slot and never blocks the
event loop (each frame composes in a worker thread).

What makes a composite honest enough to train on:

* Donors are objects a person ruled on (`source='human'`, `state='accepted'`) with a polygon mask, on a real
  frame, at least `MIN_DONOR_SIDE_PX` on the short side. Imported `rle`/`png_url` masks are skipped rather than
  guessed at.
* Backgrounds are real, selected frames with a drivable-surface mask and a label set the trainer already
  treats as complete: at least one object in a trainable state and no `annotate` object (the machine's own
  "this frame needs a person" flag). Frames with no object at all are not backgrounds: on the live corpus
  7,160 such frames had zero predictions too, so they were never labelled, and a composite made from one
  would teach every unlabelled object as background, the lesson that cost 36 recall points once already.
  The stricter rule (no `review` objects either) left 24 frames on the same corpus, which is why it is not
  the rule.
* Placement lands on the drivable surface, at the size the row implies: apparent height scales with the
  distance below the horizon row, so a donor is rescaled by the ratio of its own ground row's horizon
  offset to the target row's. A paste that would cover more than `MAX_COVER_FRAC` of any background box is
  refused and re-placed; after `PLACEMENT_TRIES` refusals the pair is dropped and counted.
* Reinhard colour transfer toward the local background patch and a Gaussian-feathered edge, both partial,
  so a white cow on a dark road is still a white cow.

Every composite is a new `Frame` on one synthetic `Session` per build, with `source_frame_id` pointing at the
background. The background's labels are copied onto it (state and source `synthetic`, the original state kept
in provenance) and the pasted instance is added with the mask the paste produced. Each `SYNTH_BATCH` frames
is one committed `AgentRun(kind='synth_batch')` whose `changes` lists the frame ids, so revert deletes exactly
what it made; the build itself is the parent run and reverts by cascading its children.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import cv2
import numpy as np
from sqlalchemy import String, cast, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.origin import REAL, SYNTHETIC, SYNTHETIC_SOURCE, SYNTHETIC_STATE
from core.storage import get_object_store
from db.models import AgentRun, DrivableMask, Frame, Object, Session

log = get_logger("synth.copy_paste")

KIND = "synth_build"          # the parent run: one per build request
BATCH_KIND = "synth_batch"    # one per committed batch of frames; the unit revert works on
SYNTH_BATCH = 200
MIN_DONOR_SIDE_PX = 48
MAX_COVER_FRAC = 0.30
PLACEMENT_TRIES = 12
# Rescale bounds: past these the row-depth model is asking for a donor it does not have the pixels for.
MIN_SCALE = 0.35
MAX_SCALE = 3.0
MIN_PASTE_SIDE_PX = 24
# Fraction of the frame a paste may span; larger is a donor standing where the camera is.
MAX_PASTE_W_FRAC = 0.6
MAX_PASTE_H_FRAC = 0.8
# The band of rows a paste may stand on, as fractions of the image height.
ROW_BAND = (0.30, 0.97)
# Partial colour transfer and feather strength; 1.0 would be full Reinhard, 0 none.
COLOUR_TRANSFER = 0.5
# Host memory above which a batch waits rather than starts. The generator is CPU work, but a 4K background,
# its mask and the LAB copies are tens of MB per frame and the API shares the machine with training.
MEMORY_CEILING_FRAC = 0.90
MEMORY_WAIT_S = 30.0
MEMORY_MAX_WAIT_S = 900.0
# Label states a background's objects are copied from; everything the trainer's default uses.
COPIED_STATES = ("accepted", "auto_accept", "settled", "review")
JPEG_QUALITY = 92
_SALT = "synth-copy-paste"


# ---------------------------------------------------------------- data


@dataclass
class Donor:
    object_id: uuid.UUID
    frame_id: uuid.UUID
    session_id: uuid.UUID
    cam_id: str
    class_id: int
    bbox: list[float]
    mask_uri: str
    img_uri: str
    width: int
    height: int
    attrs: dict = field(default_factory=dict)


@dataclass
class Background:
    frame_id: uuid.UUID
    session_id: uuid.UUID
    vehicle_id: str
    cam_id: str
    ts_ns: int
    img_uri: str
    width: int
    height: int
    drivable_uri: str
    objects: list[dict] = field(default_factory=list)   # {object_id, class_id, bbox, state, source, ...}


@dataclass
class Composite:
    image: np.ndarray
    bbox: list[float]
    polygons: list[list[float]]
    scale: float
    flipped: bool
    covered: dict[str, float]     # background object id -> fraction of its box the paste covers
    tries: int


@dataclass
class Refusal:
    reason: str
    tries: int = 0


# ---------------------------------------------------------------- geometry helpers


def rasterize_polygons(polys: list[list[float]], height: int, width: int) -> np.ndarray:
    """Flat [x,y,...] rings to a uint8 {0,1} mask, even-odd so a nested ring reads as a hole."""
    m = np.zeros((height, width), dtype=np.uint8)
    rings = []
    for poly in polys or []:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) >= 3:
            rings.append(np.round(pts).astype(np.int32))
    if rings:
        cv2.fillPoly(m, rings, 1)
    return m


def polygons_of(mask: np.ndarray, epsilon_px: float = 1.0) -> list[list[float]]:
    """External contours of a {0,1} mask as flat [x,y,...] rings, the persisted polygon encoding."""
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out: list[list[float]] = []
    for c in contours:
        if cv2.contourArea(c) < 4:
            continue
        approx = cv2.approxPolyDP(c, epsilon_px, True)
        if len(approx) >= 3:
            out.append([float(v) for pt in approx.reshape(-1, 2) for v in pt])
    return out


def horizon_row(cy: float, fy: float, pitch_deg: float) -> float:
    """The image row where the ground plane vanishes: cy - fy * tan(pitch). A camera pitched down (positive
    pitch toward the road) puts the horizon above the principal point."""
    return cy - fy * math.tan(math.radians(pitch_deg))


def cover_fraction(paste: list[float], box: list[float]) -> float:
    """Fraction of `box` hidden under `paste`, both xyxy."""
    ix0, iy0 = max(paste[0], box[0]), max(paste[1], box[1])
    ix1, iy1 = min(paste[2], box[2]), min(paste[3], box[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / area


def reinhard_transfer(src: np.ndarray, src_mask: np.ndarray, ref: np.ndarray, strength: float) -> np.ndarray:
    """Move the donor's LAB statistics toward the reference patch by `strength` in [0, 1]."""
    if strength <= 0 or ref.size == 0 or src_mask.sum() < 16:
        return src
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    sel = s[src_mask > 0]
    s_mean, s_std = sel.mean(axis=0), sel.std(axis=0) + 1e-3
    r_mean, r_std = r.mean(axis=0), r.std(axis=0) + 1e-3
    # Match the spread of the chroma channels only up to a factor of two: matching a road patch's tiny
    # colour variance onto a textured animal flattens it into a cutout.
    ratio = np.clip(r_std / s_std, 0.5, 2.0)
    moved = (s - s_mean) * ratio + r_mean
    out = s + strength * (moved - s)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- the pure compose step


def compose(background: np.ndarray, bg_boxes: dict[str, list[float]], drivable: np.ndarray,
            donor: np.ndarray, donor_mask: np.ndarray, donor_bbox: list[float], *,
            donor_horizon: float, target_horizon: float, rng: random.Random,
            max_cover: float = MAX_COVER_FRAC, tries: int = PLACEMENT_TRIES,
            colour_strength: float = COLOUR_TRANSFER) -> Composite | Refusal:
    """Paste one masked donor onto a background at a drivable, depth-consistent spot. Pure: no I/O.

    `drivable` is a {0,1} mask of the background's drivable surface at background resolution; `donor_mask` is
    the donor frame's {0,1} instance mask; `donor_bbox` its xyxy box in donor-frame pixels. Horizons are the
    ground vanishing rows of the two cameras. Returns a `Refusal` when no placement satisfies the cover
    rule, or when the donor has too little mask to be worth pasting.
    """
    H, W = background.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in donor_bbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(donor.shape[1], x1), min(donor.shape[0], y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return Refusal("donor box is degenerate")
    crop = donor[y0:y1, x0:x1]
    cmask = donor_mask[y0:y1, x0:x1]
    if cmask.sum() < 0.2 * cmask.size:
        return Refusal("donor mask fills under 20% of its box")
    ch, cw = crop.shape[:2]
    donor_ground = y1  # the box bottom is where the donor met the ground in its own frame
    donor_offset = donor_ground - donor_horizon
    if donor_offset <= 1.0:
        return Refusal("donor stands above its own horizon; the row-depth model has no scale for it")

    lo, hi = int(ROW_BAND[0] * H), int(ROW_BAND[1] * H)
    rows = [v for v in range(lo, hi) if drivable[v].any()]
    if not rows:
        return Refusal("no drivable surface in the placement band")

    flipped = rng.random() < 0.5
    if flipped:
        crop = np.ascontiguousarray(crop[:, ::-1])
        cmask = np.ascontiguousarray(cmask[:, ::-1])

    last = "no drivable placement fit the frame"
    for t in range(1, tries + 1):
        v = rows[rng.randrange(len(rows))]
        cols = np.flatnonzero(drivable[v])
        u = int(cols[rng.randrange(len(cols))])
        scale = (v - target_horizon) / donor_offset
        if not (MIN_SCALE <= scale <= MAX_SCALE):
            last = f"row {v} implies scale {scale:.2f}, outside [{MIN_SCALE}, {MAX_SCALE}]"
            continue
        pw, ph = int(round(cw * scale)), int(round(ch * scale))
        if min(pw, ph) < MIN_PASTE_SIDE_PX or pw > MAX_PASTE_W_FRAC * W or ph > MAX_PASTE_H_FRAC * H:
            last = f"scaled paste {pw}x{ph} is outside the size bounds"
            continue
        px0, py0 = u - pw // 2, v - ph
        px1, py1 = px0 + pw, v
        if px0 < 0 or py0 < 0 or px1 > W or py1 > H:
            last = "paste would leave the frame"
            continue
        paste_box = [float(px0), float(py0), float(px1), float(py1)]
        covered = {oid: cover_fraction(paste_box, b) for oid, b in bg_boxes.items()}
        worst = max(covered.values(), default=0.0)
        if worst > max_cover:
            last = f"paste would cover {worst:.0%} of a background box (limit {max_cover:.0%})"
            continue

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        fg = cv2.resize(crop, (pw, ph), interpolation=interp)
        fm = cv2.resize(cmask.astype(np.uint8) * 255, (pw, ph), interpolation=cv2.INTER_LINEAR)
        fm = (fm > 127).astype(np.uint8)

        # Colour: the patch the paste lands on plus a margin, so the reference is the light the object
        # would actually be standing in.
        mx, my = pw // 2, ph // 2
        rx0, ry0 = max(0, px0 - mx), max(0, py0 - my)
        rx1, ry1 = min(W, px1 + mx), min(H, py1 + my)
        ref = background[ry0:ry1, rx0:rx1]
        fg = reinhard_transfer(fg, fm, ref, colour_strength)

        # Feather: blur the hard mask by a kernel proportional to the paste size.
        k = max(3, int(round(min(pw, ph) * 0.04)) | 1)
        alpha = cv2.GaussianBlur(fm.astype(np.float32), (k, k), 0)[..., None]
        out = background.copy()
        region = out[py0:py1, px0:px1].astype(np.float32)
        out[py0:py1, px0:px1] = np.clip(region * (1 - alpha) + fg.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        # The label is the visible mask, tight, in background coordinates.
        full = np.zeros((H, W), dtype=np.uint8)
        full[py0:py1, px0:px1] = fm
        ys, xs = np.nonzero(full)
        bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        polys = polygons_of(full)
        if not polys:
            last = "pasted mask produced no polygon"
            continue
        return Composite(image=out, bbox=bbox, polygons=polys, scale=round(scale, 4), flipped=flipped,
                         covered={k2: round(c, 4) for k2, c in covered.items() if c > 0}, tries=t)
    return Refusal(last, tries=tries)


# ---------------------------------------------------------------- corpus queries


def _min_side():
    # Postgres arrays are 1-indexed: bbox[1..4] = x0, y0, x1, y1.
    return func.least(Object.bbox[3] - Object.bbox[1], Object.bbox[4] - Object.bbox[2])


async def load_donors(db: AsyncSession, class_ids: list[int], *, limit: int = 500) -> list[Donor]:
    stmt = (select(Object.object_id, Object.frame_id, Frame.session_id, Frame.cam_id, Object.class_id,
                   Object.bbox, Object.mask_uri, Frame.img_uri, Frame.width, Frame.height, Object.attrs)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .where(Object.class_id.in_(class_ids), Object.source == "human", Object.state == "accepted",
                   Object.mask_uri.is_not(None), Object.mask_encoding == "polygon",
                   Frame.origin == REAL, _min_side() >= MIN_DONOR_SIDE_PX)
            .order_by(Object.object_id).limit(limit))
    rows = (await db.execute(stmt)).all()
    return [Donor(object_id=r[0], frame_id=r[1], session_id=r[2], cam_id=r[3], class_id=r[4],
                  bbox=list(r[5]), mask_uri=r[6], img_uri=r[7], width=r[8], height=r[9], attrs=dict(r[10] or {}))
            for r in rows]


def _background_filter(stmt):
    unruled = exists().where(Object.frame_id == Frame.frame_id, Object.state == "annotate")
    labelled = exists().where(Object.frame_id == Frame.frame_id, Object.state.in_(COPIED_STATES))
    return stmt.where(Frame.origin == REAL, Frame.selected.is_(True), ~unruled, labelled)


async def count_backgrounds(db: AsyncSession) -> int:
    stmt = select(func.count()).select_from(Frame).join(DrivableMask, DrivableMask.frame_id == Frame.frame_id)
    return int((await db.execute(_background_filter(stmt))).scalar_one())


async def load_backgrounds(db: AsyncSession, *, n: int, seed: str, offset: int = 0) -> list[Background]:
    """`n` backgrounds in a deterministic shuffled order keyed by `seed`, so two builds with different seeds
    draw different frames and one build can page through its own order with `offset`."""
    key = func.md5(func.concat(cast(Frame.frame_id, String), seed))
    stmt = (select(Frame.frame_id, Frame.session_id, Frame.cam_id, Frame.ts_ns, Frame.img_uri, Frame.width,
                   Frame.height, DrivableMask.mask_uri, Session.vehicle_id)
            .join(DrivableMask, DrivableMask.frame_id == Frame.frame_id)
            .join(Session, Session.session_id == Frame.session_id))
    stmt = _background_filter(stmt).order_by(key).offset(offset).limit(n)
    rows = (await db.execute(stmt)).all()
    if not rows:
        return []
    ids = [r[0] for r in rows]
    objs = (await db.execute(
        select(Object.object_id, Object.frame_id, Object.class_id, Object.bbox, Object.state, Object.source,
               Object.mask_uri, Object.mask_encoding, Object.attrs, Object.conf, Object.track_id,
               Object.rot_deg, Object.sign_type, Object.sign_category)
        .where(Object.frame_id.in_(ids), Object.state.in_(COPIED_STATES)))).all()
    by_frame: dict[uuid.UUID, list[dict]] = {}
    for o in objs:
        by_frame.setdefault(o[1], []).append({
            "object_id": o[0], "class_id": o[2], "bbox": list(o[3]), "state": o[4], "source": o[5],
            "mask_uri": o[6], "mask_encoding": o[7], "attrs": dict(o[8] or {}), "conf": float(o[9]),
            "track_id": o[10], "rot_deg": float(o[11] or 0.0), "sign_type": o[12], "sign_category": o[13]})
    return [Background(frame_id=r[0], session_id=r[1], vehicle_id=r[8], cam_id=r[2], ts_ns=r[3], img_uri=r[4],
                       width=r[5], height=r[6], drivable_uri=r[7], objects=by_frame.get(r[0], [])) for r in rows]


async def plan_build(db: AsyncSession, *, class_names: list[str], n_frames: int) -> dict:
    """What a build of `n_frames` for these classes could draw on, and why it might refuse. No writes."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    classes: list[dict] = []
    ok: list[int] = []
    for name in class_names:
        try:
            cid = onto.by_name(name).id
        except KeyError:
            classes.append({"class_name": name, "class_id": None, "donors": 0,
                            "reason": "not in the ontology"})
            continue
        donors = await load_donors(db, [cid], limit=10_000)
        entry = {"class_name": name, "class_id": cid, "donors": len(donors)}
        if not donors:
            entry["reason"] = ("no human-accepted object of this class has a polygon mask on a real frame "
                               f"with a short side of at least {MIN_DONOR_SIDE_PX} px")
        else:
            ok.append(cid)
        classes.append(entry)
    backgrounds = await count_backgrounds(db)
    feasible = bool(ok) and backgrounds > 0 and n_frames > 0
    reason = None
    if not ok:
        reason = "no requested class has a usable donor"
    elif backgrounds == 0:
        reason = "no real, selected, labelled frame with a drivable mask and no annotate object"
    elif n_frames <= 0:
        reason = "n_frames must be positive"
    return {"classes": classes, "backgrounds": backgrounds, "n_frames": n_frames,
            "batches": math.ceil(n_frames / SYNTH_BATCH) if n_frames > 0 else 0,
            "feasible": feasible, "reason": reason, "buildable_class_ids": ok}


# ---------------------------------------------------------------- the build


def _frame_key(session_id: uuid.UUID, cam_id: str, ts_ns: int) -> str:
    return f"frames/{session_id}/{cam_id}/{ts_ns}.jpg"


def _mask_key(session_id: uuid.UUID, frame_id: uuid.UUID, object_id: uuid.UUID) -> str:
    return f"masks/{session_id}/{frame_id}/{object_id}.json"


def _decode(store, uri: str) -> np.ndarray | None:
    buf = np.frombuffer(store.get_bytes(uri), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _drivable_mask(store, uri: str, height: int, width: int) -> np.ndarray | None:
    blob = json.loads(store.get_bytes(uri).decode("utf-8"))
    classes = blob.get("classes") or {}
    polys = classes.get("drivable") or []
    if not polys:
        return None
    mw, mh = int(blob.get("width") or width), int(blob.get("height") or height)
    m = rasterize_polygons(polys, mh, mw)
    if (mh, mw) != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return m


def _without_hood(drivable: np.ndarray, vehicle_id: str, cam_id: str) -> tuple[np.ndarray, bool]:
    """The drivable surface with the ego vehicle's own bonnet removed.

    The segmenter reads the bottom of most dashcam frames as road because the bonnet is smooth, grey and
    continuous with it, so a placement drawn from the raw drivable mask can stand a rider on the hood of the
    recording car (the first 500-frame build did exactly that on frames whose drivable polygon reached the
    bottom edge). The per-camera hood mask that the detector's cleanup sweep already uses is the same
    answer here: cut it out of the surface before a footprint row is drawn. Returns the surface and whether
    a hood mask existed for this camera; cameras without one keep the raw surface and the report says so.
    """
    from services.autolabel.ego_mask import get_ego_mask

    ego = get_ego_mask(vehicle_id, cam_id)
    if ego is None or ego.area_frac <= 0:
        return drivable, False
    grid = np.asarray(ego.grid, dtype=np.uint8)
    hood = cv2.resize(grid, (drivable.shape[1], drivable.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = drivable.copy()
    out[hood > 0] = 0
    return out, True


def _instance_mask(store, uri: str, height: int, width: int) -> np.ndarray | None:
    blob = json.loads(store.get_bytes(uri).decode("utf-8"))
    if blob.get("encoding") != "polygon":
        return None
    mw, mh = int(blob.get("width") or width), int(blob.get("height") or height)
    m = rasterize_polygons(blob.get("polygons") or [], mh, mw)
    if (mh, mw) != (height, width):
        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
    return m


async def _horizon_for(session_id: uuid.UUID, cam_id: str, width: int, height: int) -> float:
    from services.calibration.resolve import resolve_calibration

    cal = await resolve_calibration(session_id, cam_id, width, height)
    return horizon_row(cal.cy, cal.fy, cal.rpy_deg[1])


async def _wait_for_memory(report: dict) -> bool:
    """Hold the next batch while host memory is above the ceiling; give up after MEMORY_MAX_WAIT_S."""
    from services.hardening.resources import host

    waited = 0.0
    while True:
        frac = host().get("memory_used_frac")
        if frac is None or frac < MEMORY_CEILING_FRAC:
            return True
        if waited >= MEMORY_MAX_WAIT_S:
            report["stopped"] = f"host memory at {frac:.0%} for {int(waited)}s; not starting another batch"
            return False
        report["memory_waits"] = report.get("memory_waits", 0) + 1
        await asyncio.sleep(MEMORY_WAIT_S)
        waited += MEMORY_WAIT_S


class _DonorCache:
    """Decoded donor pixels and masks, loaded once per donor per build."""

    def __init__(self, store):
        self.store = store
        self.pixels: dict[uuid.UUID, tuple[np.ndarray, np.ndarray] | None] = {}
        self.horizons: dict[tuple[uuid.UUID, str, int, int], float] = {}

    def get(self, d: Donor) -> tuple[np.ndarray, np.ndarray] | None:
        if d.object_id not in self.pixels:
            img = _decode(self.store, d.img_uri)
            mask = _instance_mask(self.store, d.mask_uri, d.height, d.width) if img is not None else None
            self.pixels[d.object_id] = (img, mask) if img is not None and mask is not None else None
        return self.pixels[d.object_id]

    async def horizon(self, d: Donor) -> float:
        key = (d.session_id, d.cam_id, d.width, d.height)
        if key not in self.horizons:
            self.horizons[key] = await _horizon_for(*key)
        return self.horizons[key]


def _seeded_rng(seed: str, frame_id: uuid.UUID) -> random.Random:
    return random.Random(int(hashlib.md5(f"{_SALT}:{seed}:{frame_id}".encode()).hexdigest()[:16], 16))


async def _new_session(db: AsyncSession, template: Session, *, class_names: list[str], run_id: uuid.UUID) -> Session:
    s = Session(session_id=uuid.uuid4(), pack_id=template.pack_id, project_id=template.project_id,
                vehicle_id="synthetic", start_ts_ns=template.start_ts_ns, end_ts_ns=template.end_ts_ns,
                city=template.city, route=f"copy-paste:{','.join(class_names)}",
                sensors={"synthetic": {"generator": "copy_paste", "classes": class_names,
                                       "agent_run_id": str(run_id)}},
                raw_uri=None, mcap_uri=None, ontology_version=template.ontology_version, origin=SYNTHETIC)
    db.add(s)
    await db.flush()
    return s


async def build(run_id: uuid.UUID, *, class_names: list[str], n_frames: int, seed: str,
                created_by: str) -> dict:
    """The worker behind `POST /synth/build` and `maybe_synth_starved`. Composes `n_frames` in batches of
    `SYNTH_BATCH`, each its own committed child run; the parent run's report is returned."""
    from db.session import get_sessionmaker
    from services.agent.runtime.report import finish_run
    from services.autolabel.ontology import get_ontology

    maker = get_sessionmaker()
    store = get_object_store()
    onto = get_ontology()
    report: dict = {"classes": class_names, "requested": n_frames, "seed": seed, "frames": 0, "batches": 0,
                    "refusals": {}, "donors": {}, "backgrounds_used": 0, "hood_masked": 0, "hood_unknown": 0,
                    "child_runs": []}
    status = "committed"
    child_runs: list[str] = []
    try:
        async with maker() as db:
            plan = await plan_build(db, class_names=class_names, n_frames=n_frames)
            report["plan"] = {k: v for k, v in plan.items() if k != "buildable_class_ids"}
            if not plan["feasible"]:
                await finish_run(run_id, status="refused", report={**report, "reason": plan["reason"]})
                return {**report, "reason": plan["reason"], "status": "refused"}
            class_ids = plan["buildable_class_ids"]
            donors = await load_donors(db, class_ids, limit=10_000)
            for d in donors:
                name = onto.by_id(d.class_id).name
                report["donors"][name] = report["donors"].get(name, 0) + 1
            first = await load_backgrounds(db, n=1, seed=seed)
            template = await db.get(Session, first[0].session_id)
            synth_session = await _new_session(db, template, class_names=class_names, run_id=run_id)
            session_id = synth_session.session_id
            await db.commit()

        cache = _DonorCache(store)
        rng_pick = random.Random(int(hashlib.md5(f"{_SALT}:{seed}".encode()).hexdigest()[:16], 16))
        offset = 0
        ts_seen: list[int] = []
        while report["frames"] < n_frames:
            if not await _wait_for_memory(report):
                break
            want = min(SYNTH_BATCH, n_frames - report["frames"])
            async with maker() as db:
                # Draw more backgrounds than the batch needs so refusals do not shrink it.
                bgs = await load_backgrounds(db, n=want * 2, seed=seed, offset=offset)
            if not bgs:
                report["stopped"] = "ran out of backgrounds"
                break
            offset += len(bgs)
            made = await _build_batch(maker, store, cache, onto, bgs, donors, want, session_id=session_id,
                                      seed=seed, rng_pick=rng_pick, parent_run_id=run_id, created_by=created_by,
                                      report=report, ts_seen=ts_seen)
            if made["run_id"]:
                child_runs.append(made["run_id"])
                report["child_runs"] = child_runs
                report["batches"] += 1
                async with maker() as db:
                    parent = await db.get(AgentRun, run_id)
                    if parent is not None:
                        parent.changes = {"child_runs": child_runs, "session_id": str(session_id)}
                        parent.counts = dict(report)
                        parent.heartbeat_at = datetime.now(UTC)
                        await db.commit()
            if made["n"] == 0 and made["run_id"] is None:
                report["stopped"] = report.get("stopped") or "a whole batch of backgrounds was refused"
                break
        async with maker() as db:
            s = await db.get(Session, session_id)
            if s is not None:
                if ts_seen:
                    s.start_ts_ns, s.end_ts_ns = min(ts_seen), max(ts_seen)
                if report["frames"] == 0:
                    await db.delete(s)
                await db.commit()
        report["session_id"] = str(session_id) if report["frames"] else None
        if report["frames"] == 0:
            status = "refused"
            report["reason"] = report.get("stopped") or "no composite could be made"
    except Exception as exc:  # noqa: BLE001 - the run must record its failure, not vanish
        log.exception("synth.build.failed", run_id=str(run_id))
        status = "error"
        report["error"] = str(exc)
    await finish_run(run_id, status=status, report=report,
                     changes={"child_runs": child_runs, "session_id": report.get("session_id")})
    return {**report, "status": status}


async def _build_batch(maker, store, cache: _DonorCache, onto, bgs: list[Background], donors: list[Donor],
                       want: int, *, session_id: uuid.UUID, seed: str, rng_pick: random.Random,
                       parent_run_id: uuid.UUID, created_by: str, report: dict, ts_seen: list[int]) -> dict:
    """Compose up to `want` frames from `bgs` and commit them as one child run. Returns {n, run_id}."""
    batch_run_id = uuid.uuid4()
    frame_ids: list[str] = []
    rows: list[tuple[Frame, list[Object]]] = []
    refusals = report["refusals"]

    for bg in bgs:
        if len(rows) >= want:
            break
        donor = donors[rng_pick.randrange(len(donors))]
        rng = _seeded_rng(seed, bg.frame_id)
        try:
            donor_px = await asyncio.to_thread(cache.get, donor)
            if donor_px is None:
                refusals["donor_unreadable"] = refusals.get("donor_unreadable", 0) + 1
                continue
            bg_img = await asyncio.to_thread(_decode, store, bg.img_uri)
            if bg_img is None:
                refusals["background_unreadable"] = refusals.get("background_unreadable", 0) + 1
                continue
            drivable = await asyncio.to_thread(_drivable_mask, store, bg.drivable_uri, bg.height, bg.width)
            if drivable is None:
                refusals["no_drivable_polygon"] = refusals.get("no_drivable_polygon", 0) + 1
                continue
            drivable, hood_known = _without_hood(drivable, bg.vehicle_id, bg.cam_id)
            report["hood_masked" if hood_known else "hood_unknown"] += 1
            d_h = await cache.horizon(donor)
            t_h = await _horizon_for(bg.session_id, bg.cam_id, bg.width, bg.height)
            boxes = {str(o["object_id"]): o["bbox"] for o in bg.objects}
            res = await asyncio.to_thread(compose, bg_img, boxes, drivable, donor_px[0], donor_px[1], donor.bbox,
                                          donor_horizon=d_h, target_horizon=t_h, rng=rng)
        except Exception as exc:  # noqa: BLE001 - one bad blob must not end the batch
            log.warning("synth.compose.error", frame_id=str(bg.frame_id), error=str(exc))
            refusals["error"] = refusals.get("error", 0) + 1
            continue
        if isinstance(res, Refusal):
            key = res.reason.split(";")[0][:80]
            refusals[key] = refusals.get(key, 0) + 1
            continue

        frame_id = uuid.uuid4()
        ok, enc = cv2.imencode(".jpg", res.image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            refusals["encode_failed"] = refusals.get("encode_failed", 0) + 1
            continue
        img_uri = await asyncio.to_thread(store.put_bytes, _frame_key(session_id, bg.cam_id, bg.ts_ns),
                                          enc.tobytes(), "image/jpeg")
        frame = Frame(frame_id=frame_id, session_id=session_id, ts_ns=bg.ts_ns, cam_id=bg.cam_id,
                      img_uri=img_uri, width=bg.width, height=bg.height, quality=0.0, origin=SYNTHETIC,
                      source_frame_id=bg.frame_id, selected=True, tags=["synthetic"])
        objs: list[Object] = []
        for o in bg.objects:
            prov = {"agent_run_id": str(batch_run_id),
                    "synthetic": {"source_frame_id": str(bg.frame_id), "source_object_id": str(o["object_id"]),
                                  "source_state": o["state"], "source_source": o["source"], "pasted": False,
                                  "covered_frac": res.covered.get(str(o["object_id"]), 0.0)}}
            objs.append(Object(object_id=uuid.uuid4(), frame_id=frame_id, class_id=o["class_id"], bbox=o["bbox"],
                               mask_uri=o["mask_uri"], mask_encoding=o["mask_encoding"], attrs=o["attrs"],
                               conf=o["conf"], source=SYNTHETIC_SOURCE, state=SYNTHETIC_STATE, provenance=prov,
                               rot_deg=o["rot_deg"], sign_type=o["sign_type"], sign_category=o["sign_category"]))
        pasted_id = uuid.uuid4()
        payload = {"encoding": "polygon", "polygons": res.polygons, "height": bg.height, "width": bg.width}
        mask_uri = await asyncio.to_thread(store.put_bytes, _mask_key(session_id, frame_id, pasted_id),
                                           json.dumps(payload).encode(), "application/json")
        objs.append(Object(object_id=pasted_id, frame_id=frame_id, class_id=donor.class_id, bbox=res.bbox,
                           mask_uri=mask_uri, mask_encoding="polygon", attrs=dict(donor.attrs), conf=1.0,
                           source=SYNTHETIC_SOURCE, state=SYNTHETIC_STATE,
                           provenance={"agent_run_id": str(batch_run_id),
                                       "synthetic": {"source_frame_id": str(bg.frame_id),
                                                     "source_object_id": str(donor.object_id),
                                                     "donor_frame_id": str(donor.frame_id), "pasted": True,
                                                     "scale": res.scale, "flipped": res.flipped,
                                                     "tries": res.tries}}))
        rows.append((frame, objs))
        frame_ids.append(str(frame_id))
        ts_seen.append(bg.ts_ns)
        name = onto.by_id(donor.class_id).name
        report.setdefault("pasted", {})[name] = report.get("pasted", {}).get(name, 0) + 1

    if not rows:
        return {"n": 0, "run_id": None}

    async with maker() as db:
        db.add(AgentRun(run_id=batch_run_id, kind=BATCH_KIND, scope={"parent_run_id": str(parent_run_id),
                                                                       "session_id": str(session_id)},
                        status="committed", policy={"seed": seed, "max_cover_frac": MAX_COVER_FRAC},
                        counts={"frames": len(rows), "objects": sum(len(o) for _, o in rows)},
                        changes={"frame_ids": frame_ids, "session_id": str(session_id)}, critic={},
                        created_by=created_by))
        for frame, _objs in rows:
            db.add(frame)
        await db.flush()
        for _, objs in rows:
            db.add_all(objs)
        await db.commit()
    report["frames"] += len(rows)
    report["backgrounds_used"] += len(rows)
    log.info("synth.batch.committed", run_id=str(batch_run_id), frames=len(rows), total=report["frames"])
    return {"n": len(rows), "run_id": str(batch_run_id)}


# ---------------------------------------------------------------- revert


async def revert_batch(db: AsyncSession, run: AgentRun) -> dict:
    """Delete the frames a batch composed (objects cascade), drop their blobs best-effort, and remove the
    synthetic session once nothing of it remains."""
    store = get_object_store()
    frame_ids = [uuid.UUID(f) for f in (run.changes or {}).get("frame_ids") or []]
    reverted = skipped = 0
    session_id = (run.changes or {}).get("session_id")
    for fid in frame_ids:
        frame = await db.get(Frame, fid)
        if frame is None or frame.origin == REAL:
            skipped += 1
            continue
        masks = (await db.execute(select(Object.mask_uri).where(Object.frame_id == fid,
                                                                 Object.mask_uri.is_not(None)))).scalars().all()
        own_prefix = f"masks/{frame.session_id}/"
        for m in masks:
            # Only masks written under the synthetic session; copied labels share their source's blob.
            if m and own_prefix in m:
                store.remove(m)
        store.remove(frame.img_uri)
        await db.delete(frame)
        reverted += 1
    await db.flush()
    if session_id:
        sid = uuid.UUID(session_id)
        left = (await db.execute(select(func.count()).select_from(Frame).where(Frame.session_id == sid))).scalar_one()
        if left == 0:
            s = await db.get(Session, sid)
            if s is not None and s.origin != REAL:
                await db.delete(s)
    run.status = "reverted"
    run.reverted_at = datetime.now(UTC)
    await db.commit()
    log.info("synth.batch.reverted", run_id=str(run.run_id), frames=reverted, skipped=skipped)
    return {"run_id": str(run.run_id), "reverted": reverted, "skipped": skipped}
