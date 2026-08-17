"""Re-check a frame's redaction by looking where the annotations say the people and vehicles are.

The whole-frame pass misses PII that is simply too small at the frame's own resolution. That is the same
failure `_detect_size` was written for: a plate 150x40 pixels arrives as 50x13 after a 3x downsample and the
detector sees nothing. Faces have the same floor, and the face detector was never given a second look.

Measured on this corpus: 82.4% of frames holding an annotated person have zero faces redacted, and a probe
over 14 already-redacted frames found 0 faces on the whole frame and 7 inside crops of the annotated people
in those same frames. The plate counts went the other way on some frames (4 whole-frame, 3 region-of-
interest), so the two passes are complementary and the answer is their union, never a replacement.

Why the annotations are the right place to look: they already say where every person and vehicle is. A face
20 pixels wide in a 1920-wide frame is 200 pixels wide once cropped to the person carrying it, which is
above the detector's floor rather than below it. Nothing in this codebase had ever used the labels to
re-examine the redaction, and nothing verifies that a blur covered anything: the DPDPA gate checks that an
audit row exists, and the proof signs a count.

This only ever adds blur. It cannot restore an over-blur, because the unredacted original is deliberately
never stored, which is why the region filters below are conservative rather than generous.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, OntologyClass, PiiAudit
from services.anonymize.backfill import ROI_SUFFIX, StoreRefused, _audit_of, _put_with_backoff
from services.quality.iaa import iou
from services.recall.backends import load_image_bgr

log = get_logger("anonymize.recheck")

# Which annotated classes are worth cropping, by ontology group rather than by name, so a class added to the
# ontology later inherits the behaviour instead of silently falling outside a hardcoded list.
#
# Faces include the two- and three-wheeler groups because a rider's head routinely sits inside the vehicle's
# box rather than inside their own, and on a motorcycle the two boxes are nearly the same rectangle.
_FACE_GROUPS = frozenset({"vru", "two_wheeler", "three_wheeler"})
_PLATE_GROUPS = frozenset({"two_wheeler", "three_wheeler", "four_wheeler", "heavy"})

# A face sits at the top of a person's box and a plate straddles the edge of a tightly drawn vehicle box, so
# a crop taken at the annotation's exact bounds cuts off the thing being looked for.
_MARGIN = 0.15

# Below this the detector sees nothing at all, which is the entire reason the whole-frame pass missed these
# regions in the first place. Upscaling a small crop to it is what makes the second look different from the
# first rather than a repeat of it.
_UPSCALE_FLOOR_PX = 320

# A crop smaller than this carries too few real pixels for an upscale to recover anything; what comes back
# from one is noise, and noise here becomes permanent blur.
_MIN_CROP_PX = 16

# A detection covering this much of its own crop is the detector agreeing with the crop, not finding
# something inside it. Faces are the stricter of the two because a "face" the size of the whole person is
# always wrong, whereas a plate genuinely can fill a crop of the back of a truck.
_MAX_AREA_FRACTION = {"face": 0.5, "plate": 0.7}

# The confidence a detection found inside an upscaled crop must clear, above whatever the whole-frame pass
# uses. The crop is interpolated pixels the detector was not calibrated on, and it is asked to look at every
# annotated person on every frame, so its opportunities to be wrong are far greater than the single
# whole-frame call beside it.
#
# The numbers are measured, and the asymmetry between them is the measurement's finding rather than a guess.
# Sampling the face candidates this pass proposed and looking at them:
#
#   at the configured 0.50, 24 candidates included an autorickshaw body, grass, a taillight and foliage,
#     roughly a third of them not faces at all
#   at 0.70, 24 candidates were faces with one or two exceptions, at the cost of 91 candidates becoming 24
#
# Plates were sampled the same way and needed nothing: at scores from 0.36 upward the boxes sat on real
# registration plates, one of them legible ("KA01MG9647") on a frame whose whole-frame pass found none. So
# the plate detector keeps its configured floor and only faces are held higher.
#
# Losing two thirds of the face candidates is the right side to err on. This blur is permanent, a face
# missed today is still reachable by a later pass with better weights, and a face blurred in error is not.
_ROI_MIN_SCORE = {"face": 0.70}

# Overlap at which a candidate counts as already covered by an existing blurred region. Deliberately low:
# a partial overlap with an existing blur usually means the same PII found again with slightly different
# bounds, and blurring it twice adds nothing while widening the redaction.
_COVERED_IOU = 0.3


@dataclass(frozen=True)
class Candidate:
    """One region the re-check believes holds PII, in full-frame pixel coordinates."""

    type: str          # the redaction target's name, as stored in PiiAudit.regions
    bbox: list[float]  # x1, y1, x2, y2
    score: float
    where: str         # "frame" or "roi", kept in the audit so a reviewer can see which pass found it


def _crop_window(box: list[float], w: int, h: int) -> tuple[int, int, int, int] | None:
    """A margined, frame-clamped crop around one annotation, or None if too small to be worth reading."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    mx, my = (x2 - x1) * _MARGIN, (y2 - y1) * _MARGIN
    cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    cx2, cy2 = min(w, int(round(x2 + mx))), min(h, int(round(y2 + my)))
    if cx2 - cx1 < _MIN_CROP_PX or cy2 - cy1 < _MIN_CROP_PX:
        return None
    return cx1, cy1, cx2, cy2


def _upscaled(crop: np.ndarray) -> tuple[np.ndarray, float]:
    """The crop enlarged to the detector's floor, and the factor used, so detections can be mapped back."""
    short_side = max(1, min(crop.shape[0], crop.shape[1]))
    scale = max(1.0, _UPSCALE_FLOOR_PX / short_side)
    if scale <= 1.0:
        return crop, 1.0
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), scale


async def roi_windows(db: AsyncSession, frame_id: uuid.UUID, w: int, h: int,
                      ) -> list[tuple[tuple[int, int, int, int], frozenset[str]]]:
    """Crop windows for one frame, each with the detectors worth running inside it.

    A three-wheeler carries both a rider's face and a plate, so a window can want more than one detector and
    the detectors are a set rather than a single kind.
    """
    rows = (await db.execute(
        select(Object.bbox, OntologyClass.l0, OntologyClass.l1)
        .join(OntologyClass, OntologyClass.id == Object.class_id)
        .where(Object.frame_id == frame_id))).all()
    out: list[tuple[tuple[int, int, int, int], frozenset[str]]] = []
    for bbox, l0, l1 in rows:
        if l0 != "object" or not bbox:
            continue
        wants = {d for d, groups in (("face", _FACE_GROUPS), ("plate", _PLATE_GROUPS)) if l1 in groups}
        if not wants:
            continue
        window = _crop_window(list(bbox), w, h)
        if window is not None:
            out.append((window, frozenset(wants)))
    return out


def _detect_in_windows(anonymizer, img: np.ndarray,
                       windows: list[tuple[tuple[int, int, int, int], frozenset[str]]],
                       targets_by_detector: dict[str, str]) -> list[Candidate]:
    """Run each window's detectors on its upscaled crop, mapped back to full-frame coordinates."""
    detectors = {"face": anonymizer.face, "plate": anonymizer.plate}
    found: list[Candidate] = []
    for (cx1, cy1, cx2, cy2), wants in windows:
        crop, scale = _upscaled(img[cy1:cy2, cx1:cx2])
        crop_area = float((cx2 - cx1) * (cy2 - cy1))
        for kind in wants:
            det = detectors.get(kind)
            target_name = targets_by_detector.get(kind)
            if det is None or target_name is None or not det.available:
                continue
            floor = _ROI_MIN_SCORE.get(kind)
            for x1, y1, x2, y2, score in det.detect(crop):
                if floor is not None and float(score) < floor:
                    continue
                # Back to the frame: undo the upscale, then the crop's offset.
                fx1, fy1 = cx1 + x1 / scale, cy1 + y1 / scale
                fx2, fy2 = cx1 + x2 / scale, cy1 + y2 / scale
                area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
                if crop_area > 0 and area / crop_area > _MAX_AREA_FRACTION.get(kind, 0.7):
                    continue
                found.append(Candidate(type=target_name,
                                       bbox=[round(fx1, 1), round(fy1, 1), round(fx2, 1), round(fy2, 1)],
                                       score=round(float(score), 3), where="roi"))
    return found


def _dedup(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse candidates that are the same region seen twice.

    Neighbouring crops overlap, so one pedestrian standing beside a motorcycle produces the same face from
    two windows. Keeping both would be harmless for the pixels and misleading in the audit, which is the
    record a reviewer reads.
    """
    kept: list[Candidate] = []
    for c in sorted(candidates, key=lambda c: c.score, reverse=True):
        if any(k.type == c.type and iou(k.bbox, c.bbox) >= _COVERED_IOU for k in kept):
            continue
        kept.append(c)
    return kept


def new_regions(candidates: list[Candidate], existing: list[dict]) -> list[Candidate]:
    """The candidates not already covered by a blurred region on this frame.

    This is what makes the action idempotent: after one run the additions are in the audit, so a second run
    matches them here and adds nothing. It is also what stops a re-check from re-blurring an already blurred
    region, which would darken it a little further on every pass.
    """
    prior = [(r.get("type"), r.get("bbox") or []) for r in existing]
    fresh: list[Candidate] = []
    for c in candidates:
        covered = any(t == c.type and len(b) >= 4 and iou(b, c.bbox) >= _COVERED_IOU for t, b in prior)
        if not covered:
            fresh.append(c)
    return fresh


async def recheck_frame(db: AsyncSession, store, anonymizer, frame: Frame, *, apply: bool = True) -> dict:
    """Look again at one frame, blur what the first pass missed, and record it. Returns counts.

    With `apply=False` nothing is written and nothing is blurred, which is what the plan endpoint serves: a
    blur cannot be undone, so the operation has to be inspectable before it is taken.
    """
    from services.domain import redaction_targets

    targets_by_detector = {t.detector: t.name for t in redaction_targets()}
    existing = await _audit_of(db, frame.frame_id)
    prior_regions: list[dict] = list(existing.regions or []) if existing is not None else []

    try:
        img = load_image_bgr(store, frame.img_uri)
    except Exception as exc:  # noqa: BLE001
        log.warning("recheck.load_failed", frame_id=str(frame.frame_id), error=str(exc))
        return {"status": "unreadable", "faces_added": 0, "plates_added": 0, "regions_added": [],
                "already_covered": 0, "windows": 0}
    h, w = img.shape[:2]

    # The whole-frame pass as well as the crops: on some frames it is the one that finds the plate, so the
    # union beats either alone. Detections it repeats are collapsed by the dedup below.
    candidates: list[Candidate] = []
    for detector_kind, target_name in targets_by_detector.items():
        det = {"face": anonymizer.face, "plate": anonymizer.plate}.get(detector_kind)
        if det is None or not det.available:
            continue
        for x1, y1, x2, y2, score in det.detect(img):
            candidates.append(Candidate(type=target_name,
                                        bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                        score=round(float(score), 3), where="frame"))

    windows = await roi_windows(db, frame.frame_id, w, h)
    candidates.extend(_detect_in_windows(anonymizer, img, windows, targets_by_detector))

    fresh = new_regions(_dedup(candidates), prior_regions)
    counts = {"face": 0, "plate": 0}
    for c in fresh:
        counts[c.type] = counts.get(c.type, 0) + 1

    method_version = f"{anonymizer.method_version}{ROI_SUFFIX}"
    result = {
        "status": "planned" if not apply else "applied",
        "faces_added": counts.get("face", 0),
        "plates_added": counts.get("plate", 0),
        "regions_added": [{"type": c.type, "bbox": c.bbox, "score": c.score, "where": c.where} for c in fresh],
        "already_covered": len(candidates) - len(fresh),
        "windows": len(windows),
        "method_version": method_version,
    }
    if not apply or not fresh:
        # Finding nothing still means the frame has now been looked at properly, so the audit records the
        # method that looked. Otherwise every sweep would re-examine the same clean frames forever, and a
        # frame that had no audit at all would stay outside the DPDPA gate despite having been checked.
        if apply and existing is not None:
            existing.method_version = method_version
        elif apply:
            db.add(PiiAudit(frame_id=frame.frame_id, session_id=frame.session_id, n_faces=0, n_plates=0,
                            regions=[], method_version=method_version, ts_ns=frame.ts_ns))
        return result

    for c in fresh:
        anonymizer._blur_region(img, *c.bbox)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return {**result, "status": "encode_failed", "faces_added": 0, "plates_added": 0,
                "regions_added": []}
    _bucket, key = store.parse_uri(frame.img_uri)
    # If the store will not take the image, the audit must not claim these regions are blurred: a row saying
    # a face is redacted when the stored bytes still show it is worse than the miss it was fixing.
    if not await _put_with_backoff(store, key, buf.tobytes(), "image/jpeg"):
        raise StoreRefused(key)

    merged = prior_regions + [{"type": c.type, "bbox": c.bbox, "score": c.score, "where": c.where}
                              for c in fresh]
    if existing is not None:
        existing.n_faces = int(existing.n_faces or 0) + counts.get("face", 0)
        existing.n_plates = int(existing.n_plates or 0) + counts.get("plate", 0)
        existing.regions = merged
        existing.method_version = method_version
    else:
        db.add(PiiAudit(frame_id=frame.frame_id, session_id=frame.session_id,
                        n_faces=counts.get("face", 0), n_plates=counts.get("plate", 0),
                        regions=merged, method_version=method_version, ts_ns=frame.ts_ns))
    log.info("recheck.applied", frame_id=str(frame.frame_id), faces=counts.get("face", 0),
             plates=counts.get("plate", 0), windows=len(windows))
    return result
