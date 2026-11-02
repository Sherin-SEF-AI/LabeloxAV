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
from services.anonymize.coverage import (
    FACE_GROUPS,
    MARGIN,
    MIN_CROP_PX,
    PLATE_GROUPS,
    crop_window,
    inside_fraction,
)
from services.quality.iaa import iou
from services.recall.backends import load_image_bgr

log = get_logger("anonymize.recheck")

# Which annotated classes are worth cropping, by ontology group rather than by name, so a class added to the
# ontology later inherits the behaviour instead of silently falling outside a hardcoded list.
#
# Faces include the two- and three-wheeler groups because a rider's head routinely sits inside the vehicle's
# box rather than inside their own, and on a motorcycle the two boxes are nearly the same rectangle.
# Imported from services/anonymize/coverage.py, which the export gate also reads. If these two disagreed,
# the gate could refuse a frame for a region the re-check would never look for.
_FACE_GROUPS = FACE_GROUPS
_PLATE_GROUPS = PLATE_GROUPS

# A face sits at the top of a person's box and a plate straddles the edge of a tightly drawn vehicle box, so
# a crop taken at the annotation's exact bounds cuts off the thing being looked for.
_MARGIN = MARGIN

# Below this the detector sees nothing at all, which is the entire reason the whole-frame pass missed these
# regions in the first place. Upscaling a small crop to it is what makes the second look different from the
# first rather than a repeat of it.
_UPSCALE_FLOOR_PX = 320

# A crop smaller than this carries too few real pixels for an upscale to recover anything; what comes back
# from one is noise, and noise here becomes permanent blur.
_MIN_CROP_PX = MIN_CROP_PX

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
# Plates were sampled the same way on a handful of frames and looked fine at the configured 0.35. A larger
# measurement said otherwise once this pass started running on every auto-labelled frame: on 91 raw
# DASHCAM-01 frames from 30 sessions, with every candidate region judged by eye, the window pass at 0.35
# blurred 31 regions that were not plates (barricade lettering, bus route displays, railings, which inside
# an upscaled vehicle crop look like plate rectangles), and on an elevated junction view it blurred the same
# roadside railing frame after frame, which the floor alone does not stop (see the placement rule below).
# Held at 0.55 it blurred 7, and the frame as a whole still lost only 2 of the 112 readable plates it covered
# at 0.35 (107 to 105, two-line plates 14 of 17 either way). The whole-frame pass keeps the configured 0.35;
# only detections this pass adds are held higher.
#
# Losing two thirds of the face candidates is the right side to err on. This blur is permanent, a face
# missed today is still reachable by a later pass with better weights, and a face blurred in error is not.
_ROI_MIN_SCORE = {"face": 0.70, "plate": 0.55}

# How much of a plate found inside a crop must lie on SOME annotated vehicle or person on the frame, rather
# than merely inside the crop. The crop is margined because a plate straddles the edge of a tightly drawn
# vehicle box, and the margin is also where the worst false plates come from: on an elevated junction view in
# INDIA-CTC-01, every plate this pass found at 0.55 was the same roadside railing, sitting in a car's margin
# and on no annotated thing at all, frame after frame.
#
# Any annotated box rather than the window's own vehicle, and people as well as vehicles. Tied to the
# window's own vehicle, the rule lost a readable two-line plate on the 91 DASHCAM-01 frames: a scooter
# annotated only as its rider, whose plate was found in the margin of the bus beside it. Anchored on any
# vehicle or person, those frames keep all 105 readable plates and the railing hits on the six junction frames
# go from 7 to 0. Half rather than any overlap, because touching a box let 4 of the 7 back in. The 7
# non-plates the dashcam sample blurs at 0.55 are untouched: they are lettering and stickers on the vehicles
# themselves, which no placement rule can tell from a plate.
_PLATE_MIN_ON_ANNOTATION = 0.5

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


# The same window the export gate measures coverage in, by construction rather than by agreement.
_crop_window = crop_window


def _upscaled(crop: np.ndarray) -> tuple[np.ndarray, float]:
    """The crop enlarged to the detector's floor, and the factor used, so detections can be mapped back."""
    short_side = max(1, min(crop.shape[0], crop.shape[1]))
    scale = max(1.0, _UPSCALE_FLOOR_PX / short_side)
    if scale <= 1.0:
        return crop, 1.0
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), scale


async def _redaction_boxes(db: AsyncSession, frame_id: uuid.UUID) -> list[tuple[list[float], frozenset[str]]]:
    """Every annotated box on the frame that implies a redaction, with the detectors it implies.

    A three-wheeler carries both a rider's face and a plate, so a box can want more than one detector and
    the detectors are a set rather than a single kind.
    """
    rows = (await db.execute(
        select(Object.bbox, OntologyClass.l0, OntologyClass.l1)
        .join(OntologyClass, OntologyClass.id == Object.class_id)
        .where(Object.frame_id == frame_id))).all()
    out: list[tuple[list[float], frozenset[str]]] = []
    for bbox, l0, l1 in rows:
        if l0 != "object" or not bbox or len(bbox) < 4:
            continue
        wants = {d for d, groups in (("face", _FACE_GROUPS), ("plate", _PLATE_GROUPS)) if l1 in groups}
        if wants:
            out.append(([float(v) for v in bbox[:4]], frozenset(wants)))
    return out


def _windows(boxes: list[tuple[list[float], frozenset[str]]], w: int, h: int,
             ) -> list[tuple[tuple[int, int, int, int], frozenset[str]]]:
    out: list[tuple[tuple[int, int, int, int], frozenset[str]]] = []
    for bbox, wants in boxes:
        window = _crop_window(bbox, w, h)
        if window is not None:
            out.append((window, wants))
    return out


async def roi_windows(db: AsyncSession, frame_id: uuid.UUID, w: int, h: int,
                      ) -> list[tuple[tuple[int, int, int, int], frozenset[str]]]:
    """Crop windows for one frame, each with the detectors worth running inside it."""
    return _windows(await _redaction_boxes(db, frame_id), w, h)


def _detect_in_windows(anonymizer, img: np.ndarray,
                       windows: list[tuple[tuple[int, int, int, int], frozenset[str]]],
                       targets_by_detector: dict[str, str],
                       anchors: list[list[float]] | None = None) -> list[Candidate]:
    """Run each window's detectors on its upscaled crop, mapped back to full-frame coordinates.

    `anchors` are the frame's annotated vehicle and person boxes. Given, a plate must lie on one of them
    (`_PLATE_MIN_ON_ANNOTATION`); None skips that check, for a caller with no annotations to hold it to.
    """
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
                if kind == "plate" and anchors is not None and not any(
                        inside_fraction((fx1, fy1, fx2, fy2), a) >= _PLATE_MIN_ON_ANNOTATION for a in anchors):
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

    boxes = await _redaction_boxes(db, frame.frame_id)
    windows = _windows(boxes, w, h)
    candidates.extend(_detect_in_windows(anonymizer, img, windows, targets_by_detector,
                                         anchors=[b for b, _wants in boxes]))

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
