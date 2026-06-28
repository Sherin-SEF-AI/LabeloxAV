"""The Label Fusion Engine. The component almost nobody builds, and where Indian-road accuracy is
won (Principle 03: never trust one detector). Per frame it:

1. Matches detections across paths by IoU/IoM and centroid distance into candidate clusters.
2. Votes on class with per-path, per-superclass reliability priors, respecting the hierarchy.
3. Reconciles geometry: prefer the SAM mask, derive the box from it; flag box-vs-mask disagreement.
4. Calibrates raw scores plus the agreement signal into one trustworthy confidence.
5. Assembles provenance: every proposal, what agreed, what was overruled, and each model version.

Output is one UnifiedObject per cluster, pre-gate, carrying its mask for persistence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.config import Settings, get_settings
from core.logging import get_logger
from core.schemas import BBox, MaskEncoding, ObjectSource, PathProposal, Provenance, UnifiedObject
from services.autolabel.calibrate import calibrate_confidence
from services.autolabel.ontology import Ontology, get_ontology
from services.autolabel.paths.base import RawDetection, mask_to_bbox

log = get_logger("fusion")


@dataclass
class FusedObject:
    obj: UnifiedObject
    mask: np.ndarray | None


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iom(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    m = min(area_a, area_b)
    return inter / m if m > 0 else 0.0


def _centroid_dist(a: tuple, b: tuple) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return float(np.hypot(acx - bcx, acy - bcy))


class FusionEngine:
    def __init__(self, settings: Settings | None = None, ontology: Ontology | None = None) -> None:
        self.settings = settings or get_settings()
        self.onto = ontology or get_ontology()
        self.cfg = self.settings.fusion

    def _matches(self, da: RawDetection, db: RawDetection) -> bool:
        iou = _iou(da.bbox, db.bbox)
        if iou >= self.cfg.iou_match:
            return True
        if _iom(da.bbox, db.bbox) >= self.cfg.iom_match:
            return True
        return _centroid_dist(da.bbox, db.bbox) <= self.cfg.centroid_px and iou > 0.0

    def _cluster(self, dets_a: list[RawDetection], dets_b: list[RawDetection]) -> list[list[RawDetection]]:
        """Greedy one-to-one matching between the two paths; leftovers are singletons."""
        clusters: list[list[RawDetection]] = []
        used_b: set[int] = set()

        for da in dets_a:
            best_j, best_iou = -1, 0.0
            for j, db in enumerate(dets_b):
                if j in used_b or not self._matches(da, db):
                    continue
                iou = _iou(da.bbox, db.bbox)
                if iou >= best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                used_b.add(best_j)
                clusters.append([da, dets_b[best_j]])
            else:
                clusters.append([da])

        for j, db in enumerate(dets_b):
            if j not in used_b:
                clusters.append([db])
        return clusters

    def _prior(self, path: str, class_id: int) -> float:
        l1 = self.onto.by_id(class_id).l1
        table = self.cfg.class_priors.get(path, {})
        return float(table.get(l1, table.get("default", 0.5)))

    def _vote(self, cluster: list[RawDetection]) -> tuple[int, float, bool]:
        """Return (winning_class_id, raw_conf, class_disagreement)."""
        weights: dict[int, float] = {}
        per_path_class: dict[str, int] = {}
        for d in cluster:
            if d.class_id is None:
                continue
            w = self._prior(d.path, d.class_id) * d.conf
            weights[d.class_id] = weights.get(d.class_id, 0.0) + w
            # highest-conf class for each path
            if d.path not in per_path_class or d.conf > 0:
                per_path_class.setdefault(d.path, d.class_id)

        if not weights:
            # No class anywhere: fallback.
            fb = self.onto.by_name("object_fallback").id
            return fb, max((d.conf for d in cluster), default=0.3), False

        winner = max(weights, key=weights.get)
        raw = max((d.conf for d in cluster if d.class_id == winner), default=0.3)
        distinct = {c for c in per_path_class.values()}
        class_disagreement = len(distinct) > 1
        return winner, raw, class_disagreement

    def _geometry(self, cluster: list[RawDetection]) -> tuple[tuple, np.ndarray | None, bool]:
        """Return (bbox xyxy, mask or None, mask_box_disagree)."""
        b_masks = [d for d in cluster if d.mask is not None]
        a_boxes = [d for d in cluster if d.path == "path_a_yolo26"]

        if b_masks:
            best = max(b_masks, key=lambda d: d.conf)
            mb = mask_to_bbox(best.mask)
            box = mb if mb is not None else best.bbox
            mask = best.mask
        else:
            best = max(cluster, key=lambda d: d.conf)
            box = best.bbox
            mask = None

        disagree = False
        if a_boxes and (b_masks or len(cluster) > 1):
            other = max(a_boxes, key=lambda d: d.conf)
            if _iou(other.bbox, box) < self.cfg.mask_box_disagree_iou:
                disagree = True
        return box, mask, disagree

    def fuse_frame(
        self, frame_id, dets_a: list[RawDetection], dets_b: list[RawDetection]
    ) -> list[FusedObject]:
        clusters = self._cluster(dets_a, dets_b)
        out: list[FusedObject] = []

        for cluster in clusters:
            class_id, raw_conf, class_disagreement = self._vote(cluster)
            box, mask, mask_box_disagree = self._geometry(cluster)
            paths_present = {d.path for d in cluster}
            agreement = len(paths_present) >= 2 and not class_disagreement

            conf = calibrate_confidence(
                raw_conf=raw_conf,
                agreement=agreement,
                class_disagreement=class_disagreement,
                mask_box_disagree=mask_box_disagree,
                cfg=self.settings.calibrate,
            )

            class_name = self.onto.by_id(class_id).name
            provenance = self._provenance(
                cluster, class_id, agreement, mask_box_disagree, raw_conf
            )

            obj = UnifiedObject(
                frame_id=frame_id,
                class_id=class_id,
                class_name=class_name,
                bbox=BBox.from_list(list(box)),
                mask_encoding=MaskEncoding.polygon if mask is not None else None,
                attrs={},
                conf=conf,
                source=ObjectSource.fused,
                provenance=provenance,
            )
            out.append(FusedObject(obj=obj, mask=mask))
        return out

    def _provenance(
        self,
        cluster: list[RawDetection],
        winner: int,
        agreement: bool,
        mask_box_disagree: bool,
        raw_conf: float,
    ) -> Provenance:
        proposals: list[PathProposal] = []
        raw: dict[str, float] = {}
        for d in cluster:
            verdict = "agree" if d.class_id == winner else "overruled"
            proposals.append(
                PathProposal(
                    path=d.path,
                    class_name=d.class_name,
                    conf=d.conf,
                    verdict=verdict,
                    model_version=d.model_version,
                )
            )
            raw[d.path] = d.conf
        return Provenance(
            proposals=proposals,
            agreement=agreement,
            mask_box_disagree=mask_box_disagree,
            raw_conf=raw,
            calibrated_from=raw_conf,
            ontology_version=self.onto.version,
        )
