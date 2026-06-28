"""3D proposals through the existing fusion, confidence gate, and governed ontology. A 3D cuboid is wrapped
as a UnifiedObject and passed through the SAME gate_object as 2D, so the fallback and safety-critical
discipline carry into 3D unchanged: a rare or fallback class forces review, the kill switch suppresses
auto-accept, and 3D detection can never invent an unsupported class (lifted inherits the governed 2D class;
native maps through the ontology to a typed fallback).

The 3D calibrated confidence is the 2D calibrated confidence modulated by the cuboid fit quality (how densely
the frustum fills the fitted footprint); native proposals carry the detector's own score.
"""

from __future__ import annotations

import numpy as np

from core.config import get_settings
from core.logging import get_logger
from core.schemas import BBox, GateState, ObjectSource, PathProposal, Provenance, UnifiedObject
from services.autolabel.gate import gate_object, is_rare
from services.autolabel.ontology import Ontology, get_ontology

log = get_logger("lidar_fuse3d")


def gate_cuboid(cuboid: dict, *, class_id: int, conf_2d: float, frame_id, box_source: str,
                bbox_2d: list[float] | None = None, agreement_2d: bool = True, track_id=None,
                model_version: str = "lift-3d-0.1", calibration_version: str | None = None,
                auto_accept_enabled: bool = True, onto: Ontology | None = None) -> dict:
    """Gate one 3D proposal. Returns the calibrated 3D confidence, the gate state, the inherited class, and a
    one-walk provenance. The cuboid's class is NEVER changed here: governance happens on the inherited or
    mapped ontology class."""
    onto = onto or get_ontology()
    cfg = get_settings()
    cls = onto.by_id(class_id)

    if box_source == "native":
        conf3d = float(np.clip(cuboid.get("conf", conf_2d), 0.0, 1.0))
        agreement = False                       # a single native head; no cross-path agreement
    else:
        fit = float(cuboid.get("fill", 0.5))
        conf3d = float(np.clip(conf_2d * (0.7 + 0.3 * fit), 0.0, 1.0))   # fit-modulated 2D confidence
        agreement = agreement_2d                # lifted inherits the agreement of its governed 2D object

    bb = bbox_2d or [0.0, 0.0, 1.0, 1.0]
    prov = Provenance(
        proposals=[PathProposal(path=f"{box_source}_3d", class_name=cls.name, conf=conf3d,
                                verdict="proposed", model_version=model_version)],
        agreement=agreement, mask_box_disagree=False, raw_conf={f"{box_source}_3d": conf3d},
        calibrated_from=conf_2d, ontology_version=onto.version,
        notes=[f"box_source={box_source}", f"n_points={cuboid.get('n_points')}", f"fill={cuboid.get('fill')}",
               f"calibration_version={calibration_version}", f"model_version={model_version}",
               f"track_id={track_id}"])
    uo = UnifiedObject(frame_id=frame_id, track_id=track_id, class_id=class_id, class_name=cls.name,
                       bbox=BBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]), conf=conf3d,
                       source=ObjectSource.fused, provenance=prov)
    state: GateState = gate_object(uo, onto, cfg.gate, auto_accept_enabled=auto_accept_enabled)

    out = {"class_id": class_id, "class_name": cls.name, "conf": conf3d, "state": state.value,
           "box_source": box_source, "is_rare": is_rare(class_id, onto), "is_fallback": onto.is_fallback(class_id),
           "agreement": agreement, "provenance": prov.model_dump(mode="json")}
    log.info("lidar.gate3d", cls=cls.name, conf=round(conf3d, 3), state=state.value, source=box_source,
             rare=out["is_rare"])
    return out
