"""Native 3D detection for real LiDAR: CenterPoint and PV-RCNN++ via OpenPCDet, BEVFusion for camera-LiDAR.
These are trained on dense real LiDAR and degrade on pseudo-LiDAR, so config gates them to real-LiDAR sources
and they run as the lidar_perception A100 burst job. OpenPCDet plus spconv are CUDA and dense-LiDAR bound and
are not installed on the interactive box; this is the real integration plus a fail-loud seam (never a fake
executor), exactly like the training cloud seam. Native class names map through the governed ontology, and an
unsupported class lands in the typed object fallback, never invented.
"""

from __future__ import annotations

from core.config import get_settings
from core.logging import get_logger
from services.autolabel.ontology import Ontology, get_ontology
from services.lidar.ingest.normalize import Cloud

log = get_logger("lidar_native3d")


class NativeDetectionUnavailable(RuntimeError):
    """Raised when OpenPCDet is not present locally. The lidar_perception job catches this and queues the
    work for the A100 burst node rather than pretending to detect."""


# nuScenes and KITTI detector class names to the project ontology; unknown lands in the object fallback.
_NATIVE_TO_ONTOLOGY = {
    "car": "sedan", "truck": "truck", "bus": "bus", "trailer": "truck", "construction_vehicle": "truck",
    "pedestrian": "pedestrian", "motorcycle": "motorcycle", "bicycle": "cycle",
    "traffic_cone": "cone", "barrier": "barrier",
    "Car": "sedan", "Pedestrian": "pedestrian", "Cyclist": "cycle", "Van": "sedan", "Truck": "truck",
}


def _object_fallback_id(onto: Ontology) -> int:
    """The typed object fallback by name, so a native detection lands in the object catch-all (not the
    vehicle fallback). Falls back to any fallback class, raising if the ontology has none."""
    for name in ("object_fallback", "object"):
        try:
            return onto.by_name(name).id
        except Exception:
            continue
    fids = onto.fallback_ids()
    if not fids:
        raise ValueError(f"ontology {onto.version} has no fallback class; cannot govern an unsupported class")
    return fids[0]


def native_class_to_ontology(name: str, onto: Ontology | None = None) -> int:
    """Map a native detector class name to an ontology class id, falling back to the typed object fallback so
    native 3D detection can never invent an unsupported class (the same discipline as the 2D YOLO path)."""
    onto = onto or get_ontology()
    mapped = _NATIVE_TO_ONTOLOGY.get(name)
    if mapped:
        try:
            return onto.by_name(mapped).id
        except Exception:
            pass
    return _object_fallback_id(onto)


def native_available() -> bool:
    try:
        import pcdet  # noqa: F401
        return True
    except Exception:
        return False


def detect_native(cloud: Cloud, model_name: str | None = None, ckpt: str | None = None) -> list[dict]:
    """Run native 3D detection on a cloud. Returns oriented cuboids with ontology class ids. Requires
    OpenPCDet on the burst node; raises NativeDetectionUnavailable on the interactive box."""
    cfg = get_settings().lidar
    model_name = model_name or cfg.native_detector
    ckpt = ckpt or cfg.native_ckpt
    if not native_available():
        raise NativeDetectionUnavailable(
            f"native 3D detection ({model_name}, {ckpt}) needs OpenPCDet + spconv on the A100 burst node. "
            "Run via the lidar_perception job; the local worker will not execute it.")

    import numpy as np  # local import: only on the burst node where the framework is present
    import torch
    from pcdet.config import cfg as pcdet_cfg
    from pcdet.config import cfg_from_yaml_file
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    onto = get_ontology()
    cfg_from_yaml_file(_model_cfg_path(model_name), pcdet_cfg)
    logger = common_utils.create_logger()
    net = build_network(model_cfg=pcdet_cfg.MODEL, num_class=len(pcdet_cfg.CLASS_NAMES), dataset=None)
    net.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
    net.cuda().eval()

    pts = np.concatenate([cloud.xyz, cloud.intensity.reshape(-1, 1)], axis=1).astype(np.float32)
    batch = {"points": np.concatenate([np.zeros((len(pts), 1), np.float32), pts], axis=1), "batch_size": 1}
    load_data_to_gpu(batch)
    with torch.no_grad():
        pred_dicts, _ = net.forward(batch)
    boxes = pred_dicts[0]["pred_boxes"].cpu().numpy()      # [x, y, z, l, w, h, yaw]
    scores = pred_dicts[0]["pred_scores"].cpu().numpy()
    labels = pred_dicts[0]["pred_labels"].cpu().numpy()
    names = pcdet_cfg.CLASS_NAMES
    out = []
    for b, s, lab in zip(boxes, scores, labels, strict=False):
        name = names[int(lab) - 1] if 0 < int(lab) <= len(names) else "unknown"
        out.append({"center": [float(b[0]), float(b[1]), float(b[2])],
                    "dims": [float(b[3]), float(b[4]), float(b[5])], "yaw": float(b[6]),
                    "pitch": 0.0, "roll": 0.0, "conf": float(s), "box_source": "native",
                    "class_id": native_class_to_ontology(name, onto), "native_class": name})
    log.info("lidar.native_detect", model=model_name, boxes=len(out))
    return out


def _model_cfg_path(model_name: str) -> str:
    return f"OpenPCDet/tools/cfgs/nuscenes_models/cbgs_{model_name}.yaml"
