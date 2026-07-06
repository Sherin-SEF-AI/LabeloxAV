"""LabeloxAV learned three-path reconciler (M13). The heuristic fusion of YOLO26 + SAM 3.1 + Qwen3-VL stays
the default and live; this adds a learned fusion head over the same per-detection features, kept behind the
labelox.learned_reconcile flag and blocked from switchover by a parity gate: the learned head must meet or
beat the heuristic on a held-out set before it is promoted. The learned model is capability-gated (it trains
only when labeled fusion outcomes exist) and never fabricated; with the flag off or no model, reconcile()
returns the heuristic result unchanged.

reconcile_features() and reconcile_parity() are pure, so the feature extraction and the promotion gate are
testable without a trained model."""

from __future__ import annotations

import importlib.util

from core.config import get_settings
from core.logging import get_logger

log = get_logger("labelox.learned_reconcile")


def reconcile_features(paths: list[dict]) -> list[float]:
    """The per-detection features the learned head fuses: per-path confidences, class agreement, box IoU
    spread, mask/box agreement, and VLM agreement. paths is the list of the three auto-label path outputs for
    one candidate object, each {path, class_id, conf, bbox, mask_iou, vlm_agree}."""
    confs = [float(p.get("conf", 0.0)) for p in paths]
    classes = [p.get("class_id") for p in paths if p.get("class_id") is not None]
    class_agree = (len(set(classes)) == 1) if classes else False
    from services.oraclyx.consensus import iou
    boxes = [p.get("bbox") for p in paths if p.get("bbox")]
    ious = [iou(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))]
    box_agree = (sum(ious) / len(ious)) if ious else 0.0
    mask_iou = max((float(p.get("mask_iou", 0.0)) for p in paths), default=0.0)
    vlm_agree = 1.0 if any(p.get("vlm_agree") for p in paths) else 0.0
    return [
        max(confs) if confs else 0.0, min(confs) if confs else 0.0, sum(confs) / len(confs) if confs else 0.0,
        1.0 if class_agree else 0.0, box_agree, mask_iou, vlm_agree, float(len(paths)),
    ]


def reconcile_parity(learned_metrics: dict, heuristic_metrics: dict, margin: float | None = None,
                     key: str = "map50") -> dict:
    """The promotion gate: the learned reconciler may replace the heuristic only if it meets or beats it by at
    least the configured margin on the held-out set. Fail-closed."""
    m = margin if margin is not None else get_settings().labelox.reconcile_parity_margin
    lh = float(learned_metrics.get(key, 0.0))
    hh = float(heuristic_metrics.get(key, 0.0))
    delta = round(lh - hh, 5)
    promote = delta >= m
    return {"promote": promote, "delta": delta, "learned": lh, "heuristic": hh,
            "reason": (f"learned {lh:.3f} >= heuristic {hh:.3f} + {m}") if promote
            else f"learned {lh:.3f} does not clear heuristic {hh:.3f} + {m}"}


class LearnedReconciler:
    """Serving-side reconciler. Active only when the flag is on AND a trained model is present; otherwise it is
    a pass-through that defers to the heuristic fuser, so nothing changes until parity is proven."""

    def __init__(self, model=None):
        self.model = model
        self.enabled = get_settings().labelox.learned_reconcile and model is not None

    def reconcile(self, paths: list[dict], heuristic_result: dict) -> dict:
        if not self.enabled:
            return heuristic_result
        feats = reconcile_features(paths)
        prob = float(self.model.predict_proba([feats])[0][1])
        out = dict(heuristic_result)
        out["conf"] = round(prob, 4)
        out["reconciler"] = "learned"
        return out


def train(labeled: list[tuple[list[dict], int]]):
    """Fit the learned fusion head over (paths, accepted) outcomes. Capability-gated: raises without
    scikit-learn or with too few labels, rather than fabricating a model."""
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError("learned reconciler needs scikit-learn; the heuristic fuser is the default")
    if len(labeled) < 50:
        raise RuntimeError(f"need >= 50 labeled fusion outcomes to train, got {len(labeled)}")
    from sklearn.linear_model import LogisticRegression

    X = [reconcile_features(paths) for paths, _ in labeled]
    y = [int(acc) for _, acc in labeled]
    return LogisticRegression(max_iter=1000).fit(X, y)
