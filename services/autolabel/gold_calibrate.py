"""Gold-based confidence calibration (the M-E calibration leg, done right).

The confidence gate is only trustworthy if a calibrated 0.95 means ~95% empirical precision. Fitting that curve
needs an UNBIASED sample of (predicted_confidence, was_it_correct) across the whole confidence range. The stored
human-review trail cannot supply it: reviewers accepted almost every auto-label (223 accept / 6 reject in this
corpus), so a curve fit on it collapses to a near-constant ~0.97 and would wrongly auto-accept low-confidence
detections. The leftover fused predictions on gold frames are the mirror bias (almost all rejected).

The correct source is the frozen gold set itself. This module re-runs the serving detector over the gold frames,
matches each prediction to the human-verified truth by class and IoU (greedy, one truth per prediction), and
labels it correct or a false positive. That yields real negatives at every confidence level. The curve is fit on
a frame-split TRAIN fold and validated on a held-out fold (frame-level split, so no prediction leaks across
folds), and it is only activated when the held-out reliability says it actually calibrates.
"""

from __future__ import annotations

import json

import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, GoldSet, Object
from db.session import get_sessionmaker
from services.autolabel.isotonic import reliability_report

log = get_logger("autolabel.gold_calibrate")


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _match_frame(dets: list[dict], truth: list[dict], iou_thr: float) -> list[tuple[float, int]]:
    """Greedy detection-to-truth match within one frame. Each detection becomes (conf, correct): correct when
    it matches an as-yet-unclaimed truth box of the same class at IoU >= threshold, else a false positive."""
    pairs: list[tuple[float, int]] = []
    used = [False] * len(truth)
    for d in sorted(dets, key=lambda x: -x["conf"]):
        best_j, best_iou = -1, iou_thr
        for j, t in enumerate(truth):
            if used[j] or t["class_id"] != d["class_id"]:
                continue
            i = _iou(d["bbox"], t["bbox"])
            if i >= best_iou:
                best_iou, best_j = i, j
        if best_j >= 0:
            used[best_j] = True
            pairs.append((d["conf"], 1))
        else:
            pairs.append((d["conf"], 0))
    return pairs


async def collect_gold_pairs(gold_id: str, iou_thr: float = 0.5, weights: str | None = None) -> dict:
    """Run the serving detector over the gold frames and label every prediction correct or false-positive
    against the human-verified truth. Returns per-frame (conf, correct) pairs plus the detector used."""
    from services.autolabel.paths.path_a_detect import YoloPath
    from services.autolabel.runner import load_image, resolve_detector_weights

    maker = get_sessionmaker()
    async with maker() as db:
        gs = await db.get(GoldSet, gold_id)
        if gs is None:
            raise RuntimeError(f"gold set {gold_id} not found")
        truth_rows = (await db.execute(
            select(Object.frame_id, Object.class_id, Object.bbox)
            .where(Object.object_id.in_(gs.object_ids or [])))).all()
        truth_by_frame: dict = {}
        for fid, cid, bbox in truth_rows:
            truth_by_frame.setdefault(fid, []).append({"class_id": int(cid), "bbox": [float(x) for x in bbox]})
        frame_rows = (await db.execute(
            select(Frame.frame_id, Frame.img_uri).where(Frame.frame_id.in_(list(truth_by_frame))))).all()
        if weights is None:
            weights = await resolve_detector_weights(db)

    path = YoloPath(weights=weights)
    path.load()
    per_frame: dict = {}
    n_det = 0
    for fid, img_uri in frame_rows:
        try:
            img = load_image(img_uri)
        except Exception as exc:  # noqa: BLE001
            log.warning("gold_cal.image_failed", frame=str(fid), error=str(exc))
            continue
        dets = [{"class_id": int(d.class_id), "bbox": [float(x) for x in d.bbox], "conf": float(d.conf)}
                for d in path.infer(img)]
        n_det += len(dets)
        per_frame[str(fid)] = _match_frame(dets, truth_by_frame.get(fid, []), iou_thr)
    path.unload()
    log.info("gold_cal.collected", gold_id=gold_id, frames=len(per_frame), detections=n_det, weights=weights)
    return {"gold_id": gold_id, "weights": weights, "iou_thr": iou_thr, "per_frame": per_frame}


async def collect_vlm_pairs(n_frames: int = 220, max_dets: int = 320, seed: int = 7,
                            weights: str | None = None, margin: float = 0.15,
                            agree_level: str = "class") -> dict:
    """Build per-detection calibration pairs with the VLM as judge. Confidence calibration only needs
    P(correct | confidence) PER DETECTION, not complete-frame truth, so missed objects are irrelevant and no
    exhaustive human labeling is required. For each detection the VLM is shown the crop and a shortlist (the
    detector's class, a few confusable road classes, and a none option); the detection is correct when the VLM
    confidently returns the detector's class. This yields real negatives across the whole confidence range."""
    from services.autolabel.ontology import get_ontology
    from services.autolabel.paths.path_a_detect import YoloPath
    from services.autolabel.paths.path_c_vlm import OllamaVlmClient, crop_object
    from services.autolabel.runner import load_image, resolve_detector_weights

    onto = get_ontology()
    confusable = ["sedan", "hatchback", "suv", "autorickshaw", "motorcycle", "truck", "bus", "cycle",
                  "pedestrian", "cart", "animal", "traffic_sign", "pole"]
    # name -> top-level super-class (l0). agree_level="class" scores exact-class correctness (the trustworthy
    # default, and the granularity the stored labels use); "l0" scores super-class correctness (looser: a
    # sedan/hatchback mix-up counts as agreement). l0 tends to be degenerate here because the detector nearly
    # always finds SOME object of the right super-type, so the harness's trustworthiness gate rejects it.
    name_to_l0 = {c.name.strip().lower(): c.l0 for c in onto.classes}

    def _agrees(vlm_name: str, det_name: str) -> bool:
        if not vlm_name:
            return False
        if vlm_name == det_name:
            return True
        if agree_level == "l0":
            a, b = name_to_l0.get(vlm_name), name_to_l0.get(det_name)
            return a is not None and a == b
        return False

    maker = get_sessionmaker()
    async with maker() as db:
        rng = np.random.default_rng(seed)
        ids = [r[0] for r in (await db.execute(select(Frame.frame_id).where(Frame.img_uri.isnot(None)))).all()]
        rng.shuffle(ids)
        ids = ids[:n_frames]
        frame_rows = (await db.execute(
            select(Frame.frame_id, Frame.img_uri).where(Frame.frame_id.in_(ids)))).all()
        if weights is None:
            weights = await resolve_detector_weights(db)

    path = YoloPath(weights=weights)
    path.load()
    vlm = OllamaVlmClient()
    per_frame: dict = {}
    n_judged = 0
    for fid, img_uri in frame_rows:
        if n_judged >= max_dets:
            break
        try:
            img = load_image(img_uri)
        except Exception:  # noqa: BLE001
            continue
        pairs: list[tuple[float, int]] = []
        for d in path.infer(img):
            if n_judged >= max_dets:
                break
            det_name = d.class_name
            shortlist = list(dict.fromkeys([det_name, *confusable, "none_of_these"]))
            crop = crop_object(img, tuple(float(x) for x in d.bbox), margin)
            res = vlm.verify(crop, shortlist, {})
            n_judged += 1
            # correct when the VLM agrees the crop is the detector's class at the super-class granularity the
            # gate cares about. The shortlist carries a none_of_these escape, so this is a real yes/no on "is
            # there a <this kind of> object here", not a forced pick.
            vlm_name = (res.class_name or "").strip().lower()
            correct = 1 if _agrees(vlm_name, det_name.strip().lower()) else 0
            pairs.append((float(d.conf), correct))
        if pairs:
            per_frame[str(fid)] = pairs
    path.unload()
    npos = sum(y for ps in per_frame.values() for _, y in ps)
    log.info("vlm_cal.collected", frames=len(per_frame), detections=n_judged, positives=npos, weights=weights)
    return {"source": "vlm_judge", "weights": weights, "per_frame": per_frame}


def _ece(pred: np.ndarray, ys: np.ndarray, bins: int = 10) -> float:
    e, n = 0.0, len(ys)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (pred >= lo) & (pred < hi) if b < bins - 1 else (pred >= lo) & (pred <= hi)
        if m.sum():
            e += abs(float(pred[m].mean()) - float(ys[m].mean())) * int(m.sum()) / n
    return float(e)


def _fit_report(collected: dict, *, tag: str, storage_key: str, val_frac: float, seed: int) -> dict:
    """Shared fit + held-out validation + storage for any per-frame (conf, correct) collection. Splits by frame
    (so no detection leaks across folds), fits isotonic on train, validates on the held-out fold, gates on
    trustworthiness, then refits the DEPLOYED curve on all pairs. Does not activate."""
    from sklearn.isotonic import IsotonicRegression

    per_frame = collected["per_frame"]
    frames = sorted(per_frame)
    rng = np.random.default_rng(seed)
    rng.shuffle(frames)
    n_val = max(1, int(len(frames) * val_frac))
    val_frames = set(frames[:n_val])

    def fold(want_val: bool):
        xs, ys = [], []
        for f, pairs in per_frame.items():
            if (f in val_frames) == want_val:
                for c, y in pairs:
                    xs.append(c)
                    ys.append(y)
        return np.asarray(xs, float), np.asarray(ys, float)

    xtr, ytr = fold(False)
    xva, yva = fold(True)
    if len(xtr) < 20 or len(xva) < 10:
        raise RuntimeError(f"too few predictions to calibrate (train={len(xtr)}, val={len(xva)})")

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(xtr, ytr)
    kx = np.asarray(iso.X_thresholds_, float)
    ky = np.asarray(iso.y_thresholds_, float)

    bins = get_settings().m9.ece_bins
    raw_val_ece = _ece(xva, yva, bins)                 # ECE of the RAW detector score on val
    cal_val = np.clip(iso.predict(xva), 0.0, 1.0)
    cal_val_ece = _ece(cal_val, yva, bins)             # ECE after applying the fitted curve on val
    val_report = reliability_report(xva, yva, kx, ky, bins)

    pos_val, neg_val = int(yva.sum()), int(len(yva) - yva.sum())
    non_degenerate = pos_val >= 3 and neg_val >= 3 and float(ky.max() - ky.min()) >= 0.1
    improves = cal_val_ece <= raw_val_ece + 1e-6
    trustworthy = bool(non_degenerate and improves)

    # held-out proved it calibrates; the DEPLOYED curve is refit on all pairs to use every prediction.
    xall = np.concatenate([xtr, xva])
    yall = np.concatenate([ytr, yva])
    iso_all = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_all.fit(xall, yall)
    kxa = np.asarray(iso_all.X_thresholds_, float)
    kya = np.asarray(iso_all.y_thresholds_, float)

    payload = {
        "kind": "isotonic-knots-v1", "source": collected.get("source", tag),
        "x": [round(float(v), 6) for v in kxa], "y": [round(float(v), 6) for v in kya],
        "n_train": int(len(xall)), "weights": collected.get("weights"), "val_ece": round(cal_val_ece, 6),
    }
    uri = get_object_store().put_bytes(storage_key, json.dumps(payload).encode(), "application/json")

    curve = {r: round(float(np.clip(iso_all.predict([r])[0], 0, 1)), 3) for r in (0.2, 0.4, 0.6, 0.8, 0.9, 0.95)}
    report = {
        "source": collected.get("source", tag), "uri": uri, "weights": collected.get("weights"),
        "n_frames": len(frames), "n_train_pred": int(len(xtr)), "n_val_pred": int(len(xva)),
        "val_positives": pos_val, "val_negatives": neg_val,
        "raw_val_ece": round(raw_val_ece, 4), "calibrated_val_ece": round(cal_val_ece, 4),
        "val_reliability": val_report, "curve": curve,
        "non_degenerate": non_degenerate, "improves_over_raw": improves, "trustworthy": trustworthy,
    }
    log.info("cal.fitted", tag=tag, n_train=len(xtr), n_val=len(xva),
             raw_val_ece=report["raw_val_ece"], cal_val_ece=report["calibrated_val_ece"], trustworthy=trustworthy)
    return report


async def fit_gold_calibration(gold_id: str, val_frac: float = 0.3, iou_thr: float = 0.5,
                               weights: str | None = None, seed: int = 7) -> dict:
    """Calibrate against a gold set by matching detections to its human-verified truth. NOTE: only valid when
    the gold set is EXHAUSTIVELY labeled per frame; a capped/sampled gold set over-counts false positives."""
    collected = await collect_gold_pairs(gold_id, iou_thr, weights)
    rep = _fit_report(collected, tag="gold_inference", storage_key=f"calibration/{gold_id}/isotonic_gold.json",
                      val_frac=val_frac, seed=seed)
    rep["gold_id"] = gold_id
    return rep


async def fit_vlm_calibration(n_frames: int = 220, max_dets: int = 320, val_frac: float = 0.3,
                              weights: str | None = None, seed: int = 7, agree_level: str = "class") -> dict:
    """Calibrate the detector's confidence using the VLM as a per-detection judge (no exhaustive labels needed).
    Fits + validates on a frame-level split and stores the curve. Does not activate; check trustworthy first."""
    collected = await collect_vlm_pairs(n_frames, max_dets, seed, weights, agree_level=agree_level)
    return _fit_report(collected, tag="vlm_judge", storage_key="calibration/vlm_judge/isotonic_vlm.json",
                       val_frac=val_frac, seed=seed)


def activate_calibration(uri: str) -> dict:
    """Point the live gate at a fitted curve by writing the calibrate block in configs/default.yaml
    (method: isotonic + isotonic_uri). The gate reads settings.calibrate, so only that block matters. Only call
    after fit_gold_calibration reports trustworthy=True. A backend restart applies it."""
    from pathlib import Path

    cfg_path = Path("configs/default.yaml")
    text = cfg_path.read_text()
    lines = text.splitlines()
    out, in_block = [], False
    changed = {"method": False, "isotonic_uri": False}
    for ln in lines:
        stripped = ln.strip()
        if stripped == "calibrate:":
            in_block = True
        elif in_block and ln and not ln[0].isspace() and stripped:
            in_block = False  # a new top-level key ends the calibrate block
        indent = ln[: len(ln) - len(ln.lstrip())]
        if in_block and stripped.startswith("method:"):
            out.append(f"{indent}method: isotonic")
            changed["method"] = True
            continue
        if in_block and stripped.startswith("isotonic_uri:"):
            out.append(f"{indent}isotonic_uri: {uri}")
            changed["isotonic_uri"] = True
            continue
        out.append(ln)
    if not (changed["method"] and changed["isotonic_uri"]):
        raise RuntimeError(f"could not locate calibrate.method / isotonic_uri to activate (changed={changed})")
    cfg_path.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""))
    log.info("gold_cal.activated", uri=uri)
    return {"activated": True, "uri": uri, "note": "restart the backend to apply"}
