"""Eval harness + regression gate (Principle: no pipeline ships if gold metrics drop).

evaluate() runs a model on a dataset's val split and returns mAP plus per-class AP. regression_gate()
compares a candidate against a baseline and decides promote/reject: overall mAP must not drop and no
class may regress past a tolerance.

Honest note: until a human-verified gold set exists, val labels are the (noisy) auto-labels, so the
numbers measure agreement-with-corpus, not absolute accuracy. The harness is unchanged once gold
arrives: point the val split at human-reviewed frames (the builder already routes them there).
"""

from __future__ import annotations

from core.config import get_settings
from core.logging import get_logger

log = get_logger("eval")


def _names(names) -> dict[int, str]:
    return {int(k): str(v) for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}


def aligned_data_yaml(model_names, data_yaml: str, split: str = "val") -> tuple[str, dict]:
    """A data yaml whose class indices are the model's own, so its predictions are scored as what they are.

    Ultralytics matches a prediction to a label by class INDEX. A model trained on its own class list and a
    dataset built from the ontology disagree on what index 3 means: the champion knows 15 classes in its own
    order, the training set carries all 156 in ontology order, and only index 0 named the same class in both.
    Its baseline therefore scored mAP50 0.0001 against a candidate's 0.05, and the training gate read every
    candidate as an improvement over a model that was in fact far better.

    When the two lists differ, the split's labels are rewritten into the model's indices by class NAME, in a
    sibling directory that links the same images. Labels of classes the model was never taught are dropped
    and counted: the model cannot be wrong about a class it does not have, and scoring it on its own classes
    is the comparison the gate needs. When the lists already agree, the original yaml is returned untouched.
    """
    import hashlib
    import os
    from pathlib import Path

    import yaml

    data_yaml_p = Path(data_yaml)
    data = yaml.safe_load(data_yaml_p.read_text())
    mnames, dnames = _names(model_names), _names(data["names"])
    info = {"remapped": False, "model_classes": len(mnames), "dataset_classes": len(dnames)}
    if mnames == dnames:
        return data_yaml, info

    by_name = {v: k for k, v in mnames.items()}
    remap = {d: by_name[n] for d, n in dnames.items() if n in by_name}
    root = Path(data.get("path") or data_yaml_p.parent)
    images = root / data[split] if not Path(data[split]).is_absolute() else Path(data[split])
    # Ultralytics' own rule (img2label_paths): the last /images/ segment becomes /labels/.
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    labels = Path(sb.join(str(images / "_").rsplit(sa, 1))).parent
    key = hashlib.sha256(repr(sorted(mnames.items())).encode()).hexdigest()[:12]
    out = root.parent / f"{root.name}__as-{key}"
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    # Each image is linked, not the directory: Ultralytics resolve()s the dataset paths, so a linked
    # directory resolves straight back to the original and its labels, the rewritten ones unread.
    for img in images.iterdir():
        dst = out / "images" / split / img.name
        if not dst.exists():
            try:
                os.link(img, dst)
            except OSError:
                dst.symlink_to(img.resolve())

    kept = dropped = 0
    for src in labels.glob("*.txt"):
        lines = []
        for line in src.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            m = remap.get(int(parts[0]))
            if m is None:
                dropped += 1
                continue
            lines.append(" ".join([str(m), *parts[1:]]))
            kept += 1
        (out / "labels" / split / src.name).write_text("\n".join(lines) + ("\n" if lines else ""))
    (out / "labels" / f"{split}.cache").unlink(missing_ok=True)
    out_yaml = out / "data.yaml"
    out_yaml.write_text(yaml.safe_dump({"path": str(out), "train": f"images/{split}", "val": f"images/{split}",
                                        split: f"images/{split}", "nc": len(mnames),
                                        "names": mnames}, sort_keys=False))
    info.update(remapped=True, shared_classes=len(remap), labels_kept=kept, labels_dropped=dropped)
    log.info("eval.class_space_aligned", **info)
    return str(out_yaml), info


def evaluate(weights: str, data_yaml: str, split: str = "val", imgsz: int = 960) -> dict:
    from ultralytics import YOLO

    device = get_settings().gpu.device
    model = YOLO(weights)
    data_yaml, class_space = aligned_data_yaml(model.names, data_yaml, split)
    res = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, verbose=False, plots=False)

    per_class: dict[str, float] = {}
    per_class_pr: dict[str, dict] = {}
    try:
        names = res.names if hasattr(res, "names") else model.names
        ap50 = res.box.ap50  # per-evaluated-class AP@50, aligned to ap_class_index by position
        p = res.box.p        # per-class precision, same alignment
        r = res.box.r        # per-class recall, same alignment
        for i, ci in enumerate(res.box.ap_class_index):
            name = names[int(ci)]
            per_class[name] = round(float(ap50[i]), 4) if i < len(ap50) else 0.0
            per_class_pr[name] = {
                "precision": round(float(p[i]), 4) if i < len(p) else 0.0,
                "recall": round(float(r[i]), 4) if i < len(r) else 0.0,
                "ap50": per_class[name],
            }
    except Exception as exc:  # noqa: BLE001
        log.warning("eval.per_class_failed", error=str(exc))

    out = {
        "map50": round(float(res.box.map50), 4),
        "map": round(float(res.box.map), 4),
        "precision": round(float(res.box.mp), 4),
        "recall": round(float(res.box.mr), 4),
        "per_class": per_class,
        "per_class_pr": per_class_pr,
        # Flat class_name -> recall, so the champion recall gate (fail-closed on safety-class recall)
        # can read it directly from gold_metrics.
        "per_class_recall": {name: pr["recall"] for name, pr in per_class_pr.items()},
        # Which classes these numbers cover; the gate reads it to compare like with like.
        "class_space": class_space,
    }
    # Segmentation models also report mask metrics; absent for detect models.
    if getattr(res, "seg", None) is not None:
        try:
            out["mask_map50"] = round(float(res.seg.map50), 4)
            out["mask_map"] = round(float(res.seg.map), 4)
        except Exception:  # noqa: BLE001
            pass
    return out


def safe_miou_report(weights: str, data_yaml: str, split: str = "val", imgsz: int = 960) -> dict:
    """Run a val pass and score the class confusion matrix with the ontology-derived Safe-mIoU, so
    unsafe confusions (pedestrian->pole) are penalized far harder than benign ones (sedan->hatchback)."""
    from ultralytics import YOLO

    from services.autolabel.ontology import get_ontology
    from services.training.safe_miou import BACKGROUND_ID, safe_miou

    settings = get_settings()
    onto = get_ontology()
    model = YOLO(weights)
    data_yaml, _ = aligned_data_yaml(model.names, data_yaml, split)
    # plots=True is required: ultralytics only accumulates the confusion matrix when plotting is on,
    # otherwise res.confusion_matrix.matrix is all zeros and Safe-mIoU reads as undefined.
    res = model.val(data=data_yaml, split=split, imgsz=imgsz, device=settings.gpu.device, verbose=False, plots=True)

    names = dict(res.names if hasattr(res, "names") else model.names)
    nc = len(names)
    # map each model class index to an ontology id (gold split names are ontology names)
    class_ids: list[int] = []
    for i in range(nc):
        try:
            class_ids.append(onto.by_name(names[i]).id)
        except Exception:  # noqa: BLE001
            class_ids.append(-1)

    try:
        matrix = res.confusion_matrix.matrix  # (nc+1, nc+1), last index is background
        # Keep the background row/col so a missed VRU (true class predicted nothing) and a false positive
        # (background predicted as a class) are scored, not discarded. Dropping them is exactly why a weak
        # detector that misses everything read as undefined; the spec wants "pedestrian versus background".
        k = matrix.shape[0]
        ids_bg = (class_ids + [BACKGROUND_ID])[:k]
        score = safe_miou(matrix, ids_bg, onto, settings.m9.safety_weight)
        offdiag = float(matrix.sum() - sum(matrix[i][i] for i in range(k)))
    except Exception as exc:  # noqa: BLE001
        log.warning("safe_miou.failed", error=str(exc))
        return {"safe_miou": None, "n_classes": nc, "offdiag_mass": None}

    return {"safe_miou": round(score, 4) if score is not None else None, "n_classes": nc,
            "offdiag_mass": offdiag, "safety_weight": settings.m9.safety_weight}


def regression_gate(
    candidate: dict, baseline: dict, min_map_delta: float = 0.0, max_class_drop: float = 0.15,
    min_safe_miou: float | None = None,
) -> dict:
    reasons: list[str] = []
    # mAP50 is a mean over whichever classes a model was scored on. A 15-class champion and a 156-class
    # candidate average over different sets, so their headline numbers are not comparable; when the scored
    # sets differ, both are re-averaged over the classes they share. Identical sets keep the headline figure.
    base_pc, cand_pc = baseline.get("per_class") or {}, candidate.get("per_class") or {}
    compared_on = "all classes"
    cand_m, base_m = candidate["map50"], baseline["map50"]
    if set(base_pc) != set(cand_pc) and base_pc and cand_pc:
        shared = sorted(set(base_pc) & set(cand_pc))
        if not shared:
            reasons.append("candidate and baseline share no scored class, so they cannot be compared")
        else:
            cand_m = sum(cand_pc[c] for c in shared) / len(shared)
            base_m = sum(base_pc[c] for c in shared) / len(shared)
            compared_on = f"{len(shared)} shared classes"
    delta = round(cand_m - base_m, 4)
    if delta < min_map_delta:
        reasons.append(f"map50 delta {delta} on {compared_on} < required {min_map_delta}")

    # Optional Safe-mIoU floor (off by default: gating on a noisy first gold set would block
    # legitimate promotions). When set, the candidate must not confuse VRUs unsafely below the floor.
    if min_safe_miou is not None:
        sm = candidate.get("safe_miou")
        # Fail-closed: when a floor is set, a MISSING Safe-mIoU cannot clear it. Treating None as a pass let a
        # candidate that never scored safety through the gate the operator explicitly asked to enforce.
        if sm is None:
            reasons.append(f"safe_miou required (floor {min_safe_miou}) but not measured")
        elif sm < min_safe_miou:
            reasons.append(f"safe_miou {sm} < required {min_safe_miou}")

    # Only classes present in BOTH vocabularies are comparable. A class the candidate's vocabulary
    # does not contain (e.g. a COCO baseline's 'bicycle' vs the ontology's 'cycle', or 'person' vs
    # 'pedestrian') is a naming difference, not a regression, so it is skipped.
    regressed = []
    cand_classes = candidate.get("per_class", {})
    for cls, base_ap in baseline.get("per_class", {}).items():
        if cls not in cand_classes:
            continue
        cand_ap = cand_classes[cls]
        if base_ap - cand_ap > max_class_drop:
            regressed.append({"class": cls, "from": base_ap, "to": cand_ap})
    if regressed:
        reasons.append(f"{len(regressed)} classes regressed past {max_class_drop}")

    promote = len(reasons) == 0

    # What this verdict did not look at.
    #
    # A gate that returns promote=true beside a Safe-mIoU of 0.0 reads as approval in spite of the safety
    # score, when in fact safety was never evaluated: the floor is off by default for the documented reason
    # above. That was observed on a real run and is a reasonable thing for a reader to misread, because the
    # verdict says what passed and never says what was skipped.
    #
    # So the omission is reported rather than left silent. Not by forcing the floor on, which would defeat
    # the deliberate default, but by naming the criterion that was not applied and surfacing the measured
    # value beside it. A Safe-mIoU near zero means the candidate confuses vulnerable road users with
    # everything else, and that is worth seeing whether or not somebody configured a threshold for it.
    unchecked: list[str] = []
    measured_sm = candidate.get("safe_miou")
    if min_safe_miou is None:
        if measured_sm is None:
            unchecked.append("safe_miou: not measured and no floor set")
        else:
            unchecked.append(
                f"safe_miou: measured {measured_sm} but no floor set, so it did not affect this verdict"
                + (" (a value this low means the candidate confuses safety classes; set "
                   "govern.min_safe_miou to gate on it)" if measured_sm < 0.5 else ""))

    return {
        "promote": promote,
        "map50_delta": delta,
        "compared_on": compared_on,
        "regressed_classes": regressed,
        "reasons": reasons or ["passes: mAP improved, no class regressed past tolerance"],
        # Always present, so a caller cannot mistake "nothing to report" for "everything was checked".
        "unchecked": unchecked,
        "safe_miou": measured_sm,
    }
