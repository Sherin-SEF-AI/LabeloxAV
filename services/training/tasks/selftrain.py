"""Detector self-training: learn from what the machine already agrees on, then correct on what people ruled.

The corpus has 443 human-accepted objects on real frames and hundreds of thousands of machine detections.
Training only on the human labels throws away almost everything the fleet has seen; training on raw
machine detections teaches the model its own mistakes back. Self-training is the middle: take the labels
a teacher is confident about, weight them by how confident, learn from those, and then spend the human
labels on a short corrective pass where they count for everything.

**Where the soft targets come from, and why there is a ladder.** The designed source is
`oraclyx.export_distillation`, the consensus of several auto-label paths. That table is empty on this
corpus and will stay empty until something batches `record_consensus`, which today runs one object at a
time from a router call. So the source falls back to what the corpus does have: the prediction plane
holds several independently trained models scored over the same frames, and where two of them put the
same box on the same class, that is a consensus too. The run records which rung answered, because a label
from two agreeing models and a label from a fused multi-path consensus are not the same evidence.

**Two stages, in this order.** Stage one is the large pseudo-labelled set with a soft-target weighted
loss. Stage two is the human labels alone at a low learning rate. The order matters: ending on the human
pass means the last thing the weights saw is the only thing in the dataset that a person ruled on, and
the pseudo-labels act as the prior rather than the conclusion.

The candidate goes through the unchanged champion gate. Nothing here promotes anything.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml as _yaml

from core.config import get_settings
from core.logging import get_logger
from core.origin import REAL
from core.storage import get_object_store
from services.autolabel.ontology import get_ontology
from services.training.tasks.base import ProgressFn, register

log = get_logger("task_selftrain")

# A pseudo-label below this is not evidence, it is a guess with a number attached. The product of two
# models' confidences, so 0.30 means roughly 0.55 from each.
SOFT_TARGET_FLOOR = 0.30
# Below this many pseudo-labels a self-training stage cannot teach anything the human pass would not,
# and the honest answer is a refusal with the count.
MIN_PSEUDO_LABELS = 200
# Stage two runs at a fraction of stage one's rate: it is a correction, not a retraining.
STAGE2_LR_SCALE = 0.1
SOFT_SIDECAR = "soft_targets.json"
# How many recent complete runs to consider when looking for the best-overlapping pair of models.
CANDIDATE_RUNS = 20
# Dataloader workers. Lower than the detection task's 8 because self-training runs two trainings back to
# back inside one process, and each one tears down its worker pool while the next builds another.
DATALOADER_WORKERS = 4


def soft_target_weight(soft_targets: list[float], floor: float = SOFT_TARGET_FLOOR) -> float:
    """The scalar a batch's box loss is multiplied by, from the soft targets of its instances.

    The mean of the batch's targets, so a batch of confident pseudo-labels counts close to full and a
    batch of marginal ones counts for less. An empty batch scores 1.0 rather than 0.0: a background image
    carries real information (the 92.4%-background lesson in `dataset_builder`) and zeroing its loss would
    teach the model to ignore exactly the frames where it currently hallucinates.
    """
    kept = [float(t) for t in soft_targets if float(t) >= floor]
    if not kept:
        return 1.0
    return float(sum(kept) / len(kept))


async def gather_pseudo_labels(*, limit: int = 200000, conf_floor: float = SOFT_TARGET_FLOOR) -> dict:
    """Soft-target boxes from the best consensus source this corpus has, and the name of that source.

    Tries the designed source first and falls through, so a deployment that does batch `record_consensus`
    gets the stronger evidence automatically without a code change here.
    """
    from sqlalchemy import func, select

    from db.models import InferenceRun, Prediction
    from db.session import get_sessionmaker
    from services.oraclyx.run import export_distillation
    from services.verdyx.shadow_run import agreements_for_runs

    async with get_sessionmaker()() as db:
        consensus = await export_distillation(db, min_score=conf_floor, limit=limit)
        if consensus["n"] >= MIN_PSEUDO_LABELS:
            return {"source": "oraclyx_consensus", "n": consensus["n"],
                    "manifest": consensus["manifest"]}

        # Two runs from two different models, chosen by how many frames they have in common. Two runs
        # of one model agree with themselves and would launder a single model's confidence as consensus,
        # and two runs that scored different frames cannot agree on anything however large they are:
        # picking by prediction count alone chose a pair overlapping on 4 frames and produced 9 labels.
        rows = (await db.execute(
            select(InferenceRun.run_id, InferenceRun.model_version,
                   func.count(Prediction.prediction_id).label("n"))
            .join(Prediction, Prediction.run_id == InferenceRun.run_id)
            .where(InferenceRun.status == "complete")
            .group_by(InferenceRun.run_id, InferenceRun.model_version)
            .order_by(func.count(Prediction.prediction_id).desc()).limit(CANDIDATE_RUNS))).all()
        if len({mv for _rid, mv, _n in rows}) < 2:
            return {"source": None, "n": 0, "manifest": [],
                    "reason": ("no consensus is available: the oraclyx pseudo-label table is empty and "
                               "fewer than two models have scored this corpus, so nothing can agree "
                               "with anything")}
        frames_of: dict[uuid.UUID, set] = {}
        model_of: dict[uuid.UUID, str] = {}
        for rid, mv, _n in rows:
            model_of[rid] = mv
            frames_of[rid] = set((await db.execute(
                select(Prediction.frame_id).where(Prediction.run_id == rid).distinct())).scalars().all())
        pair, overlap = None, 0
        ids = list(frames_of)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if model_of[a] == model_of[b]:
                    continue
                n = len(frames_of[a] & frames_of[b])
                if n > overlap:
                    pair, overlap = (a, b), n
        if pair is None or overlap == 0:
            return {"source": None, "n": 0, "manifest": [],
                    "reason": ("two models have scored this corpus but never the same frames, so there "
                               "is nothing for them to agree or disagree about")}
        agree = await agreements_for_runs(db, run_a=pair[0], run_b=pair[1],
                                          conf_floor=conf_floor, limit=limit)
        if "error" in agree:
            return {"source": None, "n": 0, "manifest": [], "reason": agree["error"]}
        return {"source": "model_agreement", "n": agree["n"], "manifest": agree["manifest"],
                "teachers": [model_of[pair[0]], model_of[pair[1]]],
                "teacher_runs": [str(pair[0]), str(pair[1])], "shared_frames": overlap}


async def _human_labels() -> list[dict]:
    """Objects a person accepted, on real frames. The only labels stage two is allowed to see."""
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Object.object_id, Object.frame_id, Object.class_id, Object.bbox)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .join(DbSession, DbSession.session_id == Frame.session_id)
            .where(Object.source == "human", Object.state == "accepted",
                   Frame.origin == REAL, DbSession.origin == REAL))).all()
    return [{"object_id": str(oid), "frame_id": str(fid), "class_id": int(cid),
             "bbox": [float(v) for v in bbox], "soft_target": 1.0} for oid, fid, cid, bbox in rows]


def _write_yolo(records: list[dict], frames: dict[str, dict], out: Path, *, idx_of: dict[int, int],
                names: dict[int, str], val_frac: float, seed: int) -> dict:
    """Write a YOLO tree plus the soft-target sidecar, and return what it contains."""
    import random

    if out.exists():
        shutil.rmtree(out)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    by_frame: dict[str, list[dict]] = {}
    for r in records:
        by_frame.setdefault(r["frame_id"], []).append(r)

    # Split at the session boundary, as everywhere else here: consecutive dashcam frames are near
    # duplicates and a per-frame split leaks the training distribution into validation.
    sessions = sorted({frames[f]["session_id"] for f in by_frame if f in frames})
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_val_sessions = max(1, int(len(sessions) * val_frac)) if len(sessions) > 2 else 0
    val_sessions = set(sessions[:n_val_sessions])

    store = get_object_store()
    soft: dict[str, list[float]] = {}
    n_train = n_val = 0
    for fid, recs in by_frame.items():
        meta = frames.get(fid)
        if meta is None:
            continue
        split = "val" if meta["session_id"] in val_sessions else "train"
        try:
            buf = np.frombuffer(store.get_bytes(meta["img_uri"]), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001 - a missing blob skips one frame, never the build
            img = None
        if img is None:
            continue
        cv2.imwrite(str(out / f"images/{split}/{fid}.jpg"), img)
        w = max(1, int(meta["width"] or img.shape[1]))
        h = max(1, int(meta["height"] or img.shape[0]))
        lines, weights = [], []
        for r in recs:
            if r["class_id"] not in idx_of:
                continue
            x1, y1, x2, y2 = r["bbox"]
            cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
            if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                continue
            lines.append(f"{idx_of[r['class_id']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            weights.append(float(r["soft_target"]))
        (out / f"labels/{split}/{fid}.txt").write_text("\n".join(lines) + "\n")
        soft[fid] = weights
        if split == "train":
            n_train += 1
        else:
            n_val += 1

    (out / SOFT_SIDECAR).write_text(json.dumps(soft))
    data_yaml = out / "data.yaml"
    data_yaml.write_text(_yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val",
        "nc": len(names), "names": names}, sort_keys=False))
    return {"data_yaml": str(data_yaml), "dir": str(out), "n_train_images": n_train,
            "n_val_images": n_val, "classes": len(names),
            "n_objects": sum(len(v) for v in soft.values()),
            "val_sessions": len(val_sessions)}


async def _frame_meta(frame_ids: set[str]) -> dict[str, dict]:
    from sqlalchemy import select

    from db.models import Frame
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Frame.frame_id, Frame.img_uri, Frame.width, Frame.height, Frame.session_id)
            .where(Frame.frame_id.in_([uuid.UUID(f) for f in frame_ids])))).all()
    return {str(fid): {"img_uri": uri, "width": w, "height": h, "session_id": str(sid)}
            for fid, uri, w, h, sid in rows}


class SelfTrainTask:
    """Two-stage detector self-training. Same gate, same registry, same propose-by-approval path."""

    task_type = "selftrain"

    def default_base_weights(self) -> str:
        return get_settings().models.yolo.weights

    async def build_dataset(self, cfg: dict, progress: ProgressFn) -> dict:
        progress({"stage": "build"})
        name = cfg["name"]
        spec = dict(cfg.get("dataset_spec") or {})
        floor = float(spec.get("soft_target_floor") or SOFT_TARGET_FLOOR)
        val_frac = float(spec.get("val_frac") or 0.2)
        seed = int(spec.get("seed") or 7)

        pseudo = await gather_pseudo_labels(conf_floor=floor)
        if pseudo["n"] < MIN_PSEUDO_LABELS:
            raise ValueError(
                pseudo.get("reason")
                or (f"only {pseudo['n']} consensus pseudo-labels are available at a soft-target floor of "
                    f"{floor}, and self-training below {MIN_PSEUDO_LABELS} teaches nothing the human pass "
                    f"would not"))
        human = await _human_labels()

        onto = get_ontology()
        present = sorted({r["class_id"] for r in pseudo["manifest"]}
                         | {r["class_id"] for r in human})
        idx_of = {cid: i for i, cid in enumerate(present)}
        names = {i: onto.by_id(cid).name for cid, i in idx_of.items()}

        root = get_settings().scratch_path() / "training"
        meta = await _frame_meta({r["frame_id"] for r in pseudo["manifest"]}
                                 | {r["frame_id"] for r in human})
        stage1 = _write_yolo(pseudo["manifest"], meta, root / f"{name}-stage1",
                             idx_of=idx_of, names=names, val_frac=val_frac, seed=seed)
        stage2 = (_write_yolo(human, meta, root / f"{name}-stage2", idx_of=idx_of, names=names,
                              val_frac=val_frac, seed=seed) if human else None)

        log.info("selftrain.built", name=name, source=pseudo["source"], pseudo=pseudo["n"],
                 human=len(human), classes=len(names))
        return {
            "name": name, "dir": stage1["dir"], "data_yaml": stage1["data_yaml"],
            "classes": len(names),
            "n_train_images": stage1["n_train_images"], "n_val_images": stage1["n_val_images"],
            "n_train_objects": stage1["n_objects"], "n_val_objects": 0,
            "gold_frames": 0, "ontology_version": onto.version,
            "pseudo_source": pseudo["source"], "pseudo_labels": pseudo["n"],
            "teachers": pseudo.get("teachers"),
            "human_labels": len(human),
            # Null rather than absent when there are no human labels: a self-training run that never got
            # its corrective pass is a different thing from one that did, and the sheet has to say so.
            "stage2": stage2,
            "soft_target_floor": floor,
        }

    def train(self, data_yaml: str, base_weights: str, hparams: dict, progress: ProgressFn) -> str:
        from ultralytics import YOLO

        _use_file_system_sharing()
        settings = get_settings()
        name = hparams["name"]
        epochs = int(hparams.get("epochs", settings.training.default_epochs))
        imgsz = int(hparams.get("imgsz", settings.training.default_imgsz))
        batch = int(hparams.get("batch", settings.training.default_batch))
        project = str(settings.scratch_path() / "training" / "runs")
        should_stop = hparams.get("_should_stop")
        # The shared executor knows only the stage-one yaml it was handed, so stage two is found beside
        # it rather than plumbed through: `build_dataset` writes the two trees as siblings under one name.
        stage2_yaml = hparams.get("stage2_data_yaml")
        if not stage2_yaml:
            sibling = Path(str(data_yaml).replace("-stage1", "-stage2"))
            stage2_yaml = str(sibling) if sibling != Path(data_yaml) and sibling.exists() else None
        stage2_epochs = int(hparams.get("stage2_epochs", max(3, epochs // 4)))

        soft = _load_soft(Path(data_yaml).parent / SOFT_SIDECAR)

        def _progress_cb(stage: str, total: int):
            def _on_epoch(trainer):
                try:
                    ep = int(getattr(trainer, "epoch", 0)) + 1
                    m = getattr(trainer, "metrics", {}) or {}
                    progress({"stage": stage, "epoch": ep, "total_epochs": total,
                              "metrics": {"map50": round(float(m.get("metrics/mAP50(B)", 0.0)), 4)}})
                    if should_stop and should_stop():
                        trainer.stop = True
                except Exception:  # noqa: BLE001 - progress must never break training
                    pass
            return _on_epoch

        model = YOLO(base_weights)
        model.add_callback("on_fit_epoch_end", _progress_cb("train:pseudo", epochs))
        attached = attach_soft_target_loss(model, soft)
        progress({"stage": "train:pseudo", "soft_target_weighting": attached})
        model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=settings.gpu.device,
                    project=project, name=f"{name}-stage1", exist_ok=True, verbose=False, plots=False,
                    batch=batch, patience=max(5, epochs // 3), workers=DATALOADER_WORKERS, seed=7)
        stage1_weights = str(Path(project) / f"{name}-stage1" / "weights" / "best.pt")

        if not stage2_yaml:
            log.warning("selftrain.no_stage2", name=name)
            return stage1_weights

        # Stage two: the human labels alone, at a fraction of the rate. Ending here is the point, so the
        # last evidence the weights saw is the only evidence a person ruled on.
        m2 = YOLO(stage1_weights)
        m2.add_callback("on_fit_epoch_end", _progress_cb("train:human", stage2_epochs))
        lr = float(hparams.get("lr0", 0.01)) * STAGE2_LR_SCALE
        m2.train(data=stage2_yaml, epochs=stage2_epochs, imgsz=imgsz, device=settings.gpu.device,
                 project=project, name=f"{name}-stage2", exist_ok=True, verbose=False, plots=False,
                 batch=batch, patience=max(3, stage2_epochs // 2), workers=DATALOADER_WORKERS,
                 seed=7, lr0=lr)
        return str(Path(project) / f"{name}-stage2" / "weights" / "best.pt")

    def evaluate(self, weights: str, data_yaml: str, imgsz: int) -> dict:
        from services.training.tasks.detection import DetectionTask

        return DetectionTask().evaluate(weights, data_yaml, imgsz)

    def gate(self, candidate: dict, baseline: dict, criteria: dict) -> dict:
        from services.training.tasks.detection import DetectionTask

        return DetectionTask().gate(candidate, baseline, criteria)


def _use_file_system_sharing() -> None:
    """Share dataloader tensors through the filesystem rather than file descriptors.

    Torch's default passes each shared tensor as a file descriptor over a socket to the worker. Two
    trainings back to back in one process tear down and rebuild that pool, and the second one died in
    `rebuild_storage_fd` with a connection reset before the first epoch. The filesystem strategy has no
    descriptor handshake to lose, and `/dev/shm` on this host has 15 GB.
    """
    try:
        import torch.multiprocessing as mp

        if mp.get_sharing_strategy() != "file_system":
            mp.set_sharing_strategy("file_system")
    except Exception as exc:  # noqa: BLE001 - a strategy that cannot be set is not a failed training
        log.warning("selftrain.sharing_strategy_failed", error=str(exc)[:200])


def _load_soft(path: Path) -> dict[str, list[float]]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001 - no sidecar means unweighted, and the run says which it was
        return {}


def attach_soft_target_loss(model, soft: dict[str, list[float]]) -> bool:
    """Scale the detection loss by the batch's mean soft target, or report that it could not.

    Ultralytics exposes no per-instance loss weight, so this wraps the criterion the trainer builds. The
    return value matters more than the wrapping: a run that silently trained unweighted and a run that
    trained weighted produce different models and the same log line, so the caller records which happened
    rather than assuming.
    """
    if not soft:
        return False

    def _on_train_start(trainer):
        try:
            det = getattr(trainer.model, "criterion", None)
            if det is None or getattr(det, "_lbx_soft", False):
                return
            weights = soft

            original = det.__call__

            def wrapped(preds, batch):
                loss, items = original(preds, batch)
                stems = [Path(str(p)).stem for p in (batch.get("im_file") or [])]
                targets = [t for s in stems for t in weights.get(s, [])]
                w = soft_target_weight(targets)
                return loss * w, items

            det.__call__ = wrapped
            det._lbx_soft = True
        except Exception as exc:  # noqa: BLE001 - a failed wrap trains unweighted, and says so
            log.warning("selftrain.soft_target_attach_failed", error=str(exc)[:200])

    model.add_callback("on_train_start", _on_train_start)
    return True


register(SelfTrainTask())
