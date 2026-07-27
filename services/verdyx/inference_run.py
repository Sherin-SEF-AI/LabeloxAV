"""The writer for the immutable prediction plane (db/models.py InferenceRun / Prediction).

A named model scores a set of frames once and every detection it produced is persisted verbatim under one run.
This is the fix for the measurement defect where "predictions" were read from live corpus rows: human review
mutates those rows in place, so a confirmed-correct detection was erased from the prediction population and the
harness scored only the residue. Inference writes here and never touches Object; review writes Object and never
touches a Prediction.

Two deliberate choices carry the reasoning:
* Inference runs at a LOW confidence floor (0.001), not the auto-accept threshold. Evaluation needs the whole
  score distribution to compute a PR curve and choose an operating point; gating belongs at scoring time.
* A run is keyed by (model_version, gold_id, code_sha, params). A complete run with the identical key is reused
  rather than recomputed, so the same evaluation is reproducible and de-duplicated against drifting state.

Degrades to None with a structured warning when the model weights are unavailable (no GPU / no registry row),
mirroring services/govern/gold_eval.py so unit tests without a model do not crash.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.version import code_sha
from db.models import Frame, InferenceRun, ModelRegistry, Prediction

log = get_logger("inference_run")

_BATCH = 16


def _load_weights_and_names(weights_uri: str, local_path: str) -> tuple[str, list[str]]:
    """Download the weights once and read the model's class order. Blocking (network + torch): worker thread.
    Same download pattern as services/govern/gold_eval.py::_load_weights_and_names."""
    from ultralytics import YOLO

    from core.storage import get_object_store

    if not Path(local_path).exists():
        Path(local_path).write_bytes(get_object_store().get_bytes(weights_uri))
    names = YOLO(local_path).names
    names_list = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    return local_path, names_list


def _infer(local_weights: str, images: list[np.ndarray], imgsz: int, conf_floor: float, device) -> list[list]:
    """Predict on a batch of decoded BGR frames. Blocking GPU work (worker thread). Returns, per frame, a list
    of (model_class_idx, conf, [x1,y1,x2,y2]) at the low conf floor so the full distribution is captured."""
    from ultralytics import YOLO

    model = YOLO(local_weights)
    res = model.predict(images, imgsz=imgsz, conf=conf_floor, device=device, verbose=False)
    out: list[list] = []
    for r in res:
        dets = []
        for b in r.boxes:
            dets.append((int(b.cls[0]), float(b.conf[0]), [float(v) for v in b.xyxy[0].tolist()]))
        out.append(dets)
    return out


def _run_params(imgsz: int, conf_floor: float, device) -> dict:
    from packs.registry import default_pack_id
    from services.autolabel.ontology import get_ontology

    return {"imgsz": int(imgsz), "conf_floor": float(conf_floor), "device": str(device),
            "pack_id": default_pack_id(), "ontology_version": get_ontology().version}


async def run_inference(db: AsyncSession, *, model_version: str, frame_ids: list[UUID],
                        gold_id: str | None = None, imgsz: int | None = None,
                        conf_floor: float = 0.001, force: bool = False) -> str | None:
    """Score `frame_ids` with `model_version`, writing one InferenceRun and its Prediction rows. Returns the
    run_id, or None when the model has no downloadable weights (nothing to score)."""
    reg = await db.get(ModelRegistry, model_version)
    if reg is None or not reg.weights_uri:
        log.warning("inference.no_weights", model_version=model_version)
        return None

    settings = get_settings()
    imgsz = int(imgsz or getattr(settings.training, "eval_imgsz", 960))
    device = settings.gpu.device
    sha = code_sha()
    params = _run_params(imgsz, conf_floor, device)

    # Idempotent by (model_version, gold_id, code_sha, params): reuse a complete run with the identical key.
    if not force:
        rows = (await db.execute(select(InferenceRun).where(
            InferenceRun.model_version == model_version, InferenceRun.gold_id == gold_id,
            InferenceRun.code_sha == sha, InferenceRun.status == "complete"))).scalars().all()
        for r in rows:
            if r.params == params:
                log.info("inference.reused", run_id=str(r.run_id), model_version=model_version, gold_id=gold_id)
                return str(r.run_id)

    run = InferenceRun(model_version=model_version, gold_id=gold_id, params=params, code_sha=sha,
                       status="running", frame_count=len(frame_ids))
    db.add(run)
    await db.commit()
    run_id = run.run_id

    try:
        loop = asyncio.get_event_loop()
        scratch = settings.scratch_path() / "inference"
        scratch.mkdir(parents=True, exist_ok=True)
        local = str(scratch / f"{model_version}.pt")
        local, names_list = await loop.run_in_executor(None, _load_weights_and_names, reg.weights_uri, local)

        from services.training.gold import align_model_to_ontology
        idx_to_onto = align_model_to_ontology(names_list)

        from core.storage import get_object_store
        store = get_object_store()

        total = 0
        for i in range(0, len(frame_ids), _BATCH):
            batch = frame_ids[i:i + _BATCH]
            frames = (await db.execute(select(Frame).where(Frame.frame_id.in_(batch)))).scalars().all()
            images, fids = [], []
            for fr in frames:
                try:
                    img = cv2.imdecode(np.frombuffer(store.get_bytes(fr.img_uri), np.uint8), cv2.IMREAD_COLOR)
                except Exception:  # noqa: BLE001 - a missing blob skips one frame, never the run
                    img = None
                if img is None:
                    continue
                images.append(img)
                fids.append(fr.frame_id)
            if not images:
                continue
            dets_per_frame = await loop.run_in_executor(None, _infer, local, images, imgsz, conf_floor, device)
            for fid, dets in zip(fids, dets_per_frame, strict=False):
                for cls_idx, conf, box in dets:
                    onto_id = idx_to_onto[cls_idx] if cls_idx < len(idx_to_onto) else None
                    if onto_id is None:
                        continue  # the model emitted a class the ontology does not have; not comparable
                    db.add(Prediction(run_id=run_id, frame_id=fid, class_id=onto_id, bbox=box, conf=conf))
                    total += 1
            await db.commit()

        run = await db.get(InferenceRun, run_id)
        run.status = "complete"
        run.finished_at = datetime.now(UTC)
        await db.commit()
        log.info("inference.complete", run_id=str(run_id), model_version=model_version,
                 frames=len(frame_ids), predictions=total)
        return str(run_id)
    except Exception as exc:  # noqa: BLE001 - record the failure on the run, never leave it "running"
        run = await db.get(InferenceRun, run_id)
        if run is not None:
            run.status = "failed"
            run.params = {**(run.params or {}), "error": type(exc).__name__ + ": " + str(exc)[:200]}
            run.finished_at = datetime.now(UTC)
            await db.commit()
        log.warning("inference.failed", run_id=str(run_id), error=str(exc))
        return None


async def run_inference_on_gold(db: AsyncSession, model_version: str, gold_id: str, force: bool = False) -> str | None:
    """Convenience: score a registered model on every frame of a sealed gold set."""
    from db.models import GoldSet, Object

    g = await db.get(GoldSet, gold_id)
    if g is None:
        raise RuntimeError(f"gold set {gold_id} not found")
    fids = (await db.execute(
        select(Object.frame_id).where(Object.object_id.in_(g.object_ids)).distinct())).scalars().all()
    return await run_inference(db, model_version=model_version, frame_ids=list(fids), gold_id=gold_id, force=force)


def main() -> None:
    import argparse

    from core.logging import setup_logging
    from db.session import get_sessionmaker

    ap = argparse.ArgumentParser(description="Run a model over a gold set and persist its predictions.")
    ap.add_argument("--model", required=True, help="model_version in the registry")
    ap.add_argument("--gold", required=True, help="gold_id to score")
    ap.add_argument("--force", action="store_true", help="recompute even if a matching complete run exists")
    args = ap.parse_args()
    setup_logging(get_settings().log_level)

    async def _run() -> None:
        async with get_sessionmaker()() as db:
            run_id = await run_inference_on_gold(db, args.model, args.gold, force=args.force)
        print({"run_id": run_id, "model": args.model, "gold": args.gold})

    asyncio.run(_run())


if __name__ == "__main__":
    main()
