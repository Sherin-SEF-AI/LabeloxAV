"""Take predictions from a model this engine did not run.

`run_inference` scores frames by downloading weights from the registry and running Ultralytics in process.
That covers models trained here and nothing else. A customer's model, a model behind their own serving
stack, a checkpoint under a licence that forbids leaving their network: all produce predictions this system
had no way to accept, which meant they could not be evaluated, could not be compared against a champion, and
could not feed failure mining.

The schema was always ready for it. `weights_uri` is nullable, `conf` is nullable, and the reproducibility
key is (model_version, gold_id, code_sha, params), none of which requires the weights to be reachable from
here. Only the write path was missing.

Two invariants are kept exactly as the internal path keeps them.

Predictions are append-only. `db/models.py` states that no code path outside the inference runner may update
or delete a Prediction, and this is now a second such path, so it only ever inserts. A caller resubmitting
the same run gets a new run id rather than a mutated one, and the old numbers stay reproducible.

A run is attributable. Scoring an anonymous model produces a number nobody can act on, so an external run
must name a registered model, and registering one records that its weights live elsewhere rather than
pretending they are absent by accident.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, InferenceRun, ModelRegistry, Prediction
from services.autolabel.ontology import get_ontology

log = get_logger("verdyx.external_run")

# Marks a registry entry whose weights are not reachable from here. Recorded rather than left NULL, because
# NULL already means "we have not got round to it" on internal rows and the two need telling apart: one is
# a model waiting to be uploaded, the other is a model that will never be.
EXTERNAL_WEIGHTS = "external://caller-hosted"

MAX_PREDICTIONS_PER_RUN = 500_000


async def register_external_model(db: AsyncSession, *, model_version: str, task: str = "detection",
                                  notes: str | None = None) -> dict:
    """Record a model whose weights this system cannot fetch, so its runs are attributable."""
    existing = await db.get(ModelRegistry, model_version)
    if existing is not None:
        return {"model_version": model_version, "created": False,
                "external": existing.weights_uri == EXTERNAL_WEIGHTS}
    db.add(ModelRegistry(model_version=model_version, task=task, weights_uri=EXTERNAL_WEIGHTS,
                         notes=notes or "registered for external prediction ingestion", gold_metrics={}))
    await db.commit()
    log.info("verdyx.external_model_registered", model_version=model_version, task=task)
    return {"model_version": model_version, "created": True, "external": True}


async def ingest_external_run(db: AsyncSession, *, model_version: str, predictions: list[dict],
                              gold_id: str | None = None, code_sha: str = "external",
                              params: dict | None = None) -> dict:
    """Write one InferenceRun and its Predictions from a caller-supplied list.

    Each prediction needs a frame_id, a class name or id, and a bbox; `conf` is optional because a model that
    reports no score is still worth evaluating at a single operating point, and the harness already refuses
    to compute AP on a run whose confidences are absent rather than inventing them.
    """
    reg = await db.get(ModelRegistry, model_version)
    if reg is None:
        raise ValueError(f"model '{model_version}' is not registered; register it first so the run is "
                         "attributable to something")
    if len(predictions) > MAX_PREDICTIONS_PER_RUN:
        raise ValueError(f"{len(predictions)} predictions exceeds the {MAX_PREDICTIONS_PER_RUN} cap for one "
                         "run; split it")

    onto = get_ontology()

    # Frames are resolved up front so an unknown one is reported rather than silently dropped: a run scored
    # against frames that are not in the corpus is not comparable with one that was.
    wanted = {str(p["frame_id"]) for p in predictions if p.get("frame_id")}
    known = {str(f) for f in (await db.execute(
        select(Frame.frame_id).where(Frame.frame_id.in_([UUID(w) for w in wanted])))).scalars().all()}
    unknown = sorted(wanted - known)

    run = InferenceRun(model_version=model_version, gold_id=gold_id,
                       params={"source": "external", **(params or {})},
                       code_sha=code_sha, status="running", frame_count=len(known))
    db.add(run)
    await db.commit()

    written = 0
    bad_class: list[str] = []
    for p in predictions:
        fid = str(p.get("frame_id") or "")
        if fid not in known:
            continue
        cid = p.get("class_id")
        if cid is None:
            name = p.get("class_name")
            if not name or not onto.has_name(str(name)):
                bad_class.append(str(name))
                continue
            cid = onto.by_name(str(name)).id
        bbox = p.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        db.add(Prediction(
            run_id=run.run_id, frame_id=UUID(fid), class_id=int(cid),
            bbox=[float(v) for v in bbox],
            # Left as given, including absent. The harness refuses AP on a run with no confidences rather
            # than substituting one, which is the behaviour that keeps a reconstructed run honest.
            conf=(float(p["conf"]) if p.get("conf") is not None else None),
            track_id=(str(p["track_id"]) if p.get("track_id") else None),
            rot_deg=p.get("rot_deg"), cuboid_3d=p.get("cuboid_3d")))
        written += 1

    run.status = "complete"
    await db.commit()

    log.info("verdyx.external_run", run_id=str(run.run_id), model_version=model_version,
             written=written, frames=len(known), unknown_frames=len(unknown), bad_class=len(bad_class))
    return {"run_id": str(run.run_id), "model_version": model_version, "predictions": written,
            "frames": len(known),
            # Named rather than dropped, for the same reason bulk review names its skips: a caller that
            # submitted 10,000 and stored 9,850 has to be able to find out which 150 and why.
            "unknown_frames": unknown[:50], "unknown_frame_count": len(unknown),
            "unknown_classes": sorted(set(bad_class))[:50]}
