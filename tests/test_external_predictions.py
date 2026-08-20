"""A model this engine did not run had no way to be evaluated by it.

`run_inference` scores frames by downloading weights from the registry and running Ultralytics in process,
which covers models trained here and nothing else. A customer's model, or one behind their own serving
stack, or a checkpoint under a licence that forbids leaving their network, produced predictions this system
could not accept: not evaluable, not comparable against a champion, and invisible to failure mining.

Nothing in the schema prevented it. `weights_uri` is nullable, `conf` is nullable, and the reproducibility
key needs no reachable weights. Only the write path was missing, and these tests pin the two invariants it
has to keep: predictions stay append-only, and a run names the model it came from.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _frames(db, onto, n: int = 3):
    from db.models import Frame, OntologyClass, OntologyVersion
    from db.models import Session as DbSession

    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts, sid = now_ns(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="EXT-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()
    out = []
    for i in range(n):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + i * 1000, cam_id="cam_f",
                     img_uri=f"s3://x/{i}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        out.append(str(fid))
    await db.flush()
    return out


@pytest.mark.asyncio
async def test_a_run_must_name_a_registered_model():
    """An anonymous run produces a number nobody can act on."""
    from db.session import get_sessionmaker
    from services.verdyx.external_run import ingest_external_run

    async with get_sessionmaker()() as db:
        with pytest.raises(ValueError, match="not registered"):
            await ingest_external_run(db, model_version="nobody-knows", predictions=[])
        await db.rollback()


@pytest.mark.asyncio
async def test_registering_records_that_the_weights_are_elsewhere():
    """NULL already means "not uploaded yet" on internal rows; these two need telling apart."""
    from db.models import ModelRegistry
    from db.session import get_sessionmaker
    from services.verdyx.external_run import EXTERNAL_WEIGHTS, register_external_model

    async with get_sessionmaker()() as db:
        mv = f"customer-det-{uuid.uuid4().hex[:8]}"
        first = await register_external_model(db, model_version=mv)
        assert first["created"] is True and first["external"] is True
        assert (await db.get(ModelRegistry, mv)).weights_uri == EXTERNAL_WEIGHTS

        again = await register_external_model(db, model_version=mv)
        assert again["created"] is False, "registering twice must not duplicate or overwrite"
        await db.rollback()


@pytest.mark.asyncio
async def test_predictions_land_and_the_run_is_complete():
    from sqlalchemy import func, select

    from db.models import InferenceRun, Prediction
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.verdyx.external_run import ingest_external_run, register_external_model

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fids = await _frames(db, onto, 3)
        mv = f"customer-det-{uuid.uuid4().hex[:8]}"
        await register_external_model(db, model_version=mv)

        out = await ingest_external_run(db, model_version=mv, predictions=[
            {"frame_id": fids[0], "class_name": "bus", "bbox": [1, 2, 30, 40], "conf": 0.8},
            {"frame_id": fids[1], "class_name": "rider", "bbox": [5, 6, 20, 60], "conf": 0.4},
            {"frame_id": fids[2], "class_name": "cattle", "bbox": [7, 8, 50, 90]},   # no conf: allowed
        ])
        assert out["predictions"] == 3 and out["frames"] == 3
        run = await db.get(InferenceRun, uuid.UUID(out["run_id"]))
        assert run.status == "complete" and run.params["source"] == "external"
        n = (await db.execute(select(func.count()).select_from(Prediction)
                              .where(Prediction.run_id == run.run_id))).scalar_one()
        assert n == 3
        await db.rollback()


@pytest.mark.asyncio
async def test_a_missing_confidence_stays_missing():
    """The harness refuses AP on a run with no confidences rather than inventing one, and that only works
    if the absence survives ingestion."""
    from sqlalchemy import select

    from db.models import Prediction
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.verdyx.external_run import ingest_external_run, register_external_model

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fids = await _frames(db, onto, 1)
        mv = f"customer-det-{uuid.uuid4().hex[:8]}"
        await register_external_model(db, model_version=mv)
        out = await ingest_external_run(db, model_version=mv, predictions=[
            {"frame_id": fids[0], "class_name": "bus", "bbox": [1, 2, 30, 40]}])
        confs = (await db.execute(select(Prediction.conf)
                                  .where(Prediction.run_id == uuid.UUID(out["run_id"])))).scalars().all()
        assert confs == [None], "a zero here would be a fabricated score"
        await db.rollback()


@pytest.mark.asyncio
async def test_unknown_frames_and_classes_are_named_not_dropped():
    """A caller who submitted ten and stored eight must be able to find out which two, and why."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.verdyx.external_run import ingest_external_run, register_external_model

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fids = await _frames(db, onto, 1)
        ghost = str(uuid.uuid4())
        mv = f"customer-det-{uuid.uuid4().hex[:8]}"
        await register_external_model(db, model_version=mv)

        out = await ingest_external_run(db, model_version=mv, predictions=[
            {"frame_id": fids[0], "class_name": "bus", "bbox": [1, 2, 30, 40], "conf": 0.9},
            {"frame_id": ghost, "class_name": "bus", "bbox": [1, 2, 30, 40], "conf": 0.9},
            {"frame_id": fids[0], "class_name": "flying_saucer", "bbox": [1, 2, 3, 4], "conf": 0.9},
        ])
        assert out["predictions"] == 1
        assert out["unknown_frames"] == [ghost]
        assert out["unknown_classes"] == ["flying_saucer"]
        await db.rollback()


@pytest.mark.asyncio
async def test_resubmitting_creates_a_new_run_rather_than_mutating_the_old_one():
    """Predictions are append-only, so a published number stays reproducible after a resubmission."""
    from sqlalchemy import func, select

    from db.models import Prediction
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.verdyx.external_run import ingest_external_run, register_external_model

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        fids = await _frames(db, onto, 1)
        mv = f"customer-det-{uuid.uuid4().hex[:8]}"
        await register_external_model(db, model_version=mv)
        body = [{"frame_id": fids[0], "class_name": "bus", "bbox": [1, 2, 30, 40], "conf": 0.5}]

        a = await ingest_external_run(db, model_version=mv, predictions=body)
        b = await ingest_external_run(db, model_version=mv, predictions=body)
        assert a["run_id"] != b["run_id"]

        first = (await db.execute(select(func.count()).select_from(Prediction)
                                  .where(Prediction.run_id == uuid.UUID(a["run_id"])))).scalar_one()
        assert first == 1, "the earlier run must be untouched by the later one"
        await db.rollback()
