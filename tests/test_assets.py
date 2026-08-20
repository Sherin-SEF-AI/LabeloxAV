"""Multi-modal spine: label config validation, per-kind payload validation, and the asset/annotation store.

The payload validators are pure, so most of this runs without infra. The store tests need a DB.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from core.config import get_settings
from services.assets.labelconfig import (
    ConfigError,
    PayloadError,
    check_label_allowed,
    validate_config,
    validate_fields,
    validate_payload,
)

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _clear_db_cache():
    from db.session import get_engine, get_sessionmaker

    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


def run_async(coro):
    _clear_db_cache()
    try:
        return asyncio.run(coro)
    finally:
        _clear_db_cache()


# ---- pure validation ------------------------------------------------------------------------------------

def test_empty_config_is_legal_and_permissive():
    """An empty config means the project inherits the AV ontology, so adopting this layer stays opt-in and
    existing projects keep working."""
    assert validate_config(None) == {}
    assert validate_config({}) == {}
    check_label_allowed({}, "anything", "bbox")   # must not raise


def test_config_rejects_malformed_declarations():
    with pytest.raises(ConfigError):
        validate_config({"labels": [{"name": "a"}, {"name": "a"}]})          # duplicate
    with pytest.raises(ConfigError):
        validate_config({"labels": [{"name": "a", "kinds": ["nope"]}]})      # unknown kind
    with pytest.raises(ConfigError):
        validate_config({"fields": [{"name": "f", "type": "wat"}]})          # unknown type
    with pytest.raises(ConfigError):
        validate_config({"fields": [{"name": "f", "type": "enum"}]})         # enum without values
    # a string label is sugar for {"name": ...}
    assert validate_config({"labels": ["speech"]})["labels"][0]["name"] == "speech"


def test_payload_validation_per_kind():
    # bbox must be ordered
    assert validate_payload("bbox", {"bbox": [1, 2, 3, 4]})["bbox"] == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(PayloadError):
        validate_payload("bbox", {"bbox": [3, 2, 1, 4]})
    with pytest.raises(PayloadError):
        validate_payload("bbox", {"bbox": [1, 2, 3]})

    # span must be a well-ordered range
    assert validate_payload("span", {"start": 0, "end": 5})["end"] == 5
    with pytest.raises(PayloadError):
        validate_payload("span", {"start": 5, "end": 5})
    with pytest.raises(PayloadError):
        validate_payload("span", {"start": -1, "end": 3})

    # region is seconds, ordered
    assert validate_payload("region", {"t_start": 0.5, "t_end": 2.0})["t_start"] == 0.5
    with pytest.raises(PayloadError):
        validate_payload("region", {"t_start": 2.0, "t_end": 0.5})

    # polygon needs 3+ points, polyline 2+
    validate_payload("polygon", {"points": [[0, 0], [1, 0], [1, 1]]})
    with pytest.raises(PayloadError):
        validate_payload("polygon", {"points": [[0, 0], [1, 0]]})

    # preference must index into its candidates
    validate_payload("preference", {"candidates": ["a", "b"], "chosen": 1})
    with pytest.raises(PayloadError):
        validate_payload("preference", {"candidates": ["a", "b"], "chosen": 2})
    with pytest.raises(PayloadError):
        validate_payload("preference", {"candidates": ["a"], "chosen": 0})

    # ranking must be a strict order
    validate_payload("ranking", {"order": ["a", "b", "c"]})
    with pytest.raises(PayloadError):
        validate_payload("ranking", {"order": ["a", "a"]})

    # relation cannot be a self-loop
    with pytest.raises(PayloadError):
        validate_payload("relation", {"from_annotation_id": "x", "to_annotation_id": "x"})

    with pytest.raises(PayloadError):
        validate_payload("not_a_kind", {})


def test_fields_are_coerced_and_enum_enforced():
    cfg = {"fields": [{"name": "speaker", "type": "enum", "values": ["a", "b"]},
                      {"name": "conf", "type": "float"},
                      {"name": "n", "type": "int"},
                      {"name": "ok", "type": "bool"},
                      {"name": "must", "type": "text", "required": True}]}
    out = validate_fields(cfg, {"speaker": "a", "conf": "0.5", "n": "3", "ok": 1, "must": "x"})
    assert out["conf"] == 0.5 and out["n"] == 3 and out["ok"] is True
    with pytest.raises(PayloadError):
        validate_fields(cfg, {"speaker": "z", "must": "x"})       # not in the enum
    with pytest.raises(PayloadError):
        validate_fields(cfg, {"speaker": "a"})                     # required field missing


def test_label_must_be_declared_and_kind_restricted():
    cfg = {"labels": [{"name": "speech", "kinds": ["region"]}], "allow_kinds": ["region"]}
    check_label_allowed(cfg, "speech", "region")
    with pytest.raises(PayloadError):
        check_label_allowed(cfg, "speech", "bbox")        # label declared, wrong kind
    with pytest.raises(PayloadError):
        check_label_allowed(cfg, "music", "region")       # undeclared label
    with pytest.raises(PayloadError):
        check_label_allowed(cfg, None, "bbox")            # kind not allowed at all


# ---- store ----------------------------------------------------------------------------------------------

@requires_infra
def test_asset_import_is_idempotent_and_annotations_validate():
    from db.session import get_sessionmaker
    from services.assets import store
    from services.labelops.jobs import create_project

    async def run():
        pid = None
        try:
            async with get_sessionmaker()() as db:
                p = await create_project(db, name=f"mm-{uuid.uuid4().hex[:6]}", modality="text")
                pid = p["project_id"]

                # a text project with a declared schema
                from db.models import LabelProject
                proj = await db.get(LabelProject, uuid.UUID(pid))
                proj.label_config = validate_config({
                    "labels": [{"name": "person", "kinds": ["span"]}],
                    "fields": [{"name": "certainty", "type": "enum", "values": ["low", "high"]}],
                })
                await db.commit()

                body = "Ravi drove the autorickshaw through Bengaluru."
                r = await store.create_assets(db, pid, [
                    {"media_type": "text", "text": body, "external_id": "doc-1"}])
                assert r["created"] == 1
                aid = r["assets"][0]["asset_id"]

                # re-import the same external id: updates, does not duplicate
                r2 = await store.create_assets(db, pid, [
                    {"media_type": "text", "text": body, "external_id": "doc-1"}])
                assert r2["created"] == 0 and r2["updated"] == 1
                assert (await store.list_assets(db, pid))["total"] == 1

                # a valid span, with the quote captured from the body
                ann = await store.create_annotation(db, aid, kind="span", label="person",
                                                    payload={"start": 0, "end": 4},
                                                    fields={"certainty": "high"})
                assert ann["payload"]["quote"] == "Ravi", ann
                assert ann["fields"]["certainty"] == "high"

                # a span past the end of the text is refused: it would export as garbage
                with pytest.raises(PayloadError):
                    await store.create_annotation(db, aid, kind="span", label="person",
                                                  payload={"start": 0, "end": 9999})

                # an undeclared label is refused
                with pytest.raises(PayloadError):
                    await store.create_annotation(db, aid, kind="span", label="vehicle",
                                                  payload={"start": 0, "end": 4})

                # the declared label cannot be used as another kind
                with pytest.raises(PayloadError):
                    await store.create_annotation(db, aid, kind="bbox", label="person",
                                                  payload={"bbox": [0, 0, 1, 1]})

                # creating an annotation moves the asset off "new"
                got = await store.get_asset(db, aid)
                assert got["state"] == "in_progress"
                assert len(got["annotations"]) == 1
                assert got["label_config"]["labels"][0]["name"] == "person"

                # optimistic concurrency on update
                with pytest.raises(store.AssetError):
                    await store.update_annotation(db, ann["annotation_id"],
                                                  payload={"start": 0, "end": 4},
                                                  expected_version=99)

                stats = await store.project_stats(db, pid)
                assert stats["total_assets"] == 1 and stats["annotations_by_kind"]["span"] == 1
        finally:
            if pid:
                from sqlalchemy import delete

                from db.models import LabelProject
                async with get_sessionmaker()() as db:
                    await db.execute(delete(LabelProject).where(
                        LabelProject.project_id == uuid.UUID(pid)))
                    await db.commit()

    run_async(run())


@requires_infra
def test_llm_eval_and_rlhf_annotations_round_trip():
    """Preference, rubric and ranking are stored on the same spine as a bounding box."""
    from db.session import get_sessionmaker
    from services.assets import store
    from services.labelops.jobs import create_project

    async def run():
        pid = None
        try:
            async with get_sessionmaker()() as db:
                p = await create_project(db, name=f"rlhf-{uuid.uuid4().hex[:6]}", modality="dialogue")
                pid = p["project_id"]
                r = await store.create_assets(db, pid, [
                    {"media_type": "dialogue", "text": "Which answer is better?",
                     "meta": {"candidates": ["left answer", "right answer"]}}])
                aid = r["assets"][0]["asset_id"]

                pref = await store.create_annotation(
                    db, aid, kind="preference",
                    payload={"candidates": ["left answer", "right answer"], "chosen": 1})
                assert pref["payload"]["chosen"] == 1

                rub = await store.create_annotation(
                    db, aid, kind="rubric", payload={"scores": {"helpfulness": 4, "accuracy": "5"}})
                assert rub["payload"]["scores"] == {"helpfulness": 4.0, "accuracy": 5.0}

                rank = await store.create_annotation(
                    db, aid, kind="ranking", payload={"order": ["b", "a"]})
                assert rank["payload"]["order"] == ["b", "a"]

                kinds = {a["kind"] for a in await store.list_annotations(db, aid)}
                assert kinds == {"preference", "rubric", "ranking"}
        finally:
            if pid:
                from sqlalchemy import delete

                from db.models import LabelProject
                async with get_sessionmaker()() as db:
                    await db.execute(delete(LabelProject).where(
                        LabelProject.project_id == uuid.UUID(pid)))
                    await db.commit()

    run_async(run())
