"""Export trigger: seal a dataset commit and render the requested formats, with reimport sanity.

Also serves the quality certificate for a commit. That is deliberately a separate call rather than part of
the export response: a certificate needs a sealed gold set and a scored evaluation run, and most exports do
not have one. Folding it in would mean either blocking exports that cannot be certified or returning a
half-certificate, and both are worse than an export that says plainly it has not been measured.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import ExportIn, db_session
from services.export.dataset import SliceSpec, export_dataset, reimport_sanity

router = APIRouter()


@router.post("/export")
async def export(payload: ExportIn):
    spec = SliceSpec(
        name=payload.name,
        states=payload.states,
        class_names=payload.class_names,
        cities=payload.cities,
        vehicle_ids=payload.vehicle_ids,
        session_id=payload.session_id,
        min_conf=payload.min_conf,
        formats=payload.formats,
        limit=payload.limit,
        val_frac=payload.val_frac,
        test_frac=payload.test_frac,
        split_group_by=payload.split_group_by,
        split_seed=payload.split_seed,
    )
    result = await export_dataset(spec)
    result["reimport_sanity"] = reimport_sanity(Path(result["out_dir"]))
    return result


@router.get("/export/{commit_id}/certificate")
async def certificate(commit_id: str, eval_id: str, gold_id: str, model_version: str,
                      fmt: str = "json", db: AsyncSession = Depends(db_session)):
    """The signed quality certificate for a release: per-class precision and recall, with intervals.

    `fmt=markdown` renders the document a buyer reads, which leads with the classes that were not measured
    rather than with the headline numbers. The JSON form carries the same manifest plus its signature, so a
    recipient can verify nothing was edited after issue.
    """
    from core.config import get_settings
    from services.export.certificate import build_certificate, render_certificate_markdown

    key = get_settings().phase4.govern.attestation_key
    cert = await build_certificate(db, commit_id=commit_id, eval_id=eval_id, gold_id=gold_id,
                                   model_version=model_version, key=key)
    if "error" in cert:
        return cert
    if fmt == "markdown":
        return Response(render_certificate_markdown(cert), media_type="text/markdown")
    return cert


@router.get("/export/scenario/event/{event_id}")
async def scenario_from_event(event_id: str, near_m: float = 60.0, pad_s: float = 4.0,
                              road_network_file: str = "map.xodr", fmt: str = "xml",
                              db: AsyncSession = Depends(db_session)):
    """An ASAM OpenSCENARIO 1.2 document for a mined event, runnable in a simulator.

    The map half of sim handover already existed: hdmap/export.py emits OpenDRIVE and Lanelet2, so a
    customer could load the road but not what happened on it. `road_network_file` is the OpenDRIVE the
    scenario references, and should be the one exported for the same session.

    `fmt=json` returns the document alongside what was excluded from it, which is the part worth reading:
    roadside furniture, tracks too short to describe a manoeuvre, and actors that never came near. A
    scenario that silently dropped the other vehicle would look complete.
    """
    from services.export.scenario_build import build_from_event

    result = await build_from_event(db, event_id, near_m=near_m, pad_s=pad_s,
                                    road_network_file=road_network_file)
    if "error" in result:
        return result
    if fmt == "json":
        return result
    return Response(result["xml"], media_type="application/xml")


@router.get("/export/scenario/session/{session_id}")
async def scenario_from_window(session_id: str, t_start_ns: int, t_end_ns: int, near_m: float = 60.0,
                               road_network_file: str = "map.xodr", fmt: str = "xml",
                               name: str = "scenario", db: AsyncSession = Depends(db_session)):
    """The same, for an arbitrary window rather than a detected event."""
    from services.export.scenario_build import build_from_window

    result = await build_from_window(db, session_id=session_id, t_start_ns=t_start_ns,
                                     t_end_ns=t_end_ns, near_m=near_m, name=name,
                                     road_network_file=road_network_file)
    if "error" in result:
        return result
    if fmt == "json":
        return result
    return Response(result["xml"], media_type="application/xml")
