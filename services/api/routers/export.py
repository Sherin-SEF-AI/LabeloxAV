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
