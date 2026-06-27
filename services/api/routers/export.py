"""Export trigger: seal a dataset commit and render the requested formats, with reimport sanity."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from services.api.deps import ExportIn
from services.export.dataset import SliceSpec, export_dataset, reimport_sanity

router = APIRouter()


@router.post("/export")
async def export(payload: ExportIn):
    spec = SliceSpec(
        name=payload.name,
        states=payload.states,
        class_names=payload.class_names,
        cities=payload.cities,
        session_id=payload.session_id,
        min_conf=payload.min_conf,
        formats=payload.formats,
        limit=payload.limit,
    )
    result = await export_dataset(spec)
    result["reimport_sanity"] = reimport_sanity(Path(result["out_dir"]))
    return result
