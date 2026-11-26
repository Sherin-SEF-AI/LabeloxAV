"""Background job kinds for the sensors tools (agent E), on the shared tool queue (services/tools/registry.py).

- sensors_reconstruct (GPU, held in office hours): train Gaussian splats for a session window and render views.
- sensors_pointmap (CPU): the no-training reconstruction, a colourised LiDAR point map, same views.
- sensors_radar_import (CPU): nuScenes RADAR_* sweeps or an MCAP's radar topics into `radar_sweep` rows.
- sensors_radar_fuse (CPU): Doppler from the session's radar onto its Track3D speeds.
- sensors_events_import (CPU): an event-camera file into an `event_stream` row.
- sensors_events_accumulate (CPU): an event stream into ordinary annotatable frames.
- sensors_variants (CPU): rain / night / fog / blur variants of hard frames, labels carried over, quarantined.
"""

from __future__ import annotations

import uuid

from core.logging import get_logger
from services.tools.registry import JobCanceled, JobContext, register

log = get_logger("kinds_sensors")


async def _reconstruct(ctx: JobContext, device: str) -> dict:
    from services.sensors.reconstruct.run import build_reconstruction, mark_failed

    recon_id = uuid.UUID(str(ctx.params["recon_id"]))
    try:
        return await build_reconstruction(recon_id, progress=ctx.progress, device=device)
    except JobCanceled:
        await mark_failed(recon_id, "canceled", None)
        raise
    except Exception as exc:
        await mark_failed(recon_id, "error", f"{type(exc).__name__}: {exc}")
        raise


@register("sensors_reconstruct", gpu=True, autolabel=True, label="Scene reconstruction (Gaussian splats)")
async def reconstruct_gsplat(ctx: JobContext) -> dict:
    """Heavy GPU training: held by the queue during office hours (10:00-18:00 IST) like auto-labelling."""
    return await _reconstruct(ctx, "cuda")


@register("sensors_pointmap", label="Scene reconstruction (point map)")
async def reconstruct_pointmap(ctx: JobContext) -> dict:
    return await _reconstruct(ctx, "cpu")


@register("sensors_radar_import", label="Radar import")
async def radar_import(ctx: JobContext) -> dict:
    from services.sensors.radar import import_radar

    return await import_radar(ctx.params, progress=ctx.progress)


@register("sensors_radar_fuse", label="Radar speed fusion")
async def radar_fuse(ctx: JobContext) -> dict:
    from services.sensors.radar import fuse_session_tracks

    return await fuse_session_tracks(uuid.UUID(str(ctx.params["session_id"])),
                                     tol_mps=float(ctx.params.get("tol_mps", 3.0)),
                                     apply=bool(ctx.params.get("apply", True)), progress=ctx.progress)


@register("sensors_events_import", label="Event camera import")
async def events_import(ctx: JobContext) -> dict:
    from services.sensors.events import import_events

    return await import_events(ctx.params, user_id=ctx.created_by, progress=ctx.progress)


@register("sensors_events_accumulate", label="Event frames")
async def events_accumulate(ctx: JobContext) -> dict:
    from services.sensors.events import accumulate_stream

    p = ctx.params
    return await accumulate_stream(uuid.UUID(str(p["stream_id"])), mode=str(p.get("mode", "count")),
                                   window_ms=float(p.get("window_ms", 33.0)), tau_ms=float(p.get("tau_ms", 30.0)),
                                   max_frames=int(p.get("max_frames", 600)), replace=bool(p.get("replace", True)),
                                   progress=ctx.progress)


@register("sensors_variants", label="Synthetic weather variants")
async def variants(ctx: JobContext) -> dict:
    from services.synth.variants import build_variants

    p = ctx.params
    return await build_variants([uuid.UUID(str(f)) for f in p.get("frame_ids") or []],
                                [str(x) for x in p.get("perturbations") or []],
                                strengths=dict(p.get("strengths") or {}), seed=int(p.get("seed", 7)),
                                include_review=bool(p.get("include_review", False)),
                                job_id=ctx.job_id, created_by=str(ctx.created_by) if ctx.created_by else None,
                                progress=ctx.progress)
