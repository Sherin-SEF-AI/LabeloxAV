"""LabeloxAV FastAPI backend. The review UI reads and writes only through these endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from core.config import get_settings
from core.logging import get_logger, setup_logging
from services.api.deps import role_rank
from services.api.routers import (
    activelearn,
    adverse,
    analytics,
    autolabel,
    calibration,
    collaborate,
    corrections,
    curation,
    datasets,
    discovery,
    drivable,
    dynamics,
    errordetect,
    export,
    govern,
    hdmap,
    imports,
    intelligence,
    jobs,
    lanes,
    lidar,
    lidar_scene,
    mapassist,
    meta,
    models,
    multicam,
    objects,
    objects3d,
    ocr,
    quality,
    recall,
    relabel,
    review,
    search,
    segment_assist,
    signs,
    tracks,
    training,
    triage,
    upload,
    users,
)

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(get_settings().log_level)
    log.info("api.startup")
    yield
    log.info("api.shutdown")


app = FastAPI(title="LabeloxAV", version="0.1.0", lifespan=lifespan)

# Role floor by path prefix for mutating requests. Reads (GET/HEAD) are always open.
_ADMIN_PREFIXES = ("/api/govern", "/api/users")
_REVIEWER_PREFIXES = (
    "/api/review", "/api/export", "/api/datasets", "/api/relabel", "/api/imports", "/api/curation",
    "/api/corrections", "/api/collaborate", "/api/objects", "/api/tracks", "/api/lanes", "/api/errordetect",
)


def _required_role(path: str) -> str:
    if path.startswith(_ADMIN_PREFIXES):
        return "admin"
    if path.startswith(_REVIEWER_PREFIXES):
        return "reviewer"
    return "annotator"  # any authenticated user (expensive but non-destructive routes)


class AuthMiddleware(BaseHTTPMiddleware):
    """Deny-by-default: every mutating /api request needs a known user, with role floors by path.
    Open for reads. A new write route added later is gated automatically (fails closed)."""

    async def dispatch(self, request, call_next):
        if not get_settings().auth.enabled:
            return await call_next(request)
        method = request.method
        path = request.url.path
        if method in ("GET", "HEAD", "OPTIONS") or path == "/api/health" or not path.startswith("/api/"):
            return await call_next(request)

        from sqlalchemy import func, select

        from db.models import User
        from db.session import get_sessionmaker

        uid = request.headers.get("x-lbx-user-id")
        role = None
        if uid:
            try:
                async with get_sessionmaker()() as db:
                    u = await db.get(User, UUID(uid))
                    role = u.role if u else None
            except Exception:  # noqa: BLE001
                role = None

        # Bootstrap: allow the very first user to be created before any user (hence any admin) exists.
        if path == "/api/users" and method == "POST":
            async with get_sessionmaker()() as db:
                n = (await db.execute(select(func.count()).select_from(User))).scalar_one()
            if n == 0:
                return await call_next(request)

        if role is None:
            return JSONResponse({"detail": "authentication required (X-Lbx-User-Id)"}, status_code=401)
        if role_rank(role) < role_rank(_required_role(path)):
            return JSONResponse({"detail": f"requires {_required_role(path)} role or higher"}, status_code=403)
        return await call_next(request)


# Order matters: add auth first so CORS (added last) is the outermost layer and a 401/403 still
# carries CORS headers for the browser.
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    """Liveness + dependency readiness: probes Postgres, Redis, and MinIO so a load balancer or operator
    sees a real status, not a static OK. Returns 200 with per-dependency detail; degraded deps are flagged."""
    deps: dict[str, str] = {}

    async def _check(name: str, coro):
        try:
            await coro
            deps[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            deps[name] = f"error: {type(exc).__name__}"

    from sqlalchemy import text

    from core.storage import get_object_store
    from db.session import get_sessionmaker

    async def _pg():
        async with get_sessionmaker()() as db:
            await db.execute(text("SELECT 1"))

    async def _redis():
        import redis.asyncio as aredis

        r = aredis.Redis.from_url(get_settings().redis.url)
        try:
            await r.ping()
        finally:
            await r.aclose()

    async def _minio():
        get_object_store().ensure_bucket()

    await _check("postgres", _pg())
    await _check("redis", _redis())
    await _check("minio", _minio())
    overall = "ok" if all(v == "ok" for v in deps.values()) else "degraded"
    return {"status": overall, "deps": deps}


@app.get("/api/metrics")
async def metrics():
    """Lightweight operational counts for a dashboard: corpus size, label states, queue depth, and the
    governance auto-accept switch. JSON, not Prometheus, to stay dependency-free."""
    from sqlalchemy import func, select

    from db.models import Frame, GovernanceState, Object, TrainingJob
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        frames = (await db.execute(select(func.count()).select_from(Frame))).scalar_one()
        by_state = dict((await db.execute(select(Object.state, func.count()).group_by(Object.state))).all())
        pending_jobs = (await db.execute(
            select(func.count()).select_from(TrainingJob).where(TrainingJob.status == "pending"))).scalar_one()
        gov = await db.get(GovernanceState, 1)
    return {
        "frames": int(frames),
        "objects_by_state": {k: int(v) for k, v in by_state.items()},
        "objects_in_review": int(by_state.get("review", 0)),
        "pending_training_jobs": int(pending_jobs),
        "auto_accept_enabled": bool(gov.auto_accept_enabled) if gov else True,
    }


app.include_router(meta.router, prefix="/api", tags=["meta"])
app.include_router(triage.router, prefix="/api", tags=["triage"])
app.include_router(objects.router, prefix="/api", tags=["objects"])
app.include_router(review.router, prefix="/api", tags=["review"])
app.include_router(intelligence.router, prefix="/api", tags=["intelligence"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(analytics.router, prefix="/api", tags=["analytics"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(export.router, prefix="/api", tags=["export"])
app.include_router(quality.router, prefix="/api", tags=["quality"])
app.include_router(recall.router, prefix="/api", tags=["recall"])
app.include_router(adverse.router, prefix="/api", tags=["adverse"])
app.include_router(segment_assist.router, prefix="/api", tags=["segment"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(imports.router, prefix="/api", tags=["imports"])
app.include_router(training.router, prefix="/api", tags=["training"])
app.include_router(tracks.router, prefix="/api", tags=["tracks"])
app.include_router(autolabel.router, prefix="/api", tags=["autolabel"])
app.include_router(jobs.router, prefix="/api", tags=["jobs"])
app.include_router(users.router, prefix="/api", tags=["users"])
app.include_router(datasets.router, prefix="/api", tags=["datasets"])
app.include_router(curation.router, prefix="/api", tags=["curation"])
app.include_router(corrections.router, prefix="/api", tags=["corrections"])
app.include_router(calibration.router, prefix="/api", tags=["calibration"])
app.include_router(activelearn.router, prefix="/api", tags=["activelearn"])
app.include_router(errordetect.router, prefix="/api", tags=["errordetect"])
app.include_router(relabel.router, prefix="/api", tags=["relabel"])
app.include_router(collaborate.router, prefix="/api", tags=["collaborate"])
app.include_router(govern.router, prefix="/api", tags=["govern"])
app.include_router(multicam.router, prefix="/api", tags=["multicam"])
app.include_router(mapassist.router, prefix="/api", tags=["mapassist"])
app.include_router(hdmap.router, prefix="/api", tags=["hdmap"])
app.include_router(dynamics.router, prefix="/api", tags=["dynamics"])
app.include_router(discovery.router, prefix="/api", tags=["discovery"])
app.include_router(lanes.router, prefix="/api", tags=["lanes"])
app.include_router(lidar.router, prefix="/api", tags=["lidar"])
app.include_router(objects3d.router, prefix="/api", tags=["lidar"])
app.include_router(lidar_scene.router, prefix="/api", tags=["lidar"])
app.include_router(drivable.router, prefix="/api", tags=["drivable"])
app.include_router(signs.router, prefix="/api", tags=["signs"])
app.include_router(ocr.router, prefix="/api", tags=["ocr"])
