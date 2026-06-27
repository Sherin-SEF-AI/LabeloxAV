"""LabeloxAV FastAPI backend. The review UI reads and writes only through these endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from core.logging import get_logger, setup_logging
from services.api.routers import (
    activelearn,
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
    govern,
    hdmap,
    lanes,
    mapassist,
    export,
    imports,
    intelligence,
    jobs,
    meta,
    models,
    multicam,
    objects,
    ocr,
    quality,
    relabel,
    review,
    signs,
    search,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


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
app.include_router(drivable.router, prefix="/api", tags=["drivable"])
app.include_router(signs.router, prefix="/api", tags=["signs"])
app.include_router(ocr.router, prefix="/api", tags=["ocr"])
