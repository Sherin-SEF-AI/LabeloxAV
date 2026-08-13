"""LabeloxAV FastAPI backend. The review UI reads and writes only through these endpoints."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.observability import spawn
from services.api.deps import role_rank
from services.api.media import MEDIA_COOKIE, is_media_read
from services.api.routers import (
    activelearn,
    adverse,
    agent,
    analytics,
    assets,
    autolabel,
    billing,
    calibration,
    campaigns,
    checkpoints,
    cloud,
    collaborate,
    corrections,
    curation,
    dataset_query,
    datasets,
    discovery,
    drivable,
    driving_events,
    dynamics,
    edge,
    errordetect,
    events,
    experiments,
    explore,
    export,
    govern,
    hdmap,
    identity_routes,
    imports,
    inbox,
    inertial,
    inspector,
    integrations,
    intelligence,
    jobs,
    labelops,
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
    op_eval,
    predictions,
    quality,
    reasoner,
    recall,
    relabel,
    review,
    search,
    security,
    secv2,
    segment_assist,
    segmentation,
    service_accounts,
    signs,
    tracks,
    training,
    triage,
    upload,
    users,
    workforce,
)
from services.api.routers import (
    auth as auth_router,
)
from services.api.routers import (
    calyx as calyx_router,
)
from services.api.routers import (
    flywheel as flywheel_router,
)
from services.api.routers import (
    forgyx as forgyx_router,
)
from services.api.routers import (
    hardening as hardening_router,
)
from services.api.routers import (
    labelox as labelox_router,
)
from services.api.routers import (
    oraclyx as oraclyx_router,
)
from services.api.routers import (
    release as release_router,
)
from services.api.routers import (
    sanyx as sanyx_router,
)
from services.api.routers import (
    sievyx as sievyx_router,
)
from services.api.routers import (
    verdyx as verdyx_router,
)
from services.export.dataset import DpdpaRefusal, UnknownExportFormat

log = get_logger("api")


async def _cloud_watchdog():
    """Enforce the warm-session idle / max-session guards even when no UI is polling status, so a connected
    cloud GPU cannot linger past its limits while the operator is away. refresh() is a no-op when there is
    no live session and safe when RUNPOD_API_KEY is unset."""
    from compute.runpod.session import get_manager

    while True:
        try:
            await get_manager().refresh()
        except Exception as exc:  # noqa: BLE001
            log.error("cloud.watchdog_error", error=str(exc))
        await asyncio.sleep(30)


def _assert_auth_floors(app: FastAPI) -> int:
    """Fail closed at startup: refuse to boot if any mounted /api read is public without being in the tiny
    approved allowlist. The route-table test enforces this on every commit, but a deployment that skipped the
    test (or a route mounted dynamically) still cannot silently expose corpus data, presigned URLs, or the
    user list. Writes need no check here: they can never reach a route without clearing the token gate, since
    _required_role never returns an anonymous floor. Returns the count of routes inspected."""
    checked = 0
    leaked: list[str] = []
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", "")
        if not path.startswith("/api/"):
            continue
        checked += 1
        if ("GET" in methods or "HEAD" in methods) and _is_public_read(path) \
                and not path.startswith(_APPROVED_PUBLIC_READ_PREFIXES):
            leaked.append(path)
    # Every unauthenticated credential route must be one that actually exists, or the allowlist is drifting
    # away from the route table and naming something that no longer means anything.
    mounted = {getattr(r, "path", "") for r in app.routes}
    missing = sorted(p for p in _CREDENTIAL_PATHS if p not in mounted)
    if missing:
        raise RuntimeError(f"credential allowlist names routes that are not mounted: {missing}")
    if leaked:
        raise RuntimeError(f"public read routes without an approved floor: {sorted(leaked)}")
    return checked


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(get_settings().log_level)
    # Inert without configuration: no SENTRY_DSN means Sentry is never initialised, no OTLP endpoint means no
    # exporter. Turning telemetry on is then a deployment change rather than a code change.
    from core.observability import init_observability

    enabled = init_observability(app)
    checked = _assert_auth_floors(app)
    log.info("api.startup", api_routes_checked=checked, **enabled)
    # Background jobs run as tasks inside this process, so every one that was live when the previous process
    # ended is now gone with no trace but a row still marked running. Startup is the one moment that is
    # certain, so the sweep happens here rather than on a timer. It only touches rows whose heartbeat has
    # already aged past the staleness window, so a job belonging to another live worker is never reaped.
    # Best effort: an unreachable database must not stop the API from starting and reporting why.
    try:
        from db.session import get_sessionmaker
        from services.agent.resume import reap_interrupted
        from services.job_reaper import reap_stale_jobs

        async with get_sessionmaker()() as db:
            await reap_interrupted(db)
            # The five job tables that are not AgentRun. One of them is not cosmetic: the autolabel router
            # refuses to start while any row reads `running`, so a single row left behind by a restart
            # disabled auto-labeling for everyone until somebody ran an UPDATE by hand.
            await reap_stale_jobs(db)
    except Exception as exc:  # noqa: BLE001
        log.warning("api.reap_interrupted_failed", error=str(exc))
    watchdog = spawn(_cloud_watchdog(), name="_cloud_watchdog")
    try:
        yield
    finally:
        watchdog.cancel()
        # mandatory teardown on shutdown: terminate any live warm pod so a connected GPU can never linger
        # past the app (no-op when nothing is connected).
        try:
            from compute.runpod.session import get_manager

            await get_manager().disconnect(terminate=True)
        except Exception as exc:  # noqa: BLE001
            log.error("cloud.shutdown_teardown_error", error=str(exc))
        log.info("api.shutdown")


app = FastAPI(title="LabeloxAV", version="0.1.0", lifespan=lifespan)

# Role floor by path prefix for mutating requests.
_ADMIN_PREFIXES = ("/api/govern", "/api/users", "/api/service-accounts")
_REVIEWER_PREFIXES = (
    "/api/review", "/api/export", "/api/datasets", "/api/relabel", "/api/imports", "/api/curation",
    "/api/corrections", "/api/collaborate", "/api/objects", "/api/tracks", "/api/lanes", "/api/errordetect",
    # C7: model promotion, paid-GPU control, and corpus-wide re-detection are not annotator actions.
    "/api/training", "/api/cloud", "/api/autolabel",
)

# Reads fail closed. A GET is open only if its path is in this small allowlist; everything else under /api/
# needs a valid token. This is the inversion of the old denylist: previously a new read route was open unless
# someone remembered to add it to a gated-prefix list, so a route that mints presigned download URLs or
# returns governance state could leak by omission. With the allowlist a new read route is gated automatically
# and has to be opted into public access deliberately. The frontend rides the Authorization header on every
# request (web/lib/api.ts userHeaders), so gating reads does not lock out a signed-in user. Only the two
# load-balancer probes are public (/api/health liveness, /api/readyz readiness): a probe cannot carry a token. Non-/api paths (openapi.json, /docs, /metrics) are not
# ours to gate and pass through below. openapi is the API's own schema, deliberately public.
_PUBLIC_READ_PREFIXES = ("/api/health", "/api/readyz")

# The security-reviewed baseline the startup backstop checks the operative allowlist against. It is a separate
# constant on purpose: widening _PUBLIC_READ_PREFIXES to expose a data route also has to widen this reviewed
# baseline, so a public read can never be opened without a deliberate edit here. Keep the two in sync only when
# the exposure is intended.
_APPROVED_PUBLIC_READ_PREFIXES = ("/api/health", "/api/readyz")


# Self-service routes: gated (need a valid token) but reachable by any authenticated user, so they must not
# inherit the admin/reviewer floor of their prefix. GET /users/me lets a user resolve their own identity.
_SELF_PATHS = ("/api/users/me",)


# The credential routes: how somebody who holds no token obtains one. A sign-in route behind the token gate
# is a deadlock, so these are reachable unauthenticated by necessity rather than by preference.
#
# Listed exactly, never by prefix. "/api/auth" as a prefix would silently expose every future route added
# under it, including the ones that change a password or disable a second factor. Each entry below carries
# its own protection: login and reset answer identically for a real and an absent account, signup refuses
# unless the operator enabled it or the deployment has no users at all, and the OIDC pair are worthless
# without the provider's signature.
_CREDENTIAL_PATHS = frozenset({
    "/api/auth/methods",              # which methods exist; configuration, never account existence
    "/api/auth/login",
    "/api/auth/login/mfa",            # finishing a sign-in that login itself began
    "/api/auth/mfa/setup-pending",    # enrolment forced at the door, before any token exists
    "/api/auth/mfa/confirm",
    "/api/auth/signup",
    "/api/auth/password/reset-request",
    "/api/auth/password/reset",
    "/api/auth/oidc/start",
    "/api/auth/oidc/callback",
    "/api/auth/logout",               # clearing cookies must work even with a lapsed token
})


def _is_public_read(path: str) -> bool:
    """A read that is reachable without a token. Kept tiny and explicit so the fail-closed default holds."""
    return path.startswith(_PUBLIC_READ_PREFIXES)


def _is_credential_path(path: str) -> bool:
    """A route somebody with no token must reach in order to get one."""
    return path in _CREDENTIAL_PATHS



def _required_role(path: str) -> str:
    if path in _SELF_PATHS:
        return "annotator"
    if path.startswith(_ADMIN_PREFIXES):
        return "admin"
    if path.startswith(_REVIEWER_PREFIXES):
        return "reviewer"
    return "annotator"  # any authenticated user (expensive but non-destructive routes)


class AuthMiddleware(BaseHTTPMiddleware):
    """Deny-by-default: every /api request, read or write, needs a valid signed Bearer token unless its path is
    a public read (_is_public_read) or an explicit bootstrap exception below. Role floors apply by path. A new
    route added later, read or write, is gated automatically (fails closed)."""

    async def dispatch(self, request, call_next):
        settings = get_settings()
        if not settings.auth.enabled:
            return await call_next(request)
        method = request.method
        path = request.url.path
        # CORS preflight carries no credentials by design; the CORS layer answers it, but allow it here too so
        # the auth gate never turns a preflight into a 401. Non-/api paths are not ours to gate.
        if method == "OPTIONS" or not path.startswith("/api/"):
            return await call_next(request)
        if method in ("GET", "HEAD") and _is_public_read(path):
            return await call_next(request)
        # The credential routes. Reaching a sign-in page requires no token by definition; each of these is
        # individually listed and individually defended, rather than a prefix that would swallow whatever
        # gets added under /api/auth next.
        if _is_credential_path(path):
            return await call_next(request)

        from sqlalchemy import func, select

        from db.models import User
        from db.session import get_sessionmaker
        from services.api.auth_token import bearer_uid

        # The credential is a signed token in the Authorization header, not a plaintext user id: an attacker
        # who reads the public user list cannot mint one without the server signing key (C1).
        authz = request.headers.get("authorization")
        # SSE is the one exception. The browser EventSource API cannot set a request header at all, so an
        # event stream has no way to present a bearer token except in the URL. This is accepted ONLY for the
        # /api/events/ prefix, and those streams carry job progress and nothing sensitive, because a URL can
        # reach a proxy or access log in a way a header does not. Everywhere else the header is the only
        # accepted form, so this does not widen the credential surface generally.
        if not authz and path.startswith("/api/events/"):
            qs_token = request.query_params.get("token")
            if qs_token:
                authz = f"Bearer {qs_token}"
        uid = bearer_uid(authz, settings.auth.signing_key)

        # Image subresources: an <img> cannot set a header, so accept the media cookie for those paths and
        # only those. bearer_uid above has already refused a media token presented as a Bearer, so this is
        # the single place the restricted credential is honoured, and it can buy nothing but an image.
        if uid is None and method in ("GET", "HEAD") and is_media_read(path):
            from services.api.auth_token import MEDIA_SCOPE, verify_token

            cookie = request.cookies.get(MEDIA_COOKIE)
            payload = verify_token(cookie, settings.auth.signing_key) if cookie else None
            if payload is not None and payload.scope == MEDIA_SCOPE:
                uid = payload.uid

        role = None
        if uid:
            try:
                async with get_sessionmaker()() as db:
                    u = await db.get(User, UUID(uid))
                    role = u.role if u else None
            except Exception:  # noqa: BLE001
                role = None
        elif authz:
            # A machine credential. This gate resolves identity independently of the route dependency, so a
            # credential the dependency understands and this does not is refused here and never reaches the
            # route: service accounts authenticated perfectly in unit tests and answered 401 to every real
            # request until this branch existed. Both paths call the same verifier so they cannot drift again.
            from services.identity.service_accounts import split_key
            from services.identity.service_accounts import verify as verify_key

            raw = authz.split(" ", 1)[1].strip() if " " in authz else ""
            if split_key(raw) is not None:
                try:
                    async with get_sessionmaker()() as db:
                        u = await verify_key(db, raw)
                        role = u.role if u else None
                except Exception:  # noqa: BLE001
                    role = None

        # Dev-login must be reachable without a token (it exists to hand out the first one); the route itself
        # is hard-gated to env == "local", so allowlisting it here does not open a hole on a real deployment.
        if path == "/api/auth/dev-login" and method == "POST":
            return await call_next(request)

        # Bootstrap: allow the very first user to be created before any user (hence any admin) exists.
        if path == "/api/users" and method == "POST":
            async with get_sessionmaker()() as db:
                n = (await db.execute(select(func.count()).select_from(User))).scalar_one()
            if n == 0:
                return await call_next(request)

        if role is None:
            return JSONResponse({"detail": "authentication required (Bearer token)"}, status_code=401)
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

# Request metrics for /metrics (Prometheus). Registered as an http middleware so it times every request under
# its route template.
from services.api.metrics import MetricsMiddleware  # noqa: E402

app.middleware("http")(MetricsMiddleware())


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus scrape target: request rates/latencies (in-process) plus live corpus gauges (review-queue
    depth, embedding backlog). Unauthenticated like a conventional metrics port; bind it internally in prod."""
    from fastapi import Response

    from services.api.metrics import render, sample_db_gauges

    body = await render(await sample_db_gauges())
    return Response(body, media_type="text/plain; version=0.0.4")


@app.exception_handler(DpdpaRefusal)
async def _dpdpa_handler(request, exc: DpdpaRefusal):
    # A DPDPA pre-sale refusal is an expected, client-actionable outcome (un-redacted PII in the slice), not a
    # server fault: return 422 with the per-session blockers instead of letting it escape as a 500.
    payload = exc.args[0] if exc.args else {"detail": "DPDPA pre-sale gate refused export"}
    return JSONResponse(payload, status_code=422)


@app.exception_handler(UnknownExportFormat)
async def _unknown_format_handler(request, exc: UnknownExportFormat):
    # Requesting a format no adapter implements is a client error, not a server fault. It used to be ignored
    # silently, which shipped a commit claiming formats the archive never contained; now it is a 400 naming
    # the unsupported target and the supported set.
    return JSONResponse({"detail": str(exc)}, status_code=400)


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


@app.get("/api/readyz")
async def readyz():
    """Readiness probe: 200 only when every dependency is reachable, 503 otherwise.

    /api/health is a liveness probe and deliberately returns 200 with a degraded body so an operator can see
    which dependency is down. A load balancer reading that as healthy would keep routing traffic to a node
    whose Postgres is gone, so readiness needs its own endpoint that actually fails. Public for the same
    reason /api/health is: a probe cannot carry a token.
    """
    body = await health()
    ready = body["status"] == "ok"
    return JSONResponse(body, status_code=200 if ready else 503)


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


app.include_router(agent.router, prefix="/api", tags=["agent"])
app.include_router(meta.router, prefix="/api", tags=["meta"])
app.include_router(triage.router, prefix="/api", tags=["triage"])
app.include_router(objects.router, prefix="/api", tags=["objects"])
app.include_router(predictions.router, prefix="/api", tags=["predictions"])
app.include_router(dataset_query.router, prefix="/api", tags=["datasets"])
app.include_router(review.router, prefix="/api", tags=["review"])
app.include_router(intelligence.router, prefix="/api", tags=["intelligence"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(analytics.router, prefix="/api", tags=["analytics"])
app.include_router(cloud.router, prefix="/api", tags=["cloud"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(export.router, prefix="/api", tags=["export"])
app.include_router(billing.router, prefix="/api", tags=["billing"])
app.include_router(workforce.router, prefix="/api", tags=["workforce"])
app.include_router(explore.router, prefix="/api", tags=["explore"])
app.include_router(quality.router, prefix="/api", tags=["quality"])
app.include_router(recall.router, prefix="/api", tags=["recall"])
app.include_router(adverse.router, prefix="/api", tags=["adverse"])
app.include_router(segment_assist.router, prefix="/api", tags=["segment"])
app.include_router(segmentation.router, prefix="/api", tags=["segmentation"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(imports.router, prefix="/api", tags=["imports"])
app.include_router(training.router, prefix="/api", tags=["training"])
app.include_router(tracks.router, prefix="/api", tags=["tracks"])
app.include_router(autolabel.router, prefix="/api", tags=["autolabel"])
app.include_router(jobs.router, prefix="/api", tags=["jobs"])
app.include_router(events.router, prefix="/api", tags=["events"])
app.include_router(labelops.router, prefix="/api", tags=["labelops"])
app.include_router(assets.router, prefix="/api", tags=["assets"])
app.include_router(integrations.router, prefix="/api", tags=["integrations"])
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
app.include_router(service_accounts.router, prefix="/api", tags=["service-accounts"])
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
app.include_router(inertial.router, prefix="/api", tags=["inertial"])
app.include_router(driving_events.router, prefix="/api", tags=["driving-events"])
app.include_router(checkpoints.router, prefix="/api", tags=["checkpoints"])
app.include_router(inspector.router, prefix="/api", tags=["inspector"])
app.include_router(sanyx_router.router, prefix="/api", tags=["sanyx"])
app.include_router(calyx_router.router, prefix="/api", tags=["calyx"])
app.include_router(sievyx_router.router, prefix="/api", tags=["sievyx"])
app.include_router(labelox_router.router, prefix="/api", tags=["labelox"])
app.include_router(oraclyx_router.router, prefix="/api", tags=["oraclyx"])
app.include_router(release_router.router, prefix="/api", tags=["release"])
app.include_router(verdyx_router.router, prefix="/api", tags=["verdyx"])
app.include_router(forgyx_router.router, prefix="/api", tags=["forgyx"])
app.include_router(auth_router.router, prefix="/api", tags=["auth"])
app.include_router(flywheel_router.router, prefix="/api", tags=["flywheel"])
app.include_router(hardening_router.router, prefix="/api", tags=["hardening"])
app.include_router(campaigns.router, prefix="/api", tags=["campaigns"])
app.include_router(reasoner.router, prefix="/api", tags=["reasoner"])
app.include_router(edge.router, prefix="/api", tags=["edge"])
app.include_router(experiments.router, prefix="/api", tags=["experiments"])
app.include_router(op_eval.router, prefix="/api", tags=["eval"])
app.include_router(secv2.router, prefix="/api", tags=["security"])
app.include_router(identity_routes.router, prefix="/api", tags=["identity"])
app.include_router(inbox.router, prefix="/api", tags=["inbox"])
app.include_router(security.router, prefix="/api", tags=["security"])
app.include_router(signs.router, prefix="/api", tags=["signs"])
app.include_router(ocr.router, prefix="/api", tags=["ocr"])
