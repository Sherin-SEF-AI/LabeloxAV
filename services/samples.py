"""Sample drives a new team can load to see the whole loop before they have footage of their own.

Real Indian street video under a licence that allows it: each is a Wikimedia Commons file released under
CC BY-SA 3.0, which permits copying and adaptation with attribution and share-alike. The credit travels with
the data: it is in GET /api/samples, on the page that offers the sample, and in the session's route
(`sample:<id>`), so nobody mistakes a sample for their own recording.

Not bundled with the install: the two files are 35 and 58 MB, and most installs never need them. The server
downloads one on request, checks it against the SHA-256 recorded here (the exact bytes whose licence was
checked), stores it in the object store, and imports it through the same video path an upload takes, so it is
frame-sampled and has faces and plates blurred like any other footage. An install without internet access
says so plainly.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass

from core.logging import get_logger

log = get_logger("samples")

# Wikimedia asks automated clients to identify themselves.
USER_AGENT = "LabeloxAV/0.1 (sample-data loader; https://github.com/Sherin-SEF-AI/LabeloxAV)"
ROUTE_PREFIX = "sample:"


@dataclass(frozen=True)
class Sample:
    id: str
    title: str
    description: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    license_url: str
    author: str
    source_page: str
    vehicle_id: str
    city: str


SAMPLES: tuple[Sample, ...] = (
    Sample(
        id="delhi-stray-cattle",
        title="Stray cattle in Lutyens' Delhi",
        description="A street-level drive past cattle on the carriageway, with cars and autorickshaws around them.",
        url="https://upload.wikimedia.org/wikipedia/commons/d/d6/Stray_Cattle_in_Lutyens_Delhi.webm",
        sha256="ca03d510e16ce9d1f3930c96736ac1529495cbf7b2ce0aa02007132fe4c1993a",
        size_bytes=58252539,
        license="CC BY-SA 3.0",
        license_url="https://creativecommons.org/licenses/by-sa/3.0",
        author="Fowler&fowler",
        source_page="https://commons.wikimedia.org/wiki/File:Stray_Cattle_in_Lutyens_Delhi.webm",
        vehicle_id="SAMPLE-DEL",
        city="DEL",
    ),
    Sample(
        id="cuttack-link-road",
        title="Moving vehicles on Link Road, Cuttack",
        description="Mixed traffic on an Odisha city road: two-wheelers, buses, autorickshaws and pedestrians.",
        url="https://upload.wikimedia.org/wikipedia/commons/6/6d/Moving_vehicles_in_Link_road%2C_Cuttack%2C_Odisha.webm",
        sha256="3e35d10f72a75c9360007bfe8666995c46ee8fd67e1f771cb46bb1bf4167de2f",
        size_bytes=35099101,
        license="CC BY-SA 3.0",
        license_url="https://creativecommons.org/licenses/by-sa/3.0",
        author="Subhashish Panigrahi",
        source_page="https://commons.wikimedia.org/wiki/File:Moving_vehicles_in_Link_road,_Cuttack,_Odisha.webm",
        vehicle_id="SAMPLE-CTC",
        city="CTC",
    ),
)


class SampleError(RuntimeError):
    """A sample could not be fetched or did not match; the message is meant for the person who asked."""


def get_sample(sample_id: str) -> Sample | None:
    return next((s for s in SAMPLES if s.id == sample_id), None)


def public(s: Sample) -> dict:
    d = asdict(s)
    d["route"] = f"{ROUTE_PREFIX}{s.id}"
    d["credit"] = f'"{s.title}" by {s.author}, {s.license}, via Wikimedia Commons'
    return d


async def fetch_to_store(s: Sample, store) -> str:
    """Download the sample into the object store, once. Returns its s3 uri. Raises SampleError on any failure,
    including a file whose bytes are not the ones recorded here."""
    import asyncio

    import httpx

    key = f"samples/{s.id}.webm"
    uri = store.uri(key)
    try:
        head = await asyncio.to_thread(store._client.head_object, Bucket=store.bucket, Key=key)
        if int(head.get("ContentLength", -1)) == s.size_bytes:
            return uri
    except Exception:  # noqa: BLE001 - not stored yet
        pass

    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True,
                                     headers={"User-Agent": USER_AGENT}) as client:
            async with client.stream("GET", s.url) as r:
                if r.status_code != 200:
                    raise SampleError(f"Wikimedia Commons answered {r.status_code} for {s.title}")
                async for chunk in r.aiter_bytes():
                    digest.update(chunk)
                    chunks.append(chunk)
    except httpx.HTTPError as exc:
        raise SampleError(
            f"could not download the sample from Wikimedia Commons ({type(exc).__name__}). This needs internet "
            "access from the server; on an offline network, upload your own footage instead.") from exc
    data = b"".join(chunks)
    if digest.hexdigest() != s.sha256:
        raise SampleError(f"the downloaded file does not match the recorded sample ({len(data)} bytes, "
                          f"sha256 {digest.hexdigest()[:12]}); it was not imported")
    await asyncio.to_thread(store.put_bytes, key, data, "video/webm")
    log.info("samples.fetched", sample=s.id, bytes=len(data))
    return uri


async def load_sample(s: Sample, job_id: uuid.UUID) -> None:
    """Background task: fetch, then import through the ordinary video path. Failures land on the job row."""
    from core.storage import get_object_store
    from services.imports.records import ImportSpec
    from services.imports.run import _bump_job, run_import_guarded
    from services.job_control import JobCanceled

    try:
        await _bump_job(job_id, status="running", progress=0.0, counts={"stage": "downloading sample"})
        uri = await fetch_to_store(s, get_object_store())
        await _bump_job(job_id, source_uri=uri, counts={"stage": "importing"})
    except JobCanceled:
        log.info("samples.canceled", sample=s.id)
        return
    except Exception as exc:  # noqa: BLE001
        log.error("samples.load_failed", sample=s.id, error=str(exc))
        try:
            await _bump_job(job_id, status="error", error=str(exc))
        except JobCanceled:
            pass
        return
    spec = ImportSpec(format="video", source_uri=uri, target_vehicle=s.vehicle_id, city=s.city,
                      route=f"{ROUTE_PREFIX}{s.id}")
    await run_import_guarded(spec, job_id)


def sample_for_route(route: str | None) -> Sample | None:
    """The sample a session came from, by its route, or None for a team's own recording."""
    if not route or not route.startswith(ROUTE_PREFIX):
        return None
    return get_sample(route[len(ROUTE_PREFIX):])
