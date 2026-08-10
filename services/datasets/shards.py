"""Stream a query straight into a dataloader, as WebDataset shards with a Parquet index.

An export today is a job that produces a zip. That is the right shape for handing a dataset to somebody, and
the wrong shape for training against one: it has to finish before anything can start, it lands on a disk
somebody has to manage, and the thing a training config ends up referencing is a path, which says nothing
about what is inside it.

Shards invert that. A `.tar` of frames and labels is what `webdataset` reads natively, so a PyTorch
dataloader can begin on the first shard while the rest are still being built, and the config references the
query and the version instead of a path. When the config says
`labelox.load("night AND vru", version="2026.07.1")`, the dataset is reproducible by construction and the
reader needs no LabeloxAV code at all: the format is ordinary tar.

The Parquet index sits beside the shards, one row per sample, so the same selection can be inspected with
DuckDB before a single image is downloaded. "How many riders at night does this actually contain" is a
question people ask before training, not after.

Version pinning goes through `DatasetCommit`, which already seals a slice spec with an ontology version and
a commit id, so a pinned load resolves to the frames that commit sealed rather than to whatever the
predicate matches today.
"""

from __future__ import annotations

import io
import json
import tarfile

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.storage import get_object_store
from db.models import DatasetCommit, Frame, Object

log = get_logger("datasets.shards")

# Samples per shard. WebDataset wants shards big enough that the reader is not opening files constantly and
# small enough that a worker can start on one without waiting: a few hundred megabytes is the usual advice,
# which at dashcam frame sizes is roughly this many.
SAMPLES_PER_SHARD = 256

MAX_SAMPLES = 200_000


async def resolve_selection(db: AsyncSession, predicate: dict, *, version: str | None = None,
                            limit: int = MAX_SAMPLES) -> dict:
    """The frames a query selects, and the objects on each.

    A pinned version resolves through the sealed commit rather than re-running the predicate, because the
    point of pinning is that the answer cannot move when the corpus does.
    """
    from services.explore.query import frame_select

    pinned = None
    if version:
        pinned = await db.get(DatasetCommit, version)
        if pinned is None:
            raise ValueError(f"no dataset commit '{version}'; pin a version that exists or omit it to take "
                             "the corpus as it stands now")
        # The commit's own spec wins. A caller who pins a version and passes a different query is asking two
        # incompatible questions, and answering the newer one silently would make the pin meaningless.
        predicate = dict(pinned.slice_spec or {}) or predicate

    stmt = frame_select(predicate, Frame.frame_id, Frame.img_uri, Frame.width, Frame.height,
                        Frame.ts_ns, Frame.cam_id, Frame.session_id).limit(limit)
    frames = (await db.execute(stmt)).all()
    fids = [r[0] for r in frames]

    objs: dict = {}
    if fids:
        from services.autolabel.ontology import get_ontology
        onto = get_ontology()
        rows = (await db.execute(
            select(Object.frame_id, Object.object_id, Object.class_id, Object.bbox,
                   Object.state, Object.conf, Object.attrs)
            .where(Object.frame_id.in_(fids)))).all()
        # Class-filtered queries constrain which frames come back, not which objects ride along on them: a
        # sample is a frame and its labels, and silently dropping the other classes would produce a training
        # set where every unlabelled car is background.
        for fid, oid, cid, bbox, state, conf, attrs in rows:
            objs.setdefault(str(fid), []).append({
                "object_id": str(oid), "class_id": int(cid),
                "class_name": onto.by_id(cid).name if cid else None,
                "bbox": [float(v) for v in (bbox or [])], "state": state,
                "conf": (float(conf) if conf is not None else None), "attrs": attrs or {}})

    return {"frames": frames, "objects": objs, "predicate": predicate,
            "version": version, "sealed": bool(pinned)}


def _index_rows(frames, objs) -> list[dict]:
    """One row per sample for the Parquet index: enough to answer "what is in here" without the images."""
    out = []
    for fid, _uri, w, h, ts, cam, sid in frames:
        os_ = objs.get(str(fid), [])
        out.append({
            "frame_id": str(fid), "session_id": str(sid), "cam_id": cam, "ts_ns": int(ts or 0),
            "width": int(w or 0), "height": int(h or 0),
            "n_objects": len(os_),
            "classes": sorted({o["class_name"] for o in os_ if o["class_name"]}),
            "states": sorted({o["state"] for o in os_ if o["state"]}),
        })
    return out


async def build_shards(db: AsyncSession, predicate: dict, *, name: str, version: str | None = None,
                       samples_per_shard: int = SAMPLES_PER_SHARD, limit: int = MAX_SAMPLES) -> dict:
    """Write WebDataset shards plus a Parquet index to the object store. Returns their URIs."""
    sel = await resolve_selection(db, predicate, version=version, limit=limit)
    frames, objs = sel["frames"], sel["objects"]
    if not frames:
        return {"name": name, "shards": [], "index_uri": None, "samples": 0,
                "predicate": sel["predicate"],
                "reason": "the query selected no frames, which is an answer worth seeing rather than an "
                          "empty shard nobody can distinguish from a failed build"}

    store = get_object_store()
    prefix = f"datasets/{name}"
    shard_uris: list[str] = []
    written = missing = 0

    buf, tar, in_shard, shard_i = None, None, 0, 0

    def _open():
        nonlocal buf, tar
        buf = io.BytesIO()
        tar = tarfile.open(fileobj=buf, mode="w")

    def _close() -> str | None:
        nonlocal buf, tar, shard_i
        if tar is None:
            return None
        tar.close()
        uri = store.put_bytes(f"{prefix}/shard-{shard_i:05d}.tar", buf.getvalue(), "application/x-tar")
        shard_i += 1
        return uri

    _open()
    for fid, uri, _w, _h, _ts, _cam, _sid in frames:
        try:
            img = store.get_bytes(uri)
        except Exception:  # noqa: BLE001  a frame whose blob is gone must not lose the shard
            missing += 1
            continue
        key = str(fid)
        # WebDataset pairs files by the part of the name before the first dot, so a sample is `<key>.jpg`
        # beside `<key>.json`. Nothing here is LabeloxAV-specific; any tar reader can consume it.
        for ext, payload, in (("jpg", img),
                              ("json", json.dumps({"frame_id": key,
                                                   "objects": objs.get(key, [])}).encode())):
            info = tarfile.TarInfo(f"{key}.{ext}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        written += 1
        in_shard += 1
        if in_shard >= samples_per_shard:
            shard_uris.append(_close())
            _open()
            in_shard = 0
    if in_shard:
        shard_uris.append(_close())
    else:
        tar.close()

    index = _index_rows(frames, objs)
    index_uri = None
    try:
        import pandas as pd
        b = io.BytesIO()
        pd.DataFrame(index).to_parquet(b, index=False)
        index_uri = store.put_bytes(f"{prefix}/index.parquet", b.getvalue(),
                                    "application/vnd.apache.parquet")
    except Exception as exc:  # noqa: BLE001
        # The index is a convenience over the shards, not the dataset. Losing it must not lose the build,
        # but it must be reported rather than quietly absent.
        log.warning("datasets.index_failed", name=name, error=str(exc))

    log.info("datasets.shards_built", name=name, samples=written, shards=len(shard_uris),
             missing_media=missing, sealed=sel["sealed"])
    return {"name": name, "shards": [u for u in shard_uris if u], "index_uri": index_uri,
            "samples": written, "shard_count": len(shard_uris),
            "missing_media": missing, "predicate": sel["predicate"],
            "version": version, "sealed": sel["sealed"]}
