"""Dataset-as-query: turn a sentence into shards a dataloader can stream.

The existing export path produces a zip, which is right for handing a dataset to somebody and wrong for
training against one. These routes let a training config reference the query and the version instead of a
path, so what the config records is what the dataset means rather than where somebody happened to put it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.deps import db_session, require_role

router = APIRouter()


class BuildIn(BaseModel):
    query: str
    name: str | None = None
    version: str | None = None          # a sealed DatasetCommit id; pins the selection
    samples_per_shard: int = 256
    limit: int = 200_000


@router.get("/datasets/blob")
async def dataset_blob(uri: str):
    """Fetch a shard or index this API produced.

    Routed through the API rather than handing out bucket credentials, so one token governs access to the
    images, the labels and the artifacts alike. Confined to the datasets prefix: the parameter is caller
    supplied, and a blob reader that will fetch any URI it is given is a way to read the whole object store.
    """
    from core.storage import get_object_store

    if "/datasets/" not in uri:
        raise HTTPException(400, "this route serves dataset artifacts only")
    try:
        data = get_object_store().get_bytes(uri)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, "no such dataset artifact") from exc
    media = "application/x-tar" if uri.endswith(".tar") else "application/octet-stream"
    return Response(content=data, media_type=media)


@router.get("/datasets/vocabulary")
async def vocabulary():
    """Every term a query can use. Returned because the compiler refuses unknown terms rather than ignoring
    them, and a refusal is only helpful next to the list of what would have worked."""
    from services.datasets.query_lang import vocabulary as vocab

    return vocab()


@router.get("/datasets/compile")
async def compile_only(q: str):
    """The predicate a query compiles to, without building anything.

    Separate from the build because a dataset whose contents cannot be explained is one nobody can defend,
    and the cheapest moment to notice that a term expanded to the wrong classes is before the shards exist.
    """
    from services.datasets.query_lang import QueryError, compile_query

    try:
        return compile_query(q)
    except QueryError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/datasets/preview")
async def preview(q: str, version: str | None = None, limit: int = 200_000,
                  db: AsyncSession = Depends(db_session)):
    """How many frames and objects the query selects, and the class mix, without writing a byte.

    "How many riders at night does this actually contain" is a question asked before training, not after.
    """
    from collections import Counter

    from services.datasets.query_lang import QueryError, compile_query
    from services.datasets.shards import resolve_selection

    try:
        compiled = compile_query(q)
    except QueryError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        sel = await resolve_selection(db, compiled["predicate"], version=version, limit=limit)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    classes: Counter = Counter()
    n_obj = 0
    for objs in sel["objects"].values():
        n_obj += len(objs)
        classes.update(o["class_name"] for o in objs if o["class_name"])
    return {"query": q, "predicate": sel["predicate"], "terms": compiled["terms"],
            "version": version, "sealed": sel["sealed"],
            "frames": len(sel["frames"]), "objects": n_obj,
            "classes": dict(classes.most_common(30))}


@router.post("/datasets/shards", dependencies=[Depends(require_role("reviewer"))])
async def build(payload: BuildIn, db: AsyncSession = Depends(db_session)):
    """Build WebDataset shards plus a Parquet index for a query."""
    import re

    from services.datasets.query_lang import QueryError, compile_query
    from services.datasets.shards import build_shards

    try:
        compiled = compile_query(payload.query)
    except QueryError as exc:
        raise HTTPException(400, str(exc)) from exc

    # A name derived from the query keeps the artifact self-describing when somebody finds it in the bucket
    # a year later, which a uuid does not.
    name = payload.name or re.sub(r"[^a-z0-9]+", "-", payload.query.lower()).strip("-")[:60]
    try:
        return await build_shards(db, compiled["predicate"], name=name, version=payload.version,
                                  samples_per_shard=payload.samples_per_shard, limit=payload.limit)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/datasets/lake/publish", dependencies=[Depends(require_role("admin"))])
async def publish_lake(db: AsyncSession = Depends(db_session)):
    """Publish the annotation plane to Iceberg for the customer's own query engine.

    Admin, not reviewer: this writes the whole label corpus to a location outside the API's access control,
    which is a data-egress decision rather than a reporting one.
    """
    from services.datasets.lake import publish

    return await publish(db)


@router.get("/datasets/lake/{table}")
async def read_lake(table: str, limit: int = 100):
    """A sample of a published table, so the lake is checkable without a query engine to hand."""
    from services.datasets.lake import scan

    try:
        return {"table": table, "rows": scan(table, limit=min(limit, 1000))}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(409, f"table not published yet: {exc}") from exc
