"""Choosing the frames worth an open-vocabulary pass, instead of running one everywhere.

`run_recall` sources missed objects from three channels. Two of them are cheap: trackgap is interpolation and
costs nothing. The other two load models, and with no frame list it runs them over every frame in the
session. On 36,905 frames that is a bulk pre-labelling job wearing a recovery job's clothes, and bulk is
exactly where a locality-optimised server is the wrong tool.

The question is which frames are actually likely to be missing something. Two candidate signals, both
measured against the corpus before either was built, because the first one looked obvious and is wrong.

The scene tags cannot carry it. Frames tagged `sparse` hold a median of 36 objects and frames tagged `dense`
hold 13, so the density axis is inverted, mislabelled, or measuring something other than what its name says.
Only 1,780 of 36,905 frames carry a density tag at all. A selector built on that would target the opposite of
what it meant to and cover 5% of the corpus while doing it.

What does carry it is the frame's own neighbours. This footage is 3fps from a moving vehicle, so the scene
does not empty between consecutive frames: a frame holding far fewer objects than the frames either side of
it is a detection failure, not a quiet stretch of road. It needs no scene tag, no ontology, and no model, and
it is the same reasoning trackgap already uses per track, applied to the frame as a whole.

Measured, that selects 248 frames, 0.67% of the corpus, including twelve holding nothing at all while their
neighbours average five or more. The strongest case holds zero objects between frames averaging 87.9. Running
the model channels on 248 frames rather than 36,905 is the difference between a targeted assist and a second
pre-labelling pass, and it is what makes locality worth optimising for.

The selection rate is returned with the frames, because "targeted" is a claim about proportion and a
selector that quietly widens is indistinguishable from the blanket pass it replaced.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("recall.targeting")

# Frames either side to compare against. Three at 3fps is two seconds of context, long enough to smooth a
# genuine gap between vehicles and short enough that the scene has not changed.
WINDOW = 3

# Below this share of the neighbourhood mean, a frame is suspected of missing objects rather than of being
# emptier. Chosen against the measured distribution: at 0.4 it selects 0.67% of the corpus, which is a
# shortlist. Loosening it toward 1.0 turns the assist back into a blanket pass one notch at a time.
RATIO = 0.4

# Neighbourhoods thinner than this cannot support the inference. Two objects dropping to zero is noise; a
# frame is only anomalous against neighbours that had something to lose.
MIN_NEIGHBOURS = 5.0


def deficit(n_objects: int, neighbour_mean: float) -> float:
    """How much of its neighbourhood a frame is missing, in [0, 1].

    1.0 is a frame holding nothing where its neighbours are busy. Pure, so the ranking can be tested without
    a database, and expressed as a fraction rather than a raw difference so a quiet session and a crowded one
    are ranked on the same scale.
    """
    if neighbour_mean <= 0:
        return 0.0
    return round(max(0.0, min(1.0, 1.0 - (float(n_objects) / float(neighbour_mean)))), 4)


def is_suspicious(n_objects: int, neighbour_mean: float, *, ratio: float = RATIO,
                  min_neighbours: float = MIN_NEIGHBOURS) -> bool:
    """Whether a frame is worth paying a model to look at again."""
    if neighbour_mean < min_neighbours:
        return False
    return float(n_objects) < float(neighbour_mean) * ratio


_SELECT = """
with counted as (
  select f.frame_id, f.session_id, f.ts_ns,
         (select count(*) from object o where o.frame_id = f.frame_id) n
  from frame f
  where f.img_uri ~ '^s3://[^/]+/.+'
    {session_filter}
),
windowed as (
  select frame_id, session_id, n,
         avg(n) over (partition by session_id order by ts_ns
                      rows between :window preceding and :window following) nb
  from counted
)
select frame_id, session_id, n, nb
from windowed
where nb >= :min_neighbours and n < nb * :ratio
order by (1.0 - n / nullif(nb, 0)) desc, nb desc
limit :limit
"""


async def frames_worth_recovering(db: AsyncSession, *, session_id: str | uuid.UUID | None = None,
                                  limit: int = 500, window: int = WINDOW, ratio: float = RATIO,
                                  min_neighbours: float = MIN_NEIGHBOURS) -> dict:
    """Frames holding materially fewer objects than their temporal neighbours, worst first.

    Unfetchable frames are excluded here rather than discovered by the model channels, for the same reason
    the relabel walk excludes them: they can never succeed, and they would occupy a slot in a shortlist whose
    whole point is that it is short.
    """
    params: dict = {"window": int(window), "ratio": float(ratio),
                    "min_neighbours": float(min_neighbours), "limit": int(limit)}
    sql = _SELECT.format(session_filter="and f.session_id = :sid" if session_id else "")
    if session_id:
        params["sid"] = str(session_id)

    rows = (await db.execute(text(sql), params)).all()
    total = int((await db.execute(text(
        "select count(*) from frame where img_uri ~ '^s3://[^/]+/.+'"
        + (" and session_id = :sid" if session_id else "")),
        ({"sid": str(session_id)} if session_id else {}))).scalar() or 0)

    frames = [{"frame_id": str(fid), "session_id": str(sid), "objects": int(n),
               "neighbour_mean": round(float(nb), 2), "deficit": deficit(int(n), float(nb))}
              for fid, sid, n, nb in rows]

    share = (len(frames) / total * 100.0) if total else 0.0
    log.info("recall.targeting", selected=len(frames), of=total, pct=round(share, 3))
    return {
        "frames": frames,
        "selected": len(frames),
        "considered": total,
        # Returned because "targeted" is a claim about proportion. A selector that quietly widens is
        # indistinguishable from the blanket pass it exists to replace.
        "selected_pct": round(share, 3),
        "detail": (f"{len(frames)} of {total} frames hold under {ratio:.0%} of what their neighbours hold "
                   f"({share:.2f}% of the corpus)"),
    }


async def run_targeted_recall(db: AsyncSession, *, session_id: str | uuid.UUID | None = None,
                              limit: int = 200, **kw) -> dict:
    """Select the suspicious frames, then run recovery on only those.

    Grouped by session because `run_recall` reasons about tracks, which are per session: handing it a mixed
    list would have it look for temporal neighbours that belong to another vehicle on another day.
    """
    from services.recall.recover import run_recall

    picked = await frames_worth_recovering(db, session_id=session_id, limit=limit, **kw)
    by_session: dict[str, list[str]] = {}
    for f in picked["frames"]:
        by_session.setdefault(f["session_id"], []).append(f["frame_id"])

    results = []
    persisted = 0
    for sid, fids in by_session.items():
        out = await run_recall(db, sid, frame_ids=fids)
        persisted += int(out.get("persisted", 0))
        results.append({"session_id": sid, "frames": len(fids), **out})

    log.info("recall.targeted_run", sessions=len(by_session), frames=picked["selected"],
             persisted=persisted)
    return {
        "selected": picked["selected"],
        "considered": picked["considered"],
        "selected_pct": picked["selected_pct"],
        "sessions": len(by_session),
        "persisted": persisted,
        "detail": (f"ran recovery on {picked['selected']} frames ({picked['selected_pct']:.2f}% of the "
                   f"corpus) across {len(by_session)} sessions, persisting {persisted} candidates"),
        "per_session": results,
    }
