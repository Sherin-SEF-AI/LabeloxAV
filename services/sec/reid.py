"""Cross-camera person re-identification, as appearance linking and nothing more.

The question a multi-camera security deployment asks is "did the same person appear at both gates". The
question it must not be able to answer is "who is that". Those are separated here by construction, not by
policy: an identity row has a signature and no name, no reference photograph, and no enrolment path. There
is nowhere to write a name, so the system can say two tracks are the same person and can never say which
person, and that is the boundary between re-identification for an authorised security deployment and
building a face database.

Three further constraints, each because the alternative is worse:

- **Gated on the `reid` capability, which only the Sec pack declares.** The AV pack must not be able to
  follow a pedestrian between dashcams.
- **Signatures are derived from crop embeddings the corpus already holds**, not from faces. A body-crop
  appearance vector links a person across a few hours of the same clothing and degrades naturally after
  that, which is the right lifetime for a security event and the wrong one for identification.
- **A match needs a real margin, not merely the best score.** The nearest signature is always something;
  requiring it to beat the runner-up by a margin is what stops two similar coats becoming one person.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import PersonIdentity, PersonSighting

log = get_logger("sec_reid")

# Cosine similarity a candidate must reach to be considered the same person at all.
MATCH_THRESHOLD = 0.78
# And how far it must beat the next best. Without this, two people in similar coats collapse into one
# identity the moment one of them is slightly closer.
MATCH_MARGIN = 0.05
# Beyond this, an appearance match means nothing: clothing is the signal, and clothing changes.
MAX_LINK_HOURS = 12


class ReidError(Exception):
    """A re-identification operation refused."""


def _require_reid(pack_id: str | None) -> str:
    from services.anpr.recognize import AnprNotAuthorised
    from services.domain import default_pack_id, has_capability

    pid = pack_id or default_pack_id()
    if not has_capability("reid", pid):
        raise AnprNotAuthorised(
            f"cross-camera re-identification is not authorised for pack {pid!r}. It is a security-domain "
            "capability; a pack must declare 'reid'. Under the AV pack, following a pedestrian between "
            "dashcams is exactly what the privacy plane exists to prevent.")
    return pid


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def track_signature(vectors: list[list[float]]) -> list[float] | None:
    """One appearance vector for a whole track.

    The median rather than the mean, per dimension. A track picks up occluded and motion-blurred crops,
    and a mean is dragged by them; the median describes the appearance the track mostly had.
    """
    usable = [np.asarray(v, dtype=np.float32) for v in vectors if v is not None and len(v)]
    if not usable:
        return None
    dims = {len(v) for v in usable}
    if len(dims) > 1:
        # Mixed dimensions mean two embedders produced these, and averaging across them is meaningless.
        raise ReidError(f"signature vectors have mixed dimensions {sorted(dims)}")
    stacked = np.stack(usable)
    sig = np.median(stacked, axis=0)
    norm = np.linalg.norm(sig)
    return (sig / norm).tolist() if norm > 1e-9 else sig.tolist()


async def _track_vectors(db: AsyncSession, track_id: str, limit: int = 40) -> list[list[float]]:
    """Crop embeddings for the objects on one track."""
    from db.models import Object, ObjectEmbedding

    rows = (await db.execute(
        select(ObjectEmbedding.dino_vec)
        .join(Object, Object.object_id == ObjectEmbedding.object_id)
        .where(Object.track_id == uuid.UUID(track_id))
        .limit(limit))).scalars().all()
    return [list(v) for v in rows if v is not None]


async def match_track(db: AsyncSession, track_id: str, *, camera_id: str | None = None,
                      ts_ns: int | None = None, pack_id: str | None = None,
                      create_if_new: bool = True) -> dict:
    """Attribute one track to an existing signature, or mint a new one.

    Returns the decision and why it was made, including the runner-up. A match with no visible margin is
    the case an operator most needs to see, and reporting only the winner hides it.
    """
    pid = _require_reid(pack_id)
    vectors = await _track_vectors(db, track_id)
    sig = track_signature(vectors)
    if sig is None:
        return {"matched": False, "reason": "the track has no crop embeddings to compare",
                "track_id": track_id}

    query = np.asarray(sig, dtype=np.float32)
    candidates = (await db.execute(
        select(PersonIdentity).where(PersonIdentity.pack_id == pid))).scalars().all()

    scored: list[tuple[float, PersonIdentity]] = []
    for cand in candidates:
        if not cand.signature or len(cand.signature) != len(sig):
            continue
        if ts_ns and cand.last_ts_ns and \
                abs(int(ts_ns) - int(cand.last_ts_ns)) > MAX_LINK_HOURS * 3600 * 1_000_000_000:
            # Beyond the clothing horizon. Skipped rather than scored low, so it cannot become a match by
            # being the only candidate.
            continue
        scored.append((cosine(query, np.asarray(cand.signature, dtype=np.float32)), cand))
    scored.sort(key=lambda p: p[0], reverse=True)

    best = scored[0] if scored else None
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    margin = (best[0] - runner_up) if best else 0.0

    if best and best[0] >= MATCH_THRESHOLD and margin >= MATCH_MARGIN:
        identity = best[1]
        await _record_sighting(db, identity, track_id, camera_id, ts_ns, best[0], sig)
        log.info("sec.reid_matched", track=track_id[:8], similarity=round(best[0], 3))
        return {"matched": True, "identity_id": str(identity.identity_id),
                "similarity": round(best[0], 4), "margin": round(margin, 4),
                "runner_up": round(runner_up, 4), "track_id": track_id}

    if not create_if_new:
        return {"matched": False, "track_id": track_id,
                "reason": ("no candidate cleared the threshold" if not best or best[0] < MATCH_THRESHOLD
                           else "the best candidate did not beat the runner-up by enough"),
                "best": round(best[0], 4) if best else None,
                "runner_up": round(runner_up, 4)}

    identity = PersonIdentity(signature=sig, n_tracks=0, cameras=[], pack_id=pid,
                              first_ts_ns=ts_ns, last_ts_ns=ts_ns)
    db.add(identity)
    await db.flush()
    await _record_sighting(db, identity, track_id, camera_id, ts_ns, 1.0, sig)
    log.info("sec.reid_new_identity", track=track_id[:8])
    return {"matched": False, "created": True, "identity_id": str(identity.identity_id),
            "best": round(best[0], 4) if best else None, "track_id": track_id}


async def _record_sighting(db: AsyncSession, identity: PersonIdentity, track_id: str,
                           camera_id: str | None, ts_ns: int | None, similarity: float,
                           sig: list[float]) -> None:
    db.add(PersonSighting(identity_id=identity.identity_id,
                          track_id=uuid.UUID(track_id) if track_id else None,
                          camera_id=camera_id, ts_ns=int(ts_ns or 0),
                          similarity=float(similarity)))
    identity.n_tracks = int(identity.n_tracks or 0) + 1
    if camera_id and camera_id not in (identity.cameras or []):
        identity.cameras = [*(identity.cameras or []), camera_id]
    if ts_ns:
        identity.first_ts_ns = min(int(identity.first_ts_ns or ts_ns), int(ts_ns))
        identity.last_ts_ns = max(int(identity.last_ts_ns or ts_ns), int(ts_ns))

    # The signature drifts toward each confirmed sighting rather than being replaced. A person's appearance
    # changes gradually across a shift (a jacket comes off), and a signature pinned to the first track stops
    # matching them by the third camera.
    if similarity < 1.0:
        prev = np.asarray(identity.signature, dtype=np.float32)
        new = np.asarray(sig, dtype=np.float32)
        blended = 0.8 * prev + 0.2 * new
        norm = np.linalg.norm(blended)
        identity.signature = (blended / norm).tolist() if norm > 1e-9 else blended.tolist()
    await db.commit()


async def link_session(db: AsyncSession, session_id: str, *, pack_id: str | None = None) -> dict:
    """Attribute every track in a session, and raise an incident when one appears on a second camera."""
    from db.models import Frame, Object

    pid = _require_reid(pack_id)
    rows = (await db.execute(
        select(Object.track_id, Frame.cam_id, func.min(Frame.ts_ns))
        .join(Frame, Object.frame_id == Frame.frame_id)
        .where(Frame.session_id == uuid.UUID(session_id), Object.track_id.isnot(None))
        .group_by(Object.track_id, Frame.cam_id))).all()

    matched = created = cross_camera = 0
    for track_id, cam, ts in rows:
        out = await match_track(db, str(track_id), camera_id=cam, ts_ns=int(ts or 0), pack_id=pid)
        if out.get("matched"):
            matched += 1
            identity = await db.get(PersonIdentity, uuid.UUID(out["identity_id"]))
            if identity and len(identity.cameras or []) > 1:
                cross_camera += 1
                from services.sec.incidents import raise_incident

                await raise_incident(
                    db, kind="reid_match", camera_id=cam, session_id=session_id,
                    severity="info", ts_ns=int(ts or 0),
                    title=f"the same person appeared on {len(identity.cameras)} cameras",
                    summary=f"cameras: {', '.join(identity.cameras)}",
                    evidence={"identity_id": str(identity.identity_id),
                              "track_id": str(track_id), "cameras": identity.cameras},
                    person_identity=str(identity.identity_id), pack_id=pid)
        elif out.get("created"):
            created += 1

    log.info("sec.session_linked", session=session_id, tracks=len(rows),
             matched=matched, created=created)
    return {"session_id": session_id, "tracks": len(rows), "matched": matched,
            "new_identities": created, "cross_camera": cross_camera}


async def list_identities(db: AsyncSession, *, min_cameras: int = 1, limit: int = 100) -> dict:
    rows = (await db.execute(
        select(PersonIdentity).order_by(PersonIdentity.last_ts_ns.desc().nullslast())
        .limit(min(max(limit, 1), 500)))).scalars().all()
    out = []
    for r in rows:
        if len(r.cameras or []) < min_cameras:
            continue
        out.append({
            "identity_id": str(r.identity_id), "n_tracks": int(r.n_tracks or 0),
            "cameras": r.cameras or [], "first_ts_ns": r.first_ts_ns, "last_ts_ns": r.last_ts_ns,
            # The signature itself is never returned. It is the only thing here that could be matched
            # against another system's database, and exporting it would defeat the boundary this module
            # is built around.
            "signature_dim": len(r.signature or []),
        })
    return {"identities": out, "total": len(out)}


async def identity_detail(db: AsyncSession, identity_id: str) -> dict:
    identity = await db.get(PersonIdentity, uuid.UUID(identity_id))
    if identity is None:
        raise ReidError("identity not found")
    sightings = (await db.execute(
        select(PersonSighting).where(PersonSighting.identity_id == identity.identity_id)
        .order_by(PersonSighting.ts_ns))).scalars().all()
    return {
        "identity_id": str(identity.identity_id), "n_tracks": int(identity.n_tracks or 0),
        "cameras": identity.cameras or [], "signature_dim": len(identity.signature or []),
        "first_ts_ns": identity.first_ts_ns, "last_ts_ns": identity.last_ts_ns,
        "sightings": [{"sighting_id": str(s.sighting_id),
                       "track_id": str(s.track_id) if s.track_id else None,
                       "camera_id": s.camera_id, "ts_ns": int(s.ts_ns),
                       "similarity": round(float(s.similarity), 4)} for s in sightings],
    }


async def forget_identity(db: AsyncSession, identity_id: str) -> dict:
    """Delete a signature and everything attributed to it.

    Present because it must be. A signature is derived from a person's appearance, so it is personal data
    under DPDPA whether or not a name is attached, and an erasure request has to be able to reach it.
    """
    identity = await db.get(PersonIdentity, uuid.UUID(identity_id))
    if identity is None:
        return {"forgotten": False, "reason": "not found"}
    n = len(identity.cameras or [])
    await db.delete(identity)      # sightings cascade
    await db.commit()
    log.info("sec.identity_forgotten", identity=identity_id[:8])
    return {"forgotten": True, "identity_id": identity_id, "cameras": n,
            "at": datetime.now(UTC).isoformat()}
