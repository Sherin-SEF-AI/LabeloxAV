"""The least-covered corners of the corpus, found from the frame embeddings (tool #14).

A scene is under-covered when nothing else in the corpus looks like it. "Nothing else" has to mean other
drives: consecutive frames of one drive are near-duplicates of each other, so a frame's nearest neighbours
inside its own session say only that the camera was running. The measure here is therefore the mean cosine
distance to the k nearest frames from OTHER sessions (a cross-session k-NN distance). A frame of a cow on a
flyover at night, filmed once, scores high however long that one drive lingered on it; a hundred drives
down the same arterial road at noon score low.

Three more numbers explain a score rather than restate it:

  cluster     k-means over the same vectors; how many frames and how many drives share this frame's cluster
  nearest     the closest frame from another drive, and how close it is (the corpus median is reported
              alongside, so "0.71" means something)
  coverage    how many objects are labelled on the frame, and the scene axes (time of day, weather) with how
              common that combination is in the corpus

Only real frames are ranked (`Frame.origin == 'real'`): a composite or a perturbed frame is rare by
construction, and surfacing one as a coverage gap would point labelling effort at the generator. There is no
other fixture filter in this codebase (the fixture sessions were purged from the corpus on 2026-07-30, and
the conftest guard keeps tests out of it), so origin is the filter.

The full ranking is CPU work over every vector (about 42k x 768 here), so it runs as a background tool job
and its result is the job's result. Search (by text or by example) ranks a small candidate set and scores
each candidate's rarity on the fly through pgvector, so it needs no prior run.
"""

from __future__ import annotations

import asyncio
import math
import uuid
from collections import Counter

import cv2
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from core.origin import REAL
from db.models import Frame, FrameEmbedding, Object
from db.models import Session as DbSession

log = get_logger("intelligence.rare_scenes")

SPACES = ("dino", "siglip")
DEFAULT_K = 10
MAX_FRAMES = 80_000


def _col(space: str):
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}")
    return FrameEmbedding.dino_vec if space == "dino" else FrameEmbedding.siglip_vec


async def load_vectors(db: AsyncSession, *, space: str = "dino", session_id: uuid.UUID | None = None,
                       city: str | None = None, max_frames: int = MAX_FRAMES):
    """(frame ids, session ids, L2-normalised matrix) for the real, embedded frames in scope."""
    col = _col(space)
    q = (select(FrameEmbedding.frame_id, Frame.session_id, col)
         .join(Frame, Frame.frame_id == FrameEmbedding.frame_id)
         .where(Frame.origin == REAL, col.isnot(None)))
    if session_id is not None:
        q = q.where(Frame.session_id == session_id)
    if city:
        q = q.join(DbSession, DbSession.session_id == Frame.session_id).where(DbSession.city == city)
    rows = (await db.execute(q.order_by(Frame.ts_ns).limit(max_frames))).all()
    if not rows:
        return [], [], np.zeros((0, 0), dtype=np.float32)
    ids = [str(r[0]) for r in rows]
    sids = [str(r[1]) for r in rows]
    X = np.asarray(np.stack([np.asarray(r[2], dtype=np.float32) for r in rows]), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return ids, sids, X


def cross_session_knn(X: np.ndarray, sessions: list[str], k: int = DEFAULT_K,
                      chunk: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: mean cosine distance to its k nearest rows from other sessions, plus the nearest one.

    Returns (score, nearest index, nearest similarity). A row whose session is the only one in scope has
    no other-session neighbour; it gets score NaN and index -1, and the caller says so rather than ranking
    it as infinitely rare.
    """
    n = X.shape[0]
    score = np.full(n, np.nan, dtype=np.float32)
    nn_idx = np.full(n, -1, dtype=np.int64)
    nn_sim = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return score, nn_idx, nn_sim
    _, sess_code = np.unique(np.asarray(sessions), return_inverse=True)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        sims = X[s:e] @ X.T                                   # (chunk, n)
        same = sess_code[s:e, None] == sess_code[None, :]
        sims[same] = -np.inf
        kk = min(k, n - 1)
        if kk <= 0:
            continue
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        top = np.take_along_axis(sims, part, axis=1)
        finite = np.isfinite(top)
        cnt = finite.sum(axis=1)
        mean_sim = np.where(cnt > 0, np.where(finite, top, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
        score[s:e] = (1.0 - mean_sim).astype(np.float32)
        best = np.argmax(sims, axis=1)
        bsim = sims[np.arange(e - s), best]
        ok = np.isfinite(bsim)
        nn_idx[s:e] = np.where(ok, best, -1)
        nn_sim[s:e] = np.where(ok, bsim, np.nan)
    return score, nn_idx, nn_sim


def cluster_labels(X: np.ndarray, n_clusters: int | None = None, seed: int = 0) -> np.ndarray:
    """k-means labels over the normalised vectors; about sqrt(n)/2 clusters, between 4 and 256."""
    n = X.shape[0]
    if n < 8:
        return np.zeros(n, dtype=np.int64)
    from sklearn.cluster import MiniBatchKMeans

    kc = n_clusters or int(min(256, max(4, round(math.sqrt(n) / 2))))
    km = MiniBatchKMeans(n_clusters=kc, random_state=seed, batch_size=2048, n_init=3)
    return km.fit_predict(X).astype(np.int64)


def pick_diverse(order: list[int], sessions: list[str], clusters: np.ndarray, *, top: int,
                 per_session: int, per_cluster: int) -> list[int]:
    """Walk rows rarest first, keeping at most `per_session` per drive and `per_cluster` per cluster.

    Without the caps one odd drive fills the whole grid with 300 views of the same rare thing, which is one
    coverage gap reported 300 times.
    """
    by_s: Counter = Counter()
    by_c: Counter = Counter()
    out: list[int] = []
    for i in order:
        if by_s[sessions[i]] >= per_session or by_c[int(clusters[i])] >= per_cluster:
            continue
        out.append(i)
        by_s[sessions[i]] += 1
        by_c[int(clusters[i])] += 1
        if len(out) >= top:
            break
    return out


def _scene_key(scene: dict | None) -> str | None:
    s = scene or {}
    parts = [str(s[k]) for k in ("time_of_day", "weather") if s.get(k)]
    return " + ".join(parts) if parts else None


async def _scene_shares(db: AsyncSession) -> tuple[dict[str, float], int]:
    tod, wx = Frame.scene["time_of_day"].astext, Frame.scene["weather"].astext
    rows = (await db.execute(
        select(tod, wx, func.count()).where(Frame.origin == REAL).group_by(tod, wx))).all()
    # Over the frames whose scene has been classified: most of the corpus has no scene axes yet, and counting
    # those in the denominator made "day + clear" read as 2% of the corpus when it is most of what is known.
    total = sum(int(n) for tod, wx, n in rows if _scene_key({"time_of_day": tod, "weather": wx}))
    shares: dict[str, float] = {}
    for tod, wx, n in rows:
        key = _scene_key({"time_of_day": tod, "weather": wx})
        if key:
            shares[key] = shares.get(key, 0.0) + int(n) / max(1, total)
    return shares, total


async def _frame_facts(db: AsyncSession, frame_ids: list[str]) -> dict[str, dict]:
    ids = [uuid.UUID(f) for f in frame_ids]
    if not ids:
        return {}
    facts: dict[str, dict] = {}
    for fid, sid, scene, ts, vehicle, city, w, h in (await db.execute(
            select(Frame.frame_id, Frame.session_id, Frame.scene, Frame.ts_ns, DbSession.vehicle_id,
                   DbSession.city, Frame.width, Frame.height)
            .join(DbSession, DbSession.session_id == Frame.session_id)
            .where(Frame.frame_id.in_(ids)))).all():
        facts[str(fid)] = {"session_id": str(sid), "scene": scene or {}, "ts_ns": ts, "vehicle_id": vehicle,
                           "city": city, "width": w, "height": h, "n_objects": 0, "n_reviewed": 0}
    for fid, n, nrev in (await db.execute(
            select(Object.frame_id, func.count(),
                   func.count().filter(Object.state.in_(("accepted", "rejected", "settled"))))
            .where(Object.frame_id.in_(ids)).group_by(Object.frame_id))).all():
        if str(fid) in facts:
            facts[str(fid)]["n_objects"] = int(n)
            facts[str(fid)]["n_reviewed"] = int(nrev)
    return facts


def why_rare(row: dict, *, median_nn: float | None, shares: dict[str, float]) -> list[str]:
    """Plain sentences for one ranked frame, each restating a measured number."""
    out: list[str] = []
    sim = row.get("nn_sim")
    if sim is None:
        out.append("No frame from another drive is in scope to compare it with.")
    else:
        med = f" (half of all frames have a match above {median_nn:.2f})" if median_nn is not None else ""
        low = sim < median_nn - 0.02 if median_nn is not None else sim < 0.85
        if low:
            out.append(f"The closest frame from any other drive is only {sim:.2f} similar{med}.")
        else:
            out.append(f"Well covered: a frame from another drive is {sim:.2f} similar{med}.")
    cs, cd = row.get("cluster_size"), row.get("cluster_sessions")
    if cs is not None:
        drives = "one drive" if cd == 1 else f"{cd} drives"
        out.append(f"Its visual cluster holds {cs} frame{'s' if cs != 1 else ''} from {drives}.")
    key = _scene_key(row.get("scene"))
    if key and key in shares and shares[key] < 0.05:
        out.append(f"{key.capitalize()} is {shares[key] * 100:.1f}% of the frames with a known scene.")
    cr = (row.get("scene") or {}).get("rarity")
    if isinstance(cr, int | float) and cr >= 0.3:
        out.append(f"It carries uncommon classes (class rarity {cr:.2f}).")
    n = row.get("n_objects", 0)
    if not n:
        out.append("Nothing is labelled on it yet.")
    elif not row.get("n_reviewed"):
        out.append(f"{n} label{'s' if n != 1 else ''}, none reviewed by a person.")
    return out


def pixel_screen(image_bgr: np.ndarray | None) -> str | None:
    """Why a frame is not a scene at all, or None when it looks like one.

    The rarest frames in an embedding space are not only rare scenes: a blank or corrupt decode (a flat grey
    field with a few smears) has no neighbours either, and ranking it as a coverage gap would point labelling
    at a broken frame. On the dev corpus the first pass surfaced exactly that, so the shortlist is screened by
    pixels (the method the July fixture purge used): a frame most of whose pixels sit within a few levels of
    one grey value, or that holds only a handful of distinct colours, is set aside with the reason.
    """
    if image_bgr is None or image_bgr.size == 0:
        return "unreadable"
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    med = float(np.median(gray))
    if float((np.abs(gray.astype(np.int16) - med) <= 6).mean()) >= 0.8:
        return "mostly one flat grey: a blank or corrupt frame, not a scene"
    small = image_bgr[::4, ::4].reshape(-1, image_bgr.shape[2] if image_bgr.ndim == 3 else 1)
    if len(np.unique(small, axis=0)) <= 8:
        return "only a few distinct colours: a test pattern or fill, not a scene"
    return None


async def _screen_frames(db: AsyncSession, frame_ids: list[str]) -> dict[str, str | None]:
    from core.storage import get_object_store

    store = get_object_store()
    uris = dict((await db.execute(select(Frame.frame_id, Frame.img_uri).where(
        Frame.frame_id.in_([uuid.UUID(f) for f in frame_ids])))).all())

    def one(uri: str | None) -> str | None:
        if not uri:
            return "no image"
        try:
            img = cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            return "unreadable"
        return pixel_screen(img)

    out: dict[str, str | None] = {}
    for f in frame_ids:
        out[f] = await asyncio.to_thread(one, uris.get(uuid.UUID(f)))
    return out


async def rank_corpus(db: AsyncSession, *, space: str = "dino", k: int = DEFAULT_K, top: int = 300,
                      per_session: int = 4, per_cluster: int = 6, session_id: str | None = None,
                      city: str | None = None, screen_pixels: bool = True, progress=None) -> dict:
    """Rank every real, embedded frame in scope by cross-session k-NN distance; keep the rarest, diversely."""
    sid = uuid.UUID(str(session_id)) if session_id else None
    if progress:
        await progress(0.05, "loading vectors")
    ids, sids, X = await load_vectors(db, space=space, session_id=sid, city=city)
    excluded = int((await db.execute(
        select(func.count()).select_from(Frame).where(Frame.origin != REAL))).scalar() or 0)
    unembedded = int((await db.execute(
        select(func.count()).select_from(Frame).outerjoin(FrameEmbedding, FrameEmbedding.frame_id == Frame.frame_id)
        .where(Frame.origin == REAL, _col(space).is_(None)))).scalar() or 0)
    base = {"space": space, "k": k, "n_frames": len(ids), "n_sessions": len(set(sids)),
            "excluded_not_real": excluded, "not_embedded": unembedded,
            "params": {"space": space, "k": k, "top": top, "per_session": per_session,
                       "per_cluster": per_cluster, "session_id": session_id, "city": city}}
    if len(set(sids)) < 2:
        return {**base, "frames": [], "detail": "rarity is measured against other drives, and fewer than "
                                                  "two drives are in scope"}
    if progress:
        await progress(0.25, "measuring distances")
    score, nn_idx, nn_sim = await asyncio.to_thread(cross_session_knn, X, sids, k)
    if progress:
        await progress(0.65, "clustering")
    labels = await asyncio.to_thread(cluster_labels, X)
    del X
    c_size = Counter(int(c) for c in labels)
    c_sess: dict[int, set] = {}
    for i, c in enumerate(labels):
        c_sess.setdefault(int(c), set()).add(sids[i])

    valid = np.where(np.isfinite(score))[0]
    order = valid[np.argsort(-score[valid])].tolist()
    # Over-pick, then set aside what the pixel screen says is not a scene, and keep the rest up to `top`.
    shortlist = pick_diverse(order, sids, labels, top=int(top * 1.5) + 10, per_session=per_session,
                             per_cluster=per_cluster)
    suspect: list[dict] = []
    if screen_pixels:
        if progress:
            await progress(0.75, "screening pixels")
        verdict = await _screen_frames(db, [ids[i] for i in shortlist])
        suspect = [{"frame_id": ids[i], "session_id": sids[i], "score": round(float(score[i]), 4),
                    "reason": verdict[ids[i]], "image_url": f"/api/frames/{ids[i]}/image"}
                   for i in shortlist if verdict.get(ids[i])]
        shortlist = [i for i in shortlist if not verdict.get(ids[i])]
    picked = shortlist[:top]
    fin = score[valid]
    pct_of = {int(i): float((fin < score[i]).mean()) for i in picked}
    sims_ok = nn_sim[np.isfinite(nn_sim)]
    median_nn = float(np.median(sims_ok)) if sims_ok.size else None

    if progress:
        await progress(0.85, "explaining")
    facts = await _frame_facts(db, [ids[i] for i in picked])
    shares, _total = await _scene_shares(db)
    frames = []
    for i in picked:
        f = facts.get(ids[i], {})
        row = {"frame_id": ids[i], "session_id": sids[i], "score": round(float(score[i]), 4),
               "percentile": round(pct_of[i], 4),
               "nn_frame_id": ids[int(nn_idx[i])] if nn_idx[i] >= 0 else None,
               "nn_sim": round(float(nn_sim[i]), 4) if np.isfinite(nn_sim[i]) else None,
               "cluster": int(labels[i]), "cluster_size": c_size[int(labels[i])],
               "cluster_sessions": len(c_sess[int(labels[i])]),
               "scene": f.get("scene") or {}, "vehicle_id": f.get("vehicle_id"), "city": f.get("city"),
               "n_objects": f.get("n_objects", 0), "n_reviewed": f.get("n_reviewed", 0),
               "image_url": f"/api/frames/{ids[i]}/image"}
        row["why"] = why_rare(row, median_nn=round(median_nn, 4) if median_nn is not None else None, shares=shares)
        frames.append(row)
    smallest = sorted(c_size.items(), key=lambda kv: kv[1])[:12]
    out = {**base, "median_nn_sim": round(median_nn, 4) if median_nn is not None else None,
           "score_quantiles": {q: round(float(np.quantile(fin, q / 100)), 4) for q in (50, 90, 99)} if fin.size else {},
           "n_clusters": len(c_size),
           "smallest_clusters": [{"cluster": c, "size": n, "sessions": len(c_sess[c])} for c, n in smallest],
           "frames": frames, "suspect_frames": suspect[:100], "n_suspect": len(suspect)}
    log.info("rare_scenes.ranked", n=len(ids), kept=len(frames), space=space)
    return out


async def rarity_of(db: AsyncSession, frame_ids: list[str], *, space: str = "dino",
                    k: int = DEFAULT_K) -> dict[str, dict]:
    """Cross-session k-NN rarity for a handful of frames, through pgvector (no prior ranking needed)."""
    from core.embeddings import tune_recall

    col = _col(space)
    ids = [uuid.UUID(str(f)) for f in frame_ids]
    if not ids:
        return {}
    rows = (await db.execute(
        select(FrameEmbedding.frame_id, Frame.session_id, col)
        .join(Frame, Frame.frame_id == FrameEmbedding.frame_id)
        .where(FrameEmbedding.frame_id.in_(ids), col.isnot(None)))).all()
    out: dict[str, dict] = {}
    await tune_recall(db)
    for fid, sid, vec in rows:
        dist = col.cosine_distance([float(x) for x in vec]).label("d")
        # Over-fetch then drop the frame's own drive: the HNSW scan cannot filter on a joined column
        # without losing recall, and a drive is rarely more than a few hundred frames.
        nbrs = (await db.execute(
            select(FrameEmbedding.frame_id, Frame.session_id, dist)
            .join(Frame, Frame.frame_id == FrameEmbedding.frame_id)
            .where(col.isnot(None), Frame.origin == REAL, Frame.session_id != sid)
            .order_by(dist).limit(k))).all()
        if not nbrs:
            out[str(fid)] = {"score": None, "nn_frame_id": None, "nn_sim": None}
            continue
        sims = [1.0 - float(d) for _f, _s, d in nbrs]
        out[str(fid)] = {"score": round(1.0 - sum(sims) / len(sims), 4), "nn_frame_id": str(nbrs[0][0]),
                         "nn_sim": round(sims[0], 4)}
    return out


async def search(db: AsyncSession, *, text: str | None = None, frame_id: str | None = None,
                 space: str | None = None, k: int = 48, rarest_first: bool = False,
                 session_id: str | None = None, encode_text=None) -> dict:
    """Frames matching a text query (SigLIP2) or an example frame (DINOv3 by default), each with its rarity.

    `encode_text` is injectable so tests can search without loading SigLIP2.
    """
    from services.intelligence.search.similar import find_similar_frames

    sid = uuid.UUID(str(session_id)) if session_id else None
    if text and text.strip():
        space = "siglip"
        if encode_text is None:
            from core.embeddings import prompt_vector
            from services.intelligence.embed import siglip2

            qv = await asyncio.to_thread(prompt_vector, text, siglip2)
        else:
            qv = list(encode_text(text))
        mode = "text"
    elif frame_id:
        space = space or "dino"
        emb = await db.get(FrameEmbedding, uuid.UUID(str(frame_id)))
        vec = None if emb is None else (emb.dino_vec if space == "dino" else emb.siglip_vec)
        if vec is None:
            return {"mode": "example", "space": space, "results": [], "detail": "that frame is not embedded yet"}
        qv = [float(x) for x in vec]
        mode = "example"
    else:
        raise ValueError("give a text query or an example frame")

    hits = await find_similar_frames(db, qv, space=space, k=min(200, k * 2), session_id=sid,
                                     exclude_frame_id=uuid.UUID(str(frame_id)) if mode == "example" else None)
    # Real frames only, as the ranking does.
    fids = [h["frame_id"] for h in hits]
    real = {str(f) for f in (await db.execute(
        select(Frame.frame_id).where(Frame.frame_id.in_([uuid.UUID(f) for f in fids]), Frame.origin == REAL)
    )).scalars().all()} if fids else set()
    hits = [h for h in hits if h["frame_id"] in real][:k]
    rar = await rarity_of(db, [h["frame_id"] for h in hits], space="dino")
    facts = await _frame_facts(db, [h["frame_id"] for h in hits])
    shares, _ = await _scene_shares(db)
    results = []
    for h in hits:
        f = facts.get(h["frame_id"], {})
        r = rar.get(h["frame_id"], {})
        row = {"frame_id": h["frame_id"], "session_id": h.get("session_id"), "match": round(float(h["sim"]), 4),
               "score": r.get("score"), "nn_frame_id": r.get("nn_frame_id"), "nn_sim": r.get("nn_sim"),
               "scene": f.get("scene") or {}, "vehicle_id": f.get("vehicle_id"), "city": f.get("city"),
               "n_objects": f.get("n_objects", 0), "n_reviewed": f.get("n_reviewed", 0),
               "image_url": f"/api/frames/{h['frame_id']}/image"}
        row["why"] = why_rare(row, median_nn=None, shares=shares)
        results.append(row)
    if rarest_first:
        results.sort(key=lambda r: -(r["score"] if r["score"] is not None else -1.0))
    return {"mode": mode, "space": space, "query": text if mode == "text" else frame_id, "results": results}
