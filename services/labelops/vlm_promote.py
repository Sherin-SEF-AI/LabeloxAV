"""VLM-confirmed promotion of review-state detections, as a revertible agent write.

This is the iteration-6 lever (scripts/vlm_review.py) made callable by the fleet: rare classes stall
the champion gate with tens of thousands of their detections sitting unreviewed in `review`, and the
gate already trusts a VLM confirmation for rare classes. The script proved the lever (cattle recall
0.14 -> 0.59) and then required a person to remember it existed. The differences from the script are
exactly the ones an unattended caller needs:

- every promoted object is stamped `provenance.agent_run_id` and its prior state/source recorded, so
  the whole night is one `revert_run` away (the generic field-driven restore in services/agent/runs.py
  handles it with no new code);
- the VLM call runs in a thread, because the verifier is a synchronous HTTP client and the caller is
  the governance daemon's event loop;
- a dead or absent VLM endpoint is a per-class refusal with the error text, never an exception - a
  lever that cannot fire tonight reports that, and the rest of the unblock attempt proceeds.

The promotion writes `state='accepted', source='vlm_review'`: the source is the provenance that keeps
it out of every reader that means "a person ruled" (gold selection filters source=='human'; judge
calibration is built from Review rows, which this deliberately does not create). The resulting metric
gain is "under VLM review", not proof of the human loop, same as the script always said.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict

from sqlalchemy import select

from core.logging import get_logger
from db.models import Frame, Object

log = get_logger("labelops.vlm_promote")

# Chunk size for persistence: a mid-run interruption keeps its progress, and no transaction holds
# hundreds of row locks while a metered model call is in flight.
_CHUNK = 50


async def promote_class(class_name: str, *, per_class: int, min_conf: float, oversample: int,
                        agent_run_id: uuid.UUID, verifier=None) -> dict:
    """Promote up to `per_class` VLM-confirmed review-state detections of one class.

    Returns {"class", "seen", "confirmed", "promoted", "changes", ...} where `changes` is the
    object_id -> {"from_state", "from_source"} mapping the AgentRun needs for the generic revert.
    Opens its own sessions (worker context): the caller holds no transaction across model calls.
    """
    from core.storage import get_object_store
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.autolabel.paths.path_c_vlm import VlmVerifier, make_vlm_client
    from services.intelligence.embed.service import _decode

    onto = get_ontology()
    cls = onto.by_name(class_name)
    if cls is None:
        return {"class": class_name, "error": f"'{class_name}' is not in the ontology", "promoted": 0,
                "changes": {}}

    if verifier is None:
        try:
            verifier = VlmVerifier(make_vlm_client(), onto)
        except Exception as exc:  # noqa: BLE001 - no VLM tonight is a reason, not a crash
            return {"class": class_name, "error": f"no VLM client: {str(exc)[:200]}", "promoted": 0,
                    "changes": {}}

    maker = get_sessionmaker()
    async with maker() as db:
        # Highest-confidence review-state detections first: the detector is surest, so the VLM is most
        # likely to confirm, which is what we want for clean training labels (not the marginal ones).
        rows = (await db.execute(
            select(Object.object_id, Object.bbox, Object.state, Object.source, Frame.img_uri,
                   Frame.frame_id)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .where(Object.class_id == cls.id, Object.state == "review", Object.conf >= min_conf)
            .order_by(Object.conf.desc()).limit(per_class * oversample))).all()

    by_frame: dict = defaultdict(list)
    uri_of: dict = {}
    prior: dict[uuid.UUID, tuple[str, str]] = {}
    for oid, bbox, state, source, uri, fid in rows:
        by_frame[fid].append((oid, bbox))
        uri_of[fid] = uri
        prior[oid] = (state, source)

    changes: dict[str, dict] = {}

    async def _persist(ids: list[uuid.UUID]) -> int:
        if not ids:
            return 0
        async with maker() as db:
            objs = (await db.execute(select(Object).where(Object.object_id.in_(ids)))).scalars().all()
            n = 0
            for obj in objs:
                if obj.state != "review":  # somebody ruled while the VLM was looking; theirs stands
                    continue
                changes[str(obj.object_id)] = {"from_state": obj.state, "from_source": obj.source}
                obj.state, obj.source = "accepted", "vlm_review"
                obj.provenance = {**(obj.provenance or {}), "agent_run_id": str(agent_run_id)}
                obj.version = (obj.version or 0) + 1
                n += 1
            await db.commit()
        return n

    store = get_object_store()
    seen = confirmed = promoted = 0
    errors = 0
    pending: list[uuid.UUID] = []
    for fid, items in by_frame.items():
        if confirmed >= per_class:
            break
        img = _decode(store, uri_of[fid])
        if img is None:
            continue
        for oid, bbox in items:
            if confirmed >= per_class:
                break
            seen += 1
            try:
                # The verifier is a synchronous HTTP client; in the daemon's loop it goes to a thread.
                res = await asyncio.to_thread(verifier.verify_object, img, tuple(bbox), cls.id)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                if errors >= 5:  # a dead endpoint fails every call; stop paying to find out
                    return {"class": class_name, "seen": seen, "confirmed": confirmed,
                            "promoted": promoted + await _persist(pending), "changes": changes,
                            "error": f"VLM failing repeatedly, stopped: {str(exc)[:200]}"}
                continue
            # Confirmed only when the VLM's own choice is this class: detector + VLM agree.
            if res.class_name == class_name:
                confirmed += 1
                pending.append(oid)
                if len(pending) >= _CHUNK:
                    promoted += await _persist(pending)
                    pending = []
    promoted += await _persist(pending)

    log.info("vlm_promote.class", cls=class_name, seen=seen, confirmed=confirmed, promoted=promoted)
    return {"class": class_name, "seen": seen, "confirmed": confirmed, "promoted": promoted,
            "changes": changes}
