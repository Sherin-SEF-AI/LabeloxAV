"""Security-domain webhook events for ANPR.

Emits `anpr.read` and `anpr.watchlist_hit` on the existing outbound-webhook infrastructure
(services/integrations/webhooks.py: HMAC-signed, fire-and-forget, per-subscription failure handling). A
downstream consumer (e.g. Sentigon) subscribes to these; we do not build the consumer, only the generic,
signed emission. Payloads carry the structured plate read plus deployment context (camera, session, pack).
"""

from __future__ import annotations

from services.anpr.recognize import PlateRead
from services.anpr.watchlist import match, normalize_watchlist


def _read_payload(read: PlateRead, camera_id: str | None, session_id: str | None,
                  pack_id: str | None) -> dict:
    p = read.parse
    return {
        "pack_id": pack_id,
        "camera_id": camera_id,
        "session_id": session_id,
        "bbox": list(read.bbox),
        "det_conf": round(read.det_conf, 4),
        "ocr_conf": round(read.ocr_conf, 4),
        "plate": {
            "text": p.normalized,
            "raw": read.ocr_text,
            "valid": p.valid,
            "type": p.plate_type,
            "state_code": p.state_code,
            "rto_district": p.rto_district,
            "series": p.series,
            "number": p.number,
            "format_confidence": p.format_confidence,
        },
    }


async def emit_anpr_read(read: PlateRead, *, camera_id: str | None = None,
                         session_id: str | None = None, pack_id: str | None = "sec") -> int:
    """Emit an `anpr.read` webhook for one plate read. Returns the number of deliveries scheduled."""
    from services.integrations.webhooks import emit

    return await emit("anpr.read", _read_payload(read, camera_id, session_id, pack_id))


async def emit_watchlist_hit(read: PlateRead, matched: str, *, camera_id: str | None = None,
                             session_id: str | None = None, pack_id: str | None = "sec") -> int:
    """Emit an `anpr.watchlist_hit` webhook when a read matches a watchlist entry."""
    from services.integrations.webhooks import emit

    payload = {**_read_payload(read, camera_id, session_id, pack_id), "matched": matched}
    return await emit("anpr.watchlist_hit", payload)


async def process_security_reads(reads: list[PlateRead], watchlist: list[str] | None = None, *,
                                 camera_id: str | None = None, session_id: str | None = None,
                                 pack_id: str | None = "sec") -> dict:
    """Emit an `anpr.read` per read and an `anpr.watchlist_hit` for each read on the watchlist. Returns a small
    summary of what was emitted. Reusable from a security pipeline after recognize_plates()."""
    wl = normalize_watchlist(watchlist or [])
    read_deliveries = 0
    hit_deliveries = 0
    hits = 0
    for r in reads:
        read_deliveries += await emit_anpr_read(r, camera_id=camera_id, session_id=session_id, pack_id=pack_id)
        m = match(r, wl)
        if m is not None:
            hits += 1
            hit_deliveries += await emit_watchlist_hit(r, m, camera_id=camera_id, session_id=session_id,
                                                       pack_id=pack_id)
    return {"reads": len(reads), "watchlist_hits": hits,
            "read_deliveries": read_deliveries, "hit_deliveries": hit_deliveries}
