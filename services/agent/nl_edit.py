"""Natural-language and VLM-assisted bulk editing (M-F.3): the annotator types an operation over the current
frame or session ("select every parked autorickshaw", "reclassify the fallback objects to push_cart", "flag
every pedestrian with no visible mask"). The command is parsed into a selection predicate and an operation,
resolved to concrete objects, and PREVIEWED. Nothing changes until the human confirms. On confirmation the edit
lands as one reversible AgentRun with per-object provenance, routes the touched objects to review, and writes an
audit entry. Agents (and language) propose; the gate disposes.

The selection uses class and attribute filters, a missing-mask predicate, and, for referential phrases the
structure cannot resolve ("near the barrier"), an optional duty-cycled VLM confirmation per candidate.
"""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import AgentRun, Frame, Object
from services.agent.class_move import refuse_reason
from services.agent.nl import _resolve_classes
from services.autolabel.ontology import get_ontology

log = get_logger("agent.nl_edit")

# attribute keywords -> the attrs values (or intent) they match. Kept small and explicit.
_ATTR_WORDS = {
    "parked": ("motion", {"parked", "static", "stopped", "stationary"}),
    "moving": ("motion", {"moving", "driving"}),
    "occluded": ("occlusion", {"partial", "heavy", "full", "occluded"}),
    "braking": ("brake", {"braking", "on"}),
}
_REFERENTIAL = re.compile(r"\bnear\b|\bnext to\b|\bbeside\b|\bbehind\b|\bin front of\b|\bunder\b|\bon the\b", re.I)


def parse_edit(text: str, onto) -> dict:
    """Parse a command into {operation, select_class_ids, to_class_id, mask_missing, attr, referential, raw}."""
    t = text.strip().lower()

    # operation
    to_class_id = None
    relabel_verb = bool(re.search(r"\breclassify|relabel|change .* to |set .* to \b", t))
    if relabel_verb or re.search(r"\bto\s+[a-z_]+\s*$", t):
        operation = "reclassify"
        # "relabel every pickup as LCV" names the target with "as" (or "into"); only after a relabel verb,
        # because "as" elsewhere in a sentence is not a target.
        target_re = r"\b(?:to|as|into)\s+([a-z][a-z0-9_ ]*)\s*$" if relabel_verb else r"\bto\s+([a-z][a-z0-9_ ]*)\s*$"
        m = re.search(target_re, t)
        if m:
            cand = m.group(1).strip().replace(" ", "_").replace("-", "_")
            if onto.has_name(cand):
                to_class_id = onto.by_name(cand).id
            else:
                hit = next((c for c in onto.classes if cand in (a.lower() for a in c.aliases)), None)
                if hit is not None:
                    to_class_id = hit.id
        sel_text = re.sub(target_re, "", t)
    else:
        if re.search(r"\bflag\b|\bmark for review\b|\bsend\b.*\breview\b|\bback (?:to|for|into) review\b"
                     r"|\bre-?review\b", t):
            operation = "flag"
        else:
            operation = "select"
        sel_text = t

    # the SELECTION classes: the "to <class>" target was stripped above so it is not also treated as a selector
    names = _resolve_classes(sel_text, onto)
    # A word that is itself a class ("autorickshaw") also names a group in the shared resolver, which widens
    # "every autorickshaw" to every three-wheeler. A bulk edit has to touch what was named, so when every
    # group phrase found is an exact class word, the exact classes win.
    from services.agent.nl import _L1_GROUPS

    exact = {n for n in names if re.search(rf"\b{re.escape(n.replace('_', ' '))}s?\b|\b{re.escape(n)}s?\b",
                                           sel_text)}
    groups_hit = [p for p in _L1_GROUPS if re.search(rf"\b{re.escape(p)}\b", sel_text)]
    if exact and groups_hit and all(onto.has_name(p.replace(" ", "_").rstrip("s")) or onto.has_name(p)
                                    for p in groups_hit):
        names = exact
    select_ids: set[int] = set()
    if "fallback" in sel_text:
        select_ids.update(onto.fallback_ids())
    for n in names:
        try:
            select_ids.add(onto.by_name(n).id)
        except Exception:  # noqa: BLE001
            pass
    # Aliases the ontology records ("force traveller" for tempo_traveller), plural-ok like the names.
    for c in onto.classes:
        for alias in c.aliases:
            spaced = re.escape(alias.lower().replace("_", " "))
            if re.search(rf"\b{spaced}s?\b", sel_text):
                select_ids.add(c.id)

    mask_missing = bool(re.search(r"\bno (visible )?(box )?mask\b|\bwithout (a )?mask\b|\bmissing mask\b", t))
    attr = None
    for word, spec in _ATTR_WORDS.items():
        if re.search(rf"\b{word}\b", t):
            attr = {"name": spec[0], "values": sorted(spec[1])}
            break

    return {"operation": operation, "select_class_ids": sorted(select_ids), "to_class_id": to_class_id,
            "mask_missing": mask_missing, "attr": attr, "referential": bool(_REFERENTIAL.search(t)),
            "raw": text.strip()}


def _matches_attr(obj: Object, attr: dict | None) -> bool:
    if not attr:
        return True
    vals = {str(v).lower() for v in (obj.attrs or {}).values()}
    return bool(vals & set(attr["values"])) or str((obj.attrs or {}).get(attr["name"], "")).lower() in attr["values"]


async def resolve(db: AsyncSession, plan: dict, *, frame_id: UUID | None = None, session_id: UUID | None = None,
                  limit: int = 300) -> dict:
    """Resolve a parsed plan to concrete objects (no mutation). Returns the preview list + count + warnings."""
    onto = get_ontology()
    q = select(Object, Frame.frame_id, Frame.session_id).join(Frame, Frame.frame_id == Object.frame_id)
    if frame_id is not None:
        q = q.where(Object.frame_id == frame_id)
    elif session_id is not None:
        q = q.where(Frame.session_id == session_id)
    q = q.where(Object.state != "rejected")
    if plan["select_class_ids"]:
        q = q.where(Object.class_id.in_(plan["select_class_ids"]))
    if plan["mask_missing"]:
        q = q.where(Object.mask_uri.is_(None))
    rows = (await db.execute(q.limit(limit * 4))).all()

    matched = []
    for obj, fid, _sid in rows:
        if not _matches_attr(obj, plan["attr"]):
            continue
        matched.append({"object_id": str(obj.object_id), "frame_id": str(fid), "class_id": obj.class_id,
                        "class_name": onto.by_id(obj.class_id).name, "bbox": [float(x) for x in obj.bbox],
                        "state": obj.state, "crop_url": f"/api/objects/{obj.object_id}/crop"})
        if len(matched) >= limit:
            break

    warnings = []
    if not plan["select_class_ids"] and not plan["mask_missing"] and not plan["attr"]:
        warnings.append("no class, attribute, or mask filter recognized; this would match everything in scope")
    if plan["operation"] == "reclassify" and plan["to_class_id"] is None:
        warnings.append("reclassify needs a valid target class after 'to'; none recognized")
    if plan["referential"]:
        warnings.append("this command references spatial context; enable VLM refine to confirm each candidate")
    return {"plan": plan, "count": len(matched), "objects": matched, "warnings": warnings}


async def vlm_refine(matched: list[dict], phrase: str, max_objs: int = 40) -> list[str]:
    """Keep the objects a VLM agrees match a referential phrase. Duty-cycled; returns kept object_ids."""
    import cv2
    import numpy as np

    from core.storage import get_object_store
    from db.session import get_sessionmaker
    from services.autolabel.paths.path_c_vlm import OllamaVlmClient, crop_object

    vlm = OllamaVlmClient()
    kept: list[str] = []
    maker = get_sessionmaker()
    async with maker() as db:
        for m in matched[:max_objs]:
            obj = await db.get(Object, UUID(m["object_id"]))
            fr = await db.get(Frame, UUID(m["frame_id"]))
            if obj is None or fr is None:
                continue
            try:
                buf = np.frombuffer(get_object_store().get_bytes(fr.img_uri), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                x1, y1, x2, y2 = (float(v) for v in obj.bbox[:4])
                res = vlm.verify(crop_object(img, (x1, y1, x2, y2), 0.4),
                                 [f"yes ({phrase})", "no"], {})
                if (res.class_name or "").lower().startswith("yes"):
                    kept.append(m["object_id"])
            except Exception as exc:  # noqa: BLE001
                log.info("nl_edit.vlm_refine_failed", error=str(exc))
    return kept


# ---- many sessions at once (#17) ------------------------------------------------------------------------------
#
# "relabel every pickup in Delhi sessions as LCV", "send all cyclists at night in Bengaluru drives back for
# review". The sentence carries a scope as well as a selection: cities, vehicles, a date range, scene and state
# terms, or an explicit dataset query ("where night AND urban", services/datasets/query_lang.py). The scope is
# parsed out first and removed, so the rest reads exactly as a single-session command does and parse_edit is
# the one parser of operations and classes.
#
# The preview counts everything the scope selects (per session and class), shows sample crops, and says what
# is left alone and why. What a reviewer approves is the list of object ids the preview found, not the filter,
# so a label that arrives after the preview is never swept in.

MAX_BULK = 20_000        # objects one approved edit may carry; above it the person narrows the scope
_SAMPLE = 24
_IST = timezone(timedelta(hours=5, minutes=30))
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                         "august", "september", "october", "november", "december"], 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_ISO = r"(\d{4}-\d{2}-\d{2})"


def is_human_confirmed(source: str | None, state: str | None) -> bool:
    """A label a person owns: drawn or edited by one, or a machine label a reviewer accepted. A VLM's
    acceptance (source vlm_review) is a machine ruling and stays editable."""
    return source == "human" or (state == "accepted" and source != "vlm_review")


def _month_range(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    end = date(year + (month == 12), month % 12 + 1, 1)
    return start, end


def _parse_dates(t: str, today: date) -> tuple[date | None, date | None, list[tuple[int, int]], str | None]:
    """A date range named in the text, as [from, to) in IST calendar days, with the spans it came from."""
    spans: list[tuple[int, int]] = []

    def take(m) -> None:
        spans.append((m.start(), m.end()))

    if m := re.search(rf"\b(?:between|from)\s+{_ISO}\s+(?:and|to|until)\s+{_ISO}", t):
        take(m)
        a, b = date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
        return a, b + timedelta(days=1), spans, f"{a} to {b}"
    if m := re.search(r"\b(?:in\s+)?(?:the\s+)?(?:last|past)\s+(\d{1,3})\s+(day|week|month)s?\b", t):
        take(m)
        n, unit = int(m.group(1)), m.group(2)
        days = n * {"day": 1, "week": 7, "month": 30}[unit]
        return today - timedelta(days=days - 1), today + timedelta(days=1), spans, f"last {n} {unit}s"
    if m := re.search(r"\b(?:from\s+)?(last|this)\s+(week|month)\b", t):
        take(m)
        if m.group(2) == "week":
            monday = today - timedelta(days=today.weekday())
            start = monday - timedelta(days=7) if m.group(1) == "last" else monday
            return start, start + timedelta(days=7), spans, f"{m.group(1)} week"
        first = today.replace(day=1)
        if m.group(1) == "last":
            prev = first - timedelta(days=1)
            a, b = _month_range(prev.year, prev.month)
        else:
            a, b = _month_range(today.year, today.month)
        return a, b, spans, f"{m.group(1)} month"
    if m := re.search(r"\b(?:from\s+)?(yesterday|today)(?:'s)?\b", t):
        take(m)
        d = today - timedelta(days=1) if m.group(1) == "yesterday" else today
        return d, d + timedelta(days=1), spans, m.group(1)
    if m := re.search(r"\b(?:in|during|from)\s+(" + "|".join(sorted(_MONTHS, key=len, reverse=True))
                      + r")(?:\s+(\d{4}))?\b", t):
        take(m)
        month = _MONTHS[m.group(1)]
        year = int(m.group(2)) if m.group(2) else (today.year if month <= today.month else today.year - 1)
        a, b = _month_range(year, month)
        return a, b, spans, f"{a.strftime('%B %Y')}"
    lo = hi = None
    label = []
    if m := re.search(rf"\b(?:since|after|from)\s+{_ISO}", t):
        take(m)
        lo = date.fromisoformat(m.group(1)) + (timedelta(days=1) if "after" in m.group(0) else timedelta(0))
        label.append(f"from {lo}")
    if m := re.search(rf"\b(?:before|until)\s+{_ISO}", t):
        take(m)
        hi = date.fromisoformat(m.group(1)) + (timedelta(days=1) if "until" in m.group(0) else timedelta(0))
        label.append(f"before {hi}")
    if lo is None and hi is None and (m := re.search(rf"\bon\s+{_ISO}", t)):
        take(m)
        lo = date.fromisoformat(m.group(1))
        hi = lo + timedelta(days=1)
        label.append(f"on {lo}")
    return lo, hi, spans, (" ".join(label) or None)


def _cut(text: str, spans: list[tuple[int, int]]) -> str:
    """Blank out the given spans (keeping positions), so later patterns cannot re-read them."""
    chars = list(text)
    for s, e in spans:
        for i in range(s, e):
            chars[i] = " "
    return "".join(chars)


def parse_scope(text: str, *, today: date | None = None) -> tuple[dict, str, list[str]]:
    """Pull the multi-session scope out of a sentence.

    Returns (scope, remaining_text, errors). `scope` holds cities (codes), city_names, vehicles, date_from /
    date_to (ISO, [from, to) in IST days), predicate (the datasets query vocabulary: scene axes and states),
    query (an explicit query string, if one was given) and a human-readable `described` list.
    """
    from services.assistant.tools import CITY_CODES
    from services.datasets.query_lang import QueryError, compile_query, terms_in_text

    today = today or datetime.now(_IST).date()
    errors: list[str] = []
    scope: dict = {"cities": [], "city_names": [], "vehicles": [], "date_from": None, "date_to": None,
                   "predicate": {}, "query": None, "described": []}
    t = text.strip()
    low = t.lower()

    # 1. An explicit dataset query at the end: "... where night AND urban", "... matching 'rain, vru'".
    if m := re.search(r"\b(?:where|matching|query\s*[:=]?)\s+[\"'“]?(.+?)[\"'”]?\s*$", low):
        q = t[m.start(1):m.end(1)]
        try:
            compiled = compile_query(q)
            scope["query"] = q
            for k, v in compiled["predicate"].items():
                if k == "class_names":
                    scope.setdefault("query_classes", []).extend(v)
                    continue
                scope["predicate"].setdefault(k, [])
                scope["predicate"][k] += [x for x in v if x not in scope["predicate"][k]]
            scope["described"].append(f"matching “{q}”")
        except QueryError as exc:
            errors.append(str(exc))
        low = _cut(low, [(m.start(), m.end())])

    # 2. Dates.
    a, b, spans, label = _parse_dates(low, today)
    if a or b:
        scope["date_from"] = a.isoformat() if a else None
        scope["date_to"] = b.isoformat() if b else None
        scope["described"].append(f"recorded {label}")
    low = _cut(low, spans)

    # 3. Vehicles: "vehicle DASHCAM-01", or any fleet id shaped like INDIA-DEL-01 in the original text.
    vspans: list[tuple[int, int]] = []
    for m in re.finditer(r"(?:\b(?:from|on|by|of|in)\s+)?(?:the\s+)?\b(?:vehicle|car|rig)\s+([a-z0-9][a-z0-9_-]{1,63})",
                         low):
        if re.search(r"\d|-", m.group(1)):
            scope["vehicles"].append(t[m.start(1):m.end(1)].upper())
            vspans.append((m.start(), m.end()))
    for m in re.finditer(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b", t):
        if any(s <= m.start() < e for s, e in vspans) or not re.search(r"\d", m.group(0)):
            continue
        scope["vehicles"].append(m.group(0))
        vspans.append((m.start(), m.end()))
    if scope["vehicles"]:
        scope["described"].append("vehicle " + ", ".join(scope["vehicles"]))
    low = _cut(low, vspans)

    # 4. Cities, by name (CITY_CODES) or by the code a fleet records (DEL, BLR) written in capitals.
    cspans: list[tuple[int, int]] = []
    tail = r"(?:\s+(?:sessions?|drives?|recordings?|footage|data|runs?))?"
    for name in sorted(CITY_CODES, key=len, reverse=True):
        for m in re.finditer(rf"(?:\b(?:in|from|of|across|for)\s+)?(?:the\s+)?\b{re.escape(name)}\b{tail}", low):
            if any(not (m.end() <= s or m.start() >= e) for s, e in cspans):
                continue
            code = CITY_CODES[name]
            if code not in scope["cities"]:
                scope["cities"].append(code)
                scope["city_names"].append(name.title())
            cspans.append((m.start(), m.end()))
    for code in sorted(set(CITY_CODES.values())):
        for m in re.finditer(rf"\b{code}\b", t):
            if any(s <= m.start() < e for s, e in cspans + vspans):
                continue
            if code not in scope["cities"]:
                scope["cities"].append(code)
                scope["city_names"].append(code)
            cspans.append((m.start(), m.end()))
    if scope["cities"]:
        scope["described"].append("in " + ", ".join(scope["city_names"]))
    low = _cut(low, cspans)

    # 5. Scene and state words ("at night", "rainy", "unreviewed").
    found = terms_in_text(low)
    for k, v in found["predicate"].items():
        scope["predicate"].setdefault(k, [])
        scope["predicate"][k] += [x for x in v if x not in scope["predicate"][k]]
    for term in found["terms"]:
        scope["described"].append(term["value"] if term["kind"] == "scene" else term["word"])
    # "at night" / "in rain": take the preposition with the word so it is not left dangling.
    pre = [(max(0, s - 3), e) if low[max(0, s - 3):s] in ("at ", "in ", "on ") else (s, e)
           for s, e in found["spans"]]
    low = _cut(low, pre)

    # 6. Leftover scope nouns ("in all sessions", "across drives").
    low = re.sub(r"\b(?:in|from|across|over)\s+(?:all\s+)?(?:the\s+)?(?:sessions?|drives?|recordings?)\b", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    # A preposition whose object was cut ("as lcv from", "in the") would hide the "as <class>" target at the end.
    low = re.sub(r"(?:\s+(?:in|from|on|at|of|for|by|during|across|with|the|all))+$", "", low)
    return scope, low, errors


def scope_is_empty(scope: dict) -> bool:
    return not (scope.get("cities") or scope.get("vehicles") or scope.get("date_from") or scope.get("date_to")
                or scope.get("predicate") or scope.get("session_ids"))


def parse_bulk_edit(text: str, onto=None, *, today: date | None = None) -> dict:
    """Parse a multi-session command: the scope first, then parse_edit on what is left."""
    onto = onto or get_ontology()
    scope, rest, errors = parse_scope(text, today=today)
    plan = parse_edit(rest, onto)
    plan["raw"] = text.strip()
    if scope.get("query_classes"):
        ids = set(plan["select_class_ids"])
        for n in scope.pop("query_classes"):
            if onto.has_name(n):
                ids.add(onto.by_name(n).id)
        plan["select_class_ids"] = sorted(ids)
    plan["scope"] = scope
    plan["errors"] = errors
    plan["kind"] = "bulk"
    return plan


def _ist_ns(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=_IST).timestamp() * 1_000_000_000)


def _bulk_where(plan: dict, hidden: frozenset) -> list:
    """The WHERE clauses for a bulk plan, over Object joined to Frame and Session."""
    from sqlalchemy import or_

    from db.models import Session as DbSession
    from services.explore.query import frame_clauses, object_clauses

    scope = plan.get("scope") or {}
    out: list = [Object.state != "rejected"]
    if hidden:
        out.append(or_(DbSession.project_id.is_(None), DbSession.project_id.not_in(list(hidden))))
    if plan.get("select_class_ids"):
        out.append(Object.class_id.in_([int(i) for i in plan["select_class_ids"]]))
    if plan.get("mask_missing"):
        out.append(Object.mask_uri.is_(None))
    if attr := plan.get("attr"):
        out.append(Object.attrs[attr["name"]].astext.in_(list(attr["values"])))
    if scope.get("cities"):
        city_or = [DbSession.city.in_(scope["cities"])]
        for code, name in zip(scope["cities"], scope.get("city_names") or [], strict=False):
            city_or += [DbSession.city.ilike(name), DbSession.vehicle_id.ilike(f"%-{code}-%")]
        out.append(or_(*city_or))
    if scope.get("vehicles"):
        out.append(or_(*[DbSession.vehicle_id.ilike(v) for v in scope["vehicles"]]))
    if scope.get("session_ids"):
        out.append(Frame.session_id.in_([UUID(str(s)) for s in scope["session_ids"]]))
    if scope.get("date_from"):
        out.append(DbSession.start_ts_ns >= _ist_ns(date.fromisoformat(scope["date_from"])))
    if scope.get("date_to"):
        out.append(DbSession.start_ts_ns < _ist_ns(date.fromisoformat(scope["date_to"])))
    pred = dict(scope.get("predicate") or {})
    # The explorer's own predicate compiler: scene axes on frame.scene, states on the object, and real frames
    # only (a synthetic or perturbed frame is never edited by a sentence).
    out += object_clauses({k: v for k, v in pred.items() if k == "states"})
    out += frame_clauses({k: v for k, v in pred.items() if k in ("weather", "time_of_day", "road_type", "density")})
    return out


def _describe(plan: dict, onto) -> list[str]:
    parts = []
    op = plan["operation"]
    names = [onto.by_id(i).name for i in plan.get("select_class_ids") or []]
    if op == "reclassify":
        to = onto.by_id(int(plan["to_class_id"])).name if plan.get("to_class_id") else "?"
        parts.append(f"change {', '.join(names) or 'everything'} to {to}")
    elif op == "flag":
        parts.append(f"send {', '.join(names) or 'everything'} back to review")
    else:
        parts.append(f"select {', '.join(names) or 'everything'}")
    if plan.get("attr"):
        parts.append(f"only where {plan['attr']['name']} is {'/'.join(plan['attr']['values'])}")
    if plan.get("mask_missing"):
        parts.append("only labels with no mask")
    parts += (plan.get("scope") or {}).get("described") or []
    return parts


async def resolve_bulk(db: AsyncSession, plan: dict, *, hidden: frozenset = frozenset(), cap: int = MAX_BULK,
                       sample: int = _SAMPLE) -> dict:
    """Resolve a bulk plan across every session its scope selects. No mutation.

    Returns the counts per session and class, the counts left alone (human-confirmed; a class change across a
    category boundary; already the target class), sample crops, warnings, and the object ids an approval
    would edit (`object_ids`, at most `cap`; `too_many` when there are more).
    """
    from sqlalchemy import and_, case, func, or_

    from db.models import Session as DbSession

    onto = get_ontology()
    op = plan["operation"]
    to_id = plan.get("to_class_id")
    where = _bulk_where(plan, hidden)
    human = or_(Object.source == "human", and_(Object.state == "accepted", Object.source != "vlm_review"))
    human_col = case((human, True), else_=False)
    base = (select(Frame.session_id, Object.class_id, human_col.label("human"), func.count())
            .select_from(Object).join(Frame, Frame.frame_id == Object.frame_id)
            .join(DbSession, DbSession.session_id == Frame.session_id))
    for c in where:
        base = base.where(c)
    rows = (await db.execute(base.group_by(Frame.session_id, Object.class_id, human_col))).all()

    by_session: dict[str, dict[str, int]] = {}
    by_class: dict[str, int] = {}
    left = {"human_confirmed": 0, "category_change": 0, "already_target": 0}
    left_by_class: dict[str, dict[str, int]] = {}
    editable_classes: set[int] = set()
    for sid, cid, is_human, n in rows:
        name = onto.by_id(int(cid)).name
        n = int(n)
        reason = None
        if is_human:
            reason = "human_confirmed"
        elif op == "reclassify" and to_id is not None and int(cid) == int(to_id):
            reason = "already_target"
        elif op == "reclassify" and to_id is not None and refuse_reason(onto, int(cid), int(to_id)):
            reason = "category_change"
        if reason:
            left[reason] += n
            left_by_class.setdefault(name, {}).setdefault(reason, 0)
            left_by_class[name][reason] += n
            continue
        editable_classes.add(int(cid))
        by_session.setdefault(str(sid), {})[name] = by_session.setdefault(str(sid), {}).get(name, 0) + n
        by_class[name] = by_class.get(name, 0) + n
    total = sum(by_class.values())

    sessions_meta: dict[str, dict] = {}
    if by_session:
        for s in (await db.execute(select(DbSession).where(
                DbSession.session_id.in_([UUID(k) for k in by_session])))).scalars():
            sessions_meta[str(s.session_id)] = {
                "vehicle": s.vehicle_id, "city": s.city, "route": s.route,
                "date": datetime.fromtimestamp(s.start_ts_ns / 1e9, _IST).date().isoformat() if s.start_ts_ns else None}
    sessions = sorted(({"session_id": k, **sessions_meta.get(k, {}), "classes": v, "total": sum(v.values())}
                       for k, v in by_session.items()), key=lambda r: -r["total"])

    object_ids: list[str] = []
    samples: list[dict] = []
    too_many = total > cap
    if total and editable_classes and not too_many:
        q = (select(Object.object_id, Object.frame_id, Frame.session_id, Object.class_id, Object.bbox, Object.state,
                    Object.conf)
             .select_from(Object).join(Frame, Frame.frame_id == Object.frame_id)
             .join(DbSession, DbSession.session_id == Frame.session_id)
             .where(~human, Object.class_id.in_(sorted(editable_classes))))
        for c in where:
            q = q.where(c)
        picked = (await db.execute(q.order_by(Frame.session_id, Object.class_id, Object.conf.desc())
                                   .limit(cap + 1))).all()
        object_ids = [str(r.object_id) for r in picked]
        # Sample round-robin over (session, class), so the crops show the spread rather than one frame's worth.
        groups: dict[tuple, list] = {}
        for r in picked:
            groups.setdefault((r.session_id, r.class_id), []).append(r)
        queues = list(groups.values())
        i = 0
        while len(samples) < sample and any(queues):
            qrow = queues[i % len(queues)]
            if qrow:
                r = qrow.pop(0)
                samples.append({"object_id": str(r.object_id), "frame_id": str(r.frame_id),
                                "session_id": str(r.session_id), "class_name": onto.by_id(int(r.class_id)).name,
                                "bbox": [float(x) for x in r.bbox], "state": r.state, "conf": round(float(r.conf), 3),
                                "crop_url": f"/api/objects/{r.object_id}/crop"})
            i += 1

    warnings = list(plan.get("errors") or [])
    if not plan.get("select_class_ids") and not plan.get("mask_missing") and not plan.get("attr"):
        warnings.append("no class recognized; name the classes to change (for example 'every pickup')")
    if op == "reclassify" and to_id is None:
        warnings.append("a class change needs a target class after 'to' or 'as'; none recognized")
    if op == "select":
        warnings.append("no change recognized: say relabel ... as <class>, or send ... back for review")
    if scope_is_empty(plan.get("scope") or {}):
        warnings.append("no city, vehicle, date or scene named: this covers every session you can see")
    if too_many:
        warnings.append(f"{total:,} labels match, more than one approval may carry ({cap:,}); narrow the scope")
    if plan.get("referential"):
        warnings.append("spatial phrases ('near the barrier') are not applied across sessions; they were ignored")
    can_propose = bool(total) and not too_many and op in ("reclassify", "flag") and bool(plan.get("select_class_ids")) \
        and not (op == "reclassify" and to_id is None) and not plan.get("errors")
    return {"plan": plan, "interpretation": _describe(plan, onto), "count": total, "by_class": by_class,
            "by_session": sessions, "sessions": len(sessions), "left_alone": left, "left_alone_by_class": left_by_class,
            "sample": samples, "warnings": warnings, "too_many": too_many, "can_propose": can_propose,
            "object_ids": object_ids if can_propose else []}


async def apply_edit(db: AsyncSession, plan: dict, object_ids: list[UUID], *, created_by: str | None = None) -> dict:
    """Apply a bulk edit to the confirmed objects as ONE reversible AgentRun. reclassify sets the class and
    routes to review; flag routes to review with a flag. Every object is stamped with the run and audited."""
    onto = get_ontology()
    op = plan["operation"]
    if op == "reclassify" and plan["to_class_id"] is None:
        return {"error": "reclassify needs a valid target class"}
    if op == "select":
        return {"error": "select is a preview only; nothing to apply"}

    run_id = uuid.uuid4()
    objs = (await db.execute(select(Object).where(Object.object_id.in_(object_ids)))).scalars().all()
    changes: dict = {}
    skipped: dict[str, int] = {}
    for obj in objs:
        # A language command must not overwrite a human's label. Every other bulk write path excludes
        # source == "human"; this one did not, so one sentence could undo a person's work. A machine label a
        # reviewer accepted is a person's ruling too, so it is left alone the same way (is_human_confirmed).
        if is_human_confirmed(obj.source, obj.state):
            skipped["human_label"] = skipped.get("human_label", 0) + 1
            continue
        if obj.state == "rejected":
            skipped["rejected"] = skipped.get("rejected", 0) + 1
            continue
        # And it may sharpen a class, not change what kind of thing it is - the same l0 boundary the
        # relabel and contamination paths already refuse to cross. A typed sentence is if anything a
        # looser filter than a detector's confidence, so it needs the guard more, not less.
        if op == "reclassify":
            refusal = refuse_reason(onto, int(obj.class_id), int(plan["to_class_id"]))
            if refusal:
                skipped["category_change"] = skipped.get("category_change", 0) + 1
                log.warning("agent.nl_edit.refused_category_change", object_id=str(obj.object_id),
                            reason=refusal, command=plan["raw"])
                continue
        changes[str(obj.object_id)] = {"from_class": int(obj.class_id), "from_state": obj.state,
                                       "from_source": obj.source}
        if op == "reclassify":
            obj.class_id = int(plan["to_class_id"])
        obj.state = "review"                    # a human confirms every language-issued edit
        obj.version = (obj.version or 0) + 1
        prov = dict(obj.provenance or {})
        prov["agent_run_id"] = str(run_id)
        prov.setdefault("nl_edit", []).append({"op": op, "command": plan["raw"],
                                               "to": onto.by_id(int(plan["to_class_id"])).name if plan["to_class_id"] else None})
        obj.provenance = prov

    scope: dict = {"command": plan["raw"], "operation": op}
    if plan.get("scope"):
        scope["bulk_scope"] = plan["scope"]
    db.add(AgentRun(run_id=run_id, kind="nl_edit", scope=scope,
                    status="committed", policy=plan, counts={"edited": len(changes)}, changes=changes,
                    critic={}, created_by=created_by))
    from services.govern.audit import record

    await record(db, actor="nl_edit", decision=f"bulk_{op}", subject=str(run_id),
                 rationale={"command": plan["raw"], "n_objects": len(changes),
                            "to_class": onto.by_id(int(plan["to_class_id"])).name if plan["to_class_id"] else None},
                 commit=False)
    await db.commit()
    log.info("nl_edit.applied", run_id=str(run_id), op=op, edited=len(changes))
    if skipped:
        # Reported rather than silent: a command that matched fifty objects and edited thirty has to say
        # what happened to the other twenty, or the preview count and the result disagree with no reason.
        log.info("nl_edit.skipped", run_id=str(run_id), **skipped)
    return {"run_id": str(run_id), "operation": op, "edited": len(changes), "routed_to": "review",
            "skipped": skipped}
