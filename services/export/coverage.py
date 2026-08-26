"""The coverage datasheet: what is in a release, and what is not known about it.

Not to be confused with `services/export/datasheet.py`, which is a pure Markdown renderer for the
buyer-diligence pack and takes its inputs from wherever the Documentation Agent harvested them. This module
does the counting, against the corpus, at export time, and ships the result inside the artifact. The two
overlap on subject and not on job: one formats numbers it is handed, this one produces them and says which
ones it could not.

A datasheet's value is entirely in the parts it refuses to guess. Every number here is counted from the
corpus at build time, and every quantity that cannot be counted honestly prints why instead of a figure. The
three that cannot, today:

  - **Recapture.** There is one `blind_audit` row and it is `status=seeded`, with 0 rows in
    `recapture_estimate`. So the sheet says the audit is seeded and how many frames it is waiting on. A
    recall figure without a recapture estimate is a statement about the labels that exist, not about the
    objects that were there.
  - **Privacy.** `PiiAudit` has no status column. A row means the frame was scanned; a row with zero counts
    means scanned and clean. There is no `verified` state and no `failed` state anywhere in the schema, so
    the sheet reports those three real categories rather than the three a reader might expect.
  - **Road class.** `Frame.road_class` is NULL on all 41,752 frames, so road-type stratification resolves
    `unresolved` for everything. Printing a distribution over one bucket would suggest the corpus had been
    checked for road-type balance.

Per-class quality reuses `per_class_precision_recall`, which already reports `measured: false` below
`MIN_GOLD_PER_CLASS` gold instances. That mechanism is left to do its job rather than being papered over: a
sheet whose weak numbers are indistinguishable from its strong ones is worse than one with gaps.
"""

from __future__ import annotations

import html
import json
from collections import Counter
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import (
    BlindAudit,
    DatasetCommit,
    Frame,
    Object,
    PiiAudit,
    Session,
    Track,
    TrackEvent,
)
from services.autolabel.ontology import get_ontology
from services.context.region import resolve_region
from services.domain import active_pack

log = get_logger("export.datasheet")

SCHEMA_VERSION = "coverage-datasheet-1"


def _pending(reason: str) -> dict:
    """A quantity that cannot be produced honestly, and the reason in the reader's hands."""
    return {"measured": False, "reason": reason}


async def _composition(db: AsyncSession, session_ids: list) -> dict:
    onto = get_ontology()
    rows = (await db.execute(
        select(Object.class_id, func.count(Object.object_id))
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Frame.session_id.in_(session_ids))
        .group_by(Object.class_id))).all()
    total = sum(n for _, n in rows) or 1
    def _name(cid: int) -> str:
        # A class id in the corpus that the current ontology does not carry is a real state: a retired class
        # still has objects. Naming it `class_{id}` keeps the row countable instead of raising.
        try:
            return onto.by_id(cid).name
        except KeyError:
            return f"class_{cid}"

    classes = sorted(
        ({"class_id": cid, "name": _name(cid), "n": n, "share": round(n / total, 5)} for cid, n in rows),
        key=lambda r: -r["n"])
    return {"n_objects": sum(n for _, n in rows), "n_classes_present": len(rows),
            "n_classes_in_ontology": len(onto.classes), "classes": classes,
            # The tail is the part a consumer needs before training on this: a class with four instances is
            # in the ontology and not in the data.
            "classes_absent": sorted(c.name for c in onto.classes if c.id not in {r[0] for r in rows}),
            "classes_under_10": sorted(c["name"] for c in classes if c["n"] < 10)}


async def _regions(db: AsyncSession, sessions: list[Session]) -> dict:
    strata: Counter = Counter()
    states: Counter = Counter()
    urban: Counter = Counter()
    raw: Counter = Counter()
    for s in sessions:
        r = resolve_region(s.city)
        strata[r.stratum()] += 1
        raw[s.city or "(none recorded)"] += 1
        if r.status == "resolved":
            states[r.state] += 1
            urban[r.urban_class] += 1
    spec = active_pack().region
    return {
        "n_sessions": len(sessions),
        "by_stratum": dict(strata.most_common()),
        "by_state": dict(states.most_common()),
        "by_urban_class": {k: urban.get(k, 0) for k in (spec.classes if spec else ())},
        # Both spellings, side by side. This row is the finding: the corpus records one city two ways, and
        # anything stratifying on the raw string sees two places.
        "raw_strings": dict(raw.most_common()),
        "concentration": (round(max(strata.values()) / len(sessions), 4) if sessions else None),
        "road_class": _pending("Frame.road_class is NULL on every frame in this release; nothing has "
                               "populated it, so no road-type stratification exists to report"),
    }


async def _context(db: AsyncSession, session_ids: list) -> dict:
    """Frame-level scene context, and how much of the release has any."""
    rows = (await db.execute(
        select(Frame.scene).where(Frame.session_id.in_(session_ids)))).scalars().all()
    n = len(rows)
    axes: dict[str, Counter] = {}
    with_any = 0
    for scene in rows:
        if not scene:
            continue
        keys = [k for k in scene if k != "confidence_per_axis"]
        if keys:
            with_any += 1
        for k in keys:
            axes.setdefault(k, Counter())[str(scene[k])] += 1
    return {"n_frames": n, "n_frames_with_context": with_any,
            "coverage": round(with_any / n, 4) if n else None,
            "axes": {k: dict(v.most_common()) for k, v in sorted(axes.items())}}


async def _track_events(db: AsyncSession, session_ids: list) -> dict:
    rows = (await db.execute(
        select(TrackEvent.event_type, TrackEvent.state, TrackEvent.source,
               func.count(TrackEvent.event_id))
        .join(Track, Track.track_id == TrackEvent.track_id)
        .where(Track.session_id.in_(session_ids))
        .group_by(TrackEvent.event_type, TrackEvent.state, TrackEvent.source))).all()
    by_type: dict[str, dict] = {}
    for etype, state, source, n in rows:
        d = by_type.setdefault(etype, {"accepted": 0, "proposed": 0, "rejected": 0, "human": 0,
                                       "heuristic": 0, "vlm": 0})
        d[state] = d.get(state, 0) + n
        d[source] = d.get(source, 0) + n
    spec = active_pack().track_events
    return {"n_events": sum(n for *_, n in rows), "by_type": by_type,
            # Absence is a coverage fact. A vocabulary of 23 with 2 populated says the labeling has not
            # started on 21 of them, which is exactly what a consumer needs to know before filtering on one.
            "types_absent": sorted(set(spec.names()) - set(by_type)) if spec else [],
            "note": "proposed events are heuristic suggestions awaiting review, not labels"}


async def _recapture(db: AsyncSession) -> dict:
    """The honest stamp. There is one seeded audit and no estimates."""
    audits = (await db.execute(select(BlindAudit))).scalars().all()
    scored = [a for a in audits if (a.status or "") not in ("seeded", "")]
    if not audits:
        return _pending("no blind audit has been seeded, so recall here is measured against the labels "
                        "that exist rather than against the objects that were present")
    if not scored:
        a = audits[0]
        return _pending(f"audit seeded, awaiting {a.n_frames} frames of independent labeling "
                        f"(audit {a.audit_id}, gold {a.gold_id}); {len(audits)} audit(s) exist and none is "
                        f"scored, so every recall figure in this release is measured against a denominator "
                        f"somebody already found")
    return {"measured": True, "n_audits_scored": len(scored)}


async def _privacy(db: AsyncSession, session_ids: list, n_frames: int) -> dict:
    """What the schema can actually say.

    `PiiAudit` has no status column. A row means the frame was scanned. A row with zero counts means scanned
    and nothing found. There is no `verified` and no `failed` state anywhere, so those are not reported.
    """
    rows = (await db.execute(
        select(PiiAudit.n_faces, PiiAudit.n_plates)
        .where(PiiAudit.session_id.in_(session_ids)))).all()
    scanned = len(rows)
    with_regions = sum(1 for f, p in rows if (f or 0) + (p or 0) > 0)
    return {
        "n_frames": n_frames,
        "scanned": scanned,
        "scanned_and_clean": scanned - with_regions,
        "scanned_with_regions_found": with_regions,
        "not_scanned": max(0, n_frames - scanned),
        "faces_found": sum(f or 0 for f, _ in rows),
        "plates_found": sum(p or 0 for _, p in rows),
        "note": ("PiiAudit records that a frame was scanned and what was found. The schema has no verified "
                 "and no failed state, so this reports scanned, scanned-and-clean, and not-scanned rather "
                 "than a pass/fail status that does not exist."),
    }


async def _quality(db: AsyncSession, commit: DatasetCommit) -> dict:
    """Per-class precision and recall, when there is an evaluation to recount from."""
    from services.govern.gold_eval import latest_gold_id

    onto = get_ontology()
    gold_id = await latest_gold_id(db, onto.version)
    if gold_id is None:
        any_gold = await latest_gold_id(db)
        return _pending(
            f"no gold set sealed at ontology {onto.version}"
            + (f" (latest sealed is {any_gold}, on an earlier version)" if any_gold else ""))
    eval_id = (commit.export_uris or {}).get("eval_id")
    if not eval_id:
        return _pending(f"this release carries no evaluation id, so there are no EvalPatch rows to recount; "
                        f"gold set {gold_id} is available to score against")
    from services.export.certificate import per_class_precision_recall

    return {"measured": True, "gold_id": gold_id, "eval_id": eval_id,
            "per_class": await per_class_precision_recall(db, eval_id)}


async def build_datasheet(db: AsyncSession, commit_id: str) -> dict:
    """Everything countable about one release, plus a stated reason for everything that is not."""
    commit = await db.get(DatasetCommit, commit_id)
    if commit is None:
        raise ValueError(f"unknown commit {commit_id}")

    spec = commit.slice_spec or {}
    # `session_id` singular is what SliceSpec actually carries; the plural is accepted for callers that
    # build a spec by hand. Reading only the plural made a single-session release's datasheet describe the
    # whole corpus, which is a sheet about the wrong data wearing the right commit id.
    session_ids = list(spec.get("session_ids") or [])
    if not session_ids and spec.get("session_id"):
        session_ids = [spec["session_id"]]
    if not session_ids and spec.get("regions"):
        from services.context.region import city_strings_for

        wanted: set[str] = set()
        for r in spec["regions"]:
            wanted |= city_strings_for(r)
        session_ids = list((await db.execute(
            select(Session.session_id).where(func.lower(Session.city).in_(sorted(wanted) or ["\x00"]))
        )).scalars())
    if not session_ids and spec.get("cities"):
        session_ids = list((await db.execute(
            select(Session.session_id).where(Session.city.in_(spec["cities"])))).scalars())
    if not session_ids:
        # A release with no explicit session list covers the corpus it was cut from. Counting over
        # everything is the honest reading, and the sheet says which it did.
        session_ids = list((await db.execute(select(Session.session_id))).scalars())
        scope = "all sessions (the release's slice spec names none, so this describes the whole corpus)"
    else:
        scope = f"{len(session_ids)} session(s) selected by the slice spec"

    sessions = list((await db.execute(
        select(Session).where(Session.session_id.in_(session_ids)))).scalars())
    n_frames = (await db.execute(
        select(func.count(Frame.frame_id)).where(Frame.session_id.in_(session_ids)))).scalar() or 0

    sheet = {
        "schema": SCHEMA_VERSION,
        "release": {
            "commit_id": commit.commit_id,
            "parent_id": commit.parent_id,
            "ontology_version": commit.ontology_version,
            "object_count": commit.object_count,
            "content_fingerprint": commit.content_fingerprint,
            "created_at": commit.created_at.isoformat() if commit.created_at else None,
            "scope": scope,
            "slice_spec": spec,
        },
        "composition": await _composition(db, session_ids),
        "regions": await _regions(db, sessions),
        "context": await _context(db, session_ids),
        "track_events": await _track_events(db, session_ids),
        "quality": await _quality(db, commit),
        "recapture": await _recapture(db),
        "privacy": await _privacy(db, session_ids, n_frames),
    }
    sheet["limitations"] = _limitations(sheet)
    return sheet


def _limitations(sheet: dict) -> list[str]:
    """The section a reader should look at first, assembled from what the counts actually showed.

    Derived rather than written, so it cannot go stale against the numbers above it.
    """
    out: list[str] = []
    reg = sheet["regions"]
    conc = reg.get("concentration")
    if conc and conc >= 0.9:
        top = next(iter(reg["by_stratum"]), "one place")
        out.append(f"{conc:.1%} of sessions are {top}. This is a single-location corpus; nothing here "
                   f"supports a claim about regional generalisation.")
    if len(reg.get("raw_strings", {})) > len(reg.get("by_stratum", {})):
        out.append("The same place is recorded under more than one string. Any stratification built on the "
                   "raw city column will overcount locations.")
    comp = sheet["composition"]
    if comp["classes_absent"]:
        out.append(f"{len(comp['classes_absent'])} of {comp['n_classes_in_ontology']} ontology classes have "
                   f"no instances in this release.")
    if comp["classes_under_10"]:
        out.append(f"{len(comp['classes_under_10'])} classes have fewer than 10 instances; per-class "
                   f"metrics for these are not measurable.")
    ctx = sheet["context"]
    if ctx["coverage"] is not None and ctx["coverage"] < 0.5:
        out.append(f"Scene context is present on {ctx['coverage']:.1%} of frames, so context-conditioned "
                   f"claims cover a minority of the release.")
    for key, label in (("quality", "Per-class quality"), ("recapture", "Recapture")):
        if sheet[key].get("measured") is False:
            out.append(f"{label} is not measured: {sheet[key]['reason']}.")
    priv = sheet["privacy"]
    if priv["not_scanned"]:
        out.append(f"{priv['not_scanned']} frames carry no PII audit row, so they have not been scanned.")
    te = sheet["track_events"]
    if te["types_absent"]:
        out.append(f"{len(te['types_absent'])} of the pack's track-event types have no instances here.")
    return out


def _rows(d: dict[str, Any], limit: int | None = None) -> str:
    items = list(d.items())[:limit] if limit else list(d.items())
    return "".join(
        f"<tr><td>{html.escape(str(k))}</td><td class=n>{html.escape(f'{v:,}' if isinstance(v, int) else str(v))}</td></tr>"
        for k, v in items)


def render_html(sheet: dict) -> str:
    """One page. Limitations first, because that is what a datasheet is for."""
    r = sheet["release"]
    comp, reg, ctx, te = sheet["composition"], sheet["regions"], sheet["context"], sheet["track_events"]
    lim = "".join(f"<li>{html.escape(x)}</li>" for x in sheet["limitations"]) or "<li>None recorded.</li>"
    top_classes = "".join(
        f"<tr><td>{html.escape(c['name'])}</td><td class=n>{c['n']:,}</td><td class=n>{c['share']:.2%}</td></tr>"
        for c in comp["classes"][:20])

    def block(title: str, body: str) -> str:
        return f"<section><h2>{html.escape(title)}</h2>{body}</section>"

    def maybe(section: dict, ok: str) -> str:
        return (f"<p class=pending><strong>Not measured.</strong> {html.escape(section['reason'])}</p>"
                if section.get("measured") is False else ok)

    cov = f" ({ctx['coverage']:.1%})" if ctx["coverage"] is not None else ""
    ctx_axes = "".join(f"<h3 style='font-size:.85rem;margin:.9rem 0 .3rem'>{html.escape(k)}</h3>"
                       f"<table>{_rows(v)}</table>" for k, v in ctx["axes"].items())

    q = sheet["quality"]
    q_body = maybe(q, f"<p>Scored against gold <code>{html.escape(str(q.get('gold_id')))}</code>, "
                      f"evaluation <code>{html.escape(str(q.get('eval_id')))}</code>. Classes below the "
                      f"gold-support floor report <code>measured: false</code> in the JSON rather than a "
                      f"number.</p>")
    priv = sheet["privacy"]
    return f"""<!doctype html><meta charset=utf-8>
<title>Datasheet {html.escape(r['commit_id'])}</title>
<style>
 body{{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
 h1{{font-size:1.4rem;margin:0}} h2{{font-size:1rem;margin:1.8rem 0 .5rem;border-bottom:1px solid #ddd;padding-bottom:.25rem}}
 table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{border-bottom:1px solid #eee;padding:.3rem .5rem;text-align:left}}
 td.n{{text-align:right;font-variant-numeric:tabular-nums}}
 code{{background:#f4f4f4;padding:.1rem .3rem}}
 .meta{{color:#666;font-size:12px}} .pending{{background:#fff8e1;border-left:3px solid #e0a800;padding:.5rem .75rem}}
 ul.lim li{{margin:.35rem 0}}
</style>
<h1>Dataset datasheet</h1>
<p class=meta><code>{html.escape(r['commit_id'])}</code> &middot; ontology {html.escape(r['ontology_version'])}
 &middot; {r['object_count']:,} objects &middot; {html.escape(str(r['created_at']))}<br>Scope: {html.escape(r['scope'])}</p>

{block("Limitations", f"<ul class=lim>{lim}</ul>")}

{block("Composition", f"<p>{comp['n_objects']:,} objects across {comp['n_classes_present']} of "
                      f"{comp['n_classes_in_ontology']} ontology classes.</p>"
                      f"<table><tr><th>class</th><th>n</th><th>share</th></tr>{top_classes}</table>"
                      f"<p class=meta>Top 20 shown; full distribution in the JSON.</p>")}

{block("Where it was captured", f"<table><tr><th>stratum</th><th>sessions</th></tr>{_rows(reg['by_stratum'])}</table>"
                                f"<h3 style='font-size:.85rem;margin:.9rem 0 .3rem'>As recorded</h3>"
                                f"<table><tr><th>raw city string</th><th>sessions</th></tr>{_rows(reg['raw_strings'])}</table>"
                                f"<p class=pending><strong>Road class not measured.</strong> "
                                f"{html.escape(reg['road_class']['reason'])}</p>")}

{block("Scene context", f"<p>Present on {ctx['n_frames_with_context']:,} of {ctx['n_frames']:,} "
                        f"frames{cov}.</p>" + ctx_axes)}

{block("Track events", f"<p>{te['n_events']:,} events. {len(te['types_absent'])} of the pack's types have "
                       f"no instances here.</p><p class=meta>{html.escape(te['note'])}</p>")}

{block("Quality", q_body)}

{block("Recapture", maybe(sheet['recapture'], "<p>Scored.</p>"))}

{block("Privacy", f"<table>{_rows({k: v for k, v in priv.items() if k != 'note'})}</table>"
                  f"<p class=meta>{html.escape(priv['note'])}</p>")}
"""


async def write_datasheet(db: AsyncSession, commit_id: str, out_dir) -> dict:
    """Build and write `datasheet.json` and `datasheet.html` beside the export artifact."""
    from pathlib import Path

    sheet = await build_datasheet(db, commit_id)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "datasheet.json").write_text(json.dumps(sheet, indent=2, sort_keys=True, default=str))
    (d / "datasheet.html").write_text(render_html(sheet))
    log.info("datasheet.written", commit_id=commit_id, limitations=len(sheet["limitations"]))
    return sheet
