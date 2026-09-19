"""Put objects back exactly as they were before a window of reviews, from the reviews' own records.

    .venv/bin/python scripts/demo_tour/restore_reviewed.py --vehicle INDIA-DEL-01 --since 2026-09-19T00:00+05:30
    .venv/bin/python scripts/demo_tour/restore_reviewed.py --vehicle INDIA-DEL-01 --since ... --apply

Reports by default and writes only with --apply, after saving every affected row to a JSON backup.

Why it exists. Rehearsing the annotation demo gave real verdicts on the Delhi clip's machine proposals and
then took them back through the review pages' own undo, the "revert" action. Until the fix that came with
this script, a revert set the object's source to "human" like any other verdict, so every rehearsed-and-
undone proposal ended up labelled as a person's label while still being an unjudged detection. Nothing
that the API offers can set a source back, because no client should ever be able to claim a machine wrote
something, so the repair reads the database's own history instead.

Every review row stores the object as it was before that review (`review.before`): class, box, attributes,
state, source, confidence and provenance. The earliest review of an object inside the window therefore
holds its exact state before the window began, and that is what is restored. Objects with no review in the
window are not touched, and neither is anything outside the named session. The review rows themselves are
kept: they are the accurate record of what was done and undone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from db.session import get_engine  # noqa: E402

FIRST = """
  select distinct on (r.object_id) r.object_id, r.before
  from review r join object o using(object_id) join frame f using(frame_id) join session s using(session_id)
  where s.vehicle_id = :vehicle and r.ts_ns >= :since and r.before ? 'source'
  order by r.object_id, r.ts_ns"""


async def run(vehicle: str, since_ns: int, apply: bool, backup: Path) -> int:
    params = {"vehicle": vehicle, "since": since_ns}
    async with get_engine().connect() as c:
        rows = (await c.execute(text(f"""
            with first as ({FIRST})
            select o.object_id::text as object_id, o.source, o.conf, o.state, o.class_id, o.bbox, o.attrs,
                   o.provenance, o.version, first.before
            from object o join first using(object_id)"""), params)).mappings().all()
    changed = [r for r in rows
               if (r["source"], r["state"], str(r["class_id"])) != (
                   r["before"].get("source"), r["before"].get("state"), str(r["before"].get("class_id")))]
    by_source: dict[str, int] = {}
    for r in changed:
        key = f"{r['source']} -> {r['before'].get('source')}"
        by_source[key] = by_source.get(key, 0) + 1
    print(f"{len(rows)} objects reviewed in the window, {len(changed)} differ from their state before it")
    for k, n in sorted(by_source.items()):
        print(f"  source {k}: {n}")
    if not apply:
        print("report only; pass --apply to restore them")
        return 0

    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(json.dumps([{k: (list(v) if isinstance(v, list | tuple) else v) for k, v in dict(r).items()}
                                  for r in rows], default=str, indent=1))
    async with get_engine().begin() as c:
        res = await c.execute(text(f"""
            with first as ({FIRST})
            update object o set
              source = first.before->>'source',
              conf = (first.before->>'conf')::float,
              state = first.before->>'state',
              class_id = (first.before->>'class_id')::int,
              bbox = array(select jsonb_array_elements_text(first.before->'bbox')::float),
              attrs = coalesce(first.before->'attrs', '{{}}'::jsonb),
              provenance = coalesce(first.before->'provenance', '{{}}'::jsonb),
              version = coalesce(o.version, 1) + 1
            from first where o.object_id = first.object_id"""), params)
    print(f"restored {res.rowcount} objects; the rows as they were are saved in {backup}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicle", required=True, help="session vehicle id, e.g. INDIA-DEL-01")
    ap.add_argument("--since", required=True, help="start of the window, ISO 8601 with offset")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup", type=Path, default=ROOT / ".scratch/demo/india/restore_backup.json")
    args = ap.parse_args()
    since_ns = int(datetime.fromisoformat(args.since).timestamp() * 1e9)
    return asyncio.run(run(args.vehicle, since_ns, args.apply, args.backup))


if __name__ == "__main__":
    sys.exit(main())
