"""Retire the duplicate and junk classes the custom sidecar accumulated.

The sidecar records whatever an annotator typed, and nothing ever reconciled it against the governed
ontology. What grew: three spellings of one traffic signal, two of one autorickshaw with one misspelled, a
`traffic_cone` beside the governed `cone`, a `small_suv` beside the governed `suv` holding 13,425 objects,
and a `test_new_vehicle` left behind by a test.

None of it was inert. `classify_crop` scores a crop against every class the ontology knows, so the relabel
agent proposed `back_side_of_autorikshasw` for real objects and spread a typo through the corpus, and it
died outright on `test_new_vehicle`, which was proposable and unstorable at the same time.

Every merge is one reversible AgentRun, so any of this can be undone individually.

    .venv/bin/python -m scripts.clean_custom_classes --dry-run
    .venv/bin/python -m scripts.clean_custom_classes --apply
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import func, select

from core.logging import get_logger, setup_logging
from db.models import Object, OntologyClass
from db.session import get_sessionmaker

log = get_logger("clean_custom_classes")

# (from_id, to_id, why). Ordered so a reader can check each one against the ontology.
MERGES: list[tuple[int, int, str]] = [
    (209, 12, "small_suv duplicates the governed suv"),
    (221, 34, "traffic_signals duplicates the governed traffic_signal"),
    (222, 34, "traffic_lights duplicates the governed traffic_signal"),
    (227, 34, "traffic_light duplicates the governed traffic_signal"),
    (225, 38, "traffic_cone duplicates the governed cone"),
    (210, 211, "back_side_of_autorikshasw is a misspelling of back_side_of_autorikshaw"),
]

# Corrected after the merge above folds the misspelling into it.
RENAMES: list[tuple[int, str]] = [
    (211, "back_side_of_autorickshaw"),
]

# Nothing references these and nothing should propose them again.
RETIRE_ONLY: list[tuple[int, str]] = [
    (219, "test_new_vehicle: a leftover from a test, and never seeded into ontology_class"),
    (214, "roasd: a typo for road"),
]


async def _counts(db, ids: list[int]) -> dict[int, int]:
    rows = (await db.execute(
        select(Object.class_id, func.count()).where(Object.class_id.in_(ids)).group_by(Object.class_id))).all()
    return {int(c): int(n) for c, n in rows}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="report what would change and write nothing")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    setup_logging("INFO")

    from services.agent.ontology_merge import merge_class, rename_in_sidecar, retire_from_sidecar

    touched = [f for f, _, _ in MERGES] + [i for i, _ in RETIRE_ONLY]
    async with get_sessionmaker()() as db:
        counts = await _counts(db, touched)
        names = {c.id: c.name for c in (await db.execute(
            select(OntologyClass).where(OntologyClass.id.in_(touched + [t for _, t, _ in MERGES])))).scalars()}

    print(f"{'':2}{'from':<30}{'->':^4}{'to':<22}{'objects':>9}")
    for frm, to, _ in MERGES:
        print(f"  {names.get(frm, frm):<30}{'->':^4}{names.get(to, to):<22}{counts.get(frm, 0):>9}")
    for cid, why in RETIRE_ONLY:
        print(f"  {names.get(cid, cid):<30}{'--':^4}{'retired':<22}{counts.get(cid, 0):>9}   {why.split(':')[1].strip()}")
    total = sum(counts.get(f, 0) for f, _, _ in MERGES)
    print(f"\n  {total:,} objects would move; predictions and eval patches keep pointing at the old classes,")
    print("  which is why the ontology_class rows are retired from the sidecar rather than deleted.")

    if args.dry_run:
        return 0

    async with get_sessionmaker()() as db:
        for frm, to, why in MERGES:
            out = await merge_class(db, from_id=frm, to_id=to, created_by="clean_custom_classes")
            print(f"  merged {out['from']} -> {out['to']}: {out['objects']} objects, {out['tracks']} tracks "
                  f"(run {out['run_id'][:8]})   {why}")

    for cid, new in RENAMES:
        print("  renamed:", rename_in_sidecar(cid, new))
    out = retire_from_sidecar({f for f, _, _ in MERGES} | {c for c, _ in RETIRE_ONLY})
    print(f"  retired from the sidecar: {', '.join(out['removed'])}")
    print(f"  custom classes remaining: {out['remaining']}")
    print("\n  Re-seal gold next: 9 objects in the active set changed class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
