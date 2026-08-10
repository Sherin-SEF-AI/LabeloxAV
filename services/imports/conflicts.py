"""What a migration would cost, before it is run.

Every importer already remaps external class names into the ontology, and every importer already counts what
it could not map. What nobody could see is the shape of that loss: which of their classes land on the same
one of ours, which fall into a fallback bucket, and how many objects each decision moves.

That matters more than it sounds during a competitive migration. A customer switching tools has years of
labels and a taxonomy they argued about internally. Telling them "imported, 4,000 unmapped" invites them to
discover a month later that their `two_wheeler_with_pillion` and their `motorcycle` both became `motorcycle`
and the distinction they were paying for is gone. Telling them beforehand turns the same fact into a
decision they get to make.

Read-only and pure over parsed records, so it can run on an upload before anything is written.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from core.logging import get_logger
from services.imports.records import ImportFrame

log = get_logger("import_conflicts")

FALLBACKS = ("vehicle_fallback", "object_fallback")


def taxonomy_report(frames: list[ImportFrame]) -> dict:
    """How the source taxonomy lands in the ontology, and what that costs."""
    from services.autolabel.ontology import get_ontology
    from services.imports.remap import remap_name

    onto = get_ontology()
    counts: Counter = Counter()
    target: dict[str, str] = {}
    mapped_ok: dict[str, bool] = {}

    for fr in frames:
        for o in fr.objects:
            counts[o.name] += 1
            if o.name not in target:
                _cid, name, ok = remap_name(o.name, onto)
                target[o.name] = name
                mapped_ok[o.name] = ok

    # Which of their names collide on one of ours. This is the finding that costs a customer a distinction,
    # so it leads the report rather than sitting in a footnote.
    collapsed: dict[str, list[str]] = defaultdict(list)
    for src, dst in target.items():
        collapsed[dst].append(src)
    merges = [
        {"ontology_class": dst,
         "source_classes": sorted(srcs, key=lambda s: -counts[s]),
         "objects": sum(counts[s] for s in srcs)}
        for dst, srcs in collapsed.items() if len(srcs) > 1
    ]
    merges.sort(key=lambda m: -m["objects"])

    unmapped = [
        {"source_class": src, "objects": counts[src], "falls_back_to": target[src]}
        for src in counts if not mapped_ok[src] or target[src] in FALLBACKS
    ]
    unmapped.sort(key=lambda u: -u["objects"])

    total = sum(counts.values())
    lost = sum(u["objects"] for u in unmapped)
    report = {
        "source_classes": len(counts),
        "objects": total,
        "mapped_cleanly": total - lost,
        "into_fallback": lost,
        "fallback_fraction": round(lost / total, 4) if total else 0.0,
        # Named, not just counted: a customer cannot act on "4,000 unmapped" and can act on the list.
        "unmapped": unmapped[:100],
        "merges": merges[:100],
        "mapping": [{"source_class": s, "ontology_class": target[s], "objects": counts[s],
                     "clean": mapped_ok[s]} for s, _ in counts.most_common(200)],
    }
    log.info("import.taxonomy_report", source_classes=len(counts), objects=total,
             into_fallback=lost, merges=len(merges))
    return report
