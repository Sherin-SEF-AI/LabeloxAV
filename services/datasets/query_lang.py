"""A dataset is a query, and the query is a sentence.

`labelox.load("night AND vru AND unoccluded")` is meant to sit in a training config, next to the learning
rate, and be read by whoever inherits that config a year later. That rules out a JSON predicate: nobody
writes one from memory, and a config full of nested dicts is a config nobody edits.

It also rules out anything clever. The terms here compile to the exact `CurationSlice.predicate` vocabulary
`services/explore/query.py` already evaluates in SQL and `services/curation/slices.py` already evaluates in
Python, so a query string is a shorthand for a cohort the rest of the system can already select, export and
version. One vocabulary, three ways in.

**The compiled predicate is always returned.** A dataset whose contents cannot be explained is one nobody
can defend in a review, and a term that quietly matched nothing is exactly how a training set ends up
missing the class it was assembled for. Unknown terms are refused rather than ignored, because
`"night AND vru AND unocluded"` silently returning every night frame is worse than an error.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("datasets.query_lang")

# Scene vocabulary, from the axes frame.scene actually carries. Aliases are included where the word somebody
# types differs from the value stored: nobody writes "time_of_day=night", they write "night".
_SCENE_TERMS: dict[str, tuple[str, str]] = {
    "night": ("time_of_day", "night"), "day": ("time_of_day", "day"),
    "dawn": ("time_of_day", "dawn"), "dusk": ("time_of_day", "dusk"),
    "rain": ("weather", "rain"), "fog": ("weather", "fog"), "clear": ("weather", "clear"),
    "overcast": ("weather", "overcast"), "glare": ("weather", "glare"),
    "highway": ("road_type", "highway"), "urban": ("road_type", "urban"),
    "rural": ("road_type", "rural"), "residential": ("road_type", "residential"),
    "dense": ("density", "dense"), "moderate": ("density", "moderate"), "sparse": ("density", "sparse"),
}

# State vocabulary. "reviewed" is the one people mean most and is not a state: it is the set of states a
# human has ruled on, which is a distinction this corpus has been bitten by before.
_STATE_TERMS: dict[str, list[str]] = {
    "reviewed": ["accepted", "rejected"],
    "accepted": ["accepted"],
    "rejected": ["rejected"],
    "unreviewed": ["review", "annotate"],
    "auto_accepted": ["auto_accept"],
}

# Groups that name a set of classes rather than one. Resolved through the ontology's own l1 field rather
# than a hardcoded list, so a class added tomorrow joins its group without editing this file.
_L1_GROUPS: dict[str, str] = {
    "vru": "vru", "two_wheeler": "two_wheeler", "two-wheeler": "two_wheeler",
    "three_wheeler": "three_wheeler", "three-wheeler": "three_wheeler",
    "four_wheeler": "four_wheeler", "four-wheeler": "four_wheeler",
    "heavy": "heavy", "animal": "animal",
}


class QueryError(ValueError):
    """A term nobody can resolve. Raised rather than skipped: see the module docstring."""


def vocabulary() -> dict:
    """Every term that resolves, so a caller can be told what it could have written."""
    from services.autolabel.ontology import get_ontology

    return {
        "scene": sorted(_SCENE_TERMS),
        "state": sorted(_STATE_TERMS),
        "group": sorted(_L1_GROUPS),
        "class": sorted(c.name for c in get_ontology().classes),
    }


def compile_query(q: str) -> dict:
    """Compile a term expression into the predicate the rest of the system already understands.

    Terms are ANDed, which is what the vocabulary supports: every clause in a predicate is a constraint and
    they intersect. OR within one axis comes free, because a clause holds a list: "night AND dusk" narrows
    time_of_day to both rather than to neither.
    """
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    pred: dict = {}
    terms = [t.strip() for t in q.replace(",", " AND ").split("AND")]
    terms = [t for t in terms if t]
    if not terms:
        raise QueryError("empty query: a dataset selected by nothing is the whole corpus, which is a "
                         "decision worth writing down rather than defaulting to")

    resolved: list[dict] = []
    for raw in terms:
        t = raw.strip().lower().replace(" ", "_")
        if t in _SCENE_TERMS:
            axis, value = _SCENE_TERMS[t]
            pred.setdefault(axis, [])
            if value not in pred[axis]:
                pred[axis].append(value)
            resolved.append({"term": raw, "kind": "scene", "axis": axis, "value": value})
        elif t in _STATE_TERMS:
            pred.setdefault("states", [])
            for s in _STATE_TERMS[t]:
                if s not in pred["states"]:
                    pred["states"].append(s)
            resolved.append({"term": raw, "kind": "state", "states": _STATE_TERMS[t]})
        elif t in _L1_GROUPS:
            l1 = _L1_GROUPS[t]
            names = sorted(c.name for c in onto.classes if c.l1 == l1)
            if not names:
                raise QueryError(f"'{raw}' names an ontology group with no classes in it")
            pred.setdefault("class_names", [])
            for n in names:
                if n not in pred["class_names"]:
                    pred["class_names"].append(n)
            resolved.append({"term": raw, "kind": "group", "l1": l1, "expands_to": names})
        elif onto.has_name(t):
            pred.setdefault("class_names", [])
            if t not in pred["class_names"]:
                pred["class_names"].append(t)
            resolved.append({"term": raw, "kind": "class", "class_name": t})
        else:
            raise QueryError(
                f"unknown term '{raw}'. A term that matched nothing would silently widen the dataset, so "
                "it is refused instead. Ask GET /api/datasets/vocabulary for what resolves.")

    log.info("datasets.query_compiled", q=q, predicate=pred)
    return {"query": q, "predicate": pred, "terms": resolved}
