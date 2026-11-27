"""Which classes an object could plausibly be instead of what it was called: the shortlist every specialist
that asks a question (zero-shot, the VLM) is restricted to.

Asking "which of these 200 classes is it?" is the trap that made `classify_crop` and the traffic-sign prompt
useless: with no negatives and no margin, a softmax over 200 bare names always has a winner, and on a dashcam
crop the winner is often absurd (a bus typed as a sign). The question people actually answer when they
correct a label is narrow: "sign or hoarding?", "truck or construction barrier?". So the shortlist for an
object predicted as `c` is:

1. `c` itself;
2. the classes people have corrected `c` to, or corrected to `c`, most often (the confusion graph, mined from
   human rulings only);
3. `c`'s siblings in the ontology's l1 taxonomy, to fill the list when the graph is thin;
4. "not an object", always, because a third of human corrections are that.

The graph is built from rulings, so when it is graded it must be built without the item being graded:
`ConfusionGraph.build` takes the items to learn from, and the benchmark passes the other half of the split.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from services.labelagent.truth import NOT_OBJECT, TruthItem

MAX_SHORTLIST = 8
MIN_CONFUSIONS = 3


@dataclass
class ConfusionGraph:
    edges: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))

    @classmethod
    def build(cls, items: list[TruthItem], *, min_count: int = MIN_CONFUSIONS) -> ConfusionGraph:
        """Undirected confusion counts between machine class and human class, from rulings only. Direction
        is dropped on purpose: if people turn trucks into barriers, a barrier called a truck and a truck
        called a barrier are the same question."""
        raw: Counter = Counter()
        for i in items:
            if i.tier == "keep" or i.pred == i.truth:
                continue
            a, b = sorted((i.pred, i.truth))
            raw[(a, b)] += 1
        g = cls()
        for (a, b), n in raw.items():
            if n >= min_count:
                g.edges[a][b] += n
                g.edges[b][a] += n
        return g

    def neighbours(self, cls_id: int) -> list[int]:
        return [c for c, _n in self.edges.get(cls_id, Counter()).most_common()]


def shortlist(pred: int, graph: ConfusionGraph, onto, *, size: int = MAX_SHORTLIST) -> list[int]:
    """The classes worth asking about for an object the machine called `pred`, `pred` first, NOT_OBJECT
    last. Fallback classes (`undefined`, `background`) are never offered as classes: they are what
    NOT_OBJECT already means."""
    from services.labelagent.truth import NOT_OBJECT_CLASS_NAMES

    def offerable(c: int) -> bool:
        if c == NOT_OBJECT:
            return False
        try:
            cd = onto.by_id(c)
        except KeyError:
            return False
        return cd.name not in NOT_OBJECT_CLASS_NAMES

    out: list[int] = [pred] if offerable(pred) else []
    for c in graph.neighbours(pred):
        if len(out) >= size - 1:
            break
        if c not in out and offerable(c):
            out.append(c)
    if len(out) < size - 1 and offerable(pred):
        l1 = onto.by_id(pred).l1
        for cd in onto.classes:
            if len(out) >= size - 1:
                break
            if cd.l1 == l1 and cd.id not in out and offerable(cd.id):
                out.append(cd.id)
    return out + [NOT_OBJECT]
