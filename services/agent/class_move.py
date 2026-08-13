"""Which class changes are a refinement, and which are a category error.

A relabel run rewrote 50,812 objects. Most were refinements: `sedan -> mpv`, `truck -> container_truck`, a
model being more specific than the one before it. Some were not. It moved **1,047 buses into
`bmtc_bus_shelter`**, a bus shelter, at confidence 0.989, over the top of every other signal agreeing on
`bus`: the detector at 0.917, the open-vocabulary path at 0.803, the reasoner recording "aspect 0.81 is
typical for bus", "the track agrees on bus across 6 neighbouring frames" and "2 independent paths agree on
bus". Across the run, 2,023 objects crossed the `l0` boundary and 2,986 were moved from a countable thing to
uncountable stuff.

The ontology already forbids the result. `bmtc_bus_shelter` is in `STUFF_NAMES`, so it is not a thing that
gets an instance box at all, and the same runner's own fusion step drops stuff detections before they are
persisted. A vehicle cannot become fixed infrastructure by re-reading the same crop.

So this is a pure check with no model in it: a proposal may sharpen a class, and may not move it to a
different kind of thing. It sits between the proposal and the write rather than inside the model, because it
is a property of the ontology rather than of any one relabeller, and because a confident wrong answer is
exactly the case a confidence threshold cannot catch.
"""

from __future__ import annotations

from services.autolabel.ontology import Ontology

# Refusals, as reasons rather than booleans: a run that skips a proposal has to be able to say what it
# skipped and why, or the count is just an unexplained shortfall.
CROSSES_L0 = "crosses the l0 boundary ({from_l0} -> {to_l0}), which is a different kind of thing"
THING_TO_STUFF = "turns a countable object into uncountable stuff"
STUFF_TO_THING = "turns uncountable stuff into a countable object"


def refuse_reason(onto: Ontology, from_class: int, to_class: int) -> str | None:
    """Why this class move must not happen, or None if it is a legitimate refinement.

    Unknown classes are not this function's business: the caller already drops proposals it cannot store,
    and answering "refused" for a class the ontology has never heard of would confuse a missing class with a
    forbidden move.
    """
    if int(from_class) == int(to_class):
        return None
    try:
        a, b = onto.by_id(int(from_class)), onto.by_id(int(to_class))
    except KeyError:
        return None

    from_stuff, to_stuff = onto.is_stuff(a.id), onto.is_stuff(b.id)
    if from_stuff != to_stuff:
        # Checked before l0, because it is the sharper statement: the persist layer refuses to store stuff as
        # an instance at all, so this move produces a row the rest of the system says cannot exist.
        return THING_TO_STUFF if to_stuff else STUFF_TO_THING
    if a.l0 != b.l0:
        return CROSSES_L0.format(from_l0=a.l0, to_l0=b.l0)
    return None


def is_refinement(onto: Ontology, from_class: int, to_class: int) -> bool:
    """True when the move stays within one kind of thing, which is what a relabel is allowed to do."""
    return refuse_reason(onto, from_class, to_class) is None
