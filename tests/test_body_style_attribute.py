"""Body style as an attribute, because the classes were not delivering the distinction they claim.

The four_wheeler group names nine classes. The corpus fills one of them: 134,281 of 142,447 four-wheelers
are labelled `sedan`, which is 94.3%, on roads where the best-selling cars are hatchbacks and hatchback
holds 0.3%. `sedan` is a `car` class that happens to be spelled sedan.

That is not a labelling failure to correct by relabelling. It is a taxonomy finer than the process, and a
judge good enough to notice makes it visible rather than causing it: on a random 300-crop sample, 128 of 140
rejections proposed another four-wheeler and 2 said the label named the wrong kind of thing. Strict class
precision reads 0.51; superclass reads 0.96.

Declared as an attribute so the distinction has somewhere to live that is not the class id. Additive: no
class removed, no object relabelled, nothing downstream changed until somebody populates it.
"""

from __future__ import annotations

import pytest

from services.autolabel.ontology import get_ontology


def test_body_style_is_declared_as_an_enum():
    onto = get_ontology()
    attr = onto.attributes.get("body_style")
    assert attr is not None, "body_style must exist for the distinction to have anywhere to live"
    assert attr.type == "enum"
    # The values the judge actually proposed on this corpus, plus an explicit unknown so an unreadable crop
    # is not forced into a guess.
    for v in ("sedan", "hatchback", "suv", "pickup", "unknown"):
        assert v in attr.values


def test_it_is_scoped_to_four_wheelers_and_nowhere_else():
    """Attribute scope is per L1. A body style on a pedestrian is a nonsense the schema should refuse."""
    onto = get_ontology()
    assert "body_style" in (onto.attribute_scope.get("four_wheeler") or [])
    for group in ("vru", "animal", "two_wheeler", "fixed", "boundary"):
        assert "body_style" not in (onto.attribute_scope.get(group) or []), (
            f"body_style must not be offered on {group}")


def test_a_value_outside_the_enum_is_refused():
    onto = get_ontology()
    assert not onto.validate_attrs({"body_style": "suv"})
    assert onto.validate_attrs({"body_style": "spaceship"})


def test_adding_it_moved_no_class():
    """The point of doing this as an attribute. Collapsing the four_wheeler classes or relabelling 142,447
    objects is a much larger decision, and this is deliberately not it."""
    onto = get_ontology()
    names = {c.name for c in onto.classes if c.l1 == "four_wheeler"}
    for still_here in ("sedan", "hatchback", "suv", "pickup", "mpv"):
        assert still_here in names, f"{still_here} must still exist as a class"


@pytest.mark.parametrize("name", ["sedan", "suv", "hatchback"])
def test_the_classes_it_describes_still_resolve(name):
    """A consumer pinned to a class id must not be broken by this."""
    onto = get_ontology()
    cls = onto.by_name(name)
    assert cls.l1 == "four_wheeler"
    assert onto.by_id(cls.id).name == name
