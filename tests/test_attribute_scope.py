"""Per-class attribute scope (Plane 1 A1). An object's class restricts which global attributes apply:
a subclass with an attribute_scope entry rejects out-of-scope attributes, an unscoped subclass accepts
all, and validation with no class_id stays backward compatible. Pure, no infra, no torch.
"""

from __future__ import annotations

from services.autolabel.ontology import get_ontology

ONTO = get_ontology()


def test_scope_loaded():
    assert {"two_wheeler", "vru", "fixed"} <= set(ONTO.attribute_scope.keys())


def test_scoped_class_rejects_out_of_scope_attr():
    mc = ONTO.by_name("motorcycle")  # l1=two_wheeler
    assert ONTO.attrs_for_class(mc.id) is not None
    assert ONTO.validate_attrs({"motion": "moving"}, mc.id) == []          # applicable
    assert ONTO.validate_attrs({"signal_arrow": "left"}, mc.id)            # not applicable -> error


def test_unscoped_class_accepts_all():
    drivable = next(c for c in ONTO.classes if c.l1 == "drivable")
    assert ONTO.attrs_for_class(drivable.id) is None                       # unscoped
    assert ONTO.validate_attrs({"signal_arrow": "left"}, drivable.id) == []


def test_no_class_id_is_backward_compatible():
    assert ONTO.validate_attrs({"motion": "moving"}) == []
    assert ONTO.validate_attrs({"nope": 1})                                # unknown attribute still fails


def test_unknown_attr_fails_even_when_scoped():
    mc = ONTO.by_name("motorcycle")
    assert ONTO.validate_attrs({"nope": 1}, mc.id)
