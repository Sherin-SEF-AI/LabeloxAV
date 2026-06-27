"""Annotator-defined custom classes: add_custom_class normalizes the name, lands it in the custom id
block marked rare (so the gate forces human review), makes it resolve through get_ontology everywhere, and
is idempotent. The sidecar is restored in finally so the governed ontology is never left mutated."""

from __future__ import annotations

import json


def test_add_custom_class_resolves_normalized_and_idempotent():
    from services.autolabel.ontology import _custom_path, add_custom_class, get_ontology

    name = "test_qx_idol_cart"
    try:
        c1 = add_custom_class("Test QX Idol Cart")
        assert c1["name"] == name and c1["existed"] is False
        assert c1["id"] >= 200            # custom block, clear of the frozen governed ids
        assert c1["india"] is True        # rare by default -> gate routes it to human review

        onto = get_ontology()
        assert onto.has_name(name) and onto.by_name(name).id == c1["id"]  # resolves on the create/review path

        # idempotent: a differently-cased / spaced form returns the same class, no duplicate
        c2 = add_custom_class("  test-qx idol  cart ")
        assert c2["existed"] is True and c2["id"] == c1["id"]

        # rejects an empty / symbol-only name
        import pytest

        with pytest.raises(ValueError):
            add_custom_class("!!!")
    finally:
        p = _custom_path()
        if p.exists():
            p.write_text(json.dumps([c for c in json.loads(p.read_text()) if c["name"] != name]))
        get_ontology.cache_clear()
