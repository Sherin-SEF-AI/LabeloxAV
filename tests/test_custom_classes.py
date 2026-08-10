"""Annotator-defined custom classes: add_custom_class normalizes the name, lands it in the custom id
block marked rare (so the gate forces human review), makes it resolve through get_ontology everywhere, and
is idempotent.

This test writes to the real governed sidecar, `ontology/custom_classes.json`, because that is the file the
production path writes and pointing it elsewhere would stop testing what ships. That makes the restore
load-bearing: it runs against a tracked file in the working tree.

The restore used to re-serialize the parsed list with a bare `json.dumps`, which preserved the content and
destroyed the formatting, collapsing the file to a single line. Every full suite run therefore left the repo
dirty with a 191-line diff that was pure whitespace, and the app's own writer uses `indent=2`, so the two
fought each other. Keeping the original bytes and putting them back is the only thing that actually restores.
"""

from __future__ import annotations


def test_add_custom_class_resolves_normalized_and_idempotent():
    from services.autolabel.ontology import _custom_path, add_custom_class, get_ontology

    name = "test_qx_idol_cart"
    sidecar = _custom_path()
    # Read before the first mutation, so a failure anywhere below still restores the file byte for byte.
    original = sidecar.read_text() if sidecar.exists() else None
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
        if original is None:
            sidecar.unlink(missing_ok=True)
        else:
            sidecar.write_text(original)
        get_ontology.cache_clear()
