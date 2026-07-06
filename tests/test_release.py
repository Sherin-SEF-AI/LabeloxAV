"""M6 release tests: the content fingerprint is deterministic and order-independent (reproducible rebuilds),
and changes when an annotation's content is mutated (the original release stays byte-stable, a rebuild has a
distinct id)."""

from services.release.fingerprint import content_fingerprint, object_fingerprint

SPEC = {"name": "demo", "state": "accepted"}
ONTO = "labelox-in-0.1.0"


def _obj(oid, cls, box, state="accepted", version=1):
    return {"object_id": oid, "class_id": cls, "bbox": box, "state": state, "version": version}


def test_deterministic_same_inputs_same_hash():
    objs = [_obj("a", 6, [1, 2, 3, 4]), _obj("b", 11, [5, 6, 7, 8])]
    assert content_fingerprint(objs, SPEC, ONTO) == content_fingerprint(objs, SPEC, ONTO)


def test_order_independent():
    a = [_obj("a", 6, [1, 2, 3, 4]), _obj("b", 11, [5, 6, 7, 8])]
    b = list(reversed(a))
    assert content_fingerprint(a, SPEC, ONTO) == content_fingerprint(b, SPEC, ONTO)


def test_mutated_annotation_changes_hash():
    original = [_obj("a", 6, [1, 2, 3, 4])]
    orig_hash = content_fingerprint(original, SPEC, ONTO)
    # mutate the box of the same object id: the release must be distinguishable
    mutated = [_obj("a", 6, [1, 2, 3, 5])]
    assert content_fingerprint(mutated, SPEC, ONTO) != orig_hash
    # reclassification also changes it
    reclassed = [_obj("a", 11, [1, 2, 3, 4])]
    assert content_fingerprint(reclassed, SPEC, ONTO) != orig_hash
    # rebuilding the ORIGINAL is byte-stable
    assert content_fingerprint(original, SPEC, ONTO) == orig_hash


def test_float_noise_does_not_change_hash():
    a = [_obj("a", 6, [1.0, 2.0, 3.0, 4.0])]
    b = [_obj("a", 6, [1.0000001, 2.0, 3.0, 4.0])]  # sub-millipixel noise, rounded away
    assert object_fingerprint(a[0]) == object_fingerprint(b[0])


def test_spec_or_ontology_change_changes_hash():
    objs = [_obj("a", 6, [1, 2, 3, 4])]
    base = content_fingerprint(objs, SPEC, ONTO)
    assert content_fingerprint(objs, {"name": "other"}, ONTO) != base
    assert content_fingerprint(objs, SPEC, "labelox-in-0.2.0") != base
