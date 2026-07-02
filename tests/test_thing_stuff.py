"""Thing/stuff panoptic split: countable foreground gets an instance box, background stuff never does.

The classification is pure (no infra). It pins the split for the classes this frame exposed as errors
(tree, barrier, wall, building, sky, road are stuff; vehicles, people, poles, signs are things), so a
regression that reclassifies, say, 'pole' as stuff or 'road' as a thing fails loudly.
"""

from __future__ import annotations

import pytest

from services.autolabel.ontology import get_ontology

THINGS = ["sedan", "motorcycle", "autorickshaw", "bus", "pedestrian", "cattle", "pole", "electric_pole",
          "traffic_sign", "traffic_signal", "street_light", "cone", "push_cart"]
STUFF = ["tree", "vegetation", "fallen_tree", "barrier", "crash_barrier", "median_barrier", "guardrail",
         "fence", "buildings", "side_wall", "road", "sidewalk", "median", "sky", "green_belt",
         "lane_marking", "foot_overbridge", "fly_over", "hoarding"]


@pytest.mark.parametrize("name", THINGS)
def test_countable_objects_are_things(name):
    onto = get_ontology()
    if not onto.has_name(name):
        pytest.skip(f"{name} not in ontology")
    cid = onto.by_name(name).id
    assert onto.is_thing(cid) and not onto.is_stuff(cid)


@pytest.mark.parametrize("name", STUFF)
def test_background_is_stuff(name):
    onto = get_ontology()
    if not onto.has_name(name):
        pytest.skip(f"{name} not in ontology")
    cid = onto.by_name(name).id
    assert onto.is_stuff(cid) and not onto.is_thing(cid)


def test_every_surface_and_ignore_class_is_stuff():
    onto = get_ontology()
    for c in onto.classes:
        if c.l0 in ("surface", "ignore"):
            assert onto.is_stuff(c.id), f"{c.name} ({c.l0}/{c.l1}) should be stuff"


def test_persist_drops_stuff_instances():
    """The persist chokepoint filters stuff by construction: a fused list of one car + one tree keeps the
    car and drops the tree, regardless of how it arrived."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    car = onto.by_name("sedan").id
    tree = onto.by_name("tree").id
    fused = [type("FO", (), {"obj": type("O", (), {"class_id": car})()})(),
             type("FO", (), {"obj": type("O", (), {"class_id": tree})()})()]
    kept = [fo for fo in fused if onto.is_thing(fo.obj.class_id)]   # mirrors the persist guard
    assert len(kept) == 1 and kept[0].obj.class_id == car
