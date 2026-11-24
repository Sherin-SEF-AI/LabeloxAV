"""nuScenes category names map to the classes they mean, and to nothing when no class means them."""

from __future__ import annotations

import pytest

from services.autolabel.ontology import get_ontology
from services.imports.remap import NUSCENES_TO_ONTOLOGY, remap_name


@pytest.mark.parametrize("source,expected", [
    ("vehicle.car", "sedan"), ("vehicle.truck", "truck"), ("vehicle.bus.rigid", "bus"),
    ("vehicle.bus.bendy", "bus"), ("vehicle.motorcycle", "motorcycle"), ("vehicle.bicycle", "cycle"),
    ("human.pedestrian.adult", "pedestrian"), ("human.pedestrian.child", "pedestrian"),
    ("human.pedestrian.police_officer", "traffic_police"), ("movable_object.trafficcone", "cone"),
    ("movable_object.barrier", "barrier"), ("vehicle.emergency.ambulance", "ambulance"),
])
def test_a_category_maps_to_the_class_it_means(source, expected):
    _cid, name, mapped = remap_name(source, get_ontology())
    assert (name, mapped) == (expected, True)


@pytest.mark.parametrize("source", ["animal", "static_object.bicycle_rack", "human.pedestrian.stroller"])
def test_a_category_with_no_counterpart_is_counted_unmapped_not_filed_under_a_wrong_class(source):
    # `animal` used to become cattle, and a bicycle rack a vehicle.
    _cid, name, mapped = remap_name(source, get_ontology())
    assert mapped is False and name == "object_fallback"


def test_every_target_in_the_table_is_a_real_class():
    onto = get_ontology()
    missing = {k: v for k, v in NUSCENES_TO_ONTOLOGY.items() if v and not onto.has_name(v)}
    assert not missing, f"the table names classes the ontology does not have: {missing}"


def test_the_table_covers_all_23_categories():
    assert len(NUSCENES_TO_ONTOLOGY) == 23
