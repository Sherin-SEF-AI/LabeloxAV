"""Name -> ontology remap for imports. Reuses the IDD mapping table and the _map_name idiom from
scripts/idd_to_yolo.py so external vocabularies collapse into the ontology consistently. Unmapped
names fall back to vehicle_fallback (if the token looks vehicular) or object_fallback, and are counted
so an import never silently drops a class.
"""

from __future__ import annotations

from scripts.idd_to_yolo import IDD_TO_ONTOLOGY, _map_name, _normalize
from services.autolabel.ontology import Ontology

_VEHICLE_HINTS = ("car", "truck", "bus", "van", "vehicle", "auto", "rickshaw", "bike",
                  "motor", "scooter", "cycle", "lorry", "tractor", "trailer", "taxi")


# The 23 nuScenes categories, by their dotted names, which no token heuristic maps correctly: before this
# table `vehicle.car` became vehicle_fallback, `human.pedestrian.adult` became object_fallback, and
# `animal` became cattle. None is a deliberate "no class here says this": it lands in the fallback and is
# counted as unmapped, which is the truth, rather than being filed under the nearest wrong name.
# `vehicle.car` follows the same convention as COCO `car` (sedan, the head guess).
NUSCENES_TO_ONTOLOGY: dict[str, str | None] = {
    "animal": None,
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "construction_worker",
    "human.pedestrian.personal_mobility": None,
    "human.pedestrian.police_officer": "traffic_police",
    "human.pedestrian.stroller": None,
    "human.pedestrian.wheelchair": "wheelchair_user",
    "movable_object.barrier": "barrier",
    "movable_object.debris": "debris",
    "movable_object.pushable_pullable": "push_cart",
    "movable_object.trafficcone": "cone",
    "static_object.bicycle_rack": None,
    "vehicle.bicycle": "cycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "sedan",
    "vehicle.construction": "construction_veh",
    "vehicle.emergency.ambulance": "ambulance",
    "vehicle.emergency.police": "police_van",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.trailer": "multi_axle_trailer",
    "vehicle.truck": "truck",
}


def remap_name(name: str, onto: Ontology) -> tuple[int, str, bool]:
    """Return (ontology_class_id, ontology_name, mapped). mapped=False means it hit a fallback."""
    # Direct ontology hit (our own exports already use ontology names).
    if onto.has_name(name):
        c = onto.by_name(name)
        return c.id, c.name, True

    if name in NUSCENES_TO_ONTOLOGY:
        target = NUSCENES_TO_ONTOLOGY[name]
        if target and onto.has_name(target):
            c = onto.by_name(target)
            return c.id, c.name, True
        fallback = ("vehicle_fallback" if name.startswith("vehicle.") and onto.has_name("vehicle_fallback")
                    else "object_fallback")
        c = onto.by_name(fallback if onto.has_name(fallback) else onto.classes[0].name)
        return c.id, c.name, False

    mapped = _map_name(name, onto)  # uses IDD table + normalization, else object_fallback/None
    norm = _normalize(name)
    if mapped and mapped != "object_fallback" and onto.has_name(mapped):
        c = onto.by_name(mapped)
        return c.id, c.name, True

    # Fallback: choose a vehicular bucket when the token looks like a vehicle.
    fallback = "vehicle_fallback" if any(h in norm for h in _VEHICLE_HINTS) else "object_fallback"
    if not onto.has_name(fallback):
        fallback = "object_fallback" if onto.has_name("object_fallback") else onto.classes[0].name
    c = onto.by_name(fallback)
    return c.id, c.name, False


# re-export for callers that want the raw table
__all__ = ["remap_name", "IDD_TO_ONTOLOGY", "NUSCENES_TO_ONTOLOGY"]
