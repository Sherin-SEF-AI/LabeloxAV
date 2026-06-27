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


def remap_name(name: str, onto: Ontology) -> tuple[int, str, bool]:
    """Return (ontology_class_id, ontology_name, mapped). mapped=False means it hit a fallback."""
    # Direct ontology hit (our own exports already use ontology names).
    if onto.has_name(name):
        c = onto.by_name(name)
        return c.id, c.name, True

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
__all__ = ["remap_name", "IDD_TO_ONTOLOGY"]
