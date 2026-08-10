"""The README states a rule about adapters; this is what enforces it.

"Adapters are written as mirrored pairs and tested by round trip, since a format adapter that only works one
way is a trap." That was true of the interchange formats and had four exceptions with nothing explaining
them, which is worse than either holding the rule or dropping it: a reader cannot tell a deliberate
asymmetry from an unfinished one.

services/formats.py names the three categories. These tests hold the code to them, so a new adapter has to
declare which kind it is and an interchange format cannot quietly lose its pair.
"""

from __future__ import annotations

import pytest

from services.export.dataset import SUPPORTED_EXPORT_FORMATS
from services.formats import ALL, DERIVED, INTERCHANGE, MIGRATION, RAW, SELF_SERVED, category
from services.imports.run import ADAPTERS, RAW_FORMATS


def test_every_interchange_format_goes_both_ways():
    """The rule itself. Pascal VOC and Mapillary were importable and not exportable once, and closing that
    asymmetry is what uncovered a latent frame-naming collision."""
    missing_export = sorted(INTERCHANGE - set(SUPPORTED_EXPORT_FORMATS))
    missing_import = sorted(INTERCHANGE - set(ADAPTERS))
    assert missing_export == [], f"interchange formats with no exporter: {missing_export}"
    assert missing_import == [], f"interchange formats with no importer: {missing_import}"


def test_a_migration_adapter_is_import_only_by_design():
    """Writing an exporter for a proprietary platform means guessing at its import schema with no way to
    verify it, and a file the target rejects is worse than no file. Not a trap, because a migrated team can
    leave through any interchange format, including lossless Parquet."""
    for fmt in MIGRATION:
        assert fmt in ADAPTERS, f"{fmt} is classified as a migration adapter but cannot be imported"
        assert fmt not in SUPPORTED_EXPORT_FORMATS, (
            f"{fmt} now has an exporter, so it is no longer migration-only; reclassify it")


def test_a_derived_target_is_export_only_by_design():
    """A derived artifact is a product of the corpus, not a representation of it. Re-importing one is a
    different and weaker operation, and allowing it would let somebody believe they had recovered a source
    they had not."""
    for fmt in DERIVED:
        assert fmt not in ADAPTERS, (
            f"{fmt} is classified as derived but has an importer; re-importing a derived artifact is not a "
            f"round trip")


def test_every_registered_format_is_classified():
    """The guard that makes the taxonomy worth having. An unclassified format is one nobody decided about,
    and the decision is the point."""
    registered = set(ADAPTERS) | set(RAW_FORMATS) | set(SUPPORTED_EXPORT_FORMATS)
    unclassified = sorted(registered - ALL)
    assert unclassified == [], (
        f"these formats are registered but not classified in services/formats.py: {unclassified}. "
        f"Decide whether each is interchange (both ways, round-trip tested), migration (in only), or "
        f"derived (out only).")


def test_nothing_is_classified_twice():
    groups = {"interchange": INTERCHANGE, "migration": MIGRATION, "derived": DERIVED, "raw": RAW}
    for a, sa in groups.items():
        for b, sb in groups.items():
            if a < b:
                assert not (sa & sb), f"{sorted(sa & sb)} is in both {a} and {b}"


def test_the_taxonomy_describes_only_formats_that_exist():
    """A stale entry would let a deleted adapter keep its blessing, and the next reader would trust it.

    SELF_SERVED is counted as registered: those targets have their own endpoint rather than living in the
    dataset export driver, because their scope is an event or a window rather than a slice of objects.
    """
    registered = (set(ADAPTERS) | set(RAW_FORMATS) | set(SUPPORTED_EXPORT_FORMATS) | SELF_SERVED)
    phantom = sorted(ALL - registered)
    assert phantom == [], f"classified but not registered anywhere: {phantom}"


def test_a_self_served_target_is_reachable_even_though_the_driver_does_not_dispatch_it():
    """Otherwise the taxonomy would claim a capability nothing provides."""
    import inspect

    from services.api.routers import export as export_router

    src = inspect.getsource(export_router)
    for fmt in SELF_SERVED:
        assert fmt in src or "scenario" in src, f"{fmt} is classified but has no route"


@pytest.mark.parametrize("fmt,expected", [
    ("coco", "interchange"), ("parquet", "interchange"),
    ("labelbox", "migration"), ("scale", "migration"),
    ("panoptic", "derived"), ("hdmap", "derived"),
    ("video", "raw"),
])
def test_category_answers_for_the_formats_a_reader_would_ask_about(fmt, expected):
    assert category(fmt) == expected


def test_an_unknown_format_is_unclassified_rather_than_assumed():
    """Defaulting to interchange would assert a round trip that does not exist."""
    assert category("some_new_tool") is None


def test_a_migrated_customer_can_still_leave():
    """The claim that makes one-way migration adapters defensible rather than lock-in: whatever came in
    through a migration adapter can leave through a lossless interchange format."""
    assert "parquet" in INTERCHANGE
    assert "parquet" in ADAPTERS and "parquet" in SUPPORTED_EXPORT_FORMATS
