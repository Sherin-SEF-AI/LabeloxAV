"""Which formats go which way, and why each one is allowed to.

The README states a principle: "adapters are written as mirrored pairs and tested by round trip, since a
format adapter that only works one way is a trap." It is a good principle and it had four exceptions with
nothing anywhere explaining them, which is worse than either holding it or dropping it: a reader cannot tell
a deliberate asymmetry from an unfinished one, and neither can a reviewer.

The exceptions are not exceptions. They are two other categories that the principle was never about, and
naming them is what makes it enforceable.

**INTERCHANGE** formats represent the corpus, so they go both ways and the round trip is the test. This is
what the README's principle is about, and it holds for all eleven of them. Pascal VOC and Mapillary were
importable and not exportable once, and closing that asymmetry is what uncovered a latent frame-naming
collision, which is the argument for the rule.

**MIGRATION** adapters read a competitor's export. One-way in, and not a trap, because the trap the
principle describes is data that cannot get out *in any shape*. A team migrating from Labelbox can leave
through any of the eleven interchange formats, including Parquet, which is lossless. What they cannot do is
round-trip back into Labelbox's own format, and writing that would mean guessing at a proprietary import
schema with no way to verify it. An exporter that produces a file the target platform rejects is worse than
no exporter: it is a promise that fails at the worst moment.

**DERIVED** targets are products of the corpus rather than representations of it. An OpenSCENARIO document
is a recording turned into something a simulator can run; an HD map is GeoJSON built from many sessions; a
panoptic raster is a dense label with no object rows behind it. Importing one back is not a round trip, it
is a different and much weaker operation, and pretending otherwise would let somebody re-import a derived
artifact and believe they had recovered the source.

The test in tests/test_format_taxonomy.py enforces all of this against the live registries: every format
must be classified, every interchange format must exist in both directions, and nothing may be classified
into two categories. A new adapter therefore has to declare which kind it is, which is the question this
file exists to make somebody answer.
"""

from __future__ import annotations

# Both directions, round-trip tested. The README's principle, and it holds without exception here.
INTERCHANGE = frozenset({
    "coco", "yolo", "pascalvoc", "cvat", "labelstudio", "openlabel",
    "nuscenes", "kitti", "bdd", "mapillary", "parquet",
})

# A competitor's export, read once when a team switches. One-way by design; see the module docstring.
MIGRATION = frozenset({"labelbox", "scale", "superannotate", "encord"})

# Products of the corpus rather than representations of it. One-way out.
DERIVED = frozenset({"panoptic", "lanes", "drivable", "hdmap", "masks", "openscenario"})

# Derived targets served by their own endpoint rather than by the dataset export driver, because their scope
# is not a slice of objects. A scenario is bounded by an event or a time window, so asking the slice-shaped
# driver for one would mean inventing a scope it does not have. Listed so the taxonomy stays complete: a
# reader looking for OpenSCENARIO should find it here rather than conclude it does not exist.
SELF_SERVED = frozenset({"openscenario"})

# Raw media, which has no annotations to mirror.
RAW = frozenset({"images", "video", "mcap"})

ALL = INTERCHANGE | MIGRATION | DERIVED | RAW


def category(fmt: str) -> str | None:
    """Which kind a format is, or None when it has not been classified.

    None rather than a default, because a format nobody has classified is exactly the case the taxonomy
    exists to surface: silently calling it interchange would assert a round trip that does not exist.
    """
    for name, members in (("interchange", INTERCHANGE), ("migration", MIGRATION),
                          ("derived", DERIVED), ("raw", RAW)):
        if fmt in members:
            return name
    return None
