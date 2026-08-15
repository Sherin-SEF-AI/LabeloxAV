"""Nine import formats landed on the wrong importer, in silence.

`File > Import` deep-links a format per menu entry, and the menu listed all nineteen the backend registers.
The import page carried its own hardcoded list of ten and fell back to `"coco"` for anything not in it, with
no message. So choosing CVAT XML, Label Studio, KITTI, BDD100K, Labelbox, Scale AI, SuperAnnotate or Encord
started a COCO import: it read nothing it understood and reported success.

The page now asks the server. This test guards the other half, the static menu, which cannot ask because it
is configuration rather than a component: a format the backend gains and the menu does not is invisible, and
one the menu offers and the backend has never heard of is a dead entry that now says so instead of quietly
importing as something else.
"""

from __future__ import annotations

import re
from pathlib import Path

from services.imports.run import ADAPTERS, ALL_FORMATS, RAW_FORMATS

MENUS = Path("web/lib/menus.ts")


def _menu_import_formats() -> set[str]:
    """The format ids the File > Import menu offers, read from the source rather than a duplicate list."""
    src = MENUS.read_text()
    block = src.split("const IMPORT_FORMATS", 1)[1].split("];", 1)[0]
    return set(re.findall(r'\["([a-z0-9_]+)",', block))


class TestTheMenuMatchesTheImporter:
    def test_every_menu_format_is_one_the_importer_can_read(self):
        """A menu entry the backend does not know is a dead link that used to import as COCO."""
        extra = _menu_import_formats() - set(ALL_FORMATS)
        assert not extra, f"the menu offers formats the importer cannot read: {sorted(extra)}"

    def test_every_format_the_importer_reads_is_reachable_from_the_menu(self):
        """The direction that hides capability: an adapter nobody can select is work that shipped and
        cannot be used."""
        missing = set(ALL_FORMATS) - _menu_import_formats()
        assert not missing, f"the importer reads formats the menu does not offer: {sorted(missing)}"

    def test_the_competitor_migrations_are_all_offered(self):
        # These are the four the silent COCO fallback hit hardest: a team migrating years of labels would
        # have watched an import succeed and produce nothing.
        assert {"labelbox", "scale", "superannotate", "encord"} <= _menu_import_formats()


class TestTheServedList:
    def test_it_is_the_union_of_adapters_and_raw_media(self):
        assert set(ALL_FORMATS) == set(ADAPTERS) | RAW_FORMATS

    def test_raw_media_is_distinguishable_from_an_annotated_format(self):
        """A video brings frames and no labels. Saying so on the option is the difference between an empty
        import and a surprise."""
        assert RAW_FORMATS and not (RAW_FORMATS & set(ADAPTERS))

    def test_the_page_no_longer_carries_a_second_list(self):
        """The fallback exists only so the select is not empty on first paint. If it ever regrows into the
        full list, the drift comes back."""
        page = Path("web/app/import/page.tsx").read_text()
        block = page.split("const FALLBACK_FORMATS", 1)[1].split("]", 1)[0]
        assert len(re.findall(r'"[a-z0-9_]+"', block)) < len(ALL_FORMATS)
