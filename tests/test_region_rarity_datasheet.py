"""WP4: where a drive happened, how unusual a frame is, and what a release will not claim.

The theme is refusing to guess. A corpus that records one city under two spellings looks like two strata; a
frame nobody has scored is not a frame with rarity zero; and a datasheet's value is entirely in the sections
that say why a number is missing rather than printing one.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


class TestRegion:
    def test_one_city_recorded_two_ways_is_one_stratum(self):
        """372 sessions say BLR and one says Bengaluru. Anything stratifying on the raw string sees two
        places, which is the whole reason this module exists."""
        from services.context.region import resolve_region

        a, b = resolve_region("BLR"), resolve_region("Bengaluru")
        assert a.status == b.status == "resolved"
        assert a.stratum() == b.stratum() == "Karnataka/Bengaluru"

    def test_a_non_indian_import_is_outside_and_not_a_class_1_town(self):
        """A KITTI drive through Karlsruhe given an Indian urban class puts German motorway footage in an
        Indian stratum. `outside` is a fact; inventing a class would be a fabrication."""
        from services.context.region import resolve_region

        for city, country in (("BERKELEY", "United States"), ("Karlsruhe", "Germany")):
            r = resolve_region(city)
            assert r.status == "outside" and r.outside == country
            assert r.urban_class is None

    def test_absent_and_unknown_are_different_answers(self):
        """A session with no city is a gap in capture; a session whose city the pack does not model is a gap
        in the region table. Merging them tells whoever reads the report to fix the wrong thing."""
        from services.context.region import resolve_region

        assert resolve_region(None).status == "absent"
        assert resolve_region("").status == "absent"
        assert resolve_region("Atlantis").status == "unknown"

    def test_the_stratum_label_is_never_none(self):
        """A group-by on this must not silently drop rows."""
        from services.context.region import resolve_region

        for c in ("BLR", "Karlsruhe", None, "Atlantis", "GLOBAL"):
            assert isinstance(resolve_region(c).stratum(), str)

    def test_a_filter_for_the_city_finds_the_code_the_fleet_records(self):
        """`cities: ["Bengaluru"]` matches one session of 373. This is what makes the region filter usable."""
        from services.context.region import city_strings_for

        assert {"blr", "bengaluru", "bangalore"} <= city_strings_for("Bengaluru")

    def test_a_filter_for_the_state_reaches_its_other_cities(self):
        from services.context.region import city_strings_for

        got = city_strings_for("Karnataka")
        assert {"blr", "mysuru", "mysore", "mangaluru"} <= got

    def test_normalisation_matches_how_the_corpus_actually_writes_places(self):
        from services.context.region import normalise

        assert normalise("  New Delhi ") == "new delhi"
        assert normalise("Bengaluru,") == "bengaluru"
        assert normalise(None) == ""

    def test_road_class_is_unresolved_rather_than_defaulted(self):
        """Frame.road_class is NULL on all 41,752 frames. A default here would make the corpus look
        stratified by road type when nothing has ever populated it."""
        from types import SimpleNamespace

        from services.context.region import road_class_of

        assert road_class_of(SimpleNamespace(road_class=None)) == "unresolved"
        assert road_class_of(SimpleNamespace(road_class="arterial")) == "arterial"


class TestRarity:
    @pytest.mark.asyncio
    async def test_the_sweep_does_not_skip_frames_with_no_scene(self):
        """`~scene.has_key(k)` is NULL, not true, when scene is NULL. Written without the NULL arm this
        skipped the 39,972 frames that have no scene at all - 96% of the corpus - and reported remaining 0.
        It looked finished."""
        from sqlalchemy import select

        from db.models import Frame, Object, OntologyClass, OntologyVersion
        from db.models import Session as DbSession
        from db.session import get_sessionmaker
        from services.autolabel.ontology import get_ontology
        from services.context.rarity import sweep_rarity
        from services.intelligence.search.rarity import reset_cache

        onto = get_ontology()
        async with get_sessionmaker()() as db:
            if await db.get(OntologyVersion, onto.version) is None:
                db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
                await db.flush()
                for c in onto.classes:
                    db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                         india=c.india, map_to={}))
                await db.flush()
            t0, sid = now_ns(), uuid.uuid4()
            db.add(DbSession(session_id=sid, vehicle_id="RAR-1", start_ts_ns=t0,
                             end_ts_ns=t0 + seconds_to_ns(10), city="BLR", sensors={},
                             ontology_version=onto.version))
            await db.flush()
            # One frame with a scene, one with none at all. The second is the case that was skipped.
            with_scene, without = uuid.uuid4(), uuid.uuid4()
            db.add(Frame(frame_id=with_scene, session_id=sid, ts_ns=t0, cam_id="c", img_uri="s3://a.jpg",
                         width=64, height=64, quality=0.9, scene={"weather": "clear"}))
            db.add(Frame(frame_id=without, session_id=sid, ts_ns=t0 + 1, cam_id="c", img_uri="s3://b.jpg",
                         width=64, height=64, quality=0.9, scene=None))
            await db.flush()
            for fid in (with_scene, without):
                db.add(Object(frame_id=fid, class_id=onto.by_name("cattle").id,
                              bbox=[1.0, 1.0, 9.0, 9.0], conf=0.9, source="fused", state="review"))
            await db.commit()
            reset_cache()
            await sweep_rarity(db, limit=5000, session_id=sid, force=True)
            rows = {f.frame_id: f.scene for f in
                    (await db.execute(select(Frame).where(Frame.frame_id.in_([with_scene, without])))).scalars()}

        assert "rarity" in (rows[without] or {}), "a frame with no scene was skipped"
        assert "rarity" in rows[with_scene]
        # The merge, not a replace: a rarity pass that overwrote scene would delete the ingest classifier's
        # weather, density, road_type and time_of_day from every frame that had them.
        assert rows[with_scene]["weather"] == "clear"

    def test_a_frame_takes_the_rarity_of_its_rarest_object(self):
        """The maximum rather than the mean. A cattle crossing on a road full of sedans is a cattle frame,
        and averaging lets the sedans bury it."""
        from services.intelligence.search.rarity import frame_rarity

        idf = {1: 0.05, 31: 0.9}
        assert frame_rarity([1, 1, 1, 31], idf) == 0.9
        assert frame_rarity([], idf) == 0.0

    def test_an_unscored_frame_is_excluded_from_a_band_not_treated_as_zero(self):
        """Zero is a real score meaning nothing unusual is here. Conflating it with not-yet-measured fills
        every low-rarity cohort with frames nobody has scored."""
        from services.curation.slices import matches_predicate

        unscored = {"scene": {}, "city": "BLR"}
        zero = {"scene": {"rarity": 0.0}, "city": "BLR"}
        assert matches_predicate(zero, {"rarity_min": 0.0, "rarity_max": 0.1})
        assert not matches_predicate(unscored, {"rarity_min": 0.0, "rarity_max": 0.1})


class TestTheCurationTwinAgreesWithExport:
    """A cohort that previews as 900 frames and exports as 40,000 is worse than one that cannot export."""

    def test_every_new_slice_clause_exists_on_both_sides(self):
        from services.export.dataset import SliceSpec

        fields = set(SliceSpec.model_fields)
        assert {"regions", "context", "rarity_min", "rarity_max", "track_event_types"} <= fields

    def test_the_region_clause_resolves_rather_than_matching_the_raw_string(self):
        from services.curation.slices import matches_predicate

        rec = {"scene": {}, "city": "BLR"}
        assert matches_predicate(rec, {"regions": ["Bengaluru"]})
        assert matches_predicate(rec, {"regions": ["Karnataka"]})
        assert not matches_predicate(rec, {"regions": ["Chennai"]})
        # The raw-string clause is the one that misses it, which is why `regions` exists beside `cities`.
        assert not matches_predicate(rec, {"cities": ["Bengaluru"]})

    def test_a_context_axis_beyond_the_four_ingest_ones_filters(self):
        from services.curation.slices import matches_predicate

        rec = {"scene": {"waterlogging": True, "weather": "rain"}, "city": "BLR"}
        assert matches_predicate(rec, {"context": {"waterlogging": [True]}})
        assert not matches_predicate(rec, {"context": {"waterlogging": [False]}})

    def test_a_track_event_clause_needs_one_of_the_named_types(self):
        from services.curation.slices import matches_predicate

        rec = {"scene": {}, "city": "BLR", "track_event_types": ["hard_brake"]}
        assert matches_predicate(rec, {"track_event_types": ["hard_brake", "cut_in"]})
        assert not matches_predicate(rec, {"track_event_types": ["cut_in"]})

    def test_events_default_to_accepted_only(self):
        """A proposal is a suggestion. Training on unreviewed heuristics is how a threshold becomes a label,
        and the hard_brake proposer's first sweep produced 16,436 of them."""
        from services.export.dataset import SliceSpec

        assert SliceSpec(name="t").track_event_states == ["accepted"]


class TestDatasheet:
    @pytest.mark.asyncio
    async def test_it_states_a_reason_for_every_number_it_cannot_produce(self):
        from db.models import DatasetCommit
        from db.session import get_sessionmaker
        from services.export.coverage import build_datasheet, render_html

        cid = f"lbx-test-{uuid.uuid4().hex[:12]}"
        async with get_sessionmaker()() as db:
            db.add(DatasetCommit(commit_id=cid, slice_spec={}, object_count=0,
                                 ontology_version="labelox-in-0.2.0"))
            await db.commit()
            sheet = await build_datasheet(db, cid)

        for key in ("quality", "recapture"):
            section = sheet[key]
            if section.get("measured") is False:
                # The reason has to be actionable, not the word "unavailable".
                assert len(section["reason"].split()) >= 6, section
        assert sheet["regions"]["road_class"]["measured"] is False
        assert isinstance(sheet["limitations"], list)
        assert render_html(sheet).startswith("<!doctype html>")

    @pytest.mark.asyncio
    async def test_privacy_reports_the_three_states_the_schema_actually_has(self):
        """PiiAudit has no status column: a row means scanned, a zero-count row means scanned and clean.
        There is no verified state and no failed state anywhere, so neither is reported."""
        from db.models import DatasetCommit
        from db.session import get_sessionmaker
        from services.export.coverage import build_datasheet

        cid = f"lbx-test-{uuid.uuid4().hex[:12]}"
        async with get_sessionmaker()() as db:
            db.add(DatasetCommit(commit_id=cid, slice_spec={}, object_count=0,
                                 ontology_version="labelox-in-0.2.0"))
            await db.commit()
            priv = (await build_datasheet(db, cid))["privacy"]

        assert {"scanned", "scanned_and_clean", "scanned_with_regions_found", "not_scanned"} <= set(priv)
        assert "verified" not in priv and "failed" not in priv

    def test_limitations_are_derived_from_the_counts_and_cannot_go_stale(self):
        from services.export.coverage import _limitations

        sheet = {
            "regions": {"concentration": 0.99, "by_stratum": {"Karnataka/Bengaluru": 373},
                        "raw_strings": {"BLR": 372, "Bengaluru": 1}},
            "composition": {"classes_absent": ["a", "b"], "classes_under_10": ["c"],
                            "n_classes_in_ontology": 199},
            "context": {"coverage": 0.04},
            "quality": {"measured": False, "reason": "no evaluation"},
            "recapture": {"measured": False, "reason": "audit seeded, awaiting 60 frames"},
            "privacy": {"not_scanned": 26},
            "track_events": {"types_absent": ["x"]},
        }
        out = " ".join(_limitations(sheet))
        assert "single-location corpus" in out
        assert "more than one string" in out
        assert "60 frames" in out
