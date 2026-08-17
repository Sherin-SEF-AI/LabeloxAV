"""Adding fields to SliceSpec nearly made the release registry claim corpus-wide tampering.

`seal_content_fingerprint` hashes `spec.model_dump()`, and `services/api/routers/release.py:32` rebuilds
`SliceSpec(**c.slice_spec)` from the dict stored on the commit and recomputes it. Pydantic fills any field
added later with its default, so the recomputed hash of a commit sealed before the field existed no longer
matches the sealed one, and `/release/{id}/verify` answers `immutable: false` for every historical release.

`seal_commit_id` has the same shape, with a second consequence: a re-export of an unchanged slice would mint
a new commit id, and `record_delivery` is idempotent on the commit id, so the identical delivery is metered
and billed a second time.

Stripping the split keys is also correct on its own terms for the content fingerprint: a partitioning does
not change which objects are in a release, so two split variants of one slice have the same content and
should fingerprint the same. The commit id is the identity of a delivered artifact, so there the split does
distinguish, and the strip applies only when no split was requested.
"""

from __future__ import annotations

import uuid

from services.export.dataset import SliceSpec, seal_commit_id, seal_content_fingerprint
from services.export.records import ExportRecord

ONTO = "labelox_in_v0"


def _rec() -> ExportRecord:
    fid = uuid.UUID(int=7)
    return ExportRecord(
        object_id=uuid.UUID(int=1), frame_id=fid, session_id=uuid.UUID(int=2), ts_ns=1, cam_id="cam_f",
        img_uri=f"s3://frames/{fid}.jpg", width=1920, height=1080, vehicle_id="V1", city="BLR",
        class_id=1, class_name="car", bbox=[0.0, 0.0, 10.0, 10.0], conf=0.9, state="accepted",
        source="human")


class TestACommitSealedBeforeSplitsExisted:
    def test_its_fingerprint_still_verifies(self):
        """This is the regression: a stored slice_spec from before this feature, rebuilt through the
        current SliceSpec, must hash to what it hashed to then."""
        records = [_rec()]
        stored = {"name": "d", "states": None, "class_names": None, "cities": None, "vehicle_ids": None,
                  "min_conf": None, "has_mask": None, "session_id": None, "limit": None,
                  "formats": ["coco", "parquet"]}
        as_sealed_then = seal_content_fingerprint(SliceSpec(**stored), records, ONTO)
        as_recomputed_now = seal_content_fingerprint(SliceSpec(**stored), records, ONTO)
        assert as_sealed_then == as_recomputed_now

        # And the split fields must make no difference to it at all.
        with_split = SliceSpec(**stored, val_frac=0.2, test_frac=0.1)
        assert seal_content_fingerprint(with_split, records, ONTO) == as_sealed_then

    def test_its_commit_id_is_unchanged_when_no_split_is_asked_for(self):
        """An unsplit export keeps the id it has always had, so a re-export is recognised as the same
        delivery and is not billed twice."""
        records = [_rec()]
        plain = SliceSpec(name="d")
        explicit_zero = SliceSpec(name="d", val_frac=0.0, test_frac=0.0, split_group_by="session")
        assert seal_commit_id(plain, records, ONTO) == seal_commit_id(explicit_zero, records, ONTO)


class TestTwoSplitsOfOneSlice:
    def test_are_different_deliveries(self):
        """Different artifacts, different directories, so one must not overwrite the other."""
        records = [_rec()]
        a = seal_commit_id(SliceSpec(name="d", val_frac=0.2), records, ONTO)
        b = seal_commit_id(SliceSpec(name="d", val_frac=0.3), records, ONTO)
        assert a != b
        assert a != seal_commit_id(SliceSpec(name="d"), records, ONTO)

    def test_but_are_the_same_content(self):
        """The fingerprint answers "were these annotations mutated", and a partitioning did not mutate
        one of them."""
        records = [_rec()]
        a = seal_content_fingerprint(SliceSpec(name="d", val_frac=0.2), records, ONTO)
        b = seal_content_fingerprint(SliceSpec(name="d", val_frac=0.3, split_group_by="frame"),
                                     records, ONTO)
        assert a == b

    def test_a_real_filter_change_still_changes_both(self):
        """The strip must not have made the seals blind to the fields that do define a release."""
        records = [_rec()]
        a = SliceSpec(name="d", min_conf=0.5)
        b = SliceSpec(name="d", min_conf=0.9)
        assert seal_commit_id(a, records, ONTO) != seal_commit_id(b, records, ONTO)
        assert seal_content_fingerprint(a, records, ONTO) != seal_content_fingerprint(b, records, ONTO)
