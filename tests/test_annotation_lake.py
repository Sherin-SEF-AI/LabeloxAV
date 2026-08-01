"""The annotation plane, published where the customer's own query engine can reach it.

Every "can we get a custom report" ask is the same ask underneath: somebody wants to join the labels against
something we have never heard of. Answering with a BI feature means guessing which joins matter and guessing
wrong a dozen times; publishing the tables answers all of them at once.

What these tests protect is the part that would break a customer's dashboard silently. The schema is written
out rather than inferred, so a snapshot with no rejections in it cannot change the column types and break a
query that worked yesterday. And a snapshot replaces rather than appends, because a reader should not have
to deduplicate by hand to find out what is true now.
"""

import pyarrow as pa
import pytest

from services.datasets.lake import NAMESPACE, TABLES, _schemas


def test_the_three_tables_match_the_three_planes_the_system_keeps_apart():
    assert set(TABLES) == {"labels", "provenance", "quality"}
    assert NAMESPACE == "labelox"


def test_schemas_are_declared_not_inferred():
    """An inferred schema changes shape when a column happens to be all-null in one snapshot, and a query
    breaking because yesterday's export had no rejections is exactly what this must not do."""
    s = _schemas()
    assert set(s) == set(TABLES)
    for name, schema in s.items():
        assert isinstance(schema, pa.Schema), name
        assert len(schema) > 0


def test_labels_carry_enough_context_to_join_without_us():
    """The point is a join nobody built a feature for, so the grain has to include the dimensions people
    actually group by: where, which vehicle, which camera, when."""
    f = set(_schemas()["labels"].names)
    assert {"object_id", "frame_id", "session_id", "class_name", "state", "source"} <= f
    assert {"city", "vehicle_id", "cam_id", "ts_ns"} <= f
    assert {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"} <= f, \
        "a bbox stored as a list cannot be aggregated in SQL without unnesting it first"


def test_provenance_carries_the_decision_not_just_the_decider():
    """Who changed what, from what, to what: a review table that records only the actor cannot answer the
    question anybody asks of it."""
    f = set(_schemas()["provenance"].names)
    assert {"reviewer", "user_id", "action", "ts_ns", "time_spent_ms"} <= f
    assert {"before_state", "after_state", "before_class_id", "after_class_id"} <= f


def test_bbox_columns_are_scalars_so_sql_can_aggregate_them():
    schema = _schemas()["labels"]
    for col in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
        assert schema.field(col).type == pa.float64()


def test_flags_stay_a_list_because_an_object_can_carry_several():
    assert _schemas()["quality"].field("flags").type == pa.list_(pa.string())


def test_every_declared_schema_accepts_an_empty_snapshot():
    """A first publish, or a corpus with no QA rows yet, must produce a valid empty table rather than an
    error. The live corpus has zero annotation_quality rows, so this is the real case, not a hypothetical.
    """
    for name, schema in _schemas().items():
        t = pa.Table.from_pylist([], schema=schema)
        assert t.num_rows == 0 and t.schema == schema, name


def test_a_row_of_each_shape_round_trips_through_its_schema():
    s = _schemas()
    labels = pa.Table.from_pylist([{
        "object_id": "o", "frame_id": "f", "session_id": "s", "track_id": None,
        "class_id": 3, "class_name": "bus", "state": "accepted", "source": "human", "conf": 0.9,
        "bbox_x1": 1.0, "bbox_y1": 2.0, "bbox_x2": 3.0, "bbox_y2": 4.0,
        "city": "BLR", "vehicle_id": "DASHCAM-01", "cam_id": "cam_f", "ts_ns": 1, "version": 2,
    }], schema=s["labels"])
    assert labels.num_rows == 1
    assert labels.column("class_name").to_pylist() == ["bus"]

    # A null track_id must survive, since most objects have none and a schema that rejected it would drop
    # the majority of the corpus.
    assert labels.column("track_id").to_pylist() == [None]

    quality = pa.Table.from_pylist([{
        "object_id": "o", "quality": 0.8, "agreement": None, "audit_verdict": None,
        "flags": ["tiny_box", "class_conflict"],
    }], schema=s["quality"])
    assert quality.column("flags").to_pylist() == [["tiny_box", "class_conflict"]]


def test_scan_refuses_a_table_that_is_not_part_of_the_lake():
    from services.datasets.lake import scan

    with pytest.raises(ValueError, match="unknown lake table"):
        scan("customers")
