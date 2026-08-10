"""A dataset selected by a sentence, and refusing to guess what the sentence meant.

`labelox.load("night AND vru")` is meant to sit in a training config next to the learning rate and be read
by whoever inherits it a year later. That rules out a JSON predicate, which nobody writes from memory, and
it rules out anything forgiving: a term that quietly matched nothing would silently widen the dataset, and a
training set missing the class it was assembled for is the hardest kind of defect to notice.

So the compiler refuses unknown terms and always returns what it compiled to. These tests pin both, plus the
expansions that make the vocabulary worth having.
"""

import pytest

from services.datasets.query_lang import QueryError, compile_query, vocabulary


def test_a_scene_word_becomes_a_scene_axis():
    out = compile_query("night")
    assert out["predicate"] == {"time_of_day": ["night"]}
    assert out["terms"][0]["kind"] == "scene"


def test_terms_intersect():
    p = compile_query("rain AND highway")["predicate"]
    assert p == {"weather": ["rain"], "road_type": ["highway"]}


def test_two_values_on_one_axis_widen_that_axis_rather_than_cancelling():
    """"night AND dusk" is a request for both, not for the empty set: a clause holds a list."""
    assert compile_query("night AND dusk")["predicate"]["time_of_day"] == ["night", "dusk"]


def test_a_group_expands_through_the_ontology_not_a_hardcoded_list():
    """So a class added tomorrow joins its group without editing the query language."""
    out = compile_query("vru")
    names = out["predicate"]["class_names"]
    assert "pedestrian" in names and "rider" in names
    assert len(names) > 3
    assert out["terms"][0]["kind"] == "group"
    # Every expanded name is a real class, which is what makes this safe to widen.
    from services.autolabel.ontology import get_ontology
    onto = get_ontology()
    assert all(onto.has_name(n) for n in names)


def test_reviewed_means_ruled_on_not_accepted():
    """A distinction this corpus has been bitten by: 'reviewed' is a set of states, and it includes
    rejections, because an object somebody looked at and refused is still evidence."""
    assert set(compile_query("reviewed")["predicate"]["states"]) == {"accepted", "rejected"}
    assert set(compile_query("unreviewed")["predicate"]["states"]) == {"review", "annotate"}


def test_a_bare_class_name_works():
    assert compile_query("cattle")["predicate"]["class_names"] == ["cattle"]


def test_an_unknown_term_is_refused_rather_than_ignored():
    """The whole reason this is strict: 'unocluded' silently returning every night frame is worse than an
    error, because the dataset would look plausible and be wrong."""
    with pytest.raises(QueryError, match="unknown term"):
        compile_query("night AND unocluded")


def test_an_empty_query_is_refused():
    with pytest.raises(QueryError, match="empty query"):
        compile_query("   ")


def test_the_compiled_predicate_is_always_returned():
    """A dataset whose contents cannot be explained is one nobody can defend in a review."""
    out = compile_query("night AND cattle")
    assert out["query"] == "night AND cattle"
    assert out["predicate"] and out["terms"]


def test_the_vocabulary_lists_what_would_have_worked():
    v = vocabulary()
    assert "night" in v["scene"] and "vru" in v["group"] and "reviewed" in v["state"]
    assert "cattle" in v["class"]


def test_the_group_vocabulary_shows_one_term_per_thing_it_selects():
    """`four-wheeler` and `four_wheeler` both parse, because people type both. Listing them as two chips
    implies they select different things, which is a UI that lies about the vocabulary."""
    v = vocabulary()
    assert "four_wheeler" in v["group"]
    assert "four-wheeler" not in v["group"], "the hyphen form is an input alias, not a second group"
    # Both still resolve, which is the point of keeping the alias at all.
    assert compile_query("four-wheeler")["predicate"] == compile_query("four_wheeler")["predicate"]


def test_the_predicate_is_the_vocabulary_the_rest_of_the_system_already_evaluates():
    """One predicate shape, three ways in: this compiler, the SQL evaluator, and the pure matcher.

    If these drift, a query and a saved slice stop meaning the same thing.
    """
    from services.explore.query import frame_select

    pred = compile_query("night AND vru AND reviewed")["predicate"]
    # Compiles to real SQL without raising, which is the contract that matters.
    stmt = frame_select(pred)
    assert stmt is not None
    assert set(pred) <= {"weather", "time_of_day", "road_type", "density", "cities", "class_names",
                         "states", "sources", "min_conf", "max_conf", "tags", "frame_tags",
                         "session_id", "object_ids", "frame_ids"}
