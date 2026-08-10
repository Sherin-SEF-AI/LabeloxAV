"""A parsed class name must not be allowed to destroy the query it came from.

`parse_query` pulls ontology class names out of a phrase, and the phrase is ordinary English, so a word can
be both at once. "a cow on the road" parses `road`, which is a surface class annotated as an object on 1 of
34,977 frames in the live corpus. Treating that as a hard filter collapsed the search to that single frame
while the actual subject was dropped, because the ontology files a cow as `cattle`.

The parse is not the problem. Using every match as a hard filter is. A filter that leaves fewer frames than
the caller asked for results is removing more than it selects, and the parsed terms are already inside the
text being embedded, so they still steer the ranking.
"""

from services.intelligence.search.query import parse_query, should_filter


def test_an_everyday_word_that_is_also_a_class_is_parsed_as_one():
    """The precondition. Without this the narrowing bug cannot arise and the fix is unmotivated."""
    _scene, classes = parse_query("a cow on the road")
    assert "road" in classes, "'road' is an ontology class as well as a preposition-phrase word"
    assert "cattle" not in classes, "and the subject is not matched, because a cow is filed as cattle"


def test_a_phrase_with_no_ontology_term_parses_to_nothing():
    scene, classes = parse_query("vehicle going the wrong way")
    assert classes == [] and scene == {}


def test_scene_words_still_parse():
    scene, _classes = parse_query("heavy rain at night")
    assert scene.get("weather") == "rain"
    assert scene.get("time_of_day") == "night"


def test_a_term_matching_too_few_frames_ranks_instead_of_filtering():
    """The defect, at the live corpus's own numbers: one candidate against a request for 24."""
    assert should_filter(1, 24, narrowed=True) is False
    assert should_filter(23, 24, narrowed=True) is False


def test_a_term_matching_plenty_of_frames_still_filters():
    """The fix must not disable filtering, only stop it destroying the result set."""
    assert should_filter(24, 24, narrowed=True) is True
    assert should_filter(12453, 24, narrowed=True) is True   # "a bus", 35.6% of the live corpus


def test_an_unparsed_query_is_never_filtered():
    # Nothing matched, so there is nothing to filter on and the whole corpus is the candidate set.
    assert should_filter(0, 24, narrowed=False) is False
    assert should_filter(9999, 24, narrowed=False) is False


def test_the_threshold_follows_the_request_size():
    # A caller asking for 3 results is satisfied by a narrower filter than one asking for 100, so the rule
    # is expressed against k rather than a constant nobody can justify.
    assert should_filter(5, 3, narrowed=True) is True
    assert should_filter(5, 100, narrowed=True) is False


def test_the_behaviour_this_replaced():
    """The old rule: any match filtered, however little it left."""
    def old_applied(candidate_count: int, narrowed: bool) -> bool:
        return narrowed  # and then `if not candidate_ids: return 0 results`

    # One frame carrying an incidental class: the old rule filtered and returned almost nothing.
    assert old_applied(1, True) is True
    assert should_filter(1, 24, narrowed=True) is False
