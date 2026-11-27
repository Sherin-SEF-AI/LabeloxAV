"""Path D's model-free logic: mapping a Grounding DINO label back to its phrase, and the per-class NMS.

The models themselves are exercised by scripts/detector_benchmark.py on gold labels. What is pinned here is
the part that silently mislabels when wrong: a returned text span mapped to the wrong phrase is a box with a
confident, wrong class, which is exactly the failure Path D exists to reduce.
"""

from __future__ import annotations

import pytest

from services.autolabel.paths.path_d_grounded import GroundedPath


@pytest.fixture
def path():
    return GroundedPath("owlv2")


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError):
        GroundedPath("yolo")


def test_every_phrase_resolves_to_one_class(path):
    assert len(path._phrases) == len(path._phrase_classes) > 10
    assert len(set(path._phrases)) == len(path._phrases)


def test_the_longest_contained_phrase_wins(path):
    path._phrases = ["bus", "school bus", "bus shelter"]
    idx = [0, 1, 2]
    assert path._phrases[path._match("school bus", idx)] == "school bus"
    assert path._phrases[path._match("a bus shelter", idx)] == "bus shelter"
    # a partial span maps to the shortest phrase containing it
    assert path._phrases[path._match("shelter", idx)] == "bus shelter"
    assert path._match("", idx) is None
    assert path._match("elephant", idx) is None


def test_duplicates_of_one_object_collapse_but_different_classes_do_not(path):
    a, b = 0, 1
    path._phrases = ["truck", "lorry"]
    path._phrase_classes = [path._phrase_classes[a], path._phrase_classes[a]]
    raw = [((0, 0, 100, 100), 0.9, 0), ((2, 2, 101, 99), 0.7, 1), ((300, 300, 400, 400), 0.6, 0)]
    out = path._finish(raw)
    assert len(out) == 2 and out[0].conf == 0.9          # the synonym hit on the same box is suppressed
    path._phrase_classes = [GroundedPath("owlv2")._phrase_classes[a], GroundedPath("owlv2")._phrase_classes[b]]
    assert len(path._finish(raw[:2])) == 2                # two classes on one box are both kept for fusion
    assert all(d.path == "path_d_grounded" and d.mask is None for d in out)
