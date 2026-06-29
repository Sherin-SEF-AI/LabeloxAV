"""M-Q.2: Safe-mIoU must score class-versus-background, not discard it. The spec calls out 'pedestrian
versus background' as the unsafe case, and a weak detector that misses everything (class -> background) was
reading as undefined because the background row/col was dropped. These prove a missed or invented VRU sinks
the score, a benign miss costs less, and the score is now defined when all the mass is class-vs-background."""

from __future__ import annotations

from services.autolabel.ontology import get_ontology
from services.training.safe_miou import BACKGROUND_ID, affinity_cost, safe_miou


def test_affinity_background_vru_is_maximal_benign_is_moderate():
    onto = get_ontology()
    assert affinity_cost(onto, onto.by_name("pedestrian").id, BACKGROUND_ID) == 1.0
    assert affinity_cost(onto, BACKGROUND_ID, onto.by_name("rider").id) == 1.0  # symmetric
    assert affinity_cost(onto, onto.by_name("sedan").id, BACKGROUND_ID) == 0.5


def test_safe_miou_is_defined_for_all_background_misses():
    onto = get_ontology()
    ped, sedan = onto.by_name("pedestrian").id, onto.by_name("sedan").id
    ids = [ped, sedan, BACKGROUND_ID]
    # 10 pedestrians missed (-> background), 5 sedans correct + 2 missed, 3 background -> pedestrian (FP)
    matrix = [[0, 0, 10], [0, 5, 2], [3, 0, 0]]
    score = safe_miou(matrix, ids, onto, safety_weight=2.0)
    assert score is not None, "background confusions must make the score defined, not None"
    assert score < 0.6, "missed and invented VRUs must sink the score"


def test_missed_vru_scores_lower_than_missed_benign():
    onto = get_ontology()
    ped, sedan = onto.by_name("pedestrian").id, onto.by_name("sedan").id
    ids = [ped, sedan, BACKGROUND_ID]
    vru_missed = [[0, 0, 10], [0, 10, 0], [0, 0, 0]]     # 10 pedestrians missed, sedans all correct
    benign_missed = [[10, 0, 0], [0, 0, 10], [0, 0, 0]]  # 10 sedans missed, pedestrians all correct
    assert safe_miou(vru_missed, ids, onto) < safe_miou(benign_missed, ids, onto)
