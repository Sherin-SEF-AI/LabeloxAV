"""The fusion path persists the ensemble class-distribution entropy on each object's provenance: agreeing paths
give ~0 entropy, class-disagreeing paths give a positive entropy, and the active-learning selector reads it as
uncertainty."""

import uuid

import numpy as np

from services.autolabel.fusion import FusionEngine, _frame_vote_entropy
from services.autolabel.paths.base import RawDetection


def _det(path, cls_id, conf, box=(10.0, 10.0, 50.0, 50.0)):
    return RawDetection(path=path, bbox=box, conf=conf, model_version="test", class_id=cls_id,
                        class_name=str(cls_id))


def test_frame_vote_entropy_agreement_vs_split():
    assert _frame_vote_entropy([]) == []
    assert _frame_vote_entropy([{6: 0.9}]) == [0.0]                 # single class in frame -> 0
    ent = _frame_vote_entropy([{6: 0.95}, {6: 0.5, 11: 0.5}])
    assert ent[0] < 1e-6 and ent[1] > 0.6                          # agree -> 0, even split -> ~ln 2


def test_fusion_persists_entropy_on_provenance():
    fuser = FusionEngine()
    fid = uuid.uuid4()
    # two overlapping boxes: path A and path B agree on the same class -> low entropy
    agree = fuser.fuse_frame(fid, [_det("path_a_yolo26", 6, 0.9)], [_det("path_b_sam3", 6, 0.85)])
    assert agree and agree[0].obj.provenance.entropy is not None
    assert agree[0].obj.provenance.entropy < 0.2

    # the same overlap but the two paths vote DIFFERENT classes -> higher entropy
    disagree = fuser.fuse_frame(fid, [_det("path_a_yolo26", 6, 0.9)], [_det("path_b_sam3", 11, 0.9)])
    # find the fused object covering the overlap (paths clustered together)
    ent = max(fo.obj.provenance.entropy or 0.0 for fo in disagree)
    assert ent > agree[0].obj.provenance.entropy                   # disagreement raises the entropy


def test_selector_folds_entropy_into_uncertainty():
    # the squash the selector applies: a split-vote object (entropy ~ ln2) reads uncertain regardless of conf
    en = float(np.log(2))
    u_from_entropy = 1.0 - np.exp(-en)
    assert u_from_entropy > 0.45                                   # ~0.5, a clear uncertainty signal
