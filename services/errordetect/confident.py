"""Confident learning over already-accepted labels (M4.1). cleanlab needs out-of-sample predicted
probabilities that can disagree with the given label; our richest local signal is the DINOv3 embedding, so
we fit a cross-validated classifier over the embeddings and let cleanlab flag the objects whose accepted
label disagrees with what their embedding neighbourhood predicts. This catches auto-accept mistakes and
human slips alike, and proposes the corrected class. Runs locally over existing embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, ObjectEmbedding
from services.autolabel.ontology import get_ontology

log = get_logger("ed_confident")
_ACCEPTED = ("accepted", "auto_accept")


async def detect_confident_learning(db: AsyncSession, session_id: str | None = None,
                                    min_per_class: int = 4, max_objects: int = 5000) -> list[dict]:
    onto = get_ontology()
    q = (select(Object.object_id, Object.class_id, ObjectEmbedding.dino_vec)
         .join(ObjectEmbedding, ObjectEmbedding.object_id == Object.object_id)
         .where(Object.state.in_(_ACCEPTED)))
    if session_id:
        q = q.join(Frame, Frame.frame_id == Object.frame_id).where(Frame.session_id == session_id)
    rows = (await db.execute(q.limit(max_objects))).all()
    if len(rows) < min_per_class * 2:
        return []

    oids = [r[0] for r in rows]
    y_raw = np.array([r[1] for r in rows])
    X = np.stack([np.asarray(r[2], dtype=np.float32) for r in rows])

    # keep only classes with enough samples for cross-validation, then densify labels to 0..K-1
    uniq, counts = np.unique(y_raw, return_counts=True)
    keep = set(int(c) for c, n in zip(uniq, counts, strict=False) if n >= min_per_class)
    if len(keep) < 2:
        return []
    mask = np.array([c in keep for c in y_raw])
    oids = [o for o, m in zip(oids, mask, strict=False) if m]
    y_raw, X = y_raw[mask], X[mask]
    classes = sorted(keep)
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[int(c)] for c in y_raw])

    cv = int(min(5, np.bincount(y).min()))
    if cv < 2:
        return []
    clf = LogisticRegression(max_iter=300, C=1.0)
    pred = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")

    from cleanlab.filter import find_label_issues

    issues = find_label_issues(labels=y, pred_probs=pred, return_indices_ranked_by="self_confidence")
    out = []
    for i in issues:
        given = int(y[i])
        proposed = int(pred[i].argmax())
        if proposed == given:
            continue
        prop_cid = classes[proposed]
        out.append({"object_id": str(oids[i]), "kind": "confident_learning",
                    "score": round(float(1.0 - pred[i][given]), 4),
                    "proposed_label": {"class_id": prop_cid, "class_name": onto.by_id(prop_cid).name},
                    "detail": {"given_class": onto.by_id(classes[given]).name,
                               "given_prob": round(float(pred[i][given]), 4),
                               "proposed_prob": round(float(pred[i][proposed]), 4)}})
    log.info("ed.confident", flagged=len(out), pool=len(oids))
    return out
