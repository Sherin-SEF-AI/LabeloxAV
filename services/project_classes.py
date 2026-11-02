"""Which ontology classes a project labels.

The ontology stays one list with global ids (services/autolabel/ontology.py); a project narrows it to the
classes it cares about. NULL means the whole ontology, which is every project until somebody chooses.

Two templates cover most teams:
  india    every class, including the Indian long tail (autorickshaws, cycle rickshaws, handcarts, cattle).
           Stored as NULL, so classes added to the ontology later are in it automatically.
  generic  the road users and infrastructure found anywhere: every class not flagged `india`, plus the
           team's own custom classes and the ignore regions, which every project needs.
Anything else is a custom list chosen class by class.

Where the set applies:
  - the editor's class pickers offer only these classes (objects already stored keep theirs);
  - auto-labelling writes no object of another class, and reports how many it left out.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Frame, LabelProject
from db.models import Session as DbSession

INDIA, GENERIC, CUSTOM = "india", "generic", "custom"
TEMPLATES = (INDIA, GENERIC)


def _always(c) -> bool:
    """Classes no project can do without: ignore regions, and classes the team added itself."""
    return c.l0 == "ignore" or c.l1 == "custom"


def template_ids(name: str) -> list[int] | None:
    """The class ids of a template. None for india: the whole ontology, now and later."""
    from services.autolabel.ontology import get_ontology

    if name == INDIA:
        return None
    if name == GENERIC:
        return sorted(c.id for c in get_ontology().classes if not c.india or _always(c))
    raise ValueError(f"unknown template {name!r}; one of {list(TEMPLATES)}")


def describe(class_ids: list[int] | None) -> dict:
    """What a stored set is, for display: which template it equals (or custom) and how many classes."""
    from services.autolabel.ontology import get_ontology

    total = len(get_ontology().classes)
    if class_ids is None:
        return {"class_set": INDIA, "class_count": total, "of": total}
    ids = sorted(set(class_ids))
    name = GENERIC if ids == template_ids(GENERIC) else CUSTOM
    return {"class_set": name, "class_count": len(ids), "of": total}


def validate(class_ids: list[int]) -> list[int]:
    """A custom set, checked against the ontology. Ignore regions and custom classes are always added."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    known = {c.id for c in onto.classes}
    unknown = sorted(set(class_ids) - known)
    if unknown:
        raise ValueError(f"not in the ontology: {unknown[:10]}")
    ids = set(class_ids) | {c.id for c in onto.classes if _always(c)}
    if not any(onto.is_thing(i) for i in ids):
        raise ValueError("a project needs at least one object class to label")
    return sorted(ids)


def keep_in_set(fused: list, class_set: frozenset[int] | None) -> tuple[list, int]:
    """Auto-labelling's filter: the fused detections whose class the project labels, and how many were left
    out. Anything with `.obj.class_id` (a FusedObject); None keeps everything."""
    if class_set is None:
        return fused, 0
    kept = [g for g in fused if g.obj.class_id in class_set]
    return kept, len(fused) - len(kept)


async def project_class_ids(db: AsyncSession, project_id: uuid.UUID | None) -> frozenset[int] | None:
    if project_id is None:
        return None
    raw = (await db.execute(select(LabelProject.class_ids)
                            .where(LabelProject.project_id == project_id))).scalar_one_or_none()
    if raw is None:
        return None
    # Joined at read time, not only when the set was saved: a class the team adds next week is in every set.
    from services.autolabel.ontology import get_ontology

    return frozenset(int(i) for i in raw) | {c.id for c in get_ontology().classes if _always(c)}


async def session_class_ids(db: AsyncSession, session_id: uuid.UUID | str) -> frozenset[int] | None:
    """The class set for a session's project, or None for the whole ontology."""
    try:
        sid = uuid.UUID(str(session_id))
    except ValueError:
        return None
    pid = (await db.execute(select(DbSession.project_id)
                            .where(DbSession.session_id == sid))).scalar_one_or_none()
    return await project_class_ids(db, pid)


async def frame_class_ids(db: AsyncSession, frame_id: uuid.UUID | str) -> frozenset[int] | None:
    try:
        fid = uuid.UUID(str(frame_id))
    except ValueError:
        return None
    pid = (await db.execute(select(DbSession.project_id).join(Frame, Frame.session_id == DbSession.session_id)
                            .where(Frame.frame_id == fid))).scalar_one_or_none()
    return await project_class_ids(db, pid)


__all__ = ["CUSTOM", "GENERIC", "INDIA", "TEMPLATES", "describe", "frame_class_ids", "keep_in_set", "project_class_ids",
           "session_class_ids", "template_ids", "validate"]
