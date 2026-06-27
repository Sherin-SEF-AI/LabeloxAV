"""VLM auto-QA + auto-attributes: a model-as-reviewer pass over already-labeled objects. Re-runs the
Qwen-VL verifier on each object's crop; when it disagrees with the current class ACROSS superclasses it
routes the object to the QA queue (state=submitted) with the VLM's suggestion recorded, and it fills the
ontology's typed attributes the detector never set. Closes the active-learning loop: the model flags its
own likely mistakes and pre-fills attributes so humans only adjudicate.

    python -m services.intelligence.vlm_qa --session <uuid> --limit 40
"""

from __future__ import annotations

import asyncio
import time
from uuid import UUID

import click
import cv2
import numpy as np
from sqlalchemy import select

from core.logging import get_logger, setup_logging
from core.config import get_settings
from core.storage import get_object_store
from db.models import Frame, Object, Review
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology

log = get_logger("vlm_qa")

# Objects worth a second opinion: machine labels that a human has not already locked in.
_QA_STATES = ("review", "annotate", "accepted", "submitted")


async def vlm_qa_session(
    session_id: UUID, limit: int = 40, fill_attrs: bool = True, flag_disagreements: bool = True
) -> dict:
    from services.autolabel.paths.path_c_qwen3vl import VlmVerifier, make_vlm_client

    onto = get_ontology()
    verifier = VlmVerifier(make_vlm_client(), onto)
    store = get_object_store()
    maker = get_sessionmaker()

    checked = flagged = filled = agreed = 0
    async with maker() as db:
        rows = (
            await db.execute(
                select(Object, Frame.img_uri)
                .join(Frame, Frame.frame_id == Object.frame_id)
                .where(Frame.session_id == session_id, Object.state.in_(_QA_STATES))
                .order_by(Object.conf.asc())  # least-confident first = highest QA value
                .limit(limit)
            )
        ).all()

        by_frame: dict[str, list] = {}
        for obj, uri in rows:
            by_frame.setdefault(uri, []).append(obj)

        for uri, objs in by_frame.items():
            try:
                img = cv2.imdecode(np.frombuffer(store.get_bytes(uri), np.uint8), cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001
                continue
            if img is None:
                continue
            for obj in objs:
                cur_name = onto.by_id(obj.class_id).name
                res = verifier.verify_object(img, tuple(obj.bbox), obj.class_id)
                checked += 1

                # auto-attributes: VlmVerifier already validated res.attrs against the ontology.
                if fill_attrs and res.attrs:
                    cur = dict(obj.attrs or {})
                    added = {k: v for k, v in res.attrs.items() if k not in cur}
                    if added:
                        cur.update(added)
                        obj.attrs = cur
                        filled += 1

                # auto-QA: flag cross-superclass disagreements into the QA queue.
                if res.class_name and onto.has_name(res.class_name):
                    if res.class_name == cur_name:
                        agreed += 1
                    elif (flag_disagreements and res.confident
                          and onto.by_id(obj.class_id).l1 != onto.by_name(res.class_name).l1):
                        prov = dict(obj.provenance or {})
                        prov["vlm_qa"] = {"suggested": res.class_name, "caption": res.caption,
                                          "agreement": round(res.agreement, 2)}
                        obj.provenance = prov
                        before_state = obj.state
                        obj.state = "submitted"
                        db.add(Review(object_id=obj.object_id, reviewer="vlm-qa", action="flag",
                                      before={"class": cur_name, "state": before_state},
                                      after={"class_suggested": res.class_name}, ts_ns=time.time_ns()))
                        flagged += 1
            await db.commit()

    out = {"checked": checked, "flagged": flagged, "attrs_filled": filled, "agreed": agreed,
           "model": get_settings().models.vlm.ollama_tag}
    log.info("vlm_qa.done", **out)
    return out


@click.command()
@click.option("--session", "session_id", required=True)
@click.option("--limit", type=int, default=40)
@click.option("--no-attrs", is_flag=True, default=False)
def main(session_id, limit, no_attrs) -> None:
    setup_logging(get_settings().log_level)
    click.echo(asyncio.run(vlm_qa_session(UUID(session_id), limit, fill_attrs=not no_attrs)))


if __name__ == "__main__":
    main()
