"""The VLM as a multiple-choice specialist: called only where the cheap specialists could not settle it.

The judge as it stands (MachineVerdict, "is this a <class>?") fixed nothing on the human-ruling benchmark and
broke 4.7% of right labels, at 5.5% coverage; 221 of its 613 verdicts were "unsure", and it calls motorcycles
"sweeper" and cattle "elephant". Two changes make it worth asking:

- **Multiple choice over the dispute.** It picks between the machine's class, the class memory and the probe
  agreed on, their confusion family, and "background" (the ontology's "not an object"), with the answer
  constrained to that list. A yes/no question about one class lets it agree with anything; a choice
  between the two names actually in dispute is the question a reviewer would be asked.
- **Self-consistency.** Two votes with different crops and temperatures (the verifier's vote plans). Only a
  unanimous answer counts.

It is only consulted on objects the agent would otherwise just ask about, so the budget is the size of the
"ask" tier, not of the session. Whether a unanimous VLM answer may turn a question into a fix is decided by
the benchmark (`agent_vlm` stage), not assumed.
"""

from __future__ import annotations

import numpy as np

from core.logging import get_logger
from services.labelagent.truth import NOT_OBJECT, NOT_OBJECT_CLASS_NAMES

log = get_logger("labelagent.vlm")

NOT_OBJECT_NAME = "background"


def choices(candidates: list[int], onto) -> list[str]:
    """Ontology names for the shortlist, "not an object" as `background`."""
    out = []
    for c in candidates:
        if c == NOT_OBJECT:
            if onto.has_name(NOT_OBJECT_NAME):
                out.append(NOT_OBJECT_NAME)
            continue
        try:
            out.append(onto.by_id(c).name)
        except KeyError:
            continue
    return list(dict.fromkeys(out))


def vote(verifier, image_bgr: np.ndarray, bbox: tuple, candidates: list[int], onto,
         votes: int = 2) -> tuple[int | None, float]:
    """(label, agreement) from the VLM over `candidates`; (None, 0) when it gave no usable answer."""
    from services.autolabel.reasoner.adjudicate import _verify_restricted

    names = choices(candidates, onto)
    if len(names) < 2:
        return None, 0.0
    r = _verify_restricted(verifier, image_bgr, bbox, names, votes)
    if not r.class_name:
        return None, 0.0
    if r.class_name in NOT_OBJECT_CLASS_NAMES:
        return NOT_OBJECT, float(r.agreement or 0.0)
    return onto.by_name(r.class_name).id, float(r.agreement or 0.0)


def unload(settings=None) -> None:
    """Ask Ollama to drop the VLM from memory now rather than after its idle timeout: the box froze once with
    the VLM resident alongside the auto-label models."""
    import requests

    from core.config import get_settings

    cfg = (settings or get_settings()).models.vlm
    try:
        requests.post(f"{cfg.ollama_url}/api/generate", json={"model": cfg.ollama_tag, "keep_alive": 0},
                      timeout=10)
    except Exception as exc:  # noqa: BLE001 - the model unloads on its own timeout anyway
        log.warning("labelagent.vlm_unload_failed", error=str(exc))
