"""Which state a review may write, given who is asking.

The two-step workflow this system is built around is that an annotator's work goes to a reviewer before it
counts. `web/lib/user.ts` encodes it as `acceptState(role)`: an annotator pressing accept means `submitted`,
which is the band the triage page's QA queue reads, and a reviewer pressing accept means `accepted`.

The server never checked. `POST /objects/{id}/review` wrote `payload.state or _ACTION_STATE[action]`, so the
state came from the request body and the caller's role was not consulted at all. The whole QA step was
therefore advisory: any annotator, and anything holding an annotator's token, could write `accepted` and
skip it. The frame editor did exactly that by accident, hardcoding `accepted` on the A key while its own
save path used `acceptState`, so the same person's two ways of approving the same box landed in different
queues.

Fixing the client alone would have left the hole, because the client is not the thing that decides. The rule
lives here, the router calls it, and both the single and the bulk review path go through it.

The ceiling is a clamp rather than a refusal. An annotator asking for `accepted` is not attacking anything;
it is a UI that has not been told about the workflow, and answering 403 would break clients over a state
name when the correct answer, `submitted`, is well defined. A request for a state that is not a real state
is a different matter and is refused, because there is no correct answer to guess.
"""

from __future__ import annotations

from services.api.deps import role_rank

# The canonical set, from `ck_object_state` in migration 0020.
OBJECT_STATES = frozenset({"review", "auto_accept", "accepted", "rejected", "annotate", "submitted"})

# What each verb means before the role is taken into account.
ACTION_STATE = {"confirm": "accepted", "accept": "accepted", "reject": "rejected"}

# Approving something so it counts as ground truth is a reviewer act. An annotator's approval is a
# submission, which is the same decision addressed to a different queue.
_ANNOTATOR_CEILING = {"accepted": "submitted", "auto_accept": "submitted"}

REVIEWER_RANK = role_rank("reviewer")


class ReviewStateError(ValueError):
    """A state that is not a state. Distinct from a state the caller may not write, which is clamped."""


def state_for(action: str | None, requested: str | None, role: str | None, current: str | None) -> str | None:
    """The state a review should write, or None to leave the object's state alone.

    `role` is None when authentication is disabled or the request carries no user. That is not treated as an
    annotator: a deployment with auth off has no roles at all, and clamping there would change behaviour for
    every test and every single-user install based on the absence of a concept rather than on a decision.
    """
    state = requested or ACTION_STATE.get(action or "", current)
    if state is None:
        return None
    if state not in OBJECT_STATES:
        raise ReviewStateError(f"unknown object state {state!r}; expected one of {sorted(OBJECT_STATES)}")
    if role is None or role_rank(role) >= REVIEWER_RANK:
        return state
    return _ANNOTATOR_CEILING.get(state, state)


def was_clamped(action: str | None, requested: str | None, role: str | None, current: str | None) -> bool:
    """Whether the answer differs from what was asked for, so a caller can be told rather than surprised."""
    asked = requested or ACTION_STATE.get(action or "", current)
    return asked is not None and asked in OBJECT_STATES and state_for(action, requested, role, current) != asked
