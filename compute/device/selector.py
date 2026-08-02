"""On-rig frame selection: deciding what is worth the uplink, one frame at a time.

`services/ingest/extract_smart.py` already makes this decision well, and makes it in the wrong place. It
runs server-side and post-ingest, which means every frame has already been driven home, stored and paid for
before anything asks whether it was worth keeping. Moving the decision onto the vehicle is the largest
scale lever this system has: a rig that uploads the interesting 8% of its day costs a twelfth as much to
operate and fills the corpus with a twelfth as much redundancy.

**The port is not a copy, because the constraint is different.** The server sees a whole session at once, so
it can rank every frame by novelty and take the top N against a budget. A rig sees frame 400 with no idea
whether frame 4000 will be more interesting, and cannot hold the day in memory to find out. Global ranking
is unavailable in a stream, and pretending otherwise is how a device agent ends up either filling its disk
by 9am or throwing away the afternoon.

So the budget is met by a running quantile instead, over the right quantity. The selector tracks how much
the picture changes from one frame to the next, and keeps a frame when that step change lands in the top
`budget` fraction of the recent window. Over a drive this spends the budget on the moments the scene is
actually changing, which is what the batch version gets for free from sorting, and it costs one bounded
deque.

**The quantile is taken over step change, not over difference from the last kept frame, and that distinction
is the whole thing.** Difference-from-last-kept grows the longer it has been since a keep, so a window of
those values is a description of the gaps the selector has been leaving rather than of the road. Ranking
against it is self-reinforcing: keep rarely, observe large differences, set a high bar, keep more rarely. It
undershot a 10% budget by a factor of eight before this was separated out. Step change is a property of the
scene and does not move when the policy does.

**Quantiles rather than an absolute threshold, because novelty has no universal scale.** The first version
of this used absolute cut-offs tuned by eye, and on the cheap descriptor it kept 0.5% against a 10% budget:
two tiled histograms of the same road a second apart sit at cosine 0.999, so every frame fell under a floor
that had been chosen while thinking about DINOv3. Nothing failed and nothing warned; the fleet would simply
have uploaded almost nothing. A quantile of the observed distribution is the same policy on any descriptor,
which also means a rig can be upgraded from histograms to DINOv3 without anybody retuning it.

**Rare classes bypass the controller entirely.** A cow on a motorway is the frame the whole pipeline exists
to find, and it must not be dropped because the preceding ten minutes of empty road already spent the hour's
budget. That asymmetry is deliberate: the cost of keeping a boring frame is a few hundred kilobytes, and the
cost of dropping the one interesting frame of the week is the week.

Deliberately free of the server stack. No database, no object store, no settings singleton, and numpy is the
only import beyond the standard library, because this has to run on a board where installing the repo is not
an option.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SelectorConfig:
    """Everything the on-rig decision needs, with no reference to server config.

    `target_keep_frac` is the uplink budget expressed the way an operator thinks about it: what fraction of
    what the camera sees should go home. Bandwidth per hour is the real constraint, but it varies with image
    size and compression, and a fraction is the part a fleet operator can reason about.
    """
    target_keep_frac: float = 0.08
    # How many recent frames the quantile is taken over. Long enough that the estimate is stable, short
    # enough that it tracks the drive: a motorway hour should not still be setting the bar in a market
    # street. About four minutes at 3fps.
    window: int = 720
    # Frames of history before the quantile is trusted. Until then only the unconditional rules fire, so a
    # device does not spend its first minute keeping everything or nothing on three samples of evidence.
    warmup: int = 60
    # Two frames closer than this in cosine terms are the same picture, and no budget argument makes one of
    # them worth uploading. Deliberately tiny: this is a de-duplicator, not the novelty policy, which is
    # what the quantile is for.
    identical_novelty: float = 1e-4
    # Frames since the last keep after which one is taken anyway, so a motionless hour still leaves a trace.
    max_gap_frames: int = 300
    # And a minimum, so a burst of camera shake cannot empty the budget in a second.
    min_gap_frames: int = 3
    # Objects on a frame past which it counts as busy and is kept: a junction with fifteen road users is
    # worth having whatever its pixel novelty says.
    dense_objects: int = 8


@dataclass
class Decision:
    keep: bool
    reason: str
    novelty: float
    threshold: float


@dataclass
class StreamingSelector:
    """Keep or drop, frame by frame, converging on a budget without seeing the future.

    Not thread-safe and not meant to be: one instance per camera, driven by that camera's frame loop.
    """

    cfg: SelectorConfig = field(default_factory=SelectorConfig)
    _last_kept_vec: np.ndarray | None = field(default=None, repr=False)
    _prev_vec: np.ndarray | None = field(default=None, repr=False)
    _recent: deque = field(default=None, repr=False)
    _seen: int = 0
    _kept: int = 0
    _since_keep: int = 0
    _reasons: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.cfg.window)

    def observe(self, vec: np.ndarray, *, rare: bool = False, object_count: int = 0) -> Decision:
        """Decide on one frame. `vec` is any unit-norm descriptor; see embed.py for what a rig can afford.

        Order matters here and is the policy. Rare first, because it must not be reachable by any budget
        argument. Then the minimum gap, because a shake burst that passes the novelty test is still not
        worth six near-identical frames. Then the ordinary tests.
        """
        self._seen += 1
        self._since_keep += 1
        v = _unit(vec)

        # Two different measures, for two different jobs.
        #
        # `step` is how much the picture moved since the previous frame. It is a property of the scene, so a
        # window of it describes the road rather than the policy, which is what makes it safe to rank
        # against.
        #
        # `novelty` is how far this frame is from the one last kept. It is what "do we already have this"
        # means, and it is used for the identical check and reported, but never for ranking: it grows with
        # the gap since the last keep, so ranking against it would be ranking against the selector's own
        # recent behaviour.
        step = 0.0 if self._prev_vec is None else round(1.0 - float(v @ self._prev_vec), 6)
        novelty = 1.0 if self._last_kept_vec is None else round(1.0 - float(v @ self._last_kept_vec), 6)
        if self._prev_vec is not None:
            self._recent.append(step)
        self._prev_vec = v

        decision = self._decide(novelty, step, rare=rare, object_count=object_count)
        if decision.keep:
            self._kept += 1
            self._since_keep = 0
            self._last_kept_vec = v
        self._reasons[decision.reason] = self._reasons.get(decision.reason, 0) + 1
        return decision

    @property
    def threshold(self) -> float:
        """The step change a frame has to beat: the (1 - budget) quantile of the recent window.

        Returns 0.0 during warmup, which lets the ordinary rules through while the window fills. The
        alternative, a guessed starting value, is the absolute threshold this design exists to avoid.
        """
        if len(self._recent) < self.cfg.warmup:
            return 0.0
        q = 100.0 * (1.0 - self.cfg.target_keep_frac)
        return float(np.percentile(np.fromiter(self._recent, dtype=np.float32), q))

    def _decide(self, novelty: float, step: float, *, rare: bool, object_count: int) -> Decision:
        c = self.cfg
        thr = self.threshold
        if self._last_kept_vec is None:
            return Decision(True, "first frame", novelty, thr)
        if rare:
            # Never gated by the budget. See the module docstring: the cost of being wrong in this direction
            # is a few hundred kilobytes, and in the other direction it is the frame the fleet exists for.
            return Decision(True, "rare class", novelty, thr)
        if self._since_keep < c.min_gap_frames:
            return Decision(False, "too soon after the last keep", novelty, thr)
        if object_count >= c.dense_objects:
            return Decision(True, "busy scene", novelty, thr)
        if self._since_keep >= c.max_gap_frames:
            # Ahead of the identical check on purpose. A parked vehicle and a frozen sensor both produce
            # identical frames forever, and from the manifest alone they are the same thing. A periodic
            # frame costs a few hundred kilobytes an hour and is the only evidence that tells them apart.
            return Decision(True, "heartbeat: nothing kept for too long", novelty, thr)
        if novelty <= c.identical_novelty:
            return Decision(False, "identical to the last kept frame", novelty, thr)
        if len(self._recent) < c.warmup:
            # Not enough history to know what change looks like on this descriptor and this route. The gap
            # rules are still running, so the device is not idle, it is just not yet ranking.
            return Decision(False, "warming up", novelty, thr)
        if step >= thr:
            return Decision(True, "scene changing faster than the recent window", novelty, thr)
        return Decision(False, "scene changing less than the recent window", novelty, thr)

    @property
    def stats(self) -> dict:
        return {
            "seen": self._seen, "kept": self._kept,
            "keep_frac": round(self._kept / self._seen, 4) if self._seen else 0.0,
            "target_keep_frac": self.cfg.target_keep_frac,
            "threshold": round(self.threshold, 6),
            "window_filled": len(self._recent),
            # Reported per reason so a drive that kept too much can be explained rather than guessed at.
            # "busy scene" dominating means the density bar is wrong for this route; "heartbeat" dominating
            # means the camera was pointed at a wall.
            "reasons": dict(sorted(self._reasons.items(), key=lambda kv: -kv[1])),
        }


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v if n < 1e-9 else v / n
