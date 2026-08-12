"""How much of a long job to report, and how often to write it down.

An autolabel job set its row to 0.05 when it started and 1.0 when it finished, and nothing in between. That
was invisible while the only reader was a table of job rows, because a table shows a status word and the
status word was right. It stopped being invisible the moment the top bar grew a progress bar: a bar frozen at
five percent for forty minutes reads as stuck work, which is the opposite of what it was added to say.

Two decisions live here rather than in the runner. The first is that per-frame progress is reported inside a
band rather than as a raw fraction: a job that reaches 1.0 while it is still persisting objects and writing
its summary has lied, and a caller that trusts the number would move on. The second is that not every frame
deserves a database write. A session of five thousand frames would otherwise commit five thousand times to
move a bar the width of a thumbnail.
"""

from __future__ import annotations


class ProgressBand:
    """Maps work-done onto the fraction a job row reports, and rations the writes.

    `start` is where the job already sits when the first item completes, and `end` is deliberately short of
    1.0: only the code that knows the job is finished may write 1.0.
    """

    def __init__(self, start: float = 0.05, end: float = 0.99, step: float = 0.01) -> None:
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(f"band must satisfy 0 <= start < end <= 1, got {start}..{end}")
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        self.start = start
        self.end = end
        self.step = step
        self._last: float | None = None

    def fraction(self, done: int, total: int) -> float:
        """Where `done` of `total` sits in the band, clamped at both ends.

        A total of zero is answered with `start` rather than an error: a session with no frames is a real
        thing that happens, and a division is not worth failing a job over.
        """
        if total <= 0:
            return self.start
        ratio = min(1.0, max(0.0, done / total))
        return self.start + (self.end - self.start) * ratio

    def due(self, fraction: float) -> bool:
        """Whether this fraction has moved far enough from the last reported one to be worth a write.

        The first call is always due, so a caller learns the job has started moving without waiting for a
        whole step. A fraction that has not advanced is never due, which also means a repeated call at the
        end of a finished job stays silent.
        """
        if self._last is None:
            self._last = fraction
            return True
        if fraction - self._last >= self.step or fraction >= self.end > self._last:
            self._last = fraction
            return True
        return False

    def next_if_due(self, done: int, total: int) -> float | None:
        """The fraction to write, or None to skip this one. The two calls above in the order callers want."""
        f = self.fraction(done, total)
        return f if self.due(f) else None
