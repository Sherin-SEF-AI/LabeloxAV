// Naming a drive, and finding one among several hundred.
//
// The frame editor never said which session you were in. The top bar prints a frame id and an object
// count, so an annotator working a fleet of 377 drives could tell which frame they were on and not which
// drive it belonged to, and there was no way to move to another one without going back to triage and in
// again.
//
// Pure and separate from the component because the two things worth testing are the label and the match:
// a session with no city must still read as something, and a search that only matched the vehicle id
// would be useless on a fleet where every drive is DASHCAM-01.

import type { SessionRow } from "./types";

/** Nanoseconds to a short local date. Sessions are drives, so the day is the useful part, not the clock. */
export function sessionDate(ts_ns: number): string {
  if (!Number.isFinite(ts_ns) || ts_ns <= 0) return "";
  return new Date(ts_ns / 1_000_000).toISOString().slice(0, 10);
}

/**
 * What to call a drive.
 *
 * Falls back through vehicle, city and id rather than rendering an empty string: 42 sessions in this
 * corpus carry no city, and a blank row in a picker is indistinguishable from a broken one.
 */
export function sessionLabel(s: SessionRow): string {
  const parts = [s.vehicle_id, s.city].filter((p) => p && String(p).trim());
  return parts.length ? parts.join(" · ") : s.session_id.slice(0, 8);
}

/**
 * The second line, which has to tell one drive from another.
 *
 * `route` holds three different things in this corpus. On the dashcam drives it is a capture label like
 * `2026-05-31 15:39 · 043719F`, which already identifies the clip and already carries a date, and a
 * DIFFERENT date from `start_ts_ns` (filmed against ingested), so appending one produced
 * "2026-06-06 10:01 · 043849F · 2026-07-01" and left the reader working out which was which.
 *
 * On 37 sessions it is the tag `import:video`, and on 147 it is null. Those rows rendered as
 * "import:video · 2026-08-12" thirty-seven times over, which names a category and not a drive, so the
 * picker showed a wall of identical lines. Where the route does not identify the session, the id does.
 */
export function sessionDetail(s: SessionRow): string {
  const route = s.route?.trim();
  if (route && /\d{4}-\d{2}-\d{2}/.test(route)) return route;
  const tail = route || sessionDate(s.start_ts_ns);
  const id = s.session_id.slice(0, 8);
  return tail ? `${id} · ${tail}` : id;
}

/**
 * Whether a query picks out this session.
 *
 * Matches the id too, because the id is what every other surface in the app shows and what somebody
 * pastes in from a queue, a log line or a URL.
 */
export function matchesSession(s: SessionRow, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return [s.vehicle_id, s.city, s.route, s.session_id, sessionDate(s.start_ts_ns)]
    .some((f) => !!f && String(f).toLowerCase().includes(q));
}

/**
 * The list to show: newest drive first, and the one you are already in pinned to the top.
 *
 * Pinned rather than hidden, because the picker is also the only place the editor says which session this
 * frame belongs to, and a current row that scrolls away answers half the question.
 */
export function orderSessions(sessions: readonly SessionRow[], currentId: string | null): SessionRow[] {
  return [...sessions].sort((a, b) => {
    if (a.session_id === currentId) return -1;
    if (b.session_id === currentId) return 1;
    return (b.start_ts_ns || 0) - (a.start_ts_ns || 0);
  });
}
