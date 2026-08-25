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

// ---- what state a drive's work is in -------------------------------------------------------------

/** One row of `GET /api/sessions/states`. */
export type SessionState = {
  session_id: string;
  frames: number;
  objects: number;
  /** Objects with a review row written by a person. Machine writers are excluded server-side. */
  reviewed_objects: number;
};

/**
 * Four states, chosen because they are the ones that actually separate drives in this corpus.
 *
 * A percentage was the obvious design and does not work: accepted-plus-auto_accept over total has a median
 * of 0.011 here, so a bar reads about 1% on nearly every drive. Measured instead: 126 sessions have no
 * camera frames at all (they are LiDAR and 3D captures, and opening one 404s), 42 have frames but no
 * detections, 125 have been ruled on by a person, and the rest are labelled and waiting.
 */
export type DriveStatus = "empty" | "unlabelled" | "ready" | "working";

export type DriveStatusOrUnknown = DriveStatus | "unknown";

/**
 * `unknown` is its own answer, and it matters more than it looks.
 *
 * The first version returned "empty" when the state was missing, and "empty" also means "cannot be
 * opened". So when the states request failed, every drive in the picker was marked "no frames" and
 * disabled: one failed aggregate turned the whole picker off, and it said the corpus was empty rather
 * than saying it did not know. Missing information must never be reported as a fact about the corpus, and
 * it must never be the thing that blocks the work.
 */
export function driveStatus(st: SessionState | undefined): DriveStatusOrUnknown {
  if (!st) return "unknown";
  if (st.frames === 0) return "empty";
  if (st.objects === 0) return "unlabelled";
  return st.reviewed_objects > 0 ? "working" : "ready";
}

/** Short label and why, so the mark is readable rather than a colour somebody has to learn. */
export const DRIVE_STATUS: Record<DriveStatusOrUnknown, { label: string; tip: string; tone: string }> = {
  unknown: { label: "", tone: "text-ink-3",
             tip: "the drive states could not be loaded, so this one is unmarked rather than guessed at" },
  empty: { label: "no frames", tone: "text-ink-3",
           tip: "this drive has no camera frames, so the editor cannot open it (LiDAR and 3D captures land here)" },
  unlabelled: { label: "not labelled", tone: "text-warn",
                tip: "frames are here but nothing has detected anything on them yet" },
  ready: { label: "ready", tone: "text-ink-2",
           tip: "labelled by a machine and nobody has reviewed any of it" },
  working: { label: "in progress", tone: "text-pass",
             tip: "a person has ruled on some of this drive" },
};

/**
 * Whether the editor can open this drive.
 *
 * Only a KNOWN zero-frame drive is refused. An unknown state lets the click through and lets the server
 * answer, because being unable to load an aggregate is not evidence that a drive has no frames.
 */
export const canOpen = (st: SessionState | undefined): boolean => driveStatus(st) !== "empty";

// ---- where you have been -------------------------------------------------------------------------

const RECENT_KEY = "lbx.editor.recentSessions";
const RECENT_MAX = 12;

/**
 * The drives opened in this browser, newest first.
 *
 * Switching drives used to lose the one you came from entirely: the picker showed 377 rows in date order
 * and nothing said which of them you had just been working in. Stored per browser rather than per account
 * because it is a navigation convenience, not a fact about the corpus.
 */
export function recentSessions(): string[] {
  try {
    const raw = globalThis.localStorage?.getItem(RECENT_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list.filter((x) => typeof x === "string").slice(0, RECENT_MAX) : [];
  } catch {
    return [];
  }
}

export function recordVisit(sessionId: string): void {
  if (!sessionId) return;
  try {
    const next = [sessionId, ...recentSessions().filter((s) => s !== sessionId)].slice(0, RECENT_MAX);
    globalThis.localStorage?.setItem(RECENT_KEY, JSON.stringify(next));
  } catch {
    /* private mode; the picker still works, it just cannot remember */
  }
}

/** The drive you were in before this one, or null on a first visit. */
export function previousSession(currentId: string): string | null {
  return recentSessions().find((s) => s !== currentId) ?? null;
}

/**
 * The list to show: the drive you are in, then the ones you have been in, then everything else by date.
 *
 * Recency before date because the question the picker exists to answer is "where was I", and a drive you
 * worked yesterday is more findable by having been worked than by when it was filmed.
 */
export function orderByVisit(sessions: readonly SessionRow[], currentId: string | null,
                             recent: readonly string[]): SessionRow[] {
  const rank = new Map(recent.map((id, i) => [id, i]));
  return [...sessions].sort((a, b) => {
    if (a.session_id === currentId) return -1;
    if (b.session_id === currentId) return 1;
    const ra = rank.get(a.session_id), rb = rank.get(b.session_id);
    if (ra != null && rb != null) return ra - rb;
    if (ra != null) return -1;
    if (rb != null) return 1;
    return (b.start_ts_ns || 0) - (a.start_ts_ns || 0);
  });
}

// ---- running a drive through auto-label, a batch at a time --------------------------------------

/**
 * How many frames one press covers.
 *
 * There is one GPU in this box, and the alternative to a bounded batch is firing an unbounded pass over a
 * thousand-frame drive from a menu. 200 frames is roughly a minute of work: long enough to be worth
 * pressing, short enough that a wrong drive costs a minute rather than an afternoon. The server also
 * refuses to start a second local job while one is running, so this cannot be stacked up by clicking.
 */
export const AUTOLABEL_BATCH = 200;

export type RunningJob = { job_id: string; progress: number | null; status: string };

/**
 * The autolabel job for this drive, if one is running.
 *
 * Matched on the session id, which the job row carries as its label truncated to eight characters. Eight
 * hex characters across 377 drives is not a collision risk, and the alternative is a second request per
 * row.
 */
export function jobForSession(rows: readonly { job_id: string; label?: string; status: string;
                                               progress?: number | null }[],
                              sessionId: string): RunningJob | null {
  const head = sessionId.slice(0, 8);
  const j = rows.find((r) => (r.label ?? "").startsWith(head)
                             && (r.status === "running" || r.status === "pending"));
  return j ? { job_id: j.job_id, progress: j.progress ?? null, status: j.status } : null;
}

/** Whether it is worth offering to label this drive at all. */
export function canAutolabel(st: SessionState | undefined): boolean {
  const s = driveStatus(st);
  return s === "unlabelled" || s === "ready" || s === "working";
}
