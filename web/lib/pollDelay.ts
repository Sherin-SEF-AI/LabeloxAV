/**
 * How long to wait before the next background poll.
 *
 * `CloudControl` runs two pollers against RunPod-backed endpoints. The status loop backed off on error and
 * the orphans loop did not: it was a plain `setInterval(..., 20000)` that kept firing while the backend was
 * unreachable. So a three-minute API restart produced 72 `ECONNREFUSED` proxy errors in the console, against
 * a component whose own comment says "a backend blip does not flood the console with retries".
 *
 * Pulled out as a pure function because it is the policy both loops should share, and because a timing rule
 * living inside a `useEffect` cannot be tested without mounting and faking timers.
 *
 * Pod state lives in RunPod's API rather than in our database, so there is no row to watch and push from and
 * polling is the honest shape here. That is exactly why the cadence has to be careful: it is a rate-limited
 * third party reached through our own backend.
 */

export type PollState = {
  /** Did the last request succeed? */
  ok: boolean;
  /** Is something actually running, so the cost and uptime meter needs to tick? */
  live?: boolean;
};

/** A live pod has a meter moving; anything else is a state that changes rarely. */
export const LIVE_MS = 3_000;
export const IDLE_MS = 20_000;
/** Long enough that a restart or a deploy does not produce a wall of errors, short enough to notice. */
export const ERROR_MS = 30_000;

/** RunPod states where a meter is genuinely moving and a 3s refresh earns its cost. */
export const LIVE_STATES: ReadonlySet<string> = new Set([
  "connected", "running_job", "provisioning", "terminating", "pausing",
]);

export function pollDelay({ ok, live = false }: PollState): number {
  // A failure outranks liveness. The last known state said "running", but the request that would confirm it
  // just failed, so polling three times a second at a backend that is not answering only fills the console.
  if (!ok) return ERROR_MS;
  return live ? LIVE_MS : IDLE_MS;
}

export const isLiveState = (state: string | null | undefined): boolean =>
  state !== null && state !== undefined && LIVE_STATES.has(state);
