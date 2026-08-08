/**
 * What to tell an operator when a console action fails, derived from the status rather than guessed.
 *
 * The agent console had thirteen call sites shaped like:
 *
 *     catch (e) { setMsg("sweep failed (needs reviewer role): " + humanizeError(e)); }
 *
 * The parenthetical is a guess, asserted unconditionally. `ApiError` has carried `.status` all along and
 * nothing read it, so a GPU-busy 503 rendered as "sweep failed (needs reviewer role): The service is busy
 * (the GPU may be in use)" - a single line that contradicts itself and sends someone to fix permissions
 * while the real answer is to wait. Twenty of these exist across the app.
 *
 * A permission hint is worth showing when it is true. This returns one only for 401 and 403, and returns a
 * cause the operator can act on for everything else.
 */

import { ApiError } from "./api";

export type ActionFailure = {
  /** "<action> failed: <reason>" - what to show. */
  message: string;
  /** Present only when the status actually says so. */
  hint?: string;
  /** null when the failure never reached the server (network, abort). */
  status: number | null;
  /** Whether trying the same thing again could plausibly work. */
  retryable: boolean;
};

/** Statuses where the operator's own permissions are genuinely the problem. */
const PERMISSION_HINT: Record<number, string> = {
  401: "you are signed out",
  403: "this needs a higher role",
};

/** Statuses where the same request may succeed later without anything being changed. */
const RETRYABLE = new Set([408, 425, 429, 500, 502, 503, 504]);

function reasonOf(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

export function describeFailure(action: string, e: unknown): ActionFailure {
  const status = e instanceof ApiError ? e.status : null;
  const reason = reasonOf(e);
  const hint = status !== null ? PERMISSION_HINT[status] : undefined;
  return {
    message: `${action} failed: ${reason}`,
    hint,
    status,
    // A request that never reached the server is worth retrying too: a dev server restart mid-click is the
    // common case here and looks identical to a network drop.
    retryable: status === null ? true : RETRYABLE.has(status),
  };
}

/** The one-line form, for a log entry or a toast. */
export function failureLine(action: string, e: unknown): string {
  const f = describeFailure(action, e);
  return f.hint ? `${f.message} (${f.hint})` : f.message;
}
