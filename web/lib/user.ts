// Lightweight current-user (no password): the chosen identity lives in localStorage and rides on every
// mutating request as the X-Lbx-User-Id header. Role drives the QA workflow (annotator submits for QA,
// reviewer/admin approves).

export type CurrentUser = { user_id: string; name: string; role: string };

const KEY = "lbx_user";
let _cache: CurrentUser | null | undefined;

export function getUser(): CurrentUser | null {
  if (_cache !== undefined) return _cache;
  if (typeof window === "undefined") return null;
  try {
    _cache = JSON.parse(localStorage.getItem(KEY) || "null");
  } catch {
    _cache = null;
  }
  return _cache ?? null;
}

export function setUser(u: CurrentUser | null): void {
  _cache = u;
  if (typeof window !== "undefined") {
    if (u) localStorage.setItem(KEY, JSON.stringify(u));
    else localStorage.removeItem(KEY);
  }
}

export function userHeaders(): Record<string, string> {
  const u = getUser();
  return u ? { "X-Lbx-User-Id": u.user_id } : {};
}

// Where an accept/confirm should land given the actor's role: annotators submit for QA, reviewers
// and admins approve straight to accepted (gold-eligible).
export function acceptState(role: string | undefined): string {
  return role === "annotator" ? "submitted" : "accepted";
}
