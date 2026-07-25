// Current-user credential: a signed Bearer token (issued by the API when the user is created) lives in
// localStorage and rides on every request as the Authorization header. Role drives the QA workflow
// (annotator submits for QA, reviewer/admin approves). The token is unforgeable, so identity can no longer
// be self-asserted by echoing a user id.

export type CurrentUser = { user_id: string; name: string; role: string; token?: string };

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
  if (u?.token) return { Authorization: `Bearer ${u.token}` };
  // No token (e.g. a dev backend with auth disabled): fall back to the legacy id header so attribution still
  // works locally. With auth enabled server-side this header is ignored and the request is treated as anon.
  return u ? { "X-Lbx-User-Id": u.user_id } : {};
}

// Where an accept/confirm should land given the actor's role: annotators submit for QA, reviewers
// and admins approve straight to accepted (gold-eligible).
export function acceptState(role: string | undefined): string {
  return role === "annotator" ? "submitted" : "accepted";
}
