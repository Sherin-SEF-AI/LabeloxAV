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

// ---- Token expiry (v2 tokens carry an exp) ----
// v2 tokens are lbx2.<base64url(payload)>.<base64url(hmac)>. The client cannot verify the signature (only the
// server holds the key), but it can read the payload's exp to roll the token before a request 401s. Decoding
// is best-effort: a legacy, absent, or malformed token yields null and the reactive 401 path still covers it.
export function decodeExp(token: string | undefined): number | null {
  if (!token) return null;
  const parts = token.split(".");
  if (parts.length !== 3 || parts[0] !== "lbx2") return null;
  try {
    const b64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    const payload = JSON.parse(atob(b64));
    return typeof payload.exp === "number" ? payload.exp : null;
  } catch {
    return null;
  }
}

export function tokenExp(): number | null {
  return decodeExp(getUser()?.token);
}

// True once the current token is at or past its expiry (minus an optional skew). Unknown expiry (legacy token)
// is treated as not-expired: the server still gates, so we do not lock the user out on a decode miss.
export function isTokenExpired(skewSeconds = 0): boolean {
  const exp = tokenExp();
  if (exp === null) return false;
  return Date.now() / 1000 >= exp - skewSeconds;
}

// Roll the token when it is within `withinSeconds` of expiry, via /auth/refresh (which mints a fresh token for
// the current user and needs a still-valid token). Deduplicated so concurrent callers share one refresh, and
// best-effort: on any failure the caller proceeds and a genuine 401 triggers the sign-in redirect in api.ts.
let _refreshing: Promise<void> | null = null;
export async function refreshTokenIfNeeded(withinSeconds = 3600): Promise<void> {
  const u = getUser();
  if (!u?.token) return;
  const exp = decodeExp(u.token);
  if (exp === null || Date.now() / 1000 < exp - withinSeconds) return; // absent expiry or still fresh
  if (_refreshing) return _refreshing;
  _refreshing = (async () => {
    try {
      const r = await fetch("/api/auth/refresh", {
        method: "POST",
        headers: { Authorization: `Bearer ${u.token}` },
      });
      if (r.ok) {
        const body = await r.json();
        if (body?.token) setUser({ ...u, token: body.token });
      }
    } catch {
      // best effort: leave the old token in place; the request path will surface a real 401 if it lapsed
    } finally {
      _refreshing = null;
    }
  })();
  return _refreshing;
}
