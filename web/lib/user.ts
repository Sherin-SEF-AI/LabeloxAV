"use client";

import { useEffect, useState } from "react";

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
  // The media cookie is bound to whoever minted it. Dropping it on any identity change is what stops the
  // next user inheriting the previous one's image access, and stops it outliving a sign-out.
  clearMediaCookie();
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
  // Awaited, not fired off: every image URL arrives inside a JSON response that came through this path, so
  // minting here means the cookie is always in place before the first <img> is even constructed. After the
  // first call it is a timestamp comparison.
  await ensureMediaCookie();
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

// ---- The media cookie ----
// An <img src> cannot set an Authorization header, so under the deny-by-default read gate every frame image,
// crop, and mask overlay 401s and the canvas renders blank while the JSON around it loads fine. A cookie is
// the only credential a browser attaches to a subresource request by itself.
//
// What goes in it is deliberately not the session token. The server mints a separate media-scoped token that
// authorises GET on the image routes and nothing else, and it expires in minutes, because a cookie rides on
// URLs that reach proxies, access logs, and Referer headers. SameSite=Strict keeps it off cross-site
// requests, and Path=/api keeps it off page navigations entirely.

const MEDIA_COOKIE = "lbx_media";
let _mediaAt = 0;                 // epoch ms when the current cookie was minted
let _mediaTtlMs = 900_000;        // replaced by whatever TTL the server actually issued
let _minting: Promise<void> | null = null;

function setCookie(name: string, value: string, maxAgeSeconds: number): void {
  if (typeof document === "undefined") return;
  const secure = window.location.protocol === "https:" ? "; Secure" : "";
  document.cookie =
    `${name}=${value}; Path=/api; Max-Age=${maxAgeSeconds}; SameSite=Strict${secure}`;
}

export function clearMediaCookie(): void {
  _mediaAt = 0;
  setCookie(MEDIA_COOKIE, "", 0);
}

// Mint (or re-mint) the media cookie. Re-mints at two thirds of its life so an image request never races the
// expiry, and deduplicates concurrent callers the way refreshTokenIfNeeded does. Best effort: a failure
// leaves images 401ing, which the caller sees, rather than blocking the page.
export async function ensureMediaCookie(): Promise<void> {
  const u = getUser();
  if (!u?.token || typeof document === "undefined") return;
  if (_mediaAt && Date.now() - _mediaAt < _mediaTtlMs * 0.66) return;
  if (_minting) return _minting;
  _minting = (async () => {
    try {
      const r = await fetch("/api/auth/media-token", {
        method: "POST",
        headers: { Authorization: `Bearer ${u.token}` },
      });
      if (r.ok) {
        const body = await r.json();
        if (body?.token) {
          _mediaTtlMs = (Number(body.expires_in) || 900) * 1000;
          setCookie(MEDIA_COOKIE, body.token, Number(body.expires_in) || 900);
          _mediaAt = Date.now();
        }
      }
    } catch {
      // leave it absent; images will 401 visibly rather than the page failing to render
    } finally {
      _minting = null;
    }
  })();
  return _minting;
}

// Read the current user without breaking hydration.
//
// getUser() reads localStorage, which does not exist on the server, so calling it in a component body makes
// the server render one thing and the client another. React responds by throwing away the server HTML for
// that subtree, which on /collaborate showed up as a page error and a flash of the wrong identity. Reading
// it in an effect means the first paint matches the server and the real value arrives a tick later.
export function useCurrentUser(): CurrentUser | null {
  const [u, setU] = useState<CurrentUser | null>(null);
  useEffect(() => { setU(getUser()); }, []);
  return u;
}
