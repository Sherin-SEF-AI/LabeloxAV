// Presence and the frame edit lock, as the server speaks them (services/api/routers/presence.py).
//
// Every call goes through the shared client in lib/api.ts, including the one made while the page is
// closing. That one is the reason this file exists apart from `api`: it has to be a keepalive request
// that still carries the Authorization header, which navigator.sendBeacon cannot do, and the server
// refuses anything without one.

import { apiPost, getBinary } from "./api";

export type PresenceUser = {
  user_id: string;
  name: string;
  role: string;
  /** This person holds the frame's edit lock. */
  editing: boolean;
};

export type LockHolder = {
  user_id: string;
  name: string;
  role: string;
  /** When this holder took the lock, epoch ms. Changes only when the lock is taken afresh. */
  since_ms: number;
  /** The holder is the signed-in user, in a different tab. */
  same_user: boolean;
};

export type PresenceState = {
  frame_id: string;
  /** The presence store is unreachable: nobody is listed, nothing is locked, and editing is not gated. */
  degraded: boolean;
  reason?: string;
  others: PresenceUser[];
  lock: { mine: boolean; holder: LockHolder | null };
  /** The server will accept a take-over from this caller. It checks again when asked. */
  can_take_over: boolean;
  ttl_s: number;
  heartbeat_s: number;
};

export type ReleaseResult = { frame_id: string; released: boolean; degraded: boolean };

const base = (frameId: string) => `/api/presence/frames/${encodeURIComponent(frameId)}`;

// Nobody is left to tell when a closing page's last request fails, and the lock's TTL is the answer to
// exactly that case. Named, so it reads as the decision it is.
function lockWillExpireByItself(): void {}

export const presenceApi = {
  /** "This tab is here." Keeps the lock alive if this tab holds it, and takes it if it is free. */
  heartbeat: (frameId: string, tab: string, acquire = true) =>
    apiPost<PresenceState>(`${base(frameId)}/heartbeat`, { tab, acquire }),

  /** Take the edit lock if nobody holds it. The answer says who does. */
  acquire: (frameId: string, tab: string) =>
    apiPost<PresenceState>(`${base(frameId)}/lock/acquire`, { tab }),

  /** Give the lock back and stop being listed. Releases nothing unless this tab holds it. */
  release: (frameId: string, tab: string) =>
    apiPost<ReleaseResult>(`${base(frameId)}/lock/release`, { tab, leave: true }),

  /** Take the frame from whoever is editing it. 403 unless reviewer, admin, or it is your own other tab. */
  takeOver: (frameId: string, tab: string) =>
    apiPost<PresenceState>(`${base(frameId)}/lock/takeover`, { tab }),

  /**
   * Release while the page is going away.
   *
   * `keepalive` lets the request outlive the document. It goes through the shared client so it carries
   * the token; that client checks token freshness first, which settles without a network round trip
   * unless the token is about to expire, and in that rare case the request may not leave before the page
   * does. The lock then lapses by its TTL, as it does for a crashed tab.
   */
  releaseOnUnload: (frameId: string, tab: string): void => {
    void getBinary(`${base(frameId)}/lock/release`, {
      method: "POST",
      keepalive: true,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tab, leave: true }),
    }).catch(lockWillExpireByItself);
  },
};
