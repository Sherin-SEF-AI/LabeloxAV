// What the editor believes about who else is on the frame, decided without React or a network.
//
// The hook (usePresence.ts) is the part that cannot be pure: timers, the visibility of the tab, requests.
// Everything it concludes from an answer is here, as a reducer over the answers, so the rules that decide
// whether somebody may edit can be tested by feeding it answers.
//
// Three rules carry most of the weight:
//
//   * Not knowing yet is not permission. Until the first answer arrives the editor is view-only, for the
//     one round trip that takes, because a box drawn in that window on a frame somebody else holds would
//     be autosaved over them.
//   * Not being able to know IS permission. If presence cannot be reached at all the editor works as it
//     did before presence existed, with the per-object version check as the only guard. But one failed
//     heartbeat is not "cannot be reached": a viewer must not become an editor for five seconds because
//     a single request was dropped, so it takes several in a row.
//   * Losing the lock never discards work. It only stops it being saved, and says so.

import type { LockHolder, PresenceState, PresenceUser } from "@/lib/presenceApi";

export type PresenceStatus = "connecting" | "live" | "degraded";

export type PresenceModel = {
  frameId: string;
  status: PresenceStatus;
  others: PresenceUser[];
  lock: { mine: boolean; holder: LockHolder | null };
  canTakeOver: boolean;
  /** Consecutive heartbeats that failed or came back degraded. */
  misses: number;
  /**
   * Set when this tab held the lock and somebody else now does. Kept until the lock comes back or the
   * frame changes, so the banner can go on saying why the editor stopped saving.
   */
  lostTo: LockHolder | null;
  /**
   * Counts the times this tab came to hold a lock it was not already holding continuously: the first
   * acquisition is 1, and every later one means the frame was out of this tab's hands for a while and
   * may have changed underneath it.
   */
  acquisitions: number;
  /** `since_ms` of the lock this tab last held, to tell a continued lock from one taken afresh. */
  heldSince: number | null;
  /**
   * Who held the frame immediately before this tab last took it, or null when nobody did (it was free,
   * or the lock had merely lapsed). Somebody here means their edits are on the server and not on screen.
   */
  regainedFrom: LockHolder | null;
  /** Seconds between heartbeats, as the server last asked for. */
  heartbeatS: number;
};

export type PresenceEvent =
  | { t: "reset"; frameId: string }
  | { t: "answer"; frameId: string; state: PresenceState }
  | { t: "failed"; frameId: string }
  /** There is nobody to be present as (no signed-in user). */
  | { t: "unavailable"; frameId: string };

/** How many heartbeats in a row may fail before the editor stops trusting what it last heard. */
export const MISSES_BEFORE_DEGRADED = 3;
export const DEFAULT_HEARTBEAT_S = 5;

export function initialPresence(frameId: string): PresenceModel {
  return {
    frameId, status: "connecting", others: [], lock: { mine: false, holder: null }, canTakeOver: false,
    misses: 0, lostTo: null, acquisitions: 0, heldSince: null, regainedFrom: null,
    heartbeatS: DEFAULT_HEARTBEAT_S,
  };
}

function degrade(m: PresenceModel): PresenceModel {
  // Whether this tab held the lock is kept through an outage, though it gates nothing while degraded:
  // when presence returns, that is what tells "somebody took the frame while I could not see" (said as a
  // loss, because there may be unsaved work behind it) from "somebody was already editing".
  return { ...m, status: "degraded", others: [], lock: { mine: m.lock.mine, holder: null }, canTakeOver: false };
}

function miss(m: PresenceModel): PresenceModel {
  const misses = m.misses + 1;
  // Never heard anything: there is no last-known state to keep trusting, so stop gating at once.
  if (m.status === "connecting") return { ...degrade(m), misses };
  if (m.status === "live" && misses < MISSES_BEFORE_DEGRADED) return { ...m, misses };
  return { ...degrade(m), misses };
}

export function presenceReducer(m: PresenceModel, e: PresenceEvent): PresenceModel {
  if (e.t === "reset") return initialPresence(e.frameId);
  // An answer for the frame the editor has already left. Applying it would put the previous frame's
  // holder in the next frame's banner.
  if (e.frameId !== m.frameId) return m;

  switch (e.t) {
    case "unavailable":
      return degrade({ ...m, misses: 0 });
    case "failed":
      return miss(m);
    case "answer": {
      const s = e.state;
      if (s.degraded) return miss(m);
      const mine = s.lock.mine;
      const holder = s.lock.holder;
      const since = mine && holder ? holder.since_ms : null;
      // A lock this tab holds now and was not holding in an unbroken line: the first time, after somebody
      // else had it, or after it lapsed in a background tab and was taken again. The lock's own start
      // time is what says so; a presence outage that the same lock survived is not a new acquisition.
      const afresh = mine && m.heldSince !== since;
      const taken = m.lock.mine && !mine && holder !== null;
      return {
        ...m,
        status: "live",
        others: s.others,
        lock: { mine, holder },
        canTakeOver: s.can_take_over,
        misses: 0,
        lostTo: mine ? null : taken ? holder : m.lostTo,
        acquisitions: afresh ? m.acquisitions + 1 : m.acquisitions,
        heldSince: mine ? since : m.heldSince,
        regainedFrom: afresh ? (m.lock.mine ? null : m.lock.holder ?? m.lostTo) : m.regainedFrom,
        heartbeatS: s.heartbeat_s > 0 ? s.heartbeat_s : m.heartbeatS,
      };
    }
    default:
      return m;
  }
}

/** Whether the editor must refuse edits. */
export function isReadOnly(m: Pick<PresenceModel, "status" | "lock">): boolean {
  if (m.status === "degraded") return false;
  if (m.status === "connecting") return true;
  return !m.lock.mine;
}

export type PresenceBanner = {
  kind: "held" | "lost" | "free";
  /** The headline. */
  text: string;
  /** What it means for the work in front of the reader, when there is something to add. */
  detail?: string;
  canTakeOver: boolean;
};

const who = (h: LockHolder) => (h.same_user ? "you in another tab" : h.name);
const whoIs = (h: LockHolder) => (h.same_user ? "your other tab is" : `${h.name} is`);

/**
 * What to say above the canvas, or null when there is nothing to say (this tab is editing, presence is
 * off, or the first answer has not arrived).
 *
 * `unsaved` is whether the editor is holding edits it has not been able to save, which changes what a
 * lost lock means to the person reading it.
 */
export function bannerFor(m: PresenceModel, unsaved: boolean): PresenceBanner | null {
  if (m.status !== "live" || m.lock.mine) return null;
  const holder = m.lock.holder;
  if (m.lostTo) {
    const now = holder ?? m.lostTo;
    return {
      kind: "lost",
      text: `${m.lostTo.same_user ? "Your other tab" : m.lostTo.name} took over this frame — view only`,
      detail: unsaved
        ? `Your unsaved changes are still here, but they cannot be saved while ${whoIs(now)} editing.`
        : "Everything you had changed was already saved.",
      canTakeOver: m.canTakeOver,
    };
  }
  if (holder) {
    return { kind: "held", text: `Being edited by ${who(holder)} — view only`, canTakeOver: m.canTakeOver };
  }
  // Nobody holds it and this tab does not either: the next heartbeat asks for it.
  return { kind: "free", text: "Nobody is editing this frame — taking it for you", canTakeOver: false };
}

/** Two letters for an avatar. "Asha Rao" is AR, "binu" is BI, an empty name is a question mark. */
export function initialsOf(name: string): string {
  const parts = name.trim().split(/[\s._-]+/).filter(Boolean);
  if (!parts.length) return "?";
  const letters = parts.length === 1 ? parts[0].slice(0, 2) : parts[0][0] + parts[parts.length - 1][0];
  return letters.toUpperCase();
}

/**
 * A token for this tab, unguessable and never reused.
 *
 * Deliberately not kept in sessionStorage. A duplicated tab copies its session storage, and two tabs
 * presenting one token would share one lock, which is the thing a per-tab token exists to prevent. The
 * cost is that a reloaded tab is a new tab; it releases on the way out, and takes its own lock back.
 *
 * getRandomValues rather than randomUUID: the second exists only in a secure context and the editor is
 * also served over plain http on a LAN.
 */
export function newTabToken(random: (bytes: Uint8Array) => Uint8Array = (b) => globalThis.crypto.getRandomValues(b)): string {
  const bytes = random(new Uint8Array(16));
  return "tab-" + Array.from(bytes, (x) => x.toString(16).padStart(2, "0")).join("");
}
