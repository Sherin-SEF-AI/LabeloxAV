import { describe, expect, it } from "vitest";

import type { LockHolder, PresenceState } from "@/lib/presenceApi";

import {
  MISSES_BEFORE_DEGRADED, bannerFor, initialPresence, initialsOf, isReadOnly, newTabToken, presenceReducer,
  type PresenceEvent, type PresenceModel,
} from "./presence";

// The rules that decide whether somebody may edit, fed answers by hand. The hook around this adds timers
// and requests (usePresence.test.tsx); nothing in here needs either.

const FRAME = "frame-1";

const holder = (name: string, over: Partial<LockHolder> = {}): LockHolder =>
  ({ user_id: `u-${name.toLowerCase()}`, name, role: "annotator", since_ms: 1000, same_user: false, ...over });

function answer(over: Partial<PresenceState> = {}): PresenceEvent {
  return {
    t: "answer", frameId: FRAME,
    state: {
      frame_id: FRAME, degraded: false, others: [], lock: { mine: false, holder: null },
      can_take_over: false, ttl_s: 15, heartbeat_s: 5, ...over,
    },
  };
}

const mine = (since = 1000) => answer({ lock: { mine: true, holder: holder("Me", { since_ms: since, same_user: true }) } });
const heldBy = (h: LockHolder, canTakeOver = false) =>
  answer({ lock: { mine: false, holder: h }, can_take_over: canTakeOver,
           others: [{ user_id: h.user_id, name: h.name, role: h.role, editing: true }] });
const degraded = () => answer({ degraded: true });
const failed: PresenceEvent = { t: "failed", frameId: FRAME };

const run = (...events: PresenceEvent[]): PresenceModel =>
  events.reduce(presenceReducer, initialPresence(FRAME));

describe("before the first answer", () => {
  it("is view-only, because not knowing yet is not permission", () => {
    const m = initialPresence(FRAME);
    expect(m.status).toBe("connecting");
    expect(isReadOnly(m)).toBe(true);
    // And says nothing: there is no holder to name for the length of one round trip.
    expect(bannerFor(m, false)).toBeNull();
  });
});

describe("holding the lock", () => {
  it("makes the editor editable and shows no banner", () => {
    const m = run(mine());
    expect(m.status).toBe("live");
    expect(m.lock.mine).toBe(true);
    expect(isReadOnly(m)).toBe(false);
    expect(bannerFor(m, true)).toBeNull();
    expect(m.acquisitions).toBe(1);
    expect(m.regainedFrom).toBeNull();
  });

  it("a heartbeat that continues the same lock is not a new acquisition", () => {
    const m = run(mine(1000), mine(1000), mine(1000));
    expect(m.acquisitions).toBe(1);
  });

  it("a lock that lapsed in a background tab and was taken again counts as a new one", () => {
    // Same tab, still `mine` on both sides of the gap, but the lock's own start time moved: for a while
    // nobody held the frame and anybody could have.
    const m = run(mine(1000), mine(9000));
    expect(m.lock.mine).toBe(true);
    expect(m.acquisitions).toBe(2);
    expect(m.regainedFrom).toBeNull();
  });
});

describe("somebody else holding the lock", () => {
  it("is view-only, names them, and offers no take-over to an annotator", () => {
    const m = run(heldBy(holder("Asha")));
    expect(isReadOnly(m)).toBe(true);
    expect(m.others.map((p) => p.name)).toEqual(["Asha"]);
    expect(bannerFor(m, false)).toEqual({ kind: "held", text: "Being edited by Asha — view only", canTakeOver: false });
  });

  it("offers the take-over when the server says this caller may", () => {
    const m = run(heldBy(holder("Asha"), true));
    expect(bannerFor(m, false)?.canTakeOver).toBe(true);
  });

  it("says so when the holder is this person's other tab", () => {
    const m = run(heldBy(holder("Me", { same_user: true }), true));
    expect(bannerFor(m, false)?.text).toBe("Being edited by you in another tab — view only");
  });

  it("becomes editable when the holder leaves and this tab gets the lock, remembering whose it was", () => {
    const asha = holder("Asha");
    const m = run(heldBy(asha), mine(5000));
    expect(isReadOnly(m)).toBe(false);
    expect(m.acquisitions).toBe(1);
    // Asha edited this frame while it was on screen here, so what is on screen is stale.
    expect(m.regainedFrom).toEqual(asha);
    expect(bannerFor(m, false)).toBeNull();
  });
});

describe("losing the lock while editing", () => {
  const ravi = holder("Ravi", { role: "reviewer", since_ms: 7000 });

  it("switches to view-only and says who took it", () => {
    const m = run(mine(), heldBy(ravi));
    expect(isReadOnly(m)).toBe(true);
    expect(m.lostTo).toEqual(ravi);
    const b = bannerFor(m, true);
    expect(b?.kind).toBe("lost");
    expect(b?.text).toBe("Ravi took over this frame — view only");
  });

  it("tells somebody with unsaved edits that they are kept and cannot be saved", () => {
    const b = bannerFor(run(mine(), heldBy(ravi)), true);
    expect(b?.detail).toMatch(/unsaved changes are still here/);
    expect(b?.detail).toMatch(/cannot be saved while Ravi is editing/);
  });

  it("tells somebody with nothing unsaved that nothing was lost", () => {
    expect(bannerFor(run(mine(), heldBy(ravi)), false)?.detail).toBe("Everything you had changed was already saved.");
  });

  it("goes on saying so on later heartbeats", () => {
    const m = run(mine(), heldBy(ravi), heldBy(ravi), heldBy(ravi));
    expect(m.lostTo).toEqual(ravi);
    expect(bannerFor(m, true)?.kind).toBe("lost");
  });

  it("clears once the lock comes back, and records who had it in between", () => {
    const m = run(mine(1000), heldBy(ravi), mine(20000));
    expect(m.lostTo).toBeNull();
    expect(isReadOnly(m)).toBe(false);
    expect(m.acquisitions).toBe(2);
    expect(m.regainedFrom).toEqual(ravi);
  });

  it("a lock that merely lapsed is not reported as taken", () => {
    // Free and not mine: the next heartbeat asks for it again. Nobody took anything.
    const m = run(mine(), answer());
    expect(m.lostTo).toBeNull();
    expect(bannerFor(m, false)?.kind).toBe("free");
    expect(isReadOnly(m)).toBe(true);
  });

  it("losing it to this person's other tab says that", () => {
    const m = run(mine(), heldBy(holder("Me", { same_user: true, since_ms: 9000 }), true));
    const b = bannerFor(m, true);
    expect(b?.text).toBe("Your other tab took over this frame — view only");
    expect(b?.detail).toMatch(/while your other tab is editing/);
    expect(b?.canTakeOver).toBe(true);
  });
});

describe("failing open", () => {
  it("a store that is down on the first answer leaves the editor editable and says degraded", () => {
    const m = run(degraded());
    expect(m.status).toBe("degraded");
    expect(isReadOnly(m)).toBe(false);
    expect(m.others).toEqual([]);
    expect(bannerFor(m, false)).toBeNull();
  });

  it("a request that fails before anything was ever heard does the same", () => {
    expect(isReadOnly(run(failed))).toBe(false);
    expect(run(failed).status).toBe("degraded");
  });

  it("nobody signed in is not a reason to lock the editor", () => {
    const m = run({ t: "unavailable", frameId: FRAME });
    expect(m.status).toBe("degraded");
    expect(isReadOnly(m)).toBe(false);
  });

  it("one dropped heartbeat does not turn a viewer into an editor", () => {
    const asha = holder("Asha");
    for (const blip of [failed, degraded()]) {
      const m = run(heldBy(asha), blip);
      expect(m.status).toBe("live");
      expect(isReadOnly(m)).toBe(true);
      expect(m.lock.holder).toEqual(asha);
    }
  });

  it("several in a row do: presence that stays down stops gating", () => {
    const blips = Array.from({ length: MISSES_BEFORE_DEGRADED }, () => failed);
    const m = run(heldBy(holder("Asha")), ...blips);
    expect(m.status).toBe("degraded");
    expect(isReadOnly(m)).toBe(false);
    expect(m.lock.holder).toBeNull();
  });

  it("an answer in between resets the count", () => {
    const asha = holder("Asha");
    const m = run(heldBy(asha), failed, failed, heldBy(asha), failed, failed);
    expect(m.status).toBe("live");
    expect(isReadOnly(m)).toBe(true);
  });

  it("an editor keeps editing through an outage and through its end", () => {
    const blips = Array.from({ length: MISSES_BEFORE_DEGRADED + 2 }, () => failed);
    const down = run(mine(1000), ...blips);
    expect(isReadOnly(down)).toBe(false);
    const back = presenceReducer(down, mine(1000));
    expect(back.status).toBe("live");
    expect(isReadOnly(back)).toBe(false);
    // The same lock survived the outage, so nothing happened behind this tab's back.
    expect(back.acquisitions).toBe(1);
  });

  it("when presence comes back and somebody else holds the frame, the editor is told it lost it", () => {
    const blips = Array.from({ length: MISSES_BEFORE_DEGRADED }, () => failed);
    const back = presenceReducer(run(mine(), ...blips), heldBy(holder("Asha")));
    expect(isReadOnly(back)).toBe(true);
    const b = bannerFor(back, true);
    expect(b?.kind).toBe("lost");
    expect(b?.text).toBe("Asha took over this frame — view only");
    expect(b?.detail).toMatch(/unsaved changes are still here/);
  });

  it("when presence comes back for somebody who was only viewing, it is the ordinary banner", () => {
    const asha = holder("Asha");
    const blips = Array.from({ length: MISSES_BEFORE_DEGRADED }, () => failed);
    const back = presenceReducer(run(heldBy(asha), ...blips), heldBy(asha));
    expect(bannerFor(back, false)?.kind).toBe("held");
  });
});

describe("changing frame", () => {
  it("reset forgets everything about the previous frame", () => {
    const m = presenceReducer(run(mine(), heldBy(holder("Ravi"))), { t: "reset", frameId: "frame-2" });
    expect(m).toEqual(initialPresence("frame-2"));
    expect(isReadOnly(m)).toBe(true);
  });

  it("a late answer for the previous frame is ignored", () => {
    const next = initialPresence("frame-2");
    expect(presenceReducer(next, mine())).toBe(next);
    expect(presenceReducer(next, failed)).toBe(next);
  });
});

describe("initialsOf", () => {
  it("takes the first and last name, or the first two letters of a single one", () => {
    expect(initialsOf("Asha Rao")).toBe("AR");
    expect(initialsOf("asha kumari rao")).toBe("AR");
    expect(initialsOf("binu")).toBe("BI");
    expect(initialsOf("rev-1a2b3c4d")).toBe("R1");
    expect(initialsOf("x")).toBe("X");
    expect(initialsOf("   ")).toBe("?");
    expect(initialsOf("")).toBe("?");
  });
});

describe("newTabToken", () => {
  it("is in the alphabet the server accepts", () => {
    const t = newTabToken((b) => b.fill(0xab));
    expect(t).toBe("tab-" + "ab".repeat(16));
    expect(t).toMatch(/^[A-Za-z0-9_-]{16,64}$/);
  });

  it("differs from one call to the next", () => {
    expect(newTabToken()).not.toBe(newTabToken());
  });
});
