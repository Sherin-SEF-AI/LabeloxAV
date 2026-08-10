/**
 * The poll cadence, which existed in one loop and not the other.
 *
 * CloudControl polls two endpoints. The status loop backed off on error; the orphans loop was a fixed
 * `setInterval` that kept firing at an unreachable backend. A three-minute API restart produced 72
 * `ECONNREFUSED` proxy errors, in a component whose comment claims a blip does not flood the console.
 *
 * The case worth pinning is the last one: a failure has to outrank liveness, or a pod that was running when
 * the backend went away keeps being polled three times a second while nothing can answer.
 */

import { describe, expect, it } from "vitest";

import { ERROR_MS, IDLE_MS, LIVE_MS, isLiveState, pollDelay } from "./pollDelay";

describe("pollDelay", () => {
  it("refreshes fast while something is actually running", () => {
    expect(pollDelay({ ok: true, live: true })).toBe(LIVE_MS);
  });

  it("slows down when nothing is running, which is the usual state", () => {
    expect(pollDelay({ ok: true, live: false })).toBe(IDLE_MS);
    expect(pollDelay({ ok: true })).toBe(IDLE_MS);
  });

  it("backs off on failure", () => {
    expect(pollDelay({ ok: false })).toBe(ERROR_MS);
  });

  it("lets a failure outrank liveness", () => {
    /* The last known state said running, but the request that would confirm it just failed. Polling at 3s
       against a backend that is not answering is what produced the flood. */
    expect(pollDelay({ ok: false, live: true })).toBe(ERROR_MS);
  });

  it("backs off further than it ever polls when healthy", () => {
    expect(ERROR_MS).toBeGreaterThan(IDLE_MS);
    expect(ERROR_MS).toBeGreaterThan(LIVE_MS);
  });

  it("keeps the error interval short enough to recover on its own", () => {
    /* A restart should heal without a page reload. */
    expect(ERROR_MS).toBeLessThanOrEqual(60_000);
  });
});

describe("isLiveState", () => {
  it.each(["connected", "running_job", "provisioning", "terminating", "pausing"])(
    "treats %s as live", (s) => expect(isLiveState(s)).toBe(true));

  it.each(["disconnected", "idle", "stopped", "unknown", ""])(
    "treats %s as not live", (s) => expect(isLiveState(s)).toBe(false));

  it("treats an absent state as not live, rather than throwing", () => {
    /* The first render has no status yet, and a missing state must not be read as a running pod. */
    expect(isLiveState(null)).toBe(false);
    expect(isLiveState(undefined)).toBe(false);
  });
});
