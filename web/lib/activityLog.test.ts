/**
 * The console transcript, which exists because one shared message slot lost results.
 *
 * Roughly fourteen operations were reported through a single `msg` string written by 64 call sites, and
 * several of them settle on the server minutes after the click. So the second action erased the first
 * action's result, and a background result arriving late erased whatever was on screen. These tests pin the
 * properties that make a transcript a transcript: nothing overwrites anything, late results land on their
 * own entry, and a still-running job is never evicted to make room.
 */

import { describe, expect, it } from "vitest";

import { MAX_ENTRIES, begin, clear, emptyLog, isRunning, record, runningCount, settle } from "./activityLog";

describe("begin and settle", () => {
  it("keeps a result instead of overwriting the previous one", () => {
    let log = emptyLog;
    const a = begin(log, "sweep", 1); log = a.log;
    log = settle(log, a.id, "ok", "queued 8 sessions", 2);
    const b = begin(log, "audit", 3); log = b.log;
    log = settle(log, b.id, "ok", "sampled 200", 4);

    expect(log.entries).toHaveLength(2);
    expect(log.entries.map((e) => e.message)).toContain("queued 8 sessions");
  });

  it("lets a slow background job settle after a later one already has", () => {
    /* The sweep and the audit both run server-side; whichever finishes first must not claim the other's
       entry. */
    let log = emptyLog;
    const sweep = begin(log, "sweep", 1); log = sweep.log;
    const audit = begin(log, "audit", 2); log = audit.log;

    log = settle(log, audit.id, "ok", "audit done", 3);
    log = settle(log, sweep.id, "ok", "sweep done", 9);

    expect(log.entries.find((e) => e.id === sweep.id)?.message).toBe("sweep done");
    expect(log.entries.find((e) => e.id === audit.id)?.message).toBe("audit done");
  });

  it("shows newest first", () => {
    let log = emptyLog;
    log = begin(log, "first", 1).log;
    log = begin(log, "second", 2).log;
    expect(log.entries[0].action).toBe("second");
  });

  it("records when an operation settled, so a transcript can show duration", () => {
    let log = emptyLog;
    const r = begin(log, "sweep", 100); log = r.log;
    log = settle(log, r.id, "ok", "done", 4100);
    const e = log.entries[0];
    expect(e.startedAt).toBe(100);
    expect(e.settledAt).toBe(4100);
  });

  it("carries a hint onto the entry when there is one", () => {
    let log = emptyLog;
    const r = begin(log, "sweep", 1); log = r.log;
    log = settle(log, r.id, "failed", "sweep failed: forbidden", 2, "this needs a higher role");
    expect(log.entries[0].hint).toBe("this needs a higher role");
  });
});

describe("double-clicks and stale results", () => {
  it("keeps two entries when the same action is started twice", () => {
    /* Collapsing them would hide a double-click that fired two sweeps, which is the thing worth seeing. */
    let log = emptyLog;
    log = begin(log, "sweep", 1).log;
    log = begin(log, "sweep", 2).log;
    expect(log.entries).toHaveLength(2);
    expect(runningCount(log)).toBe(2);
  });

  it("ignores a result for an id that is no longer present", () => {
    /* A late result whose entry was evicted must not reappear at the top as though it just started. */
    let log = emptyLog;
    const r = begin(log, "sweep", 1); log = r.log;
    log = clear(log);
    log = settle(log, r.id, "ok", "came back much later", 999);
    expect(log.entries).toHaveLength(0);
  });

  it("does not reuse ids after a clear", () => {
    /* Otherwise an in-flight operation settling after a clear would land on an unrelated entry. */
    let log = emptyLog;
    const first = begin(log, "sweep", 1); log = first.log;
    log = clear(log);
    const second = begin(log, "audit", 2); log = second.log;
    expect(second.id).not.toBe(first.id);
  });
});

describe("isRunning", () => {
  it("is true only while an entry for that action is unsettled", () => {
    let log = emptyLog;
    const r = begin(log, "sweep", 1); log = r.log;
    expect(isRunning(log, "sweep")).toBe(true);
    log = settle(log, r.id, "ok", "done", 2);
    expect(isRunning(log, "sweep")).toBe(false);
  });

  it("stays true while a second run of the same action is still going", () => {
    let log = emptyLog;
    const a = begin(log, "sweep", 1); log = a.log;
    const b = begin(log, "sweep", 2); log = b.log;
    log = settle(log, a.id, "ok", "done", 3);
    expect(isRunning(log, "sweep")).toBe(true);
    void b;
  });
});

describe("the cap", () => {
  it("bounds the transcript", () => {
    let log = emptyLog;
    for (let i = 0; i < MAX_ENTRIES + 20; i++) log = record(log, `a${i}`, "ok", "x", i);
    expect(log.entries).toHaveLength(MAX_ENTRIES);
  });

  it("evicts settled entries before a job that is still running", () => {
    /* A long sweep must not vanish from the list while it is the thing being waited on. */
    let log = emptyLog;
    const sweep = begin(log, "long-sweep", 0); log = sweep.log;
    for (let i = 0; i < MAX_ENTRIES + 20; i++) log = record(log, `noise${i}`, "ok", "x", i + 1);

    expect(log.entries.find((e) => e.id === sweep.id)?.status).toBe("running");
    expect(log.entries.length).toBeLessThanOrEqual(MAX_ENTRIES);
  });

  it("keeps the newest settled entries, not the oldest", () => {
    let log = emptyLog;
    for (let i = 0; i < MAX_ENTRIES + 5; i++) log = record(log, `a${i}`, "ok", `m${i}`, i);
    expect(log.entries[0].message).toBe(`m${MAX_ENTRIES + 4}`);
    expect(log.entries.map((e) => e.message)).not.toContain("m0");
  });
});
