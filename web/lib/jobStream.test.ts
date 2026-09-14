// The top-bar indicator showed nothing while twelve jobs were live on the server.
//
// It watched `uploadManager`, which is module-scoped browser state: right for surviving client-side
// navigation, wrong for a global indicator. A reload or a second tab empties it while the server carries on,
// and it never knew about work it did not start.
//
// The truth is on the server, and `/api/events/jobs` already carries it. What this module adds is one
// connection instead of one per caller: `useEventStream` opens an EventSource per hook call and six pages
// call `useJobStream()`, so putting the same data in the top bar, which is on every page, would have given
// each of those six two connections to the same endpoint.
//
// The counting rules are the other half. This deployment holds 67 autolabel jobs pending since late June and
// 11 parked for a cloud A100. None of them are progressing, and a chip reading "78 running" forever teaches
// people to ignore the chip.

import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  getJobs, jobsConnected, type JobsSummary, resetJobStream,
  subscribeJobs, subscribeJobSummary, summarizeJobs,
} from "./jobStream";
import type { JobStream } from "./useEventStream";

// A minimal EventSource stand-in that records how many were constructed and closed.
class FakeES {
  static made = 0;
  static closed = 0;
  static last: FakeES | null = null;
  listeners = new Map<string, (e: unknown) => void>();

  constructor(public url: string) {
    FakeES.made++;
    FakeES.last = this;
  }

  addEventListener(name: string, fn: (e: unknown) => void) {
    this.listeners.set(name, fn);
  }

  close() {
    FakeES.closed++;
  }

  push(frame: Partial<JobStream>) {
    this.listeners.get("jobs")?.({ data: JSON.stringify(frame) } as MessageEvent);
  }
}

const row = (job_id: string, status: string, progress = 0) => ({ job_id, status, progress });

const frame = (over: Partial<JobStream> = {}): JobStream => ({
  training: [], import: [], export: [], autolabel: [], ingest: [], ...over,
});

beforeEach(() => {
  resetJobStream();
  FakeES.made = 0;
  FakeES.closed = 0;
  vi.stubGlobal("EventSource", FakeES);
  vi.stubGlobal("window", globalThis);
});

// Hoisted by vitest, so it applies to the import above rather than to a call inside a hook.
vi.mock("./user", () => ({ getUser: () => ({ token: "t" }) }));

describe("one shared connection", () => {
  it("opens a single stream no matter how many subscribers attach", () => {
    const a = subscribeJobs(() => {});
    const b = subscribeJobs(() => {});
    const c = subscribeJobs(() => {});
    expect(FakeES.made).toBe(1);
    a(); b(); c();
  });

  it("closes it only when the last subscriber leaves", () => {
    const a = subscribeJobs(() => {});
    const b = subscribeJobs(() => {});
    a();
    expect(FakeES.closed).toBe(0);
    b();
    expect(FakeES.closed).toBe(1);
  });

  it("reopens after everybody has gone and someone comes back", () => {
    subscribeJobs(() => {})();
    expect(FakeES.made).toBe(1);
    subscribeJobs(() => {})();
    expect(FakeES.made).toBe(2);
  });

  it("delivers each frame to every subscriber", () => {
    const seen: number[] = [];
    const a = subscribeJobs((s) => seen.push(s?.import.length ?? -1));
    const b = subscribeJobs((s) => seen.push(s?.import.length ?? -1));
    FakeES.last!.push(frame({ import: [row("j1", "running")] }));
    a(); b();
    expect(seen.filter((n) => n === 1)).toHaveLength(2);
  });
});

describe("the snapshot", () => {
  it("is handed to a subscriber that attaches between pushes", () => {
    // The server pushes on change, so "next push" can be minutes on a quiet system. Without this a chip is
    // blank for the first minutes after every navigation, which looks broken.
    const first = subscribeJobs(() => {});
    FakeES.last!.push(frame({ autolabel: [row("a1", "running")] }));
    let got: JobStream | null = null;
    const second = subscribeJobs((s) => { got = s; });
    expect(got).not.toBeNull();
    expect(got!.autolabel).toHaveLength(1);
    first(); second();
  });

  it("survives the last subscriber leaving, so a remount does not flash empty", () => {
    const a = subscribeJobs(() => {});
    FakeES.last!.push(frame({ import: [row("j1", "running")] }));
    a();
    expect(getJobs()?.import).toHaveLength(1);
  });

  it("wakes subscribers when the connection drops, not only when a frame lands", () => {
    // A dead stream sends no frames, so a subscriber whose only wake-up is a frame would go on claiming the
    // connection is live for as long as it stays down. That is precisely when a live indicator matters.
    let woke = 0;
    const a = subscribeJobs(() => { woke++; });
    const before = woke;
    FakeES.last!.listeners.get("open")?.({} as Event);
    expect(jobsConnected()).toBe(true);
    FakeES.last!.listeners.get("error")?.({} as Event);
    expect(jobsConnected()).toBe(false);
    expect(woke).toBeGreaterThan(before + 1);
    a();
  });

  it("ignores a malformed frame rather than tearing down the stream", () => {
    const a = subscribeJobs(() => {});
    FakeES.last!.listeners.get("jobs")?.({ data: "not json" } as MessageEvent);
    FakeES.last!.push(frame({ import: [row("j1", "running")] }));
    expect(getJobs()?.import).toHaveLength(1);
    a();
  });
});

describe("summarizeJobs", () => {
  it("counts only what is actually progressing", () => {
    const s = summarizeJobs(frame({
      autolabel: [row("a1", "running", 0.5), row("a2", "pending"), row("a3", "queued-cloud")],
      training: [row("t1", "pending")],
    }));
    expect(s.running.map((r) => r.job_id)).toEqual(["a1"]);
  });

  it("counts the ones that are held but not moving, separately", () => {
    // 67 pending since June and 11 parked for a cloud A100 is the answer to "why is nothing happening",
    // so it is kept rather than discarded, just never called running.
    const s = summarizeJobs(frame({
      autolabel: [row("a2", "pending"), row("a3", "queued-cloud"), row("a4", "queued")],
    }));
    expect(s.running).toEqual([]);
    expect(s.waiting).toBe(3);
  });

  it("ignores terminal jobs entirely", () => {
    const s = summarizeJobs(frame({ import: [row("j1", "done"), row("j2", "error")] }));
    expect(s.running).toEqual([]);
    expect(s.waiting).toBe(0);
  });

  it("spans every job kind, since the bar speaks for the whole system", () => {
    const s = summarizeJobs(frame({
      import: [row("i", "running")], autolabel: [row("a", "running")],
      training: [row("t", "running")], export: [row("e", "running")],
    }));
    expect(s.running.map((r) => r.kind).sort()).toEqual(["autolabel", "export", "import", "training"]);
  });

  it("does not treat the ingest summary as a job", () => {
    // `ingest` rides the same stream but is a progress summary with no job_id, so counting it would add a
    // phantom row to the chip.
    const s = summarizeJobs(frame({
      ingest: [{ source: "import_job", active: true, finished: false, done: 1, total: 2,
                 current: null, frames: 10, progress: 0.5 }],
    }));
    expect(s.running).toEqual([]);
  });

  it("clamps a progress value rather than trusting it", () => {
    const s = summarizeJobs(frame({ import: [row("j", "running", 4)] }));
    expect(s.running[0].progress).toBe(1);
  });

  it("prefers the server's queued total over what fits in the window", () => {
    // The lists are a recent tail. 67 autolabel jobs parked for the cloud A100 are all older than the ten
    // most recent, so counting the window said 1 queued when the answer was 68, and a top bar that reports
    // a healthy system while sixty-eight jobs sit untouched is worse than one that reports nothing.
    const s = summarizeJobs(frame({
      training: [row("t1", "pending")],
      waiting: { training: 1, autolabel: 67 },
    }));
    expect(s.waiting).toBe(68);
  });

  it("falls back to counting the window when the server does not send a total", () => {
    const s = summarizeJobs(frame({ training: [row("t1", "pending")], autolabel: [row("a1", "pending")] }));
    expect(s.waiting).toBe(2);
  });

  it("an empty or absent frame is empty, not an error", () => {
    expect(summarizeJobs(null)).toEqual({ running: [], waiting: 0, connected: false });
    expect(summarizeJobs(frame())).toEqual({ running: [], waiting: 0, connected: false });
  });
});

describe("the summary fan-out", () => {
  it("tells a summary subscriber when the connection drops", () => {
    // The reason this exists. A dropped stream sends no frames, so a chip watching only the counts shows
    // "idle" - identical to a system with nothing to do, and meaning the opposite. The summary callback
    // used to return early on a null frame, which is exactly when a disconnect arrives, so the one
    // subscriber that renders a live indicator never heard about it.
    let got: JobsSummary | null = null;
    const off = subscribeJobSummary((x) => { got = x; });
    FakeES.last!.listeners.get("open")?.({} as Event);
    FakeES.last!.push(frame({ autolabel: [row("a1", "running", 0.4)] }));
    expect(got!.connected).toBe(true);
    expect(got!.running).toHaveLength(1);

    FakeES.last!.listeners.get("error")?.({} as Event);
    expect(got!.connected).toBe(false);
    // And the counts survive: the last thing the server said is still the last thing it said. Blanking
    // them would replace one wrong answer with another.
    expect(got!.running).toHaveLength(1);
    off();
  });

  it("tells a summary subscriber the stream is up before the server has said anything", () => {
    // The case that distinguishes a working implementation from the obvious broken one, and the reason
    // the first two attempts at this test were worthless.
    //
    // On a quiet system the stream connects and no frame follows, because the server pushes on change.
    // The summary callback used to return early whenever the frame was null - which is every notification
    // until the first change - so the summary kept its initial `connected: false` and the chip warned
    // "not receiving updates" about a stream that was perfectly healthy.
    //
    // Asserting `false` after a drop cannot catch that: false is also the initial value, so the test
    // passes whether or not anything was ever told. Watching it go TRUE is what requires the callback to
    // have run on a null frame.
    let got: JobsSummary | null = null;
    const off = subscribeJobSummary((x) => { got = x; });
    expect(got!.connected).toBe(false);

    FakeES.last!.listeners.get("open")?.({} as Event);
    expect(got!.connected).toBe(true);
    expect(got!.running).toHaveLength(0);

    // And back down again, still with no frame in either direction.
    FakeES.last!.listeners.get("error")?.({} as Event);
    expect(got!.connected).toBe(false);
    off();
  });

  it("hands the current summary to a subscriber that attaches between pushes", () => {
    const a = subscribeJobSummary(() => {});
    FakeES.last!.push(frame({ autolabel: [row("a1", "running", 0.4)] }));
    let got: JobsSummary | null = null;
    const b = subscribeJobSummary((s) => { got = s; });
    expect(got!.running.map((r) => r.job_id)).toEqual(["a1"]);
    a(); b();
  });

  it("a second subscriber does not blank what the first was shown", () => {
    // `subscribeJobs` replays its snapshot on every attach, and before the first frame that snapshot is
    // null. Treating null as an empty summary would clear the chip each time anything else subscribed.
    const a = subscribeJobSummary(() => {});
    FakeES.last!.push(frame({ import: [row("i1", "running")] }));
    let got: JobsSummary | null = null;
    const b = subscribeJobSummary((s) => { got = s; });
    subscribeJobSummary(() => {})();
    expect(got!.running).toHaveLength(1);
    a(); b();
  });

  it("delivers each frame to every subscriber exactly once", () => {
    // One jobs listener serves the whole group. A listener per subscriber would summarise the same frame
    // twice and deliver it four times here, which is the shape that grows quadratically.
    const seen: number[] = [];
    const a = subscribeJobSummary((s) => seen.push(s.running.length));
    const b = subscribeJobSummary((s) => seen.push(s.running.length));
    FakeES.last!.push(frame({ training: [row("t1", "running")], export: [row("e1", "running")] }));
    expect(seen.filter((n) => n === 2)).toHaveLength(2);
    a(); b();
  });

  it("lets go of the underlying stream when the last summary subscriber leaves", () => {
    const a = subscribeJobSummary(() => {});
    const b = subscribeJobSummary(() => {});
    a();
    expect(FakeES.closed).toBe(0);
    b();
    expect(FakeES.closed).toBe(1);
  });
});
