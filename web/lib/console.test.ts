// The application could not tell you whether anything was happening inside it.
//
// Job state was on one page, the machine was in a terminal, and background work was in a log file. So the
// honest answer to "why is this slow" needed three places, two of which a reviewer does not have. These are
// the rules the console reads by: what counts as work, when a reading deserves attention, and how to say a
// number so somebody can take it in at a glance.

import { describe, expect, it } from "vitest";

import { headline, level, mb, uptime, workInFlight } from "./console";
import type { JobStream, SystemStream } from "./useEventStream";

const frame = (over: Partial<JobStream> = {}): JobStream => ({
  training: [], import: [], export: [], autolabel: [], ingest: [], ...over,
});

const row = (job_id: string, status: string, progress: number | null = 0.5) =>
  ({ job_id, status, progress } as JobStream["import"][number]);

const sys = (over: Partial<SystemStream> = {}): SystemStream => ({
  ts: 0,
  gpus: [],
  host: {
    cpu_pct: 10, cpu_count: 8, load: { "1m": 1, "5m": 1, "15m": 1 },
    memory_used_mb: 8000, memory_total_mb: 32000, memory_used_frac: 0.25,
    disk: { path: "/x", used_mb: 100, total_mb: 1000, used_frac: 0.1 },
  },
  process: { pid: 1, rss_mb: 100, cpu_pct: 1, threads: 4, uptime_s: 60 },
  ...over,
});

const gpu = (over = {}) => ({
  index: 0, name: "RTX 5080", memory_used_mb: 6000, memory_total_mb: 16000,
  memory_used_frac: 0.37, utilization_pct: 3, temperature_c: 34, power_w: 20,
  processes: [], ...over,
});

describe("what counts as work", () => {
  it("collects running jobs across every kind", () => {
    const { running } = workInFlight(frame({
      training: [row("t", "running")], autolabel: [row("a", "running")],
      import: [row("i", "running")], export: [row("e", "running")],
    }));
    expect(running.map((r) => r.kind).sort()).toEqual(["autolabel", "export", "import", "training"]);
  });

  it("keeps queued work apart from running work", () => {
    const { running, waiting } = workInFlight(frame({
      autolabel: [row("a", "running"), row("b", "pending"), row("c", "queued-cloud")],
    }));
    expect(running).toHaveLength(1);
    expect(waiting).toHaveLength(2);
  });

  it("reports no progress as null rather than zero", () => {
    // A bar pinned at 0% reads as stalled work. It is unmeasured work, which is a different thing.
    const { running } = workInFlight(frame({ import: [row("i", "running", null)] }));
    expect(running[0].progress).toBeNull();
  });

  it("clamps a progress value rather than trusting it", () => {
    const { running } = workInFlight(frame({ import: [row("i", "running", 4)] }));
    expect(running[0].progress).toBe(1);
  });

  it("does not treat the ingest summary as a job", () => {
    const { running } = workInFlight(frame({
      ingest: [{ source: "import_job", active: true, finished: false, done: 1, total: 2,
                 current: null, frames: 10, progress: 0.5 }],
    }));
    expect(running).toEqual([]);
  });

  it("an absent frame is empty rather than an error", () => {
    expect(workInFlight(null)).toEqual({ running: [], waiting: [], waitingTotal: 0 });
  });

  it("uses the server's queued total rather than counting a windowed list", () => {
    // The lists are a recent tail per kind: 67 jobs parked for a GPU pod are all older than the ten most
    // recent, so counting them says 1. The top-bar chip already uses the totals, and a console that
    // disagrees with the chip about how much is queued is worse than either number alone.
    const w = workInFlight(frame({
      training: [row("t", "pending")],
      waiting: { training: 1, autolabel: 67 },
    }));
    expect(w.waiting).toHaveLength(1);
    expect(w.waitingTotal).toBe(68);
  });

  it("falls back to counting the window when the server sends no total", () => {
    expect(workInFlight(frame({ training: [row("t", "pending")] })).waitingTotal).toBe(1);
  });
});

describe("when a reading deserves attention", () => {
  it("is quiet well below the line", () => {
    expect(level(0.4)).toBe("ok");
  });

  it("warns before it is too late to act", () => {
    expect(level(0.85)).toBe("warn");
  });

  it("escalates near full", () => {
    // The disk on this machine sits at 91%, which is the reading that should be visible before an export
    // fails rather than after.
    expect(level(0.93)).toBe("critical");
  });

  it("treats an unavailable reading as quiet, not as alarming", () => {
    expect(level(null)).toBe("ok");
    expect(level(undefined)).toBe("ok");
  });
});

describe("numbers a person can read", () => {
  it("turns megabytes into gigabytes past a thousand", () => {
    expect(mb(16303)).toBe("16.3 GB");
    expect(mb(880)).toBe("880 MB");
  });

  it("says nothing rather than zero when there is no reading", () => {
    expect(mb(null)).toBe("-");
  });

  it("scales uptime to the unit that fits", () => {
    expect(uptime(30)).toBe("30s");
    expect(uptime(90)).toBe("1m");
    expect(uptime(3700)).toBe("1h 1m");
    expect(uptime(90000)).toBe("1d 1h");
  });
});

describe("the one-line answer", () => {
  it("names what is running and what kind it is", () => {
    expect(headline(frame({ autolabel: [row("a", "running")] }), sys()))
      .toContain("1 running (autolabel)");
  });

  it("says who holds the GPU when the app itself is idle", () => {
    // The most common reason work will not start here, and invisible from every other surface: the card can
    // be full while this application is doing nothing at all.
    const h = headline(frame(), sys({
      gpus: [gpu({ memory_used_frac: 0.37,
                   processes: [{ pid: 1, name: "/usr/local/lib/ollama/llama-server", used_mb: 5074 }] })],
    }));
    expect(h).toContain("llama-server");
    expect(h).toContain("5.1 GB");
  });

  it("does not blame our own process for holding the GPU", () => {
    const h = headline(frame(), sys({
      gpus: [gpu({ processes: [{ pid: 1, name: ".venv/bin/python", used_mb: 880 }] })],
    }));
    expect(h).not.toContain("python");
  });

  it("reports the true queue size when nothing is running", () => {
    expect(headline(frame({ autolabel: [row("a", "pending")], waiting: { autolabel: 68 } }), sys()))
      .toContain("68 queued");
  });

  it("says idle when it is idle", () => {
    expect(headline(frame(), sys())).toBe("idle");
  });

  it("survives having no system reading at all", () => {
    expect(headline(frame(), null)).toBe("idle");
  });
});
