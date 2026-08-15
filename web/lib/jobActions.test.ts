// The jobs dashboard offered four filters while the server emitted seven kinds, and could open a job but
// never stop one.
//
// The filter list was hardcoded, so export, map_fusion and relabel rows appeared under "all" and could not
// be isolated. And with one cancel route in the whole API, clearing a wedged auto-label meant an UPDATE
// against the database, which mattered because the start route refuses while any row reads `running`.

import { describe, expect, it } from "vitest";

import { canCancel, cancelPath, jobCounts, jobKinds } from "./jobActions";
import type { JobRow } from "./types";

const job = (over: Partial<JobRow> = {}): JobRow => ({
  job_id: "j1", kind: "autolabel", status: "running", progress: 0.5,
  label: "sess", detail: "", link: "/", error: null,
  created_at: null, updated_at: null, ...over,
});

describe("which filters to offer", () => {
  it("offers every kind that actually arrived, not a list someone remembered to update", () => {
    // The three that were missing: export, map_fusion and relabel are all emitted by /api/jobs.
    const kinds = jobKinds([
      job({ kind: "import" }), job({ kind: "export" }),
      job({ kind: "map_fusion" }), job({ kind: "relabel" }),
    ]);
    expect(kinds).toContain("export");
    expect(kinds).toContain("map_fusion");
    expect(kinds).toContain("relabel");
  });

  it("always starts with all", () => {
    expect(jobKinds([job()])[0]).toBe("all");
  });

  it("puts the busiest kind first, so a rare kind is not the first thing offered", () => {
    const kinds = jobKinds([job({ kind: "import" }), job({ kind: "autolabel" }), job({ kind: "autolabel" })]);
    expect(kinds).toEqual(["all", "autolabel", "import"]);
  });

  it("breaks a tie by name rather than by arrival order, so the row does not reshuffle between ticks", () => {
    expect(jobKinds([job({ kind: "zeta" }), job({ kind: "alpha" })])).toEqual(["all", "alpha", "zeta"]);
  });

  it("offers no dead buttons for kinds this deployment never runs", () => {
    expect(jobKinds([job({ kind: "import" })])).toEqual(["all", "import"]);
  });

  it("an empty list is just all", () => {
    expect(jobKinds([])).toEqual(["all"]);
  });
});

describe("what can be cancelled", () => {
  it("running work can be stopped", () => {
    expect(canCancel(job({ status: "running" }))).toBe(true);
  });

  it("queued work can be stopped", () => {
    expect(canCancel(job({ status: "pending" }))).toBe(true);
  });

  it("work parked for a pod can be stopped, which is the point for the 67 of them", () => {
    expect(canCancel(job({ status: "queued-cloud" }))).toBe(true);
  });

  it("finished work is left alone", () => {
    for (const status of ["done", "error", "canceled", "cancelled"]) {
      expect(canCancel(job({ status }))).toBe(false);
    }
  });
});

describe("where the cancel goes", () => {
  it("uses the unified route for the five kinds the API runs itself", () => {
    expect(cancelPath(job({ kind: "autolabel", job_id: "abc" }))).toBe("/api/jobs/autolabel/abc/cancel");
    expect(cancelPath(job({ kind: "map_fusion", job_id: "abc" }))).toBe("/api/jobs/map_fusion/abc/cancel");
  });

  it("sends training to its own route, which asks the worker rather than writing the row", () => {
    expect(cancelPath(job({ kind: "training", job_id: "abc" }))).toBe("/api/training/abc/cancel");
  });
});

describe("the header count", () => {
  it("keeps running, queued and parked apart", () => {
    // One number called "active" counted all three. On this deployment that reads 68 on an idle machine,
    // which is how a status line stops being read.
    const c = jobCounts([
      job({ status: "running" }),
      job({ status: "pending" }), job({ status: "pending" }),
      job({ status: "queued-cloud" }), job({ status: "queued-cloud" }), job({ status: "queued-cloud" }),
    ]);
    expect(c).toEqual({ running: 1, queued: 2, parked: 3 });
  });

  it("counts no finished job as anything", () => {
    expect(jobCounts([job({ status: "done" }), job({ status: "error" }), job({ status: "canceled" })]))
      .toEqual({ running: 0, queued: 0, parked: 0 });
  });

  it("an empty board is three zeroes, not an error", () => {
    expect(jobCounts([])).toEqual({ running: 0, queued: 0, parked: 0 });
  });
});
