// "where i cant see", "still not showing global indicator", "look where it is?"
//
// Three reports, all the same thing: the chip rendered null whenever nothing was running, so on a quiet
// system there was nothing in the top bar to find. Correct by its own rules and useless as a control, since
// the question "is anything happening" cannot be answered by an element that only exists when it is.

import { describe, expect, it } from "vitest";

import { indicatorView } from "./indicatorState";

const base = { running: [], waiting: 0, uploading: [], localRunning: false };

describe("there is always a chip", () => {
  it("says idle when nothing is running and nothing is queued", () => {
    const v = indicatorView({ ...base });
    expect(v.kind).toBe("idle");
    expect(v.verb).toBe("idle");
  });

  it("says queued when the system is holding work it is not doing", () => {
    // 67 autolabel jobs parked for a cloud A100 and one pending training job. This is the state the reports
    // were filed in, and the chip showed nothing at all.
    const v = indicatorView({ ...base, waiting: 68 });
    expect(v.kind).toBe("queued");
    expect(v.count).toBe(68);
  });

  it("never calls queued work running, however much of it there is", () => {
    // A chip reading "68 running" forever teaches people to ignore the chip.
    expect(indicatorView({ ...base, waiting: 68 }).kind).not.toBe("working");
  });

  it("gives every state somewhere to click, so the chip is always a way in", () => {
    for (const v of [indicatorView({ ...base }),
                     indicatorView({ ...base, waiting: 3 }),
                     indicatorView({ ...base, running: [{ kind: "import", progress: 0.5 }] })]) {
      expect(v.href).toBeTruthy();
      expect(v.tip).toContain("click");
    }
  });

  it("has no bar to show when there is no work in flight", () => {
    // A bar at zero percent looks like stalled work rather than absent work.
    expect(indicatorView({ ...base }).pct).toBeNull();
    expect(indicatorView({ ...base, waiting: 5 }).pct).toBeNull();
  });
});

describe("while work is running", () => {
  it("counts server jobs and local uploads together", () => {
    const v = indicatorView({ ...base, running: [{ kind: "import", progress: 0.2 }],
                              uploading: [0.4], localRunning: true });
    expect(v.kind).toBe("working");
    expect(v.count).toBe(2);
  });

  it("averages progress across everything in flight", () => {
    const v = indicatorView({ ...base, running: [{ kind: "import", progress: 0.2 },
                                                 { kind: "training", progress: 0.8 }] });
    expect(v.pct).toBe(50);
  });

  it("never reaches a hundred while something is still going", () => {
    const v = indicatorView({ ...base, running: [{ kind: "import", progress: 1 }] });
    expect(v.pct).toBe(99);
  });

  it("says uploading when only this browser is busy", () => {
    expect(indicatorView({ ...base, uploading: [0.3], localRunning: true }).verb).toBe("uploading");
  });

  it("says labelling when every running job is an autolabel", () => {
    expect(indicatorView({ ...base, running: [{ kind: "autolabel", progress: 0.1 },
                                              { kind: "autolabel", progress: 0.2 }] }).verb).toBe("labelling");
  });

  it("falls back to working for a mixture, rather than naming one of them", () => {
    expect(indicatorView({ ...base, running: [{ kind: "autolabel", progress: 0.1 },
                                              { kind: "training", progress: 0.2 }] }).verb).toBe("working");
  });

  it("keeps the queued count in the tooltip, since the chip has no room for both", () => {
    const v = indicatorView({ ...base, running: [{ kind: "import", progress: 0.5 }], waiting: 68 });
    expect(v.tip).toContain("68 queued and not progressing");
  });

  it("reports this tab's own share when it has one", () => {
    const v = indicatorView({ ...base, uploading: [0.5], localRunning: true, localDone: 3, localTotal: 10 });
    expect(v.tip).toContain("3/10 in this tab");
  });
});

describe("where a click goes", () => {
  it("goes to the upload page while this tab is uploading, which is where that work is visible", () => {
    expect(indicatorView({ ...base, uploading: [0.5], localRunning: true }).href).toBe("/annotate/new");
  });

  it("goes to jobs for server work, since an empty upload page would be a dead end", () => {
    expect(indicatorView({ ...base, running: [{ kind: "autolabel", progress: 0.5 }] }).href).toBe("/jobs");
  });

  it("goes to jobs once this tab's uploads are done even if its queue is still running", () => {
    // The import phase has server jobs, and those are watched on /jobs rather than on the upload page.
    const v = indicatorView({ ...base, running: [{ kind: "import", progress: 0.5 }], localRunning: true });
    expect(v.href).toBe("/jobs");
  });
});
