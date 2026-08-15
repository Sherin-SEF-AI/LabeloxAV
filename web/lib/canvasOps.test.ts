// The canvas ran real work on every interaction and looked idle while it did.
//
// SAM segmentation, mask composition, propagation across frames, drivable inference, an auto-classify on
// each new box, an autosave behind all of it: every one was fire-and-forget with a one-line flash on the way
// out. While a call was in flight nothing said so, and when one was slow there was no way to tell slow from
// stuck from never sent.

import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  beginOp, endOp, getOps, resetOps, subscribeOps, summarizeOps, trackOp, updateOp,
} from "./canvasOps";

beforeEach(() => resetOps());

describe("tracking an operation", () => {
  it("appears as running the moment it starts", () => {
    beginOp("sam", "SAM segmentation");
    expect(getOps()[0]).toMatchObject({ kind: "sam", status: "running" });
  });

  it("keeps its outcome after it finishes", () => {
    // "It succeeded and vanished" and "it never ran" look identical at the moment you look up, and the
    // question about a slow editor is what just happened.
    const id = beginOp("sam", "SAM segmentation");
    endOp(id, "ok", "1 mask");
    expect(getOps()[0]).toMatchObject({ status: "ok", detail: "1 mask" });
  });

  it("records a failure with its reason", () => {
    const id = beginOp("propagate", "propagate forward");
    endOp(id, "failed", "no track on this object");
    expect(getOps()[0]).toMatchObject({ status: "failed", detail: "no track on this object" });
  });

  it("carries progress only when the operation reports it", () => {
    // Most do not, and a made-up bar is worse than none.
    const id = beginOp("propagate", "propagate forward");
    expect(getOps()[0].progress).toBeUndefined();
    updateOp(id, { progress: 0.5 });
    expect(getOps()[0].progress).toBe(0.5);
  });
});

describe("trackOp", () => {
  it("marks success and describes the result", async () => {
    const out = await trackOp("sam", "SAM", async () => ({ n: 3 }), (r) => `${r.n} masks`);
    expect(out).toEqual({ n: 3 });
    expect(getOps()[0]).toMatchObject({ status: "ok", detail: "3 masks" });
  });

  it("marks a failure and rethrows, so the caller still handles it", async () => {
    // The console must not swallow an error the editor needs to show its own way.
    await expect(trackOp("sam", "SAM", async () => { throw new Error("model unavailable"); }))
      .rejects.toThrow("model unavailable");
    expect(getOps()[0]).toMatchObject({ status: "failed", detail: "model unavailable" });
  });

  it("never leaves an operation stuck at running", async () => {
    // Hand-written begin/end pairs leak on every early return somebody adds later, and an operation stuck
    // at running forever is exactly the lie this surface exists to remove.
    await trackOp("a", "a", async () => 1).catch(() => {});
    await trackOp("b", "b", async () => { throw new Error("x"); }).catch(() => {});
    expect(getOps().filter((o) => o.status === "running")).toHaveLength(0);
  });

  it("hands a progress reporter to the operation", async () => {
    await trackOp("propagate", "propagate", async (report) => { report({ progress: 0.7 }); return 0; });
    expect(getOps()[0].progress).toBe(0.7);
  });
});

describe("subscribers", () => {
  it("hear about every change", () => {
    const seen: number[] = [];
    const off = subscribeOps((ops) => seen.push(ops.length));
    const id = beginOp("sam", "SAM");
    endOp(id, "ok");
    off();
    expect(seen.length).toBeGreaterThanOrEqual(3);   // initial, begin, end
  });

  it("get the current state immediately on subscribing", () => {
    beginOp("sam", "SAM");
    let got = -1;
    const off = subscribeOps((ops) => { got = ops.length; });
    off();
    expect(got).toBe(1);
  });

  it("stop hearing after unsubscribing", () => {
    let calls = 0;
    subscribeOps(() => { calls++; })();
    const before = calls;
    beginOp("sam", "SAM");
    expect(calls).toBe(before);
  });
});

describe("the list does not become a log", () => {
  it("drops finished operations once they are old", () => {
    vi.useFakeTimers();
    try {
      const id = beginOp("sam", "SAM");
      endOp(id, "ok");
      vi.setSystemTime(Date.now() + 60_000);
      expect(getOps()).toHaveLength(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("never drops something still running, however long it has been going", () => {
    vi.useFakeTimers();
    try {
      beginOp("propagate", "a long propagate");
      vi.setSystemTime(Date.now() + 600_000);
      expect(getOps()).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("is bounded, so a busy session cannot grow it without limit", () => {
    for (let i = 0; i < 200; i++) beginOp("sam", `op ${i}`);
    expect(getOps().length).toBeLessThanOrEqual(40);
  });
});

describe("the summary on the toggle", () => {
  it("names the single running operation rather than counting it", () => {
    beginOp("sam", "SAM segmentation");
    expect(summarizeOps(getOps()).label).toBe("SAM segmentation");
  });

  it("counts once there is more than one", () => {
    beginOp("sam", "SAM");
    beginOp("save", "saving");
    expect(summarizeOps(getOps()).label).toBe("2 running");
  });

  it("puts a failure ahead of progress", () => {
    // Something that went wrong and was never read is the state this surface exists to prevent.
    const id = beginOp("sam", "SAM");
    endOp(id, "failed", "boom");
    beginOp("save", "saving");
    expect(summarizeOps(getOps()).label).toBe("1 failed");
  });

  it("says idle when nothing is happening", () => {
    expect(summarizeOps([])).toMatchObject({ running: 0, failed: 0, label: "idle" });
  });
});

describe("leaving a frame", () => {
  it("clears the board, so the next canvas is not showing the last one's work", () => {
    beginOp("sam", "SAM");
    resetOps();
    expect(getOps()).toEqual([]);
  });
});
