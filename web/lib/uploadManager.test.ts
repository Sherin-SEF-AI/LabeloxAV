// A 186-file, 41GB batch is over an hour of transfer, and the queue lived in the page's React state, so it
// lived exactly as long as that component stayed mounted. Clicking a row's "open" link, a breadcrumb, or any
// menu item unmounted the page and abandoned everything still uploading. Asking somebody not to touch the
// app for an hour is not a design, and losing the batch because they did is worse.
//
// Moving the store out of React is the fix, and these tests are about the properties that makes possible:
// the run continues while nothing is subscribed, a remount cannot start a second loop over the same items,
// and one failure does not stall the rest.

import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  type UploadDeps,
  clearUploads,
  enqueue,
  enqueueSessions,
  getUploadState,
  startAutolabel,
  startUploads,
  subscribeUploads,
} from "./uploadManager";

// The mime type has to match the name, or the fixture is not the thing being tested: buildQueue falls back
// to the mime when the extension is unknown, so stamping every file "video/mp4" would let a readme through
// on a path a browser would never produce.
const MIME: Record<string, string> = { mp4: "video/mp4", txt: "text/plain", json: "application/json" };
const file = (name: string, size = 1000) =>
  ({ name, size, type: MIME[name.split(".").pop() ?? ""] ?? "" }) as unknown as File;

function deps(over: Partial<UploadDeps> = {}): UploadDeps {
  return {
    upload: async (_f, onProgress) => { onProgress(1); return "s3://bucket/obj"; },
    startImport: async () => ({ job_id: "job-1" }),
    watchImport: async (_id, onProgress) => { onProgress({ progress: 1 }); return "sess-1"; },
    firstFrame: async () => ({ frame_id: "frame-1" }),
    humanizeError: (e) => String((e as Error)?.message ?? e),
    ...over,
  };
}

const TARGET = { format: "auto", vehicle: "ANNO-01", city: "BLR" };

beforeEach(() => {
  clearUploads();
});

describe("enqueue", () => {
  it("accepts a batch and reports what it dropped", () => {
    const r = enqueue([file("a.mp4"), file("b.mp4"), file("readme.txt"), file(".DS_Store")]);
    expect(r.accepted).toBe(2);
    expect(r.skipped).toHaveLength(2);
    expect(getUploadState().items).toHaveLength(2);
  });

  it("refuses to swap the batch underneath a run", async () => {
    enqueue([file("a.mp4")]);
    let released: (() => void) | null = null;
    const gate = new Promise<void>((res) => { released = res; });
    const run = startUploads(TARGET, deps({ upload: async () => { await gate; return "s3://x/y"; } }));

    const r = enqueue([file("z.mp4")]);
    expect(r.accepted).toBe(0);
    expect(getUploadState().items[0].name).toBe("a.mp4");

    released!();
    await run;
  });
});

describe("running", () => {
  it("works every item and records the session it produced", async () => {
    enqueue([file("a.mp4"), file("b.mp4")]);
    await startUploads(TARGET, deps());
    const items = getUploadState().items;
    expect(items.map((i) => i.status)).toEqual(["done", "done"]);
    expect(items[0].sessionId).toBe("sess-1");
    expect(items[0].frameId).toBe("frame-1");
  });

  it("keeps going when one item fails", async () => {
    // One bad clip in a folder of 186 must not strand the other 185.
    let n = 0;
    enqueue([file("a.mp4"), file("b.mp4"), file("c.mp4")]);
    await startUploads(TARGET, deps({
      watchImport: async () => {
        n++;
        if (n === 2) throw new Error("cannot open video");
        return "sess-1";
      },
    }));
    const st = getUploadState().items.map((i) => i.status);
    expect(st).toEqual(["done", "error", "done"]);
    expect(getUploadState().items[1].detail).toContain("cannot open video");
  });

  it("a session with no frames is done, not failed", async () => {
    enqueue([file("a.mp4")]);
    await startUploads(TARGET, deps({ firstFrame: async () => { throw new Error("404"); } }));
    const item = getUploadState().items[0];
    expect(item.status).toBe("done");
    expect(item.frameId).toBeUndefined();
    expect(item.detail).toContain("no frames");
  });

  it("continues while nothing is subscribed", async () => {
    // The whole point. React unmounting is what used to end the run; now it only ends the view.
    enqueue([file("a.mp4"), file("b.mp4")]);
    const unsub = subscribeUploads(() => {});
    const run = startUploads(TARGET, deps());
    unsub();
    await run;
    expect(getUploadState().items.every((i) => i.status === "done")).toBe(true);
  });

  it("a second start is a no-op rather than a second loop", async () => {
    // A remount calling start again must not double-upload the same files.
    const upload = vi.fn(async () => "s3://x/y");
    enqueue([file("a.mp4"), file("b.mp4")]);
    const first = startUploads(TARGET, deps({ upload }));
    await startUploads(TARGET, deps({ upload }));
    await first;
    expect(upload).toHaveBeenCalledTimes(2);
  });

  it("starting an empty queue does nothing", async () => {
    await startUploads(TARGET, deps());
    expect(getUploadState().running).toBe(false);
  });

  it("clears the running flag even when an item throws", async () => {
    enqueue([file("a.mp4")]);
    await startUploads(TARGET, deps({ upload: async () => { throw new Error("network"); } }));
    expect(getUploadState().running).toBe(false);
    expect(getUploadState().items[0].status).toBe("error");
  });
});

describe("subscription", () => {
  it("hands a subscriber the current state immediately", () => {
    enqueue([file("a.mp4")]);
    let seen: number | null = null;
    const unsub = subscribeUploads((s) => { seen = s.items.length; });
    expect(seen).toBe(1);
    unsub();
  });

  it("emits a new object so a reference comparison sees the change", async () => {
    enqueue([file("a.mp4")]);
    const seen: unknown[] = [];
    const unsub = subscribeUploads((s) => seen.push(s));
    await startUploads(TARGET, deps());
    unsub();
    expect(seen.length).toBeGreaterThan(2);
    expect(seen[0]).not.toBe(seen[seen.length - 1]);
  });

  it("stops delivering after unsubscribe", async () => {
    enqueue([file("a.mp4")]);
    let count = 0;
    const unsub = subscribeUploads(() => { count++; });
    const at = count;
    unsub();
    await startUploads(TARGET, deps());
    expect(count).toBe(at);
  });
});

describe("clear", () => {
  it("empties an idle queue", () => {
    enqueue([file("a.mp4")]);
    clearUploads();
    expect(getUploadState().items).toEqual([]);
  });

  it("refuses to empty a running one", async () => {
    enqueue([file("a.mp4")]);
    let released: (() => void) | null = null;
    const gate = new Promise<void>((res) => { released = res; });
    const run = startUploads(TARGET, deps({ upload: async () => { await gate; return "s3://x/y"; } }));
    clearUploads();
    expect(getUploadState().items).toHaveLength(1);
    released!();
    await run;
  });
});


describe("autolabel pass", () => {
  const auto = (over: Partial<UploadDeps> = {}) => deps({
    startAutolabel: async () => ({ job_id: "al-1" }),
    watchAutolabel: async (_id, onProgress) => { onProgress({ progress: 1 }); return { objects: 42 }; },
    ...over,
  });

  it("labels every session the import produced, in queue order", async () => {
    const seen: string[] = [];
    enqueue([file("a.mp4"), file("b.mp4")]);
    await startUploads(TARGET, deps());
    await startAutolabel(auto({
      startAutolabel: async (sid) => { seen.push(sid); return { job_id: "al-1" }; },
    }));
    expect(seen).toHaveLength(2);
    expect(getUploadState().items.map((i) => i.status)).toEqual(["labeled", "labeled"]);
    expect(getUploadState().items[0].labeled).toBe(42);
  });

  it("skips the clips that never imported", async () => {
    // There is no session to label, and inventing one would send autolabel at a null id.
    enqueue([file("a.mp4"), file("b.mp4")]);
    let n = 0;
    await startUploads(TARGET, deps({
      watchImport: async () => { n++; if (n === 1) throw new Error("cannot open video"); return "sess-1"; },
    }));
    const calls: string[] = [];
    await startAutolabel(auto({ startAutolabel: async (s) => { calls.push(s); return { job_id: "x" }; } }));
    expect(calls).toHaveLength(1);
    expect(getUploadState().items.map((i) => i.status)).toEqual(["error", "labeled"]);
  });

  it("a failed labelling leaves the clip imported, not failed", async () => {
    // The session plainly exists. Calling it an error would lose it in the UI over a second-pass problem.
    enqueue([file("a.mp4")]);
    await startUploads(TARGET, deps());
    await startAutolabel(auto({ watchAutolabel: async () => { throw new Error("gpu busy"); } }));
    const item = getUploadState().items[0];
    expect(item.status).toBe("done");
    expect(item.sessionId).toBe("sess-1");
    expect(item.detail).toContain("gpu busy");
  });

  it("one failure does not stop the ones after it", async () => {
    enqueue([file("a.mp4"), file("b.mp4"), file("c.mp4")]);
    await startUploads(TARGET, deps());
    let n = 0;
    await startAutolabel(auto({
      watchAutolabel: async () => { n++; if (n === 1) throw new Error("boom"); return { objects: 5 }; },
    }));
    expect(getUploadState().items.map((i) => i.status)).toEqual(["done", "labeled", "labeled"]);
  });

  it("refuses to start while the import is still running", async () => {
    // Two GPU jobs in flight for no gain, and the ordering is the thing that was asked for.
    enqueue([file("a.mp4")]);
    let released: (() => void) | null = null;
    const gate = new Promise<void>((res) => { released = res; });
    const run = startUploads(TARGET, deps({ upload: async () => { await gate; return "s3://x/y"; } }));
    const calls: string[] = [];
    await startAutolabel(auto({ startAutolabel: async (s) => { calls.push(s); return { job_id: "x" }; } }));
    expect(calls).toEqual([]);
    released!();
    await run;
  });

  it("does nothing when nothing imported", async () => {
    enqueue([file("a.mp4")]);
    await startUploads(TARGET, deps({ watchImport: async () => { throw new Error("no"); } }));
    await startAutolabel(auto());
    expect(getUploadState().items[0].status).toBe("error");
  });

  it("reports which pass is in flight", async () => {
    enqueue([file("a.mp4")]);
    await startUploads(TARGET, deps());
    const phases: string[] = [];
    const unsub = subscribeUploads((s) => phases.push(s.phase));
    await startAutolabel(auto());
    unsub();
    expect(phases).toContain("autolabeling");
    expect(getUploadState().phase).toBe("idle");
  });
});


describe("labelling sessions that were imported earlier", () => {
  const auto = () => deps({
    startAutolabel: async () => ({ job_id: "al-1" }),
    watchAutolabel: async () => ({ objects: 7 }),
  });

  it("loads existing sessions as a queue the same runner can work", async () => {
    // Autolabel used to be reachable only for a queue still in memory, so a reload or yesterday's imports
    // had no batch path at all.
    const n = enqueueSessions([
      { sessionId: "s1", name: "20260606_a.MP4" },
      { sessionId: "s2", name: "20260606_b.MP4" },
    ]);
    expect(n).toBe(2);
    await startAutolabel(auto());
    expect(getUploadState().items.map((i) => i.status)).toEqual(["labeled", "labeled"]);
    expect(getUploadState().items[0].labeled).toBe(7);
  });

  it("keeps the session names, so a row says which drive it is", () => {
    enqueueSessions([{ sessionId: "s1", name: "20260606_a.MP4" }]);
    expect(getUploadState().items[0].name).toBe("20260606_a.MP4");
  });

  it("refuses to load sessions over a batch that is still running", async () => {
    enqueue([file("a.mp4")]);
    let released: (() => void) | null = null;
    const gate = new Promise<void>((res) => { released = res; });
    const run = startUploads(TARGET, deps({ upload: async () => { await gate; return "s3://x/y"; } }));
    expect(enqueueSessions([{ sessionId: "s9", name: "x" }])).toBe(0);
    released!();
    await run;
  });
});
