// "New annotation" took exactly one file, and one file is the wrong unit for this footage. A dashcam
// session is a folder of clips and the corpus was built from 186 of them, so ingesting it a file at a time
// means waiting out an upload, a decode and an editor open before choosing the next one.
//
// Most of these tests are about the ways a queue breaks that a single upload never could.
//
// The old flow navigated to the editor the moment its import finished. Correct for one file, destructive for
// a queue: the page unmounts and everything still uploading is abandoned mid-transfer.
//
// A folder pick is not a list of media. It hands over .DS_Store, sidecar json and thumbnails, and uploading
// those produces one failed import per file, burying the real ones under failures that look like the
// importer's fault rather than the picker's.
//
// And the progress bar has to move during a single large video, or a five-minute upload looks like a hang.

import { describe, expect, it } from "vitest";

import {
  type QueueItem,
  buildQueue,
  formatForName,
  humanSize,
  nextPending,
  patchItem,
  shouldAutoOpen,
  summarize,
} from "./uploadQueue";

const f = (name: string, size = 1000, type = "") => ({ name, size, type });

describe("formatForName", () => {
  it("recognises the media this page can ingest", () => {
    expect(formatForName("clip.mp4")).toBe("video");
    expect(formatForName("drive.MOV")).toBe("video");
    expect(formatForName("log.mcap")).toBe("mcap");
    expect(formatForName("frames.zip")).toBe("images");
    expect(formatForName("shot.jpeg")).toBe("images");
  });

  it("falls back to the mime type when the name has no useful extension", () => {
    expect(formatForName("recording", "video/mp4")).toBe("video");
  });

  it("returns null for anything else, rather than guessing", () => {
    // Guessing here means uploading a readme and reporting an import failure for it.
    expect(formatForName("notes.txt")).toBeNull();
    expect(formatForName("labels.json")).toBeNull();
  });
});

describe("buildQueue", () => {
  it("takes many files at once", () => {
    const { items } = buildQueue([f("a.mp4"), f("b.mp4"), f("c.mcap")]);
    expect(items).toHaveLength(3);
    expect(items.map((i) => i.format)).toEqual(["video", "video", "mcap"]);
  });

  it("drops what a folder pick sweeps up, and says what it dropped", () => {
    const { items, skipped } = buildQueue([
      f("clip.mp4"), f(".DS_Store"), f("Thumbs.db"), f("notes.txt"), f("sidecar.json"),
    ]);
    expect(items.map((i) => i.name)).toEqual(["clip.mp4"]);
    expect(skipped).toEqual([".DS_Store", "Thumbs.db", "notes.txt", "sidecar.json"]);
  });

  it("drops a dotfile even when its extension looks like media", () => {
    // macOS resource forks are named ._clip.mp4 and are not video.
    expect(buildQueue([f("._clip.mp4")]).items).toEqual([]);
  });

  it("deduplicates, because dragging the same folder twice is a normal accident", () => {
    const { items, duplicates } = buildQueue([f("a.mp4", 10), f("a.mp4", 10), f("a.mp4", 20)]);
    expect(items).toHaveLength(2);
    expect(duplicates).toBe(1);
  });

  it("gives every item a stable id", () => {
    const { items } = buildQueue([f("a.mp4"), f("b.mp4")]);
    expect(new Set(items.map((i) => i.id)).size).toBe(2);
  });

  it("an empty or entirely unsupported selection produces an empty queue rather than throwing", () => {
    expect(buildQueue([]).items).toEqual([]);
    expect(buildQueue([f("readme.md")]).items).toEqual([]);
  });
});

describe("queue progression", () => {
  const base = () => buildQueue([f("a.mp4"), f("b.mp4"), f("c.mp4")]).items;

  it("hands out one item at a time", () => {
    let items = base();
    const first = nextPending(items)!;
    items = patchItem(items, first.id, { status: "uploading" });
    // Sequential on purpose: each import decodes video on a machine with one GPU slot, so ten at once
    // compete rather than finish sooner.
    expect(nextPending(items)!.id).not.toBe(first.id);
  });

  it("returns nothing once every item has been started", () => {
    let items = base();
    for (const i of items) items = patchItem(items, i.id, { status: "done" });
    expect(nextPending(items)).toBeNull();
  });

  it("patching leaves the other items alone", () => {
    const items = patchItem(base(), "a.mp4:1000", { status: "error", detail: "boom" });
    expect(items.filter((i) => i.status === "pending")).toHaveLength(2);
  });
});

describe("summarize", () => {
  it("counts what is done, failed and still to come", () => {
    let items = buildQueue([f("a.mp4"), f("b.mp4"), f("c.mp4"), f("d.mp4")]).items;
    items = patchItem(items, items[0].id, { status: "done" });
    items = patchItem(items, items[1].id, { status: "error" });
    items = patchItem(items, items[2].id, { status: "uploading", progress: 0.5 });
    const s = summarize(items);
    expect([s.done, s.failed, s.active, s.pending]).toEqual([1, 1, 1, 1]);
  });

  it("moves while a single large file is still uploading", () => {
    // Without partial credit a five-minute upload leaves the bar at 0 and reads as a hang.
    let items = buildQueue([f("big.mp4")]).items;
    items = patchItem(items, items[0].id, { status: "uploading", progress: 0.4 });
    expect(summarize(items).progress).toBeCloseTo(0.4, 5);
  });

  it("is finished only when nothing is left in flight", () => {
    let items = buildQueue([f("a.mp4"), f("b.mp4")]).items;
    items = patchItem(items, items[0].id, { status: "done" });
    expect(summarize(items).finished).toBe(false);
    items = patchItem(items, items[1].id, { status: "error" });
    // A failure still finishes the queue: the run is over, it just did not fully succeed.
    expect(summarize(items).finished).toBe(true);
  });

  it("an empty queue is not finished, it has not started", () => {
    expect(summarize([]).finished).toBe(false);
    expect(summarize([]).progress).toBe(0);
  });

  it("never reports more than complete", () => {
    let items = buildQueue([f("a.mp4")]).items;
    items = patchItem(items, items[0].id, { status: "uploading", progress: 5 });
    expect(summarize(items).progress).toBeLessThanOrEqual(1);
  });
});

describe("shouldAutoOpen", () => {
  const done = (extra: Partial<QueueItem> = {}): QueueItem => ({
    id: "a", name: "a.mp4", size: 1, format: "video", status: "done", progress: 1,
    frameId: "frame-1", ...extra,
  });

  it("opens the editor for a single file, which is the old behaviour", () => {
    expect(shouldAutoOpen([done()])).toBe(true);
  });

  it("never navigates away from a queue", () => {
    // The bug this module exists around: navigating unmounts the page and abandons every upload still in
    // flight. One finished item is not a finished queue.
    expect(shouldAutoOpen([done(), { ...done(), id: "b", status: "pending" }])).toBe(false);
  });

  it("does not open when the single import produced no frames", () => {
    expect(shouldAutoOpen([done({ frameId: undefined })])).toBe(false);
  });

  it("does not open on a failure", () => {
    expect(shouldAutoOpen([done({ status: "error" })])).toBe(false);
  });
});

describe("humanSize", () => {
  it("reads at a glance beside a filename", () => {
    expect(humanSize(512)).toBe("512 B");
    expect(humanSize(1536)).toBe("1.5 KB");
    expect(humanSize(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(humanSize(2_500_000_000)).toBe("2.3 GB");
  });
});
