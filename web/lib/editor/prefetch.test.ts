import { describe, expect, it } from "vitest";

import { neighbourImageUrls, prefetchImages } from "./prefetch";

// `[` and `]` each landed on a cold image. The two properties worth pinning are the ones that would be
// expensive to get wrong: it never reaches further than one frame each way (the image endpoint is also
// the access log for frames holding personal data), and it lets go when the page goes away.

const URL = "/api/frames/f-cur/image";

describe("which images are warmed", () => {
  it("is the previous and the next frame's, in the server's own URL shape", () => {
    expect(neighbourImageUrls(URL, "f-cur", "f-prev", "f-next"))
      .toEqual(["/api/frames/f-prev/image", "/api/frames/f-next/image"]);
  });

  it("is never more than one frame in each direction", () => {
    expect(neighbourImageUrls(URL, "f-cur", "f-prev", "f-next").length).toBeLessThanOrEqual(2);
  });

  it("is one at the start or the end of a drive", () => {
    expect(neighbourImageUrls(URL, "f-cur", null, "f-next")).toEqual(["/api/frames/f-next/image"]);
    expect(neighbourImageUrls(URL, "f-cur", "f-prev", undefined)).toEqual(["/api/frames/f-prev/image"]);
  });

  it("is none on a single-frame drive", () => {
    expect(neighbourImageUrls(URL, "f-cur", null, null)).toEqual([]);
  });

  it("does not fetch the same image twice when both neighbours are one frame", () => {
    expect(neighbourImageUrls(URL, "f-cur", "f-x", "f-x")).toEqual(["/api/frames/f-x/image"]);
  });

  it("does not prefetch the frame already on screen", () => {
    expect(neighbourImageUrls(URL, "f-cur", "f-cur", "f-next")).toEqual(["/api/frames/f-next/image"]);
  });

  it("keeps whatever else is in the URL", () => {
    expect(neighbourImageUrls("https://cdn.example/frames/f-cur/image?v=2", "f-cur", null, "f-next"))
      .toEqual(["https://cdn.example/frames/f-next/image?v=2"]);
  });

  it("prefetches nothing rather than guess, when the URL is not built from the frame id", () => {
    expect(neighbourImageUrls("/blobs/9f8e7d.jpg", "f-cur", "f-prev", "f-next")).toEqual([]);
    expect(neighbourImageUrls("", "f-cur", "f-prev", "f-next")).toEqual([]);
    expect(neighbourImageUrls(URL, "", "f-prev", "f-next")).toEqual([]);
  });
});

type Fake = { src: string; crossOrigin: string | null; onload: unknown; onerror: unknown };

function fakes() {
  const made: Fake[] = [];
  const make = () => {
    const im: Fake = { src: "", crossOrigin: null, onload: null, onerror: null };
    made.push(im);
    return im;
  };
  return { made, make };
}

describe("warming them", () => {
  it("starts one request per URL, in the mode the canvas will ask for it in", () => {
    const { made, make } = fakes();
    prefetchImages(["/a", "/b"], make);
    expect(made.map((m) => m.src)).toEqual(["/a", "/b"]);
    // A response cached for a no-cors request is not one a crossOrigin canvas image may reuse.
    expect(made.every((m) => m.crossOrigin === "anonymous")).toBe(true);
  });

  it("starts nothing for an empty list", () => {
    const { made, make } = fakes();
    prefetchImages([], make)();
    expect(made).toEqual([]);
  });

  it("lets go on unmount: no handler is left that could run against a page that is gone", () => {
    const { made, make } = fakes();
    const cancel = prefetchImages(["/a", "/b"], make);
    expect(made.every((m) => typeof m.onload === "function" && typeof m.onerror === "function")).toBe(true);
    cancel();
    expect(made.every((m) => m.onload === null && m.onerror === null)).toBe(true);
  });

  it("does not abort on unmount, because the next page is about to ask for the same image", () => {
    const { made, make } = fakes();
    prefetchImages(["/a"], make)();
    expect(made[0].src).toBe("/a");
  });

  it("an image that has arrived, or failed, is no longer held", () => {
    const { made, make } = fakes();
    const cancel = prefetchImages(["/a", "/b"], make);
    (made[0].onload as () => void)();
    (made[1].onerror as () => void)();
    // Neither is touched by the cancel: both were already released.
    const [a, b] = [made[0].onload, made[1].onerror];
    cancel();
    expect(made[0].onload).toBe(a);
    expect(made[1].onerror).toBe(b);
  });

  it("copes with an image that reports loaded the moment its source is set", () => {
    const made: Fake[] = [];
    const make = () => {
      const im = { crossOrigin: null, onload: null as unknown, onerror: null,
        set src(_: string) { (this.onload as (() => void) | null)?.(); }, get src() { return ""; } };
      made.push(im);
      return im;
    };
    expect(() => prefetchImages(["/a"], make)()).not.toThrow();
  });

  it("uses a real Image when it is not handed a factory", () => {
    class StubImage { src = ""; crossOrigin: string | null = null; onload: unknown = null; onerror: unknown = null; }
    const g = globalThis as unknown as { Image?: unknown };
    const before = g.Image;
    g.Image = StubImage;
    try {
      expect(() => prefetchImages(["/a"])()).not.toThrow();
    } finally {
      g.Image = before;
    }
  });
});
