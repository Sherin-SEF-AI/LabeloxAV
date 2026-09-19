import { Rect } from "konva/lib/shapes/Rect";
import { describe, expect, it } from "vitest";

import { amodalRectProps } from "./amodal";

describe("amodal box rendering", () => {
  // A translucent shape with both a fill and a stroke is drawn through an offscreen buffer the size of the
  // layer. The editor's stage mounts at 0 by 0, so that buffer was zero sized and drawImage threw, and any
  // frame carrying an amodal box crashed the editor on open.
  it("never needs Konva's offscreen buffer", () => {
    const rect = new Rect(amodalRectProps([10, 20, 110, 220], "#ff0000", 0.5));
    expect(rect.hasFill()).toBe(false);
    expect(rect.perfectDrawEnabled()).toBe(false);
    expect(rect.hasStroke()).toBe(true);
  });

  it("covers exactly the whole-extent box", () => {
    const p = amodalRectProps([10, 20, 110, 220], "#ff0000", 2);
    expect([p.x, p.y, p.width, p.height]).toEqual([10, 20, 100, 200]);
    expect(p.strokeWidth).toBe(0.5);
  });
});
