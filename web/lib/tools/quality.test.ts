import { describe, expect, it } from "vitest";

import {
  boxToScreen, classTotals, fitScale, focusWindow, heatStyle, mergeSummary, rarestShare, roleTone, sceneLine, toggleId,
  verdictLabel, verdictTone,
} from "./quality";

describe("heatStyle", () => {
  it("is blank for an empty cell and picks the hue from the rate", () => {
    expect(heatStyle(null).tone).toBe("muted");
    expect(heatStyle({ accepted: 9, disputed: 1, dropped: 0, rate: 0.9 }).tone).toBe("pass");
    expect(heatStyle({ accepted: 3, disputed: 2, dropped: 0, rate: 0.6 }).tone).toBe("warn");
    expect(heatStyle({ accepted: 1, disputed: 4, dropped: 0, rate: 0.2 }).tone).toBe("block");
  });
  it("is fainter when fewer clusters were decided", () => {
    const one = heatStyle({ accepted: 1, disputed: 0, dropped: 0, rate: 1 }).background;
    const many = heatStyle({ accepted: 20, disputed: 0, dropped: 0, rate: 1 }).background;
    expect(one).toBe("rgb(var(--c-pass) / 0.19)");
    expect(many).toBe("rgb(var(--c-pass) / 0.64)");
  });
});

describe("classTotals", () => {
  it("sums each class across groups and ignores dropped boxes in the rate", () => {
    const t = classTotals([[{ accepted: 2, disputed: 1, dropped: 3, rate: 0.67 }, null,
      { accepted: 1, disputed: 0, dropped: 0, rate: 1 }], [null, null]]);
    expect(t[0]).toEqual({ accepted: 3, disputed: 1, dropped: 3, rate: 0.75 });
    expect(t[1].rate).toBeNull();
  });
});

describe("geometry", () => {
  it("maps image boxes through the editor viewport", () => {
    expect(boxToScreen([10, 20, 110, 70], { scale: 0.5, ox: 100, oy: 40 })).toEqual({ x: 105, y: 50, w: 50, h: 25 });
    expect(boxToScreen([0, 0, 10, 10], { scale: NaN, ox: 0, oy: 0 }).w).toBe(10);
  });
  it("fits without enlarging", () => {
    expect(fitScale(1920, 1080, 480, 480)).toBeCloseTo(0.25);
    expect(fitScale(100, 100, 480, 480)).toBe(1);
  });
  it("frames the evidence with padding inside the image", () => {
    expect(focusWindow([[100, 100, 200, 200], [180, 150, 260, 220]], 1920, 1080, 0.5)).toEqual([20, 40, 340, 280]);
    expect(focusWindow([], 640, 480)).toEqual([0, 0, 640, 480]);
    expect(focusWindow([[0, 0, 10, 10]], 640, 480)[0]).toBe(0);
  });
});

describe("labels", () => {
  it("names verdicts and roles", () => {
    expect(verdictTone("incorrect")).toBe("block");
    expect(verdictTone(undefined)).toBe("muted");
    expect(verdictLabel({ verdict: "incorrect", alternative: "e_auto" })).toBe("Looks wrong, maybe e auto");
    expect(verdictLabel({ verdict: "unsure", alternative: null })).toBe("Not sure");
    expect(roleTone("subject")).toBe("block");
    expect(roleTone("overlaps")).toBe("warn");
  });
  it("summarises a merge", () => {
    expect(mergeSummary({ accepted: 3, disputed: 1, dropped: 2, created: 3 }))
      .toBe("3 agreed, 1 sent to review, 2 model-only boxes nobody drew, 3 new consensus labels");
    expect(mergeSummary(null)).toBe("");
  });
  it("toggles a selection immutably and lists scene axes", () => {
    const a = new Set(["x"]);
    const b = toggleId(a, "y");
    expect([...a]).toEqual(["x"]);
    expect([...toggleId(b, "x")]).toEqual(["y"]);
    expect(sceneLine({ time_of_day: "night", weather: "rain", rarity: 0.3 })).toBe("night · rain");
  });
});

describe("rarestShare", () => {
  it("states the share of frames at least as rare", () => {
    expect(rarestShare(0.99997)).toBe("rarest 0.1%");
    expect(rarestShare(0.95)).toBe("rarest 5.0%");
    expect(rarestShare(0.5)).toBe("rarest 50%");
    expect(rarestShare(undefined)).toBe("");
  });
});
