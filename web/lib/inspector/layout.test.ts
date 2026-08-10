// Panels could not be arranged. They rendered into `grid-cols-2 auto-rows-[minmax(200px,1fr)]`, so a camera
// got the same area as a one-line raw message, the 3D panel was clipped by a cell sized for a plot, and the
// order was whatever order they had been added in.
//
// The model is two integers per panel, a column span and a row span, with list order as layout order. These
// tests are about the parts that are easy to get subtly wrong: a reorder that scrambles instead of moving, a
// clamp that lets a panel escape the grid, and an old saved layout that must reopen looking the way it did
// when it was saved rather than rearranged by a feature that did not exist yet.

import { describe, expect, it } from "vitest";

import {
  type Arranged,
  MAX_ROWS,
  MAX_SPAN,
  MIN_ROWS,
  MIN_SPAN,
  cycleSpan,
  fromSaved,
  movePanel,
  panelStyle,
  rowsOf,
  setRows,
  setSpan,
  spanOf,
  toSaved,
} from "./layout";

const P = (id: string, extra: Partial<Arranged> = {}): Arranged => ({ id, type: "image", ...extra });

describe("movePanel", () => {
  it("moves a panel and closes the gap behind it", () => {
    const out = movePanel([P("a"), P("b"), P("c")], 2, 0);
    expect(out.map((p) => p.id)).toEqual(["c", "a", "b"]);
  });

  it("does not swap", () => {
    // A swap would send whatever was first to the end and scramble an arrangement somebody just built.
    const out = movePanel([P("a"), P("b"), P("c"), P("d")], 0, 2);
    expect(out.map((p) => p.id)).toEqual(["b", "c", "a", "d"]);
  });

  it("returns the same list when nothing moves", () => {
    const before = [P("a"), P("b")];
    expect(movePanel(before, 1, 1)).toBe(before);
  });

  it("ignores an index off the end rather than dropping a panel", () => {
    const before = [P("a"), P("b")];
    expect(movePanel(before, 0, 9)).toBe(before);
    expect(movePanel(before, -1, 0)).toBe(before);
  });

  it("does not mutate the input", () => {
    const before = [P("a"), P("b"), P("c")];
    movePanel(before, 0, 2);
    expect(before.map((p) => p.id)).toEqual(["a", "b", "c"]);
  });
});

describe("spans and rows", () => {
  it("defaults to one cell, which is how every existing panel looks today", () => {
    expect(spanOf(P("a"))).toBe(1);
    expect(rowsOf(P("a"))).toBe(1);
  });

  it("clamps a span to the grid", () => {
    expect(spanOf(P("a", { span: 99 }))).toBe(MAX_SPAN);
    expect(spanOf(P("a", { span: 0 }))).toBe(MIN_SPAN);
  });

  it("clamps height so a panel cannot become the only thing on screen", () => {
    expect(rowsOf(P("a", { rows: 50 }))).toBe(MAX_ROWS);
    expect(rowsOf(P("a", { rows: -3 }))).toBe(MIN_ROWS);
  });

  it("gives a corrupt value the smallest panel, not the largest", () => {
    // NaN and Infinity both mean "this value is not trustworthy". Clamping them upward would hand the
    // biggest cell on screen to whichever panel had the most broken state, which is backwards.
    expect(spanOf(P("a", { span: NaN }))).toBe(MIN_SPAN);
    expect(rowsOf(P("a", { rows: Infinity }))).toBe(MIN_ROWS);
    expect(rowsOf(P("a", { rows: NaN }))).toBe(MIN_ROWS);
  });

  it("changes only the panel asked for", () => {
    const out = setSpan([P("a"), P("b")], "b", 2);
    expect(spanOf(out[0])).toBe(1);
    expect(spanOf(out[1])).toBe(2);
  });

  it("setting an unknown id is a no-op, not a crash", () => {
    expect(setRows([P("a")], "nope", 3).map(rowsOf)).toEqual([1]);
  });

  it("cycles between one column and full width", () => {
    let ps = [P("a")];
    ps = cycleSpan(ps, "a");
    expect(spanOf(ps[0])).toBe(MAX_SPAN);
    ps = cycleSpan(ps, "a");
    expect(spanOf(ps[0])).toBe(MIN_SPAN);
  });
});

describe("panelStyle", () => {
  it("places a panel by span", () => {
    expect(panelStyle(P("a", { span: 2, rows: 3 }))).toEqual({ gridColumn: "span 2", gridRow: "span 3" });
  });
});

describe("persistence", () => {
  it("saves the arrangement but not the render id", () => {
    // `id` is a render-time counter; persisting it would make two layouts collide as soon as both opened.
    const saved = toSaved(P("p7", { span: 2, rows: 2, topic: "/imu/accel" }));
    expect(saved).toEqual({ type: "image", topic: "/imu/accel", field: undefined, span: 2, rows: 2 });
    expect("id" in saved).toBe(false);
  });

  it("gives every restored panel a fresh id", () => {
    let n = 0;
    const out = fromSaved([P("x"), P("y")], () => `p${++n}`);
    expect(out.map((p) => p.id)).toEqual(["p1", "p2"]);
  });

  it("reopens a layout saved before spans existed exactly as it looked", () => {
    // Those panels have no span or rows at all. Defaulting them to anything other than one cell would
    // rearrange a layout somebody saved, by a feature that did not exist when they saved it.
    const out = fromSaved([{ id: "", type: "image" }, { id: "", type: "map" }], () => "p1");
    expect(out.map(spanOf)).toEqual([1, 1]);
    expect(out.map(rowsOf)).toEqual([1, 1]);
  });
});
