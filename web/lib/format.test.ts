import { describe, expect, it } from "vitest";
import { ago, compact, duration, fromNs, humanize, num, pct, plural, shortId } from "./format";

describe("format", () => {
  it("humanizes storage codes", () => {
    expect(humanize("adjust_geometry")).toBe("Adjust geometry");
    expect(humanize("CUT_IN")).toBe("Cut in");
    expect(humanize("two-wheeler")).toBe("Two wheeler");
    expect(humanize("")).toBe("");
    expect(humanize(null)).toBe("");
    expect(humanize("suv")).toBe("SUV");
    expect(humanize("vlm_verify")).toBe("VLM verify");
    expect(humanize("lidar_3d_box")).toBe("LiDAR 3D box");
    expect(humanize("map_fusion")).toBe("Map fusion");
  });

  it("formats fractions as percentages without hiding a running job", () => {
    expect(pct(0.05)).toBe("5%");
    expect(pct(1)).toBe("100%");
    expect(pct(0)).toBe("0%");
    expect(pct(0.003)).toBe("<1%");
    expect(pct(0.8333, 1)).toBe("83.3%");
    expect(pct(null)).toBe("-");
  });

  it("formats counts", () => {
    expect(num(882079)).toBe("882,079");
    expect(compact(882079)).toBe("882k");
    expect(compact(1_250_000)).toBe("1.3M");
    expect(compact(9999)).toBe("9,999");
    expect(plural(1, "frame")).toBe("1 frame");
    expect(plural(3, "frame")).toBe("3 frames");
  });

  it("says how long ago, on one scale everywhere", () => {
    const now = Date.parse("2026-11-25T12:00:00Z");
    expect(ago("2026-11-25T11:59:50Z", now)).toBe("just now");
    expect(ago("2026-11-25T11:56:00Z", now)).toBe("4 min ago");
    expect(ago("2026-11-24T15:00:00Z", now)).toBe("21 h ago");
    expect(ago("2026-11-22T12:00:00Z", now)).toBe("3 days ago");
    expect(ago(null, now)).toBe("-");
    expect(ago("not a date", now)).toBe("-");
  });

  it("reads the frame clock", () => {
    expect(fromNs(null)).toBe("-");
    expect(fromNs("4200000000")).toBe("4.2 s");
    expect(fromNs("1786533661949000000")).toMatch(/Aug/);
  });

  it("formats durations and ids", () => {
    expect(duration(95)).toBe("1 min 35 s");
    expect(duration(3725)).toBe("1 h 2 min");
    expect(shortId("5901b4fa-69d7-4417")).toBe("5901b4fa");
  });
});
