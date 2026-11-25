import { describe, expect, it } from "vitest";
import type { ActivityEvent } from "./types";
import { groupActivity, sentence, verbLabel, windowName } from "./activityFeed";

const ev = (id: string, summary: string, verb = "reviewed", user = "admin", href: string | null = `/frame/${id}`): ActivityEvent => ({
  event_id: id, user_id: null, user_name: user, verb, label: "reviewed an object", subject_type: "object",
  subject_id: id, summary, href, meta: {}, created_at: "2026-11-25T10:00:00Z",
});

describe("activity feed", () => {
  it("turns review codes into sentences", () => {
    expect(sentence(ev("1", "revert cattle"))).toBe("Reverted a cattle label");
    expect(sentence(ev("1", "adjust_geometry bus"))).toBe("Adjusted the shape of a bus label");
    expect(sentence(ev("1", "accept traffic_sign"))).toBe("Accepted a traffic sign label");
    expect(sentence(ev("1", "revert auto_rickshaw"))).toBe("Reverted an auto rickshaw label");
    expect(sentence(ev("1", "confirm 7 objects"))).toBe("Confirmed 7 objects");
    expect(sentence(ev("1", "accept suv"))).toBe("Accepted an SUV label");
    expect(sentence(ev("1", "revert lcv"))).toBe("Reverted an LCV label");
    expect(sentence(ev("1", "revert bus"))).toBe("Reverted a bus label");
    expect(sentence(ev("1", "revert umbrella"))).toBe("Reverted an umbrella label");
    expect(sentence(ev("1", "reclassify 3 objects as two_wheeler"))).toBe("Reclassified 3 objects as two wheeler");
  });

  it("capitalises free-text summaries and humanizes bare codes", () => {
    expect(sentence(ev("1", "took over editing a frame from meera", "took_over_frame"))).toBe("Took over editing a frame from meera");
    expect(sentence({ summary: null, label: "took_over_frame", verb: "took_over_frame" })).toBe("Took over frame");
  });

  it("folds consecutive identical events into one counted line", () => {
    const lines = groupActivity([
      ev("a", "confirm 7 objects", "confirmed", "admin", "/review/grid?ids=7"),
      ...Array.from({ length: 14 }, (_, i) => ev(`r${i}`, "revert cattle")),
      ev("x", "revert cattle", "reviewed", "meera"),
      ev("b", "confirm 7 objects", "confirmed", "admin", null),
      ev("c", "confirm 7 objects", "confirmed", "admin", null),
    ]);
    expect(lines.map((l) => l.text)).toEqual([
      "Confirmed 7 objects", "Reverted 14 cattle labels", "Reverted a cattle label", "Confirmed 14 objects in 2 batches",
    ]);
    expect(lines[1].count).toBe(14);
    expect(lines[1].frames).toBe(14);
    expect(lines[1].first.event_id).toBe("r0");
    expect(lines[1].last.event_id).toBe("r13");
  });

  it("labels verbs and windows", () => {
    expect(verbLabel("took_over_frame", "took_over_frame")).toBe("Took over frame");
    expect(verbLabel("reviewed", "reviewed an object")).toBe("Reviewed an object");
    expect(windowName(24)).toBe("day");
    expect(windowName(72)).toBe("3 days");
  });
});
