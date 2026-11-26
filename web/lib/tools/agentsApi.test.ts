import { describe, expect, it } from "vitest";
import { approvalState, editTable, leftAloneText, validHHMM } from "./agentsApi";

describe("editTable", () => {
  it("orders class columns by total and puts the rest in other", () => {
    const t = editTable({
      by_class: { pickup: 10, lcv: 2, cyclist: 5 },
      by_session: [
        { session_id: "a", classes: { pickup: 6, lcv: 2 }, total: 8 },
        { session_id: "b", classes: { pickup: 4, cyclist: 5 }, total: 9 },
      ],
    }, 2);
    expect(t.classes).toEqual(["pickup", "cyclist"]);
    expect(t.rows[0].cells).toEqual([6, 0]);
    expect(t.rows[0].other).toBe(2);
    expect(t.rows[1].other).toBe(0);
  });
});

describe("leftAloneText", () => {
  it("says what is left alone and nothing when nothing is", () => {
    expect(leftAloneText({ human_confirmed: 1, category_change: 0, already_target: 3 }))
      .toBe("Left alone: 1 human-confirmed label, 3 already that class.");
    expect(leftAloneText({ human_confirmed: 0, category_change: 0, already_target: 0 })).toBeNull();
    expect(leftAloneText(undefined)).toBeNull();
  });
});

describe("approvalState", () => {
  it("tells applied from undone", () => {
    expect(approvalState({ status: "pending", run: null }).tone).toBe("warn");
    expect(approvalState({ status: "approved", run: { run_id: "r", status: "committed", edited: 3, reverted_at: null } }).label)
      .toBe("Approved and applied");
    expect(approvalState({ status: "approved", run: { run_id: "r", status: "reverted", edited: 3, reverted_at: "x" } }).label)
      .toBe("Applied, then undone");
    expect(approvalState({ status: "failed", run: null }).tone).toBe("block");
  });
});

describe("validHHMM", () => {
  it("accepts a time of day only", () => {
    expect(validHHMM("20:00")).toBe(true);
    expect(validHHMM("9:30")).toBe(true);
    expect(validHHMM("24:00")).toBe(false);
    expect(validHHMM("20")).toBe(false);
  });
});
