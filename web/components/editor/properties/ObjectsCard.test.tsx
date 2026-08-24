import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { EdObject } from "../useEditor";
import ObjectsCard from "./ObjectsCard";
import SelectionFilters from "./SelectionFilters";

// This list came out of a 2,318-line page with no coverage at all, and every row control is a two-line
// handler that keeps rendering perfectly after it stops firing. The assertions that matter are therefore
// about dispatch payloads, not about markup.
//
// Two of them guard `stopPropagation`. The hide, lock and delete buttons sit inside a row whose own click
// selects. Drop the stopPropagation and every one of them still works, and also silently changes the
// selection under the annotator, which on a multi-select is destructive and invisible.

vi.mock("@/lib/colors", () => ({ classColor: () => "#888" }));

function obj(id: string, over: Partial<EdObject> = {}): EdObject {
  return {
    id, class_id: 1, class_name: "sedan", bbox: [0, 0, 10, 10], mask: [], attrs: {},
    conf: 0.9, state: "review", visible: true, ...over,
  };
}

let dispatch: ReturnType<typeof vi.fn>;
beforeEach(() => { dispatch = vi.fn(); });

const card = (objects: EdObject[], selectedIds: string[] = []) => (
  <ObjectsCard objects={objects} selectedIds={selectedIds} dispatch={dispatch}
    filters={<SelectionFilters dispatch={dispatch} />} />
);

describe("selection", () => {
  it("a plain click selects exactly one object", async () => {
    render(card([obj("aaaaaaaa11"), obj("bbbbbbbb22")]));
    await userEvent.click(screen.getByText("aaaaaaaa"));
    expect(dispatch).toHaveBeenCalledWith({ t: "select", id: "aaaaaaaa11" });
  });

  it("ctrl, meta and shift each extend the selection instead of replacing it", () => {
    // All three, because the handler tests them with || and dropping one branch leaves the other two
    // working. Mac users reach for Meta and everyone else for Control, so a missing branch is invisible
    // to whoever wrote it.
    render(card([obj("aaaaaaaa11")]));
    const row = screen.getByText("aaaaaaaa");
    for (const mod of ["ctrlKey", "metaKey", "shiftKey"] as const) {
      dispatch.mockClear();
      fireEvent.click(row, { [mod]: true });
      expect(dispatch).toHaveBeenCalledWith({ t: "toggleSelect", id: "aaaaaaaa11" });
    }
  });
});

describe("row controls do not also change the selection", () => {
  it("hide toggles visibility and nothing else", async () => {
    render(card([obj("aaaaaaaa11", { visible: true })]));
    await userEvent.click(screen.getByRole("button", { name: "hide object" }));
    expect(dispatch).toHaveBeenCalledWith({ t: "setVisible", ids: ["aaaaaaaa11"], visible: false });
    expect(dispatch).toHaveBeenCalledTimes(1);
  });

  it("show is the same control with the opposite payload", async () => {
    render(card([obj("aaaaaaaa11", { visible: false })]));
    await userEvent.click(screen.getByRole("button", { name: "show object" }));
    expect(dispatch).toHaveBeenCalledWith({ t: "setVisible", ids: ["aaaaaaaa11"], visible: true });
  });

  it("lock toggles the lock and nothing else", async () => {
    render(card([obj("aaaaaaaa11")]));
    await userEvent.click(screen.getByRole("button", { name: "lock object" }));
    expect(dispatch).toHaveBeenCalledWith({ t: "setLocked", ids: ["aaaaaaaa11"], locked: true });
    expect(dispatch).toHaveBeenCalledTimes(1);
  });

  it("delete removes that object and nothing else", async () => {
    render(card([obj("aaaaaaaa11")]));
    await userEvent.click(screen.getByRole("button", { name: "delete object" }));
    expect(dispatch).toHaveBeenCalledWith({ t: "delete", id: "aaaaaaaa11" });
    expect(dispatch).toHaveBeenCalledTimes(1);
  });

  it("a locked object cannot be deleted from the list", async () => {
    // Locking exists to pin work somebody has already checked. A delete button that still fires on a
    // locked row makes the lock decorative.
    render(card([obj("aaaaaaaa11", { locked: true })]));
    const del = screen.getByRole("button", { name: "delete object" });
    expect(del).toBeDisabled();
    await userEvent.click(del);
    expect(dispatch).not.toHaveBeenCalled();
  });
});

describe("search and grouping", () => {
  it("filters on class name and on object id", async () => {
    render(card([obj("c11881d9xx", { class_name: "autorickshaw" }), obj("77777777yy", { class_name: "sedan" })]));
    const box = screen.getByLabelText("search objects");
    await userEvent.type(box, "c1188");
    expect(screen.getByText("c11881d9")).toBeInTheDocument();
    expect(screen.queryByText("77777777")).not.toBeInTheDocument();
    await userEvent.clear(box);
    await userEvent.type(box, "sedan");
    expect(screen.getByText("77777777")).toBeInTheDocument();
  });

  it("says the frame is empty and how to fix it, rather than showing a blank list", () => {
    render(card([]));
    expect(screen.getByText(/draw a box \(B\)/)).toBeInTheDocument();
  });

  it("distinguishes an empty frame from a query that matched nothing", async () => {
    render(card([obj("aaaaaaaa11")]));
    await userEvent.type(screen.getByLabelText("search objects"), "tanker");
    expect(screen.getByText(/no object matches that/)).toBeInTheDocument();
    expect(screen.queryByText(/draw a box/)).not.toBeInTheDocument();
  });

  it("collapsing a class group hides its rows and keeps the header", async () => {
    render(card([obj("aaaaaaaa11", { class_name: "sedan" })]));
    const header = screen.getByRole("button", { name: /sedan/i });
    await userEvent.click(header);
    expect(screen.queryByText("aaaaaaaa")).not.toBeInTheDocument();
    await userEvent.click(header);
    expect(screen.getByText("aaaaaaaa")).toBeInTheDocument();
  });
});

describe("row badges", () => {
  it("shows the mask diamond only for objects that carry one", () => {
    render(card([obj("aaaaaaaa11", { mask: [[0, 0, 1, 1]] }), obj("bbbbbbbb22")]));
    expect(screen.getAllByTitle("has mask")).toHaveLength(1);
  });

  it("shows a quality dot only when a quality score exists, toned on the shared thresholds", () => {
    render(card([
      obj("aaaaaaaa11", { quality_score: 0.5 }),
      obj("bbbbbbbb22", { quality_score: 0.1 }),
      obj("cccccccc33"),
    ]));
    expect(screen.getByTitle("label quality 0.50").className).toContain("bg-pass");
    expect(screen.getByTitle("label quality 0.10").className).toContain("bg-block");
    // Two dots for three objects: the one with no score gets none rather than a grey "unknown" dot,
    // which would read as a measured bad score.
    expect(screen.getAllByTitle(/label quality/)).toHaveLength(2);
  });

  it("marks a locally drawn object as new rather than showing a temporary id", () => {
    render(card([obj("tmp-3", { isNew: true })]));
    // Scoped to the row, because "new" is also one of the selection chips underneath.
    const row = screen.getByRole("button", { name: "delete object" }).closest("div");
    expect(row).toHaveTextContent("new");
    expect(screen.queryByText("tmp-3")).not.toBeInTheDocument();
  });
});

describe("the low-conf figure in the header", () => {
  it("counts on the same threshold the selection chip selects on", async () => {
    // If these two ever disagree the header says one number and the chip picks a different set, and
    // nothing on screen explains the gap.
    render(card([obj("a1", { conf: 0.4 }), obj("b2", { conf: 0.49 }), obj("c3", { conf: 0.5 })]));
    expect(screen.getByText("2 low conf")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "conf < 0.5" }));
    expect(dispatch).toHaveBeenCalledWith({ t: "selectBy", how: "lowConf", value: 0.5 });
  });

  it("is hidden when every object is confident", () => {
    render(card([obj("a1", { conf: 0.99 })]));
    expect(screen.queryByText(/low conf/)).not.toBeInTheDocument();
  });
});

describe("the card collapses", () => {
  it("hides the list but keeps the count visible", async () => {
    render(card([obj("aaaaaaaa11")]));
    await userEvent.click(screen.getByRole("button", { name: /objects/i }));
    expect(screen.queryByText("aaaaaaaa")).not.toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
  });
});
