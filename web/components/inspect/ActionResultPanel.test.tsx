import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import ActionResultPanel, { type ActionResult } from "./ActionResultPanel";

// The panel replaces a one-line summary with the payload the action actually returned, so the tests that
// matter are about what it does NOT drop. The coverage report is the worst previous offender: it returns
// five fields and the page rendered ten strings out of one of them.
//
// The other load-bearing behaviour is that ids become reachable. A uuid in a payload is a thing in the
// corpus, and a result that names one without letting you open it is the same defect the evidence panel
// was built to fix, one level up.

const router = { push: vi.fn() };
vi.mock("next/navigation", () => ({ useRouter: () => router }));

const at = Date.UTC(2026, 7, 20, 10, 0, 0);
const result = (data: unknown, extra: Partial<ActionResult> = {}): ActionResult =>
  ({ kind: "k", label: "coverage report", data, at, ...extra });

describe("ActionResultPanel", () => {
  it("says what it is for when nothing has been run", () => {
    render(<ActionResultPanel result={null} onOpenObject={vi.fn()} onOpenFrame={vi.fn()} />);
    expect(screen.getByText(/run an action/i)).toBeInTheDocument();
  });

  it("renders every top-level field, not the one the summary happened to mention", () => {
    // The real coverage shape. The page printed ten entries from `gaps` and dropped the other four fields.
    render(<ActionResultPanel onOpenObject={vi.fn()} onOpenFrame={vi.fn()} result={result({
      scene_frames: 41752,
      class_balance: { median: 12, missing: ["tanker"], rare: ["cattle"] },
      geo: { BLR: 300, DEL: 20 },
      gaps: ["no night highway frames"],
    })} />);
    expect(screen.getByText("scene_frames")).toBeInTheDocument();
    expect(screen.getByText("41,752")).toBeInTheDocument();
    expect(screen.getByText(/class_balance/)).toBeInTheDocument();
    expect(screen.getByText(/no night highway frames/)).toBeInTheDocument();
    expect(screen.getByText("geo")).toBeInTheDocument();
  });

  it("makes an object id clickable rather than printing it", async () => {
    const onOpenObject = vi.fn();
    render(<ActionResultPanel onOpenObject={onOpenObject} onOpenFrame={vi.fn()} result={result({
      object_id: "0f9c1a2b-3c4d-5e6f-7a8b-9c0d1e2f3a4b",
    })} />);
    await userEvent.click(screen.getByRole("button", { name: /object_id/i }));
    expect(onOpenObject).toHaveBeenCalledWith("0f9c1a2b-3c4d-5e6f-7a8b-9c0d1e2f3a4b");
  });

  it("makes a list of frame ids individually openable", async () => {
    const onOpenFrame = vi.fn();
    render(<ActionResultPanel onOpenObject={vi.fn()} onOpenFrame={onOpenFrame} result={result({
      frame_ids: ["11111111-1111-1111-1111-111111111111",
                  "22222222-2222-2222-2222-222222222222"],
    })} />);
    expect(screen.getByText(/frame_ids \(2\)/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "22222222" }));
    expect(onOpenFrame).toHaveBeenCalledWith("22222222-2222-2222-2222-222222222222");
  });

  it("does not turn every uuid into an object link", () => {
    // A run id is not an object. Opening one in the object inspector would 404 in a way that reads as
    // missing data rather than as a wrong link.
    render(<ActionResultPanel onOpenObject={vi.fn()} onOpenFrame={vi.fn()} result={result({
      run_id: "33333333-3333-3333-3333-333333333333",
    })} />);
    expect(screen.queryByRole("button", { name: /run_id/i })).not.toBeInTheDocument();
    expect(screen.getByText(/33333333-3333/)).toBeInTheDocument();
  });

  it("links to where the produced items went, instead of naming it in prose", async () => {
    // "mined 47 safety scenarios, see Scenarios" was the whole of the previous affordance.
    render(<ActionResultPanel onOpenObject={vi.fn()} onOpenFrame={vi.fn()}
      result={result({ persisted: 47 }, {
        label: "mine safety scenarios",
        destination: { href: "/scenarios", label: "open the 47 scenarios" },
      })} />);
    await userEvent.click(screen.getByRole("button", { name: /open the 47 scenarios/i }));
    expect(router.push).toHaveBeenCalledWith("/scenarios");
  });

  it("draws a counter map as bars rather than as a wall of key-value pairs", () => {
    render(<ActionResultPanel onOpenObject={vi.fn()} onOpenFrame={vi.fn()} result={result({
      by_kind: { near_miss: 30, hard_brake: 12, cut_in: 5 },
    })} />);
    // Sorted by size, so the biggest contributor is the first thing read.
    const labels = screen.getAllByTitle(/near_miss|hard_brake|cut_in/).map((e) => e.textContent);
    expect(labels).toEqual(["near_miss", "hard_brake", "cut_in"]);
  });

  it("shows an empty list as empty rather than omitting the field", () => {
    // An absent field and a field that came back empty are different results, and the second is often the
    // answer: "no drift breach" is `breached: []`.
    render(<ActionResultPanel onOpenObject={vi.fn()} onOpenFrame={vi.fn()} result={result({
      breached: [], champion: null,
    })} />);
    expect(screen.getByText(/breached: ?empty/i)).toBeInTheDocument();
    expect(screen.getByText(/champion: ?none/i)).toBeInTheDocument();
  });

  it("says so when an action returned nothing, rather than rendering blank", () => {
    render(<ActionResultPanel result={result({})} onOpenObject={vi.fn()} onOpenFrame={vi.fn()} />);
    expect(screen.getByText(/returned nothing to show/i)).toBeInTheDocument();
  });

  it("caps a very long list and says how much it did not draw", () => {
    render(<ActionResultPanel onOpenObject={vi.fn()} onOpenFrame={vi.fn()} result={result({
      gaps: Array.from({ length: 260 }, (_, i) => `gap ${i}`),
    })} />);
    expect(screen.getByText(/60 more not drawn/)).toBeInTheDocument();
  });
});
