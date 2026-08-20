import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useRef, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { OntologyClass } from "@/lib/types";
import ClassPopover from "./ClassPopover";

// The load-bearing tests here are the two about keys the popover must NOT let through.
//
// The frame page registers its keymap on `window` in the bubble phase. Its Escape branch clears the
// selection and its 1-9 branch relabels it. A popover built the way every other popover in this codebase
// is built (MenuBar, NotificationBell: a plain bubble-phase document listener) lets both of those fire,
// so dismissing the class list wipes a multi-selection and picking a class by number produces two history
// entries for one keypress. Neither is visible in review, and neither throws.
//
// `spy` below stands in for that page handler: same target, same phase. If it is ever called, the real
// editor has just had its selection cleared.

vi.mock("@/lib/colors", () => ({ classColor: () => "#888" }));

const CLASSES: OntologyClass[] = [
  { id: 1, name: "motorcycle", l0: "vehicle", l1: "two_wheeler", india: false },
  { id: 2, name: "scooter", l0: "vehicle", l1: "two_wheeler", india: false },
  { id: 3, name: "moped", l0: "vehicle", l1: "two_wheeler", india: false },
  { id: 4, name: "cycle", l0: "vehicle", l1: "two_wheeler", india: false },
  { id: 5, name: "delivery_rider_bike", l0: "vehicle", l1: "two_wheeler", india: true },
  { id: 6, name: "autorickshaw", l0: "vehicle", l1: "three_wheeler", india: true },
  { id: 7, name: "e_auto", l0: "vehicle", l1: "three_wheeler", india: true },
  { id: 8, name: "sedan", l0: "vehicle", l1: "car", india: false },
  { id: 9, name: "hatchback", l0: "vehicle", l1: "car", india: false },
  { id: 10, name: "suv", l0: "vehicle", l1: "car", india: false },
];

let spy: ReturnType<typeof vi.fn>;

beforeEach(() => {
  spy = vi.fn();
  window.addEventListener("keydown", spy);
});
afterEach(() => {
  window.removeEventListener("keydown", spy);
  vi.clearAllMocks();
});

function Harness({ onPick = vi.fn(), onAdd = vi.fn(), startOpen = true }: {
  onPick?: (c: OntologyClass) => void; onAdd?: (raw: string) => void; startOpen?: boolean;
}) {
  const anchorRef = useRef<HTMLButtonElement | null>(null);
  const [open, setOpen] = useState(startOpen);
  return (
    <div className="relative">
      <button ref={anchorRef} onClick={() => setOpen((o) => !o)}>change</button>
      <ClassPopover anchorRef={anchorRef} open={open} onClose={() => setOpen(false)}
        classes={CLASSES} currentId={1} onPick={onPick} onAdd={onAdd} />
    </div>
  );
}

describe("keys the page keymap must never see", () => {
  it("Escape closes the popover without reaching the page, which would clear the selection", () => {
    render(<Harness />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();
  });

  it("a digit picks in the popover without also relabelling on the page", async () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} />);
    fireEvent.keyDown(document, { key: "3" });
    expect(onPick).toHaveBeenCalledWith(CLASSES[2]);
    expect(spy).not.toHaveBeenCalled();
  });

  it("returns focus to the trigger when Escape closes it", () => {
    render(<Harness />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "change" }));
  });

  it("lets through keys it does not handle, so the editor keeps its shortcuts", () => {
    render(<Harness />);
    fireEvent.keyDown(document, { key: "b" });
    expect(spy).toHaveBeenCalled();
  });

  it("stops listening once closed", () => {
    render(<Harness startOpen={false} />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(spy).toHaveBeenCalledTimes(1);
  });
});

describe("number badges", () => {
  it("numbers the first nine rows while the query is empty", () => {
    render(<Harness />);
    const rows = screen.getAllByRole("button", { name: /motorcycle|scooter|moped/ });
    expect(rows[0]).toHaveTextContent("1");
    expect(screen.getByRole("button", { name: /hatchback/ })).toHaveTextContent("9");
    // The tenth row gets none: the page only binds 1-9.
    expect(screen.getByRole("button", { name: /^suv$/ }).textContent).toBe("suv");
  });

  it("drops every badge once a query filters the list", async () => {
    // The list is now a subset in a different order, and the page still indexes the raw ontology. A badge
    // here would be a promise the keymap does not keep.
    render(<Harness />);
    await userEvent.type(screen.getByLabelText(/search or add class/i), "cycle");
    const rows = screen.getAllByRole("button", { name: /cycle/ });
    for (const r of rows) expect(r.textContent).not.toMatch(/[1-9]/);
  });

  it("does not pick by number while a query is showing", () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} />);
    fireEvent.change(screen.getByLabelText(/search or add class/i), { target: { value: "car" } });
    fireEvent.keyDown(document, { key: "1" });
    expect(onPick).not.toHaveBeenCalled();
  });
});

describe("picking and adding", () => {
  it("clicking a row picks that class and closes", async () => {
    const onPick = vi.fn();
    render(<Harness onPick={onPick} />);
    await userEvent.click(screen.getByRole("button", { name: /autorickshaw/ }));
    expect(onPick).toHaveBeenCalledWith(CLASSES[5]);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("Enter on an exact normalised match relabels rather than creating a duplicate class", async () => {
    const onPick = vi.fn(); const onAdd = vi.fn();
    render(<Harness onPick={onPick} onAdd={onAdd} />);
    const input = screen.getByLabelText(/search or add class/i);
    // "E Auto" normalises to "e_auto", which exists. Creating a second class here is the bug normClass
    // exists to prevent.
    await userEvent.type(input, "E Auto{Enter}");
    expect(onPick).toHaveBeenCalledWith(CLASSES[6]);
    expect(onAdd).not.toHaveBeenCalled();
  });

  it("Enter on a name that does not exist adds it, passing the raw text", async () => {
    const onAdd = vi.fn();
    render(<Harness onAdd={onAdd} />);
    await userEvent.type(screen.getByLabelText(/search or add class/i), "tanker{Enter}");
    expect(onAdd).toHaveBeenCalledWith("tanker");
  });

  it("offers the add row only when nothing matches", async () => {
    render(<Harness />);
    const input = screen.getByLabelText(/search or add class/i);
    await userEvent.type(input, "sedan");
    expect(screen.queryByText(/as custom class/)).not.toBeInTheDocument();
    await userEvent.clear(input);
    await userEvent.type(input, "tanker");
    expect(screen.getByText(/add "tanker" as custom class/)).toBeInTheDocument();
  });

  it("marks the India-specific classes, which is why the asterisk exists", () => {
    render(<Harness />);
    expect(screen.getByRole("button", { name: /autorickshaw/ })).toHaveTextContent("*");
    expect(screen.getByRole("button", { name: /^sedan/ })).not.toHaveTextContent("*");
  });
});

describe("dismissal", () => {
  it("a click outside closes it", () => {
    render(<Harness />);
    fireEvent.mouseDown(document.body);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("a click on the trigger does not close-then-reopen in the same tick", async () => {
    render(<Harness />);
    await userEvent.click(screen.getByRole("button", { name: "change" }));
    // The trigger's own toggle is the only thing that ran, so the popover is closed rather than closed by
    // the outside handler and immediately reopened by the toggle.
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
