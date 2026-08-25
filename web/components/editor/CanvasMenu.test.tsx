import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CanvasMenu from "./CanvasMenu";
import type { CanvasTarget } from "@/lib/canvasMenu";
import type { OntologyClass } from "@/lib/types";

// The load-bearing test is the Escape one. The editor binds its keymap on window and its Escape branch
// clears the annotator's selection, so a menu that closes on a plain bubble-phase listener also wipes a
// twelve-object selection on the way out. `spy` stands in for that handler: same target, same phase.

vi.mock("@/lib/colors", () => ({ classColor: () => "#888" }));

const CLASSES: OntologyClass[] = [
  { id: 1, name: "sedan", l0: "vehicle", l1: "car", india: false },
  { id: 2, name: "autorickshaw", l0: "vehicle", l1: "three_wheeler", india: true },
];

const target = (over: Partial<CanvasTarget> = {}): CanvasTarget => ({
  object: null, selectedCount: 0, targetInSelection: false, ...over,
});
const obj = (over = {}) => ({ id: "o1", class_name: "sedan", visible: true, state: "review", ...over });

let spy: ReturnType<typeof vi.fn>;
beforeEach(() => { spy = vi.fn(); window.addEventListener("keydown", spy); });
afterEach(() => { window.removeEventListener("keydown", spy); vi.clearAllMocks(); });

const view = (t: CanvasTarget, onAction = vi.fn(), onClose = vi.fn()) => {
  render(<CanvasMenu at={{ x: 40, y: 40 }} target={t} classes={CLASSES}
    onClose={onClose} onAction={onAction} />);
  return { onAction, onClose };
};

describe("keys the page keymap must never see", () => {
  it("Escape closes the menu without reaching the page, which would clear the selection", () => {
    const { onClose } = view(target({ object: obj() }));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
    expect(spy).not.toHaveBeenCalled();
  });

  it("arrow keys walk the menu instead of nudging the canvas", () => {
    view(target());
    fireEvent.keyDown(document, { key: "ArrowDown" });
    expect(spy).not.toHaveBeenCalled();
  });

  it("lets through keys it does not handle", () => {
    view(target());
    fireEvent.keyDown(document, { key: "b" });
    expect(spy).toHaveBeenCalled();
  });
});

describe("what it acts on", () => {
  it("fires the row you click and closes", async () => {
    const { onAction, onClose } = view(target());
    await userEvent.click(screen.getByRole("menuitem", { name: /fit frame to view/ }));
    expect(onAction).toHaveBeenCalledWith("fit");
    expect(onClose).toHaveBeenCalled();
  });

  it("a disabled row is shown, explained, and does nothing", async () => {
    // Shown and greyed rather than hidden, so the menu is a stable map of what exists.
    const { onAction } = view(target({ object: obj({ locked: true }) }));
    const del = screen.getByRole("menuitem", { name: /delete/ });
    expect(del).toBeDisabled();
    expect(del).toHaveAttribute("title", expect.stringMatching(/locked/));
    await userEvent.click(del);
    expect(onAction).not.toHaveBeenCalled();
  });

  it("change class opens the class list rather than firing straight away", async () => {
    const { onAction, onClose } = view(target({ object: obj() }));
    await userEvent.click(screen.getByRole("menuitem", { name: /change class/ }));
    expect(onAction).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByLabelText("choose a class")).toBeInTheDocument();
  });

  it("picking a class reports which one", async () => {
    const { onAction } = view(target({ object: obj() }));
    await userEvent.click(screen.getByRole("menuitem", { name: /change class/ }));
    await userEvent.click(screen.getByRole("menuitem", { name: /autorickshaw/ }));
    expect(onAction).toHaveBeenCalledWith("class", CLASSES[1]);
  });

  it("Escape from the class list goes back to the menu rather than closing it", async () => {
    const { onClose } = view(target({ object: obj() }));
    await userEvent.click(screen.getByRole("menuitem", { name: /change class/ }));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole("menuitem", { name: /change class/ })).toBeInTheDocument();
  });

  it("acts on the whole selection when the click landed inside it", () => {
    view(target({ object: obj(), selectedCount: 12, targetInSelection: true }));
    expect(screen.getByRole("menuitem", { name: /delete 12/ })).toBeInTheDocument();
  });

  it("acts on one object when the click landed outside the selection", () => {
    // Aiming at one box and getting an action on twelve others is a destructive surprise.
    view(target({ object: obj(), selectedCount: 12, targetInSelection: false }));
    expect(screen.queryByRole("menuitem", { name: /delete 12/ })).not.toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /^delete/ })).toBeInTheDocument();
  });

  it("a click outside dismisses it", () => {
    const { onClose } = view(target());
    fireEvent.mouseDown(document.body);
    expect(onClose).toHaveBeenCalled();
  });
});
