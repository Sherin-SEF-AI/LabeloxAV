import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { KEYMAP_STORAGE_KEY, effectiveModes, getKeymap, resetKeymap } from "@/lib/editor/keymap";
import { toolForHotkey } from "@/lib/editor/registry";

import KeymapOverlay from "./KeymapOverlay";

// The overlay edits the keymap the keyboard handler reads. The tests that carry weight are the ones about
// what the page's own handler sees while a key is being captured: it is a bubble-phase listener on the
// same window, so `spy` stands in for it exactly as it does in CanvasMenu.test.tsx. If the key being
// offered also reached it, rebinding the box tool to N would switch tool on the way, and Escape would
// close the overlay and clear the annotator's selection.

let spy: ReturnType<typeof vi.fn>;
beforeEach(() => {
  window.localStorage.clear();
  resetKeymap();
  spy = vi.fn();
  window.addEventListener("keydown", spy);
});
afterEach(() => { window.removeEventListener("keydown", spy); });

const open = () => {
  render(<KeymapOverlay />);
  act(() => { window.dispatchEvent(new Event("lbx:shortcuts")); });
};
const keyButton = (label: RegExp) => screen.getByRole("button", { name: label });

describe("opening", () => {
  it("is closed until asked for", () => {
    render(<KeymapOverlay />);
    expect(screen.queryByText("keyboard shortcuts")).toBeNull();
  });

  it("opens on the shortcuts event the header button and the menu raise", () => {
    open();
    expect(screen.getByText("keyboard shortcuts")).toBeInTheDocument();
  });

  it("toggles on ?, and closes on Escape", () => {
    render(<KeymapOverlay />);
    fireEvent.keyDown(document.body, { key: "?" });
    expect(screen.getByText("keyboard shortcuts")).toBeInTheDocument();
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByText("keyboard shortcuts")).toBeNull();
  });

  it("does not open on a ? typed into a text field", () => {
    render(<><input aria-label="note" /><KeymapOverlay /></>);
    fireEvent.keyDown(screen.getByLabelText("note"), { key: "?" });
    expect(screen.queryByText("keyboard shortcuts")).toBeNull();
  });
});

describe("what it lists", () => {
  it("prints each tool and action with the key in force", () => {
    open();
    expect(keyButton(/^rebind box, currently B$/i)).toHaveTextContent("B");
    expect(keyButton(/^rebind fit the frame to the window, currently F$/i)).toBeInTheDocument();
  });

  it("keeps the fixed rows, and adds the keys this editor binds", () => {
    open();
    expect(screen.getByText("Cmd Z")).toBeInTheDocument();
    expect(screen.getByText("Arrows")).toBeInTheDocument();
    expect(screen.getByText("Shift Arrows")).toBeInTheDocument();
  });

  it("does not print a rebindable letter twice, once fixed and once live", () => {
    open();
    expect(screen.getAllByText(/fit the frame to the window/)).toHaveLength(1);
  });

  it("says which mode a tool belongs to when it is not in all of them", () => {
    open();
    expect(screen.getByText(/whole extent/i).textContent).toContain("Objects");
    expect(screen.getByText(/^keypoint/i).textContent).toContain("Pose and behavior");
  });

  it("filters by the search box", async () => {
    open();
    await userEvent.type(screen.getByPlaceholderText("search shortcuts..."), "polygon");
    expect(keyButton(/^rebind polygon/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^rebind box/i })).toBeNull();
    expect(screen.queryByText("Cmd Z")).toBeNull();
  });

  it("says so when nothing matches", async () => {
    open();
    await userEvent.type(screen.getByPlaceholderText("search shortcuts..."), "zzzz");
    expect(screen.getByText("no shortcuts match")).toBeInTheDocument();
  });
});

describe("rebinding", () => {
  it("click a key, press a key: the binding moves, is stored, and the handler would now resolve it", async () => {
    open();
    await userEvent.click(keyButton(/^rebind box/i));
    expect(screen.getByRole("button", { name: "press a key for Box" })).toHaveTextContent("press a key");
    fireEvent.keyDown(document.body, { key: "n" });
    expect(keyButton(/^rebind box, currently N$/i)).toHaveTextContent("N");
    expect(screen.getByText(/was B/)).toBeInTheDocument();
    expect(JSON.parse(window.localStorage.getItem(KEYMAP_STORAGE_KEY) ?? "null")).toEqual({ "tool:box": "n" });
    expect(toolForHotkey("objects", "n", effectiveModes(getKeymap()))).toBe("box");
    expect(toolForHotkey("objects", "b", effectiveModes(getKeymap()))).toBeNull();
  });

  it("the key being offered never reaches the page's own handler", async () => {
    open();
    await userEvent.click(keyButton(/^rebind box/i));
    spy.mockClear();
    fireEvent.keyDown(document.body, { key: "n" });
    expect(spy).not.toHaveBeenCalled();
  });

  it("Escape cancels the capture, keeps the overlay open, and does not reach the page", async () => {
    open();
    await userEvent.click(keyButton(/^rebind box/i));
    spy.mockClear();
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(spy).not.toHaveBeenCalled();
    expect(screen.getByText("keyboard shortcuts")).toBeInTheDocument();
    expect(keyButton(/^rebind box, currently B$/i)).toBeInTheDocument();
    expect(getKeymap()).toEqual({});
  });

  it("refuses a key another tool has in the same mode, says why, and changes nothing", async () => {
    open();
    await userEvent.click(keyButton(/^rebind box/i));
    fireEvent.keyDown(document.body, { key: "g" });
    expect(screen.getByRole("alert")).toHaveTextContent('G is already "Polygon" in Objects');
    expect(getKeymap()).toEqual({});
    expect(window.localStorage.getItem(KEYMAP_STORAGE_KEY)).toBeNull();
    // Still armed: the next key is another attempt, not a tool switch.
    fireEvent.keyDown(document.body, { key: "n" });
    expect(screen.queryByRole("alert")).toBeNull();
    expect(getKeymap()).toEqual({ "tool:box": "n" });
  });

  it("refuses a reserved key", async () => {
    open();
    await userEvent.click(keyButton(/^rebind box/i));
    fireEvent.keyDown(document.body, { key: "1" });
    expect(screen.getByRole("alert")).toHaveTextContent(/1 is reserved/);
  });

  it("refuses a chord, and waits through a modifier pressed on its own", async () => {
    open();
    await userEvent.click(keyButton(/^rebind box/i));
    fireEvent.keyDown(document.body, { key: "Shift" });
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("button", { name: "press a key for Box" })).toBeInTheDocument();
    fireEvent.keyDown(document.body, { key: "n", ctrlKey: true });
    expect(screen.getByRole("alert")).toHaveTextContent(/without Cmd, Ctrl or Alt/);
    expect(getKeymap()).toEqual({});
  });

  it("clicking the armed key again disarms it", async () => {
    open();
    await userEvent.click(keyButton(/^rebind box/i));
    await userEvent.click(screen.getByRole("button", { name: "press a key for Box" }));
    expect(keyButton(/^rebind box, currently B$/i)).toBeInTheDocument();
  });

  it("closing mid-capture leaves nothing armed behind the closed overlay", async () => {
    open();
    await userEvent.click(keyButton(/^rebind box/i));
    await userEvent.click(screen.getByRole("button", { name: "esc" }));
    spy.mockClear();
    fireEvent.keyDown(document.body, { key: "n" });
    // The key reaches the page again, and binds nothing.
    expect(spy).toHaveBeenCalled();
    expect(getKeymap()).toEqual({});
  });

  it("rebinds an action the same way", async () => {
    open();
    await userEvent.click(keyButton(/^rebind fit the frame/i));
    fireEvent.keyDown(document.body, { key: "z" });
    expect(getKeymap()).toEqual({ "action:fit": "z" });
  });
});

describe("reset to defaults", () => {
  it("is disabled while nothing has been changed", () => {
    open();
    expect(screen.getByRole("button", { name: "Reset to defaults" })).toBeDisabled();
    expect(screen.getByText("these are the default keys")).toBeInTheDocument();
  });

  it("puts every key back and clears what was stored", async () => {
    open();
    await userEvent.click(keyButton(/^rebind box/i));
    fireEvent.keyDown(document.body, { key: "n" });
    expect(screen.getByText("1 changed from the defaults")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Reset to defaults" }));
    expect(keyButton(/^rebind box, currently B$/i)).toBeInTheDocument();
    expect(getKeymap()).toEqual({});
    expect(window.localStorage.getItem(KEYMAP_STORAGE_KEY)).toBeNull();
  });

  it("reads a keymap stored on an earlier visit", () => {
    window.localStorage.setItem(KEYMAP_STORAGE_KEY, JSON.stringify({ "tool:box": "n" }));
    open();
    expect(keyButton(/^rebind box, currently N$/i)).toBeInTheDocument();
  });
});
