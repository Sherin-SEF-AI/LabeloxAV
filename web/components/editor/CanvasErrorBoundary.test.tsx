import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useReducer } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CanvasErrorBoundary from "./CanvasErrorBoundary";

// One bad polygon thrown in the renderer used to unmount the whole editor, and the editor is where the
// unsaved work is held. The property under test is the one that matters to the person at the keyboard:
// after a crash and a retry, what they had done is still there.

let fail = false;
let mounts = 0;

function Canvas({ label }: { label: string }) {
  if (fail) throw new Error("drawImage: zero-sized canvas");
  return <div>canvas showing {label}</div>;
}

function CountingCanvas() {
  // A ref-free mount counter: the initialiser of useReducer runs once per mount.
  useReducer((x: number) => x, 0, () => { mounts += 1; return 0; });
  if (fail) throw new Error("boom");
  return <div>canvas</div>;
}

// Stands in for the page: state above the boundary, the way the editor's reducer is.
function Page() {
  const [edits, edit] = useReducer((n: number) => n + 1, 0);
  return (
    <div>
      <button onClick={edit}>edit</button>
      <span>{edits} unsaved edits</span>
      <CanvasErrorBoundary><Canvas label={`${edits} edits`} /></CanvasErrorBoundary>
    </div>
  );
}

let errors: ReturnType<typeof vi.spyOn>;
beforeEach(() => {
  fail = false;
  mounts = 0;
  // React logs a caught render error, and so does the boundary. Both are expected here.
  errors = vi.spyOn(console, "error").mockImplementation(() => {});
});
afterEach(() => errors.mockRestore());

describe("a crash in the canvas", () => {
  it("renders the canvas untouched while nothing is wrong", () => {
    render(<CanvasErrorBoundary><Canvas label="frame" /></CanvasErrorBoundary>);
    expect(screen.getByText("canvas showing frame")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows a panel with a retry instead of taking the page down", () => {
    fail = true;
    render(<Page />);
    expect(screen.getByRole("alert")).toHaveTextContent("canvas crashed — reload canvas");
    expect(screen.getByRole("alert")).toHaveTextContent("drawImage: zero-sized canvas");
    expect(screen.getByRole("button", { name: "Reload canvas" })).toBeInTheDocument();
    // The rest of the page is still mounted and still works.
    expect(screen.getByText("0 unsaved edits")).toBeInTheDocument();
  });

  it("records the error, which is the only trace of which shape did it", () => {
    fail = true;
    render(<Page />);
    expect(errors.mock.calls.some((c) => c[0] === "editor canvas crashed")).toBe(true);
  });

  it("keeps the state held above it across the crash and the retry", async () => {
    render(<Page />);
    await userEvent.click(screen.getByRole("button", { name: "edit" }));
    await userEvent.click(screen.getByRole("button", { name: "edit" }));
    fail = true;
    await userEvent.click(screen.getByRole("button", { name: "edit" }));
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("3 unsaved edits")).toBeInTheDocument();
    // Still editable while the canvas is down.
    await userEvent.click(screen.getByRole("button", { name: "edit" }));
    fail = false;
    await userEvent.click(screen.getByRole("button", { name: "Reload canvas" }));
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByText("canvas showing 4 edits")).toBeInTheDocument();
    expect(screen.getByText("4 unsaved edits")).toBeInTheDocument();
  });

  it("shows the panel again if the retry crashes too", async () => {
    fail = true;
    render(<Page />);
    await userEvent.click(screen.getByRole("button", { name: "Reload canvas" }));
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("mounts the canvas fresh on retry rather than resuming the tree that threw", async () => {
    render(<CanvasErrorBoundary><CountingCanvas /></CanvasErrorBoundary>);
    expect(mounts).toBe(1);
    fail = true;
    render(<CanvasErrorBoundary><CountingCanvas /></CanvasErrorBoundary>);
    fail = false;
    const before = mounts;
    await userEvent.click(screen.getByRole("button", { name: "Reload canvas" }));
    expect(mounts).toBe(before + 1);
  });

  it("copes with something thrown that is not an Error", () => {
    function Thrower(): never { throw "a string"; }
    render(<CanvasErrorBoundary><Thrower /></CanvasErrorBoundary>);
    expect(screen.getByRole("alert")).toHaveTextContent("a string");
  });
});
