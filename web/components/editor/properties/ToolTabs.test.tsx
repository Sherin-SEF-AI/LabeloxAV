import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import ToolTabs from "./ToolTabs";

// The strip this replaces was three unlabelled buttons: no role, no aria-selected, no aria-controls, no
// key handling. None of that is visible on screen, which is why it survived, and none of it can be
// asserted by looking at the panel.

const TABS = [
  { key: "agent", label: "agent" },
  { key: "bulk", label: "bulk edit" },
  { key: "frame", label: "frame data" },
] as const;

function Harness({ onChange = vi.fn() }: { onChange?: (v: string) => void }) {
  const [tab, setTab] = useState<"agent" | "bulk" | "frame">("agent");
  return (
    <>
      <ToolTabs tabs={[...TABS]} value={tab} idPrefix="props" label="Editor tools"
        onChange={(v) => { setTab(v); onChange(v); }} />
      {TABS.map((t) => (
        <div key={t.key} role="tabpanel" hidden={t.key !== tab}
          id={`props-panel-${t.key}`} aria-labelledby={`props-tab-${t.key}`}>{t.label} body</div>
      ))}
    </>
  );
}

describe("semantics", () => {
  it("is a tablist with exactly one selected tab", () => {
    render(<Harness />);
    expect(screen.getByRole("tablist", { name: "Editor tools" })).toBeInTheDocument();
    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(3);
    expect(tabs.filter((t) => t.getAttribute("aria-selected") === "true")).toHaveLength(1);
  });

  it("each tab points at a panel that points back at it", () => {
    render(<Harness />);
    for (const t of screen.getAllByRole("tab")) {
      const panel = document.getElementById(t.getAttribute("aria-controls")!);
      expect(panel).not.toBeNull();
      expect(panel!.getAttribute("aria-labelledby")).toBe(t.id);
    }
  });

  it("uses a roving tabindex so the strip is one Tab stop, not three", () => {
    render(<Harness />);
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.getAttribute("tabindex"))).toEqual(["0", "-1", "-1"]);
  });
});

describe("keyboard", () => {
  it("arrow keys move between tabs and wrap at both ends", () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    const strip = screen.getByRole("tablist");
    fireEvent.keyDown(strip, { key: "ArrowRight" });
    expect(onChange).toHaveBeenLastCalledWith("bulk");
    fireEvent.keyDown(strip, { key: "ArrowRight" });
    expect(onChange).toHaveBeenLastCalledWith("frame");
    // Wrapping matters on a three-tab strip: stopping at the end makes the user reverse direction to
    // reach the tab one step the other way.
    fireEvent.keyDown(strip, { key: "ArrowRight" });
    expect(onChange).toHaveBeenLastCalledWith("agent");
    fireEvent.keyDown(strip, { key: "ArrowLeft" });
    expect(onChange).toHaveBeenLastCalledWith("frame");
  });

  it("Home and End jump to the ends", () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    const strip = screen.getByRole("tablist");
    fireEvent.keyDown(strip, { key: "End" });
    expect(onChange).toHaveBeenLastCalledWith("frame");
    fireEvent.keyDown(strip, { key: "Home" });
    expect(onChange).toHaveBeenLastCalledWith("agent");
  });

  it("moves focus with the selection, so the ring never sits on an inactive tab", () => {
    render(<Harness />);
    fireEvent.keyDown(screen.getByRole("tablist"), { key: "ArrowRight" });
    expect(document.activeElement).toBe(screen.getByRole("tab", { name: "bulk edit" }));
  });
});

describe("panels", () => {
  it("shows only the active tab's panel", async () => {
    render(<Harness />);
    expect(screen.getByText("agent body")).toBeVisible();
    expect(screen.getByText("bulk edit body")).not.toBeVisible();
    await userEvent.click(screen.getByRole("tab", { name: "bulk edit" }));
    expect(screen.getByText("bulk edit body")).toBeVisible();
    expect(screen.getByText("agent body")).not.toBeVisible();
  });
});
