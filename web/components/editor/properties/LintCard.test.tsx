import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import LintCard from "./LintCard";

// The panel exists to make three different things distinguishable, and a card that renders them the same
// way is worse than no card.
//
// A finding is something a reviewer can fix. A SYSTEMIC rule is one that objected to most of the frame,
// which is one fact about the pipeline and not fifty facts about boxes. A DORMANT rule is one that could
// not run at all because nobody is collecting the data it needs - measured over the corpus, all three of
// the relation and attribute rules are dormant on every frame, because object_relationship holds 98 rows
// and occupant_count is set on none.
//
// Collapsing any of those into the others produces a specific lie: dormant rendered as clean says the
// frame was checked when it was not, and systemic rendered as findings buries the four that matter under
// fifty that do not.

const finding = (over: Partial<Parameters<typeof LintCard>[0] extends never ? never : any> = {}) => ({
  object_id: "obj-1", rule: "self_intersecting_polygon", label: "Outline crosses itself",
  severity: "high", score: 0.8, reason: "the outline crosses itself", ...over,
});

describe("LintCard", () => {
  it("says a clean frame is clean", () => {
    render(<LintCard lint={{ findings: [], systemic: {}, dormant: [] }} selectedIds={[]} onSelect={vi.fn()} />);
    expect(screen.getByText("clean")).toBeTruthy();
  });

  it("distinguishes 'not yet checked' from 'checked and clean'", () => {
    // Null is the state before the request lands. Rendering it as "clean" would tell an annotator the
    // frame passed a check that has not run.
    render(<LintCard lint={null} selectedIds={[]} onSelect={vi.fn()} />);
    expect(screen.getByText("running...")).toBeTruthy();
    expect(screen.queryByText("clean")).toBeNull();
  });

  it("lists findings and selects the object one is about", async () => {
    const onSelect = vi.fn();
    render(<LintCard lint={{ findings: [finding()], systemic: {}, dormant: [] }}
      selectedIds={[]} onSelect={onSelect} />);
    await userEvent.click(screen.getByText("Outline crosses itself"));
    expect(onSelect).toHaveBeenCalledWith("obj-1");
  });

  it("reports a frame-wide rule once, not once per object", () => {
    render(<LintCard
      lint={{ findings: [], systemic: { attr_validity: 122 }, dormant: [], n_objects: 122 }}
      selectedIds={[]} onSelect={vi.fn()} />);
    expect(screen.getByText(/objected to/)).toBeTruthy();
    expect(screen.getByText(/122 of/)).toBeTruthy();
    // And it is not dressed up as a clean frame, which it is not.
    expect(screen.queryByText("clean")).toBeNull();
  });

  it("shows that a check could not run, and why", async () => {
    render(<LintCard lint={{
      findings: [], systemic: {},
      dormant: [{ rule: "rider_without_mount", label: "Rider with no rider_of",
                  reason: "no relations have been drawn on this frame" }],
    }} selectedIds={[]} onSelect={vi.fn()} />);
    // Collapsed by default so it does not compete with findings, but present and countable.
    await userEvent.click(screen.getByText(/1 check could not run/));
    expect(screen.getByText(/no relations have been drawn/)).toBeTruthy();
  });

  it("offers to open issues only when there is something to open", () => {
    const onOpenIssues = vi.fn();
    const { rerender } = render(<LintCard lint={{ findings: [], systemic: {}, dormant: [] }}
      selectedIds={[]} onSelect={vi.fn()} onOpenIssues={onOpenIssues} />);
    expect(screen.queryByText(/open .* as issue/)).toBeNull();

    rerender(<LintCard lint={{ findings: [finding()], systemic: {}, dormant: [] }}
      selectedIds={[]} onSelect={vi.fn()} onOpenIssues={onOpenIssues} />);
    expect(screen.getByText(/open 1 as issue/)).toBeTruthy();
  });

  it("does not open issues on its own", async () => {
    // The editor autosaves every 700ms and re-lints after each save. A card that opened threads by itself
    // would fill the panel within a minute.
    const onOpenIssues = vi.fn();
    render(<LintCard lint={{ findings: [finding()], systemic: {}, dormant: [] }}
      selectedIds={[]} onSelect={vi.fn()} onOpenIssues={onOpenIssues} />);
    expect(onOpenIssues).not.toHaveBeenCalled();
    await userEvent.click(screen.getByText(/open 1 as issue/));
    expect(onOpenIssues).toHaveBeenCalledTimes(1);
  });
});
