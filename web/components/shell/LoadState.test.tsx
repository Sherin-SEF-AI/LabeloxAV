import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import LoadState from "./LoadState";

// The component this remediation leaned on hardest, and it had no test.
//
// Three separate pages told a user their work was finished when a request had actually failed: triage
// rendered "Queue is clear", analytics rendered "no objects yet" over a 570k-object corpus, and the
// project board rendered as though a manager had no projects. Every one of those was fixed by routing the
// failure through this component, so what it does with each of its four states is now load-bearing in
// several places at once.
//
// The precedence is the part worth pinning: error must beat loading, and loading must beat empty. A
// component that showed "nothing here yet" while a request was still in flight, or while one had failed,
// would reintroduce the exact defect its callers adopted it to fix.

describe("LoadState", () => {
  it("shows the error, not the empty state, when both could apply", () => {
    render(<LoadState error={new Error("upstream is down")} empty emptyText="nothing here yet" />);
    expect(screen.queryByText(/nothing here yet/i)).not.toBeInTheDocument();
    expect(screen.getByText(/upstream is down/i)).toBeInTheDocument();
  });

  it("shows the error, not the loading state, when both could apply", () => {
    render(<LoadState error={new Error("upstream is down")} loading />);
    expect(screen.getByText(/upstream is down/i)).toBeInTheDocument();
  });

  it("shows loading rather than empty while a request is in flight", () => {
    const { container } = render(<LoadState loading empty emptyText="nothing here yet" />);
    expect(screen.queryByText(/nothing here yet/i)).not.toBeInTheDocument();
    expect(container).not.toBeEmptyDOMElement();
  });

  it("states the empty case in the caller's own words", () => {
    render(<LoadState empty emptyText="no frames in this session" />);
    expect(screen.getByText(/no frames in this session/i)).toBeInTheDocument();
  });

  it("renders its children when there is nothing to report", () => {
    render(<LoadState><span>the actual content</span></LoadState>);
    expect(screen.getByText("the actual content")).toBeInTheDocument();
  });

  it("offers a retry only when the caller can act on one", async () => {
    const onRetry = vi.fn();
    const { rerender } = render(<LoadState error={new Error("boom")} onRetry={onRetry} />);
    const button = screen.getByRole("button", { name: /retry/i });
    await userEvent.click(button);
    expect(onRetry).toHaveBeenCalledTimes(1);

    // Without a handler there is no button to press, rather than a dead one.
    rerender(<LoadState error={new Error("boom")} />);
    expect(screen.queryByRole("button", { name: /retry/i })).not.toBeInTheDocument();
  });

  it("turns an unknown throw into a sentence rather than [object Object]", () => {
    // humanizeError is what stands between a caller and a rendered object. A thrown string, a plain
    // object and an Error all reach this component in practice.
    render(<LoadState error={{ detail: "session not found" }} />);
    expect(screen.queryByText(/\[object Object\]/)).not.toBeInTheDocument();
  });
});
