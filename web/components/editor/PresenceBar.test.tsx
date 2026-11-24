import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { PresenceBanner } from "@/lib/editor/presence";
import type { PresenceUser } from "@/lib/presenceApi";

import { PresenceAvatars, ReadOnlyBanner } from "./PresenceBar";

const person = (name: string, editing = false): PresenceUser =>
  ({ user_id: `u-${name}`, name, role: "annotator", editing });

describe("PresenceAvatars", () => {
  it("renders nothing when nobody else is here", () => {
    const { container } = render(<PresenceAvatars others={[]} degraded={false} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows initials, names everyone for a screen reader, and marks who is editing", () => {
    render(<PresenceAvatars others={[person("Asha Rao", true), person("binu")]} degraded={false} />);
    expect(screen.getByRole("list", { name: "Also on this frame: Asha Rao, binu" })).toBeInTheDocument();
    expect(screen.getByText("AR")).toBeInTheDocument();
    expect(screen.getByText("BI")).toBeInTheDocument();
    expect(screen.getByText("Asha Rao (editing)")).toBeInTheDocument();
    expect(screen.getByTitle("Asha Rao (annotator), editing this frame")).toBeInTheDocument();
    expect(screen.getByTitle("binu (annotator), viewing")).toBeInTheDocument();
  });

  it("collapses a crowd into a count", () => {
    const crowd = ["A A", "B B", "C C", "D D", "E E", "F F"].map((n) => person(n));
    render(<PresenceAvatars others={crowd} degraded={false} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(5);
    expect(screen.getByText("+2")).toHaveAttribute("title", "E E, F F");
  });

  it("says presence is off rather than showing an empty row", () => {
    render(<PresenceAvatars others={[]} degraded />);
    expect(screen.getByText("presence off")).toBeInTheDocument();
  });
});

describe("ReadOnlyBanner", () => {
  const held: PresenceBanner = { kind: "held", text: "Being edited by Asha — view only", canTakeOver: false };

  it("renders nothing without a banner", () => {
    const { container } = render(<ReadOnlyBanner banner={null} onTakeOver={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("announces who is editing, and offers no take-over to somebody who may not", () => {
    render(<ReadOnlyBanner banner={held} onTakeOver={() => {}} />);
    expect(screen.getByRole("status")).toHaveTextContent("Being edited by Asha — view only");
    expect(screen.queryByRole("button", { name: /take over/i })).toBeNull();
  });

  it("offers Take over to a reviewer or admin, and calls back", () => {
    const onTakeOver = vi.fn();
    render(<ReadOnlyBanner banner={{ ...held, canTakeOver: true }} onTakeOver={onTakeOver} />);
    fireEvent.click(screen.getByRole("button", { name: "Take over" }));
    expect(onTakeOver).toHaveBeenCalledTimes(1);
  });

  it("disables the button while a take-over is in flight", () => {
    render(<ReadOnlyBanner banner={{ ...held, canTakeOver: true }} onTakeOver={() => {}} taking />);
    expect(screen.getByRole("button", { name: "Taking over..." })).toBeDisabled();
  });

  it("says what happened to unsaved work when the lock was lost", () => {
    render(<ReadOnlyBanner onTakeOver={() => {}} banner={{
      kind: "lost", text: "Ravi took over this frame — view only", canTakeOver: false,
      detail: "Your unsaved changes are still here, but they cannot be saved while Ravi is editing.",
    }} />);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Ravi took over this frame — view only");
    expect(status).toHaveTextContent("Your unsaved changes are still here");
  });
});
