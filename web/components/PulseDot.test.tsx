import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import PulseDot from "./PulseDot";

// A pulsing dot is a claim that something is happening right now, so the tests are about the claim rather
// than about the pixels.
//
// Three rules. It pulses only while its subject is live, because once every dot moves the one that matters
// is invisible. Motion is never the only carrier, so a reader who has turned animation off still gets the
// state from the colour and the words. And it is announced: a bare coloured circle is nothing to a screen
// reader, which is exactly how the eight existing `<span className="running-dot" />` usages read today.

describe("PulseDot", () => {
  it("gives an idle dot no animation class, so a still dot means a still system", () => {
    render(<PulseDot tone="idle" label="connected, nothing running" />);
    const dot = screen.getByLabelText("connected, nothing running");
    expect(dot.className).toContain("pulse-dot--idle");
    // The animation lives on the tone classes, and idle deliberately has none. If this starts pulsing,
    // "something is happening" stops meaning anything on every surface that uses it.
    expect(dot.className).not.toContain("pulse-dot--live");
  });

  it("carries the state in colour, not only in motion", () => {
    // Under prefers-reduced-motion the stylesheet drops the animation and keeps the dot, so the class that
    // sets the colour has to be the class that identifies the state.
    const tones = ["live", "good", "warn", "bad", "idle"] as const;
    for (const tone of tones) {
      const { unmount } = render(<PulseDot tone={tone} label={`state ${tone}`} />);
      expect(screen.getByLabelText(`state ${tone}`).className).toContain(`pulse-dot--${tone}`);
      unmount();
    }
  });

  it("is announced when it says something, and hidden when it repeats adjacent text", () => {
    const { unmount } = render(<PulseDot tone="warn" label="the live connection dropped" />);
    const named = screen.getByLabelText("the live connection dropped");
    expect(named.getAttribute("role")).toBe("img");
    expect(named.getAttribute("aria-hidden")).toBeNull();
    expect(named.getAttribute("title")).toBe("the live connection dropped");
    unmount();

    // No label means the dot sits beside text that already says it, so reading it out would say it twice.
    const { container } = render(<PulseDot tone="live" />);
    const bare = container.querySelector(".pulse-dot")!;
    expect(bare.getAttribute("aria-hidden")).toBe("true");
    expect(bare.getAttribute("role")).toBeNull();
  });

  it("does not use a live region, so a reconnecting stream is not a stream of interruptions", () => {
    const { container } = render(<PulseDot tone="warn" label="dropped" />);
    const dot = container.querySelector(".pulse-dot")!;
    expect(dot.getAttribute("aria-live")).toBeNull();
  });

  it("keeps the halo off idle, because a ring around a still dot claims attention it has not earned", () => {
    const { container: live } = render(<PulseDot tone="warn" halo label="a" />);
    expect(live.querySelector(".pulse-dot")!.className).toContain("pulse-dot--halo");

    const { container: idle } = render(<PulseDot tone="idle" halo label="b" />);
    expect(idle.querySelector(".pulse-dot")!.className).not.toContain("pulse-dot--halo");
  });

  it("sizes stay small enough to sit inside a line of text", () => {
    const { container } = render(<PulseDot tone="live" size="lg" label="x" />);
    const dot = container.querySelector(".pulse-dot") as HTMLElement;
    expect(parseInt(dot.style.width, 10)).toBeLessThanOrEqual(8);
    expect(dot.style.width).toBe(dot.style.height);
  });
});
