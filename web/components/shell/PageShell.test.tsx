import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import PageShell from "./PageShell";

// Next's router and the menu wiring are not what these assert, and both need a mount context.
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), back: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/",
  useSearchParams: () => new URLSearchParams(),
}));

// The keyboard floor, on the shell 64 of 71 pages render inside.
//
// The README's claim for this app is that it is keyboard driven - j/k through a queue, a/x to rule on an
// object, tool letters in the editor. It shipped with no focus system at all: Tailwind's preflight removes
// the default outline and nothing put one back, so tabbing moved an invisible cursor. And every page puts a
// menu bar, a header and often a filter band before the content, with no way past them.
//
// These are structural assertions rather than visual ones - jsdom does not do layout, and a screenshot
// test for a focus ring would be the wrong instrument anyway. What they pin is that the landmark and the
// skip target exist and point at each other, which is the part that silently rots.

describe("PageShell", () => {
  it("puts the content behind a main landmark", () => {
    render(<PageShell active="HOME"><p>the content</p></PageShell>);
    const main = screen.getByRole("main");
    expect(main).toBeInTheDocument();
    expect(main).toHaveAttribute("id", "content");
  });

  it("offers a skip link that actually points at the content", () => {
    const { container } = render(<PageShell active="HOME"><p>the content</p></PageShell>);
    const skip = screen.getByRole("link", { name: /skip to content/i });
    // The pair is the thing under test: a skip link pointing at an id nothing carries is worse than none,
    // because it looks like the problem is solved.
    expect(skip).toHaveAttribute("href", "#content");
    expect(container.querySelector("#content")).toBeTruthy();
  });

  it("puts the skip link first in the tab order", () => {
    const { container } = render(<PageShell active="HOME"><button>a control</button></PageShell>);
    const focusable = container.querySelectorAll("a[href], button, input, select, textarea, [tabindex]");
    expect(focusable.length).toBeGreaterThan(1);
    // Reaching it after the menu bar would defeat the point of having it.
    expect(focusable[0]).toHaveTextContent(/skip to content/i);
  });

  it("renders its children", () => {
    render(<PageShell active="HOME"><p>the content</p></PageShell>);
    expect(screen.getByText("the content")).toBeInTheDocument();
  });
});
