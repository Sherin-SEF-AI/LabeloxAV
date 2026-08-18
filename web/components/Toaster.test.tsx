import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import Toaster from "./Toaster";
import { toast } from "@/lib/toast";

// The undo affordance, rendered.
//
// ToastAction was built with a written rationale - a batch that changed fifty objects and offers no way
// back asks the operator to be certain before acting, where one that can be undone from the confirmation
// asks them only to look - and then had zero call sites for its whole life. It has one now (bulk relabel
// on the triage page), which makes the rendering load-bearing rather than decorative.
//
// The detail worth pinning is the dismiss-then-await ordering: Toaster removes the toast before awaiting
// the action, so a rejection inside `run` has nowhere left to surface. The caller is responsible for
// catching it, and that only holds if the ordering stays what it is.

describe("Toaster", () => {
  it("renders a toast and its action", async () => {
    render(<Toaster />);
    toast("reclassify: 60 objects", "success", 5000, { label: "undo", run: () => {} });

    expect(await screen.findByText(/reclassify: 60 objects/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /undo/i })).toBeInTheDocument();
  });

  it("runs the action when it is pressed, and takes the toast away", async () => {
    const run = vi.fn();
    render(<Toaster />);
    toast("deleted 12 objects", "success", 5000, { label: "undo", run });

    await userEvent.click(await screen.findByRole("button", { name: /undo/i }));
    expect(run).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.queryByText(/deleted 12 objects/)).not.toBeInTheDocument());
  });

  it("offers no action when the caller supplied none", async () => {
    render(<Toaster />);
    toast("saved", "success");
    expect(await screen.findByText("saved")).toBeInTheDocument();
    // Dismiss is always there; what must not appear is an action button with nothing behind it.
    expect(screen.getByRole("button", { name: /dismiss/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /undo/i })).not.toBeInTheDocument();
  });

  it("announces politely, because a toast must not interrupt a keyboard reviewer mid-verdict", async () => {
    const { container } = render(<Toaster />);
    toast("confirmed", "success");
    await screen.findByText("confirmed");
    expect(container.querySelector('[aria-live="polite"]')).toBeTruthy();
  });
});
