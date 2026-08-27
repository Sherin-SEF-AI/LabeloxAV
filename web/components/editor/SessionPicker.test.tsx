import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import SessionPicker from "./SessionPicker";

// A drive row carries two independent actions: open the drive, and start or stop its auto-label. Nesting
// those inside one <button> is invalid HTML - React logs "In HTML, <button> cannot be a descendant of
// <button>. This will cause a hydration error." - and the browser's parser recovers by relocating nodes,
// so what renders is not what the JSX says.
//
// It survived review because it reads correctly in the source and works on a click. So the assertion is
// structural, and the test opens the picker for real: a version that asserted on an unmounted list passed
// against the broken code too, which is worse than no test.

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("@/lib/toast", () => ({ toast: vi.fn() }));
vi.mock("@/lib/api", () => {
  // Declared inside the factory: vi.mock is hoisted above the module body, so a top-level const is not
  // yet initialised when this runs.
  const rows = [
    { session_id: "s-1", vehicle_id: "DASHCAM-01", city: "BLR", n_frames: 120 },
    { session_id: "s-2", vehicle_id: "ANNO-01", city: "BLR", n_frames: 80 },
  ];
  return { api: {
    sessionsPage: vi.fn().mockResolvedValue({ sessions: rows, total: rows.length }),
    // Every drive labelled and openable, so the rows render their full complement of controls - which is
    // exactly where the nesting was.
    sessionStates: vi.fn().mockResolvedValue(
      rows.map((r) => ({ session_id: r.session_id, n_frames: r.n_frames, n_objects: 10, reviewed: 0 })),
    ),
    jobs: vi.fn().mockResolvedValue({ autolabel: [] }),
    firstFrame: vi.fn().mockResolvedValue({ frame_id: "f-1" }),
    startAutolabel: vi.fn(),
    cancelJob: vi.fn(),
    sessions: vi.fn().mockResolvedValue(rows),
  }, humanizeError: (e: unknown) => String(e) };
});

describe("the drive row is valid HTML", () => {
  it("never nests a button inside a button", async () => {
    const { container } = render(<SessionPicker sessionId="s-1" onPick={vi.fn()} />);
    await userEvent.click(screen.getByRole("button"));
    // Wait for the real rows, so this cannot pass against an empty list.
    await waitFor(() => expect(screen.getAllByText(/DASHCAM-01/).length).toBeGreaterThan(0));

    const nested = Array.from(container.querySelectorAll("button button"));
    expect(
      nested.map((n) => n.textContent),
      "a <button> inside a <button> is invalid HTML and React will not hydrate it",
    ).toEqual([]);
  });

  it("still offers both actions on a row", async () => {
    // The fix must not have achieved validity by deleting a control.
    render(<SessionPicker sessionId="s-1" onPick={vi.fn()} />);
    await userEvent.click(screen.getByRole("button"));
    await waitFor(() => expect(screen.getAllByText(/ANNO-01/).length).toBeGreaterThan(0));
    expect(screen.getAllByRole("button", { name: /label/i }).length).toBeGreaterThan(0);
  });
});
