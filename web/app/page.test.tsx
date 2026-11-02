import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import HomePage from "./page";

const startAutolabel = vi.fn(() => Promise.resolve({ job_id: "job-1", status: "pending" }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), back: vi.fn() }),
  usePathname: () => "/",
  useSearchParams: () => new URLSearchParams(),
}));
vi.mock("@/lib/toast", () => ({ toast: vi.fn() }));
vi.mock("@/components/shell/PageShell", () => ({ default: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }));
vi.mock("@/components/SampleDrives", () => ({ default: () => <div>samples</div> }));
vi.mock("@/lib/user", () => ({ useCurrentUser: () => ({ user_id: "u", name: "admin", role: "annotator" }) }));
vi.mock("@/lib/api", () => {
  const home = {
    corpus: { sessions: 1, frames: 418, objects: 0, waiting: 0, settled: 0, by_state: {} },
    jobs: { active: { autolabel: 0, import: 0, training: 0 }, active_total: 0, running: 0, failed: [], failed_window_days: 14 },
    recent_sessions: [{ session_id: "s-ctc", vehicle_id: "SAMPLE-CTC", city: "CTC", start_ts_ns: 1_793_000_000_000_000_000,
                        frames: 418, objects: 0, waiting: 0 }],
    steps: { data: true, autolabel: false, review: false, export: false, model: false },
  };
  return {
    apiGet: () => Promise.resolve(home),
    api: { startAutolabel: (...a: unknown[]) => startAutolabel(...(a as [])), firstFrame: vi.fn() },
    humanizeError: (e: unknown) => String(e),
  };
});

describe("home, first run", () => {
  it("shows where the team is in the five steps", async () => {
    render(<HomePage />);
    await waitFor(() => expect(screen.getByText("1 of 5 done")).toBeInTheDocument());
    expect(screen.getByText("Auto-label a session")).toBeInTheDocument();
  });

  it("offers to auto-label the newest unlabelled session, and starts it on that session", async () => {
    render(<HomePage />);
    const button = await screen.findByRole("button", { name: "Auto-label SAMPLE-CTC" });
    await userEvent.click(button);
    expect(startAutolabel).toHaveBeenCalledWith("s-ctc");
    expect(await screen.findByRole("link", { name: "Follow it in Jobs" })).toHaveAttribute("href", "/jobs");
  });
});
