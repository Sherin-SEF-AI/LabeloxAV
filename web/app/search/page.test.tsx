import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import SearchPage from "./page";

// Label by example, on the page that could already find the neighbours and could not do anything with
// them.
//
// CorrectionModal is a complete version of this flow and opens only after somebody has already made a
// correction, so the way to find every other object that looks like this one was to relabel it wrongly
// and change your mind. This page had the crop grid and the rerank controls, and clicking a result
// navigated away one object at a time.
//
// Two claims are pinned. An object result selects rather than navigates, because navigating is what the
// mode replaces. And a frame result still navigates, because there is no bulk verdict for a frame and
// silently turning those tiles into a selection would break the search this page was already good at.

const OBJECTS = [0, 1, 2].map((i) => ({
  object_id: `obj-${i}`, frame_id: `frame-${i}`, class_name: "traffic_sign",
  crop_url: `/api/objects/obj-${i}/crop`, score: 0.9 - i * 0.05,
}));

const { bulkReview, searchSimilar, push, agentRevert } = vi.hoisted(() => ({
  bulkReview: vi.fn(), searchSimilar: vi.fn(), push: vi.fn(), agentRevert: vi.fn(),
}));

let query = "object=obj-seed";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push, back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(query),
  // PageShell renders PlatformSwitcher, which reads the path to decide which platform is current.
  usePathname: () => "/search",
}));
vi.mock("@/lib/toast", () => ({ toast: vi.fn() }));
vi.mock("@/lib/user", () => ({
  acceptState: () => "accepted",
  useCurrentUser: () => ({ name: "rev", role: "reviewer" }),
}));
// The shell is stubbed to its two slots. It mounts the nav, the upload indicator and the job stream,
// none of which this test is about, and each of which reaches for a different corner of the api module.
// PageShell has its own render tests; the search body is what is under test here and it still mounts.
vi.mock("@/components/shell/PageShell", () => ({
  default: ({ filters, children }: { filters?: React.ReactNode; children: React.ReactNode }) => (
    <div>{filters}{children}</div>
  ),
}));
vi.mock("@/lib/api", () => ({
  humanizeError: (e: unknown) => String(e),
  api: {
    searchSimilar,
    bulkReview,
    agentRevert,
    ontology: vi.fn(async () => ({ version: "t", classes: [
      { id: 3, name: "hoarding", l0: "static", l1: "fixed", india: false },
      { id: 4, name: "bmtc_bus_shelter", l0: "static", l1: "fixed", india: true },
    ] })),
    addClass: vi.fn(),
    parseQuery: vi.fn(async () => ({ filters: {}, classes: [] })),
  },
}));

async function mounted() {
  render(<SearchPage />);
  await waitFor(() => expect(screen.getByTitle(/sim 0\.900/)).toBeTruthy());
}

describe("label by example on the search page", () => {
  beforeEach(() => {
    query = "object=obj-seed";
    push.mockReset();
    bulkReview.mockReset().mockResolvedValue({
      updated: 2, action: "reclassify", skipped_missing: [], skipped_stale: [], run_id: "run-1",
    });
    searchSimilar.mockReset().mockResolvedValue({ kind: "object", mode: "visual", results: OBJECTS });
  });

  it("selects an object result instead of navigating away from it", async () => {
    await mounted();
    await userEvent.click(screen.getByTitle(/sim 0\.900/));
    expect(push).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.getByText("1 selected")).toBeTruthy());
  });

  it("applies a class to the whole selection through bulk review", async () => {
    await mounted();
    await userEvent.click(screen.getByTitle(/sim 0\.900/));
    await userEvent.click(screen.getByTitle(/sim 0\.850/));
    await userEvent.click(screen.getByTitle("set a class on the selection (C)"));
    await userEvent.click(await screen.findByText("bmtc_bus_shelter"));

    await waitFor(() => expect(bulkReview).toHaveBeenCalledTimes(1));
    const [ids, action, className] = bulkReview.mock.calls[0];
    // The exact confusion services/intelligence/corrections.py documents: one mistake spread across
    // bus, traffic_sign and hoarding, fixed in one action rather than three visits.
    expect(ids).toEqual(["obj-0", "obj-1"]);
    expect(action).toBe("reclassify");
    expect(className).toBe("bmtc_bus_shelter");
  });

  it("A accepts the found set and R rejects it", async () => {
    await mounted();
    await userEvent.click(screen.getByTitle(/sim 0\.900/));
    await userEvent.keyboard("a");
    await waitFor(() => expect(bulkReview).toHaveBeenCalledTimes(1));
    expect(bulkReview.mock.calls[0][1]).toBe("confirm");

    await userEvent.click(screen.getByTitle(/sim 0\.850/));
    await userEvent.keyboard("r");
    await waitFor(() => expect(bulkReview).toHaveBeenCalledTimes(2));
    expect(bulkReview.mock.calls[1][1]).toBe("reject");
    expect(bulkReview.mock.calls[1][3]).toBe("rejected");
  });

  it("a verdict with nothing selected lands on the tile under the cursor, not on everything", async () => {
    await mounted();
    await userEvent.keyboard("{ArrowRight}");
    await userEvent.keyboard("a");
    await waitFor(() => expect(bulkReview).toHaveBeenCalledTimes(1));
    expect(bulkReview.mock.calls[0][0]).toEqual(["obj-1"]);
  });

  it("still navigates for frame results, which have no bulk verdict", async () => {
    query = "frame=frame-seed";
    searchSimilar.mockResolvedValue({
      kind: "frame", mode: "visual",
      results: [{ frame_id: "frame-9", image_url: "/x.jpg", score: 0.9 }],
    });
    render(<SearchPage />);
    await waitFor(() => expect(screen.getByTitle(/sim 0\.900/)).toBeTruthy());
    await userEvent.click(screen.getByTitle(/sim 0\.900/));
    expect(push).toHaveBeenCalledWith("/frame/frame-9");
    expect(bulkReview).not.toHaveBeenCalled();
  });
});
