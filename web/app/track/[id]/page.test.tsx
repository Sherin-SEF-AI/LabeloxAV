import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import TrackEditor from "./page";

// Tube review's one keystroke, and the thing it must not become.
//
// A is the whole feature: 7,512 tracks of ten or more objects hold 549,038 objects between them, so a
// reviewer confirming a track with one key stands in for fifty per-frame verdicts. There is exactly one
// rule about what A means, it is printed above the strip, and it is the one thing here worth pinning:
// with nothing selected A confirms the whole track through the track endpoint, and with a selection it
// confirms only those frames through bulk review.
//
// Getting that backwards is silent in both directions. An A that always accepts the whole track discards
// a reviewer's careful selection of the three bad crops; an A that always accepts the selection turns the
// headline keystroke into a one-frame accept and the reviewer notices nothing until they scroll back.

const ITEMS = [0, 1, 2, 3].map((i) => ({
  object_id: `obj-${i}`,
  frame_id: `frame-${i}`,
  ts_ns: 1000 + i,
  class_id: 3,
  class_name: i === 2 ? "minivan" : "sedan",   // one flip, so the outlier styling has something to paint
  bbox: [1, 1, 50, 90],
  state: "review",
  conf: 0.5,
  source: "fused",
  is_keyframe: false,
  interp_source: null,
  crop_url: `/api/objects/obj-${i}/crop`,
}));

// vi.hoisted, because vi.mock's factory is lifted above every module-level const and would otherwise
// close over these before they are initialised.
const { acceptTrack, bulkReview, relabelTrack, cropSheet } = vi.hoisted(() => ({
  acceptTrack: vi.fn(), bulkReview: vi.fn(), relabelTrack: vi.fn(), cropSheet: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "track-abc-def-123" }),
  useRouter: () => ({ push: vi.fn(), back: vi.fn() }),
}));
vi.mock("@/lib/nav", () => ({ useSmartBack: () => vi.fn() }));
vi.mock("@/components/ConfirmProvider", () => ({ useConfirm: () => vi.fn(async () => true) }));
vi.mock("@/lib/toast", () => ({ toast: vi.fn() }));
vi.mock("@/lib/user", () => ({
  acceptState: () => "accepted",
  useCurrentUser: () => ({ name: "rev", role: "reviewer" }),
}));

vi.mock("@/lib/api", () => ({
  humanizeError: (e: unknown) => String(e),
  api: {
    track: vi.fn(async () => ({
      track_id: "track-abc-def-123", n_frames: ITEMS.length,
      classes: { sedan: 3, minivan: 1 }, dominant: "sedan", flips: true,
      items: ITEMS, intents: [],
    })),
    // Listed explicitly rather than behind a Proxy, so a call the page adds later fails loudly here
    // instead of silently resolving undefined.
    ontology: vi.fn(async () => ({ version: "test", classes: [
      { id: 3, name: "sedan", l0: "vehicle", l1: "four_wheeler", india: false },
      { id: 4, name: "minivan", l0: "vehicle", l1: "four_wheeler", india: false },
    ] })),
    intentVocab: vi.fn(async () => ({ vehicle: [], vru: [] })),
    trackEvents: vi.fn(async () => ({ events: [], event_types: [] })),
    // EventLane asks for the track's changepoints so a dragged span edge can snap to one.
    trackChangepoints: vi.fn(async () => ({
      track_id: "track-abc-def-123", source: "object_speed", samples: 0, changepoints: [],
    })),
    cropSheet,
    acceptTrack,
    bulkReview,
    relabelTrack,
    interpolateTrack: vi.fn(async () => ({ created: 0 })),
    deleteTrack: vi.fn(async () => ({ n_objects: 0 })),
    addClass: vi.fn(async () => ({ id: 4, name: "minivan", existed: true })),
  },
}));

async function mounted() {
  render(<TrackEditor />);
  // The strip is what the keymap acts on, so nothing is asserted until it is on screen.
  await waitFor(() => expect(screen.getByTitle(/frame 1: sedan/)).toBeTruthy());
}

describe("tube review strip", () => {
  beforeEach(() => {
    acceptTrack.mockReset().mockResolvedValue({
      accepted: 4, state: "accepted", clamped: false, run_id: "run-1",
      skipped_human: [], skipped_stale: [], id_switch_events: 0,
    });
    bulkReview.mockReset().mockResolvedValue({
      updated: 1, action: "confirm", skipped_missing: [], skipped_stale: [], run_id: "run-2",
    });
    relabelTrack.mockReset().mockResolvedValue({ relabeled: 4 });
    cropSheet.mockReset().mockResolvedValue({
      cell: 96, cols: 2, rows: 2, count: 4, frames_decoded: 4, crops: 4, sheet: "data:image/jpeg;base64,xx",
      placements: ITEMS.map((it, i) => ({ object_id: it.object_id, row: Math.floor(i / 2), col: i % 2, ok: true })),
    });
  });

  it("A with nothing selected confirms the whole track", async () => {
    await mounted();
    await userEvent.keyboard("a");

    await waitFor(() => expect(acceptTrack).toHaveBeenCalledTimes(1));
    expect(acceptTrack.mock.calls[0][0]).toBe("track-abc-def-123");
    // Not bulk review over four ids the server can resolve from one track id, and emphatically not
    // relabel, which would assert a class nobody stated and rewrite every box's source to "propagated".
    expect(bulkReview).not.toHaveBeenCalled();
    expect(relabelTrack).not.toHaveBeenCalled();
  });

  it("A with a selection confirms only the selected frames", async () => {
    await mounted();
    await userEvent.keyboard("{ArrowRight}");   // cursor onto frame 2
    await userEvent.keyboard(" ");              // select it
    await userEvent.keyboard("a");

    await waitFor(() => expect(bulkReview).toHaveBeenCalledTimes(1));
    const [ids, action] = bulkReview.mock.calls[0];
    expect(ids).toEqual(["obj-1"]);
    expect(action).toBe("confirm");
    expect(acceptTrack).not.toHaveBeenCalled();
  });

  it("shift+arrow extends the range the verdict lands on", async () => {
    await mounted();
    await userEvent.keyboard("{Shift>}{ArrowRight}{ArrowRight}{/Shift}");
    await userEvent.keyboard("a");

    await waitFor(() => expect(bulkReview).toHaveBeenCalledTimes(1));
    expect(bulkReview.mock.calls[0][0]).toEqual(["obj-0", "obj-1", "obj-2"]);
  });

  it("fetches crops as sprite sheets rather than one request per tile", async () => {
    await mounted();
    // The strip issued one <img src=/crop> per tile before, so a 697-frame track opened 697 connections.
    await waitFor(() => expect(cropSheet).toHaveBeenCalledTimes(1));
    expect(cropSheet.mock.calls[0][0]).toEqual(["obj-0", "obj-1", "obj-2", "obj-3"]);
    expect(document.querySelectorAll("img").length).toBe(0);
  });

  it("does not act on keys typed into the class search box", async () => {
    await mounted();
    const box = screen.getByPlaceholderText("search class...");
    await userEvent.type(box, "auto");
    expect(acceptTrack).not.toHaveBeenCalled();
    expect(bulkReview).not.toHaveBeenCalled();
  });
});
