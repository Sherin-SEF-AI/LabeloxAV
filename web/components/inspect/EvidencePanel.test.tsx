import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import EvidencePanel from "./EvidencePanel";

// The panel exists because the console was asking people to confirm or dismiss a label from a paragraph of
// prose about an image it never showed them. So the tests that matter are the ones about what it does when
// the image is NOT there: an empty grey box beside a confirm button is the same defect in a new place.
//
// The other load-bearing behaviour is the box overlay. It is an SVG in image-pixel coordinates over an
// object-contain img, and the whole reason for that is that neither layer needs scaling arithmetic. If the
// viewBox ever stops matching the frame's real dimensions the boxes drift off the objects while still
// looking plausible, which is worse than not drawing them.

const router = { push: vi.fn() };
vi.mock("next/navigation", () => ({ useRouter: () => router }));

const object = vi.fn();
const frameObjects = vi.fn();
const review = vi.fn();
vi.mock("@/lib/api", () => ({
  api: {
    object: (...a: unknown[]) => object(...a),
    frameObjects: (...a: unknown[]) => frameObjects(...a),
    review: (...a: unknown[]) => review(...a),
  },
  humanizeError: (e: unknown) => String((e as Error)?.message ?? e),
}));

const toastError = vi.fn();
vi.mock("@/lib/toast", () => ({
  toast: vi.fn(), toastError: (...a: unknown[]) => toastError(...a), toastSuccess: vi.fn(),
}));

const OBJ = {
  object_id: "obj-1", frame_id: "frame-1", session_id: "s", ts_ns: 0, cam_id: "front",
  image_url: "/api/frames/frame-1/image", width: 1920, height: 1080,
  class_id: 3, class_name: "e_auto", bbox: [100, 200, 300, 400], mask_polygons: [],
  attrs: {}, conf: 0.55, state: "review", source: "fused", provenance: {}, version: 2,
};

beforeEach(() => {
  vi.clearAllMocks();
  object.mockResolvedValue(OBJ);
  frameObjects.mockResolvedValue([]);
});

describe("EvidencePanel", () => {
  it("says what it is for when nothing is selected, rather than rendering an empty box", async () => {
    render(<EvidencePanel subject={null} />);
    expect(screen.getByText(/select a row/i)).toBeInTheDocument();
    expect(object).not.toHaveBeenCalled();
  });

  it("renders the crop and the frame for the selected object", async () => {
    render(<EvidencePanel subject={{ objectId: "obj-1" }} />);
    const crop = await screen.findByAltText("e_auto");
    expect(crop).toHaveAttribute("src", expect.stringContaining("/api/objects/obj-1/crop"));
    expect(screen.getByAltText("frame")).toHaveAttribute("src", "/api/frames/frame-1/image");
  });

  it("draws the box in image-pixel coordinates so it cannot drift from the object", async () => {
    // The viewBox must be the frame's real size. An SVG sized in CSS pixels would put every box in the
    // wrong place while still drawing something, which reads as a labelling error rather than a bug.
    const { container } = render(<EvidencePanel subject={{ objectId: "obj-1" }} />);
    await screen.findByAltText("frame");
    const svg = container.querySelector("svg");
    expect(svg).toHaveAttribute("viewBox", "0 0 1920 1080");
    expect(svg).toHaveAttribute("preserveAspectRatio", "xMidYMid meet");
    const rect = container.querySelector("rect");
    expect(rect).toHaveAttribute("x", "100");
    expect(rect).toHaveAttribute("width", "200");
  });

  it("says so when the crop cannot be loaded instead of showing an empty frame", async () => {
    // The failure this whole panel is meant to prevent: a decision made against a blank rectangle.
    render(<EvidencePanel subject={{ objectId: "obj-1" }} />);
    const crop = await screen.findByAltText("e_auto");
    // fireEvent rather than dispatchEvent: it wraps the resulting state update in act, so the assertion
    // below is against a settled render rather than a racing one.
    fireEvent.error(crop);
    expect(await screen.findByText(/crop could not be loaded/i)).toBeInTheDocument();
  });

  it("draws the frame's other boxes, because a duplicate claim needs both", async () => {
    // The largest kind in this queue claims another box covers the same object. One box cannot show that.
    frameObjects.mockResolvedValue([
      { object_id: "obj-1", bbox: [100, 200, 300, 400], class_name: "e_auto", state: "review" },
      { object_id: "obj-2", bbox: [110, 210, 310, 410], class_name: "sedan", state: "review" },
    ]);
    const { container } = render(<EvidencePanel subject={{ objectId: "obj-1" }} />);
    await screen.findByAltText("frame");
    await waitFor(() => expect(container.querySelectorAll("rect").length).toBe(2));
    expect(screen.getByText(/1 other box shown dim/i)).toBeInTheDocument();
  });

  it("shows the row's own words next to the image they are about", async () => {
    render(<EvidencePanel subject={{
      objectId: "obj-1", kind: "vlm disagrees", score: 1,
      text: "The image shows a building facade, not any type of vehicle.",
    }} />);
    expect(await screen.findByText(/building facade/i)).toBeInTheDocument();
    expect(screen.getByText(/vlm disagrees/i)).toBeInTheDocument();
  });

  it("sends the version it read, so a stale write is refused rather than silently winning", async () => {
    review.mockResolvedValue({ ...OBJ, class_name: "sedan", version: 3 });
    render(<EvidencePanel subject={{
      objectId: "obj-1", suggestion: { class_id: 9, class_name: "sedan" },
    }} />);
    await userEvent.click(await screen.findByRole("button", { name: /relabel as sedan/i }));
    await waitFor(() => expect(review).toHaveBeenCalledWith("obj-1", expect.objectContaining({
      action: "reclassify", class_name: "sedan", expected_version: 2,
    })));
  });

  it("does not offer to apply a class the object already has", async () => {
    render(<EvidencePanel subject={{
      objectId: "obj-1", suggestion: { class_id: 3, class_name: "e_auto" },
    }} />);
    expect(await screen.findByRole("button", { name: /already e_auto/i })).toBeDisabled();
  });

  it("reports a failed action instead of appearing to have worked", async () => {
    const run = vi.fn().mockRejectedValue(new Error("needs reviewer role"));
    const onResolved = vi.fn();
    render(<EvidencePanel subject={{ objectId: "obj-1" }} onResolved={onResolved}
      actions={[{ key: "confirm", label: "confirm", run }]} />);
    await userEvent.click(await screen.findByRole("button", { name: "confirm" }));
    await waitFor(() => expect(toastError).toHaveBeenCalledWith(expect.stringMatching(/needs reviewer role/)));
    // The row must not be dropped from the queue on a failure, or the work silently disappears.
    expect(onResolved).not.toHaveBeenCalled();
  });

  it("surfaces a failure to load the object rather than rendering a blank panel", async () => {
    object.mockRejectedValue(new Error("object not found"));
    render(<EvidencePanel subject={{ objectId: "gone" }} />);
    expect(await screen.findByText(/object not found/i)).toBeInTheDocument();
  });

  it("deep links into the editor for anything it cannot do itself", async () => {
    render(<EvidencePanel subject={{ objectId: "obj-1" }} />);
    await userEvent.click(await screen.findByRole("button", { name: /open editor/i }));
    expect(router.push).toHaveBeenCalledWith("/frame/frame-1?focus=obj-1");
  });
});
