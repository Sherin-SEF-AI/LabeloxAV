import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ErrorWhy } from "@/lib/tools/qualityApi";

const getFrameWhy = vi.fn();
const getVlmChecks = vi.fn();
vi.mock("@/lib/tools/qualityApi", async (orig) => ({
  ...(await orig<typeof import("@/lib/tools/qualityApi")>()),
  getFrameWhy: (...a: unknown[]) => getFrameWhy(...a),
  getVlmChecks: (...a: unknown[]) => getVlmChecks(...a),
}));
vi.mock("next/link", () => ({ default: ({ href, children, ...p }: { href: string; children: React.ReactNode }) => <a href={href} {...p}>{children}</a> }));

import ErrorOverlay from "./ErrorOverlay";

const why = (over: Partial<ErrorWhy> = {}): ErrorWhy => ({
  candidate_id: "c1", object_id: "o1", kind: "reanalyze", detector: "Flagged on re-check", rule: "duplicate",
  score: 0.7, status: "pending", proposed: null,
  subject: { object_id: "o1", frame_id: "f1", class_name: "sedan", bbox: [100, 100, 200, 150], conf: 0.5, state: "review", source: "fused" },
  frame: { frame_id: "f1", width: 1920, height: 1080, image_url: "/api/frames/f1/image" },
  reason: "Another box on this frame covers the same object with higher confidence.",
  facts: [{ label: "IoU with the other box", value: "0.91" }],
  evidence: [
    { bbox: [100, 100, 200, 150], role: "subject", label: "sedan", object_id: "o1" },
    { bbox: [102, 101, 199, 152], role: "overlaps", label: "sedan (IoU 0.91)", object_id: "o2" },
  ],
  neighbours: [], twin_frame: null, ...over,
});

beforeEach(() => {
  getFrameWhy.mockReset();
  getVlmChecks.mockReset();
  getVlmChecks.mockResolvedValue({ checks: {} });
});

describe("ErrorOverlay", () => {
  it("places a tag on each flagged object through the viewport and opens the explanation", async () => {
    getFrameWhy.mockResolvedValue({ frame_id: "f1", width: 1920, height: 1080, items: [why()] });
    const onSelect = vi.fn();
    render(<ErrorOverlay frameId="f1" viewport={{ scale: 0.5, ox: 10, oy: 20 }} onSelectObject={onSelect} />);
    const tag = await screen.findByRole("button", { name: /Likely error on this sedan/ });
    // Right edge of the box (10 + 0.5*200 = 110) minus half the tag, top (20 + 0.5*100 = 70) minus half.
    expect(tag.style.left).toBe("101px");
    expect(tag.style.top).toBe("61px");
    expect(screen.getByText("1 likely error")).toBeInTheDocument();
    fireEvent.click(tag);
    expect(onSelect).toHaveBeenCalledWith("o1");
    const card = screen.getByRole("dialog", { name: "Why this label was flagged" });
    expect(card).toHaveTextContent("covers the same object");
    expect(card).toHaveTextContent("IoU with the other box");
    expect(screen.getByText("sedan (IoU 0.91)")).toBeInTheDocument();   // the compared box, drawn on the canvas
    expect(await screen.findByText("Not checked yet.")).toBeInTheDocument();   // the vision model's slot
  });

  it("opens the selected object's flag and groups several checks on one object", async () => {
    getFrameWhy.mockResolvedValue({ frame_id: "f1", width: 1920, height: 1080,
      items: [why(), why({ candidate_id: "c2", kind: "track_inconsistent", detector: "Class changes along its track",
        reason: "Most of its track is labelled suv.", evidence: [why().evidence[0]] })] });
    render(<ErrorOverlay frameId="f1" viewport={{ scale: 1, ox: 0, oy: 0 }} selectedObjectId="o1" />);
    const card = await screen.findByRole("dialog");
    expect(card).toHaveTextContent("Most of its track is labelled suv.");
    expect(screen.getByRole("button", { name: /Likely error on this sedan/ })).toHaveTextContent("!2");
    await screen.findByText("Not checked yet.");
  });

  it("renders nothing when hidden and says so when the checks cannot load", async () => {
    const { container, rerender } = render(<ErrorOverlay frameId="f1" viewport={{ scale: 1, ox: 0, oy: 0 }} visible={false} />);
    expect(container).toBeEmptyDOMElement();
    expect(getFrameWhy).not.toHaveBeenCalled();
    getFrameWhy.mockRejectedValue(new Error("boom"));
    rerender(<ErrorOverlay frameId="f1" viewport={{ scale: 1, ox: 0, oy: 0 }} />);
    await waitFor(() => expect(screen.getByText("Error checks unavailable")).toBeInTheDocument());
  });
});
