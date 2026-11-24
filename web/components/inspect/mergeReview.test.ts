import { describe, expect, it } from "vitest";

import type { ReviewResult } from "@/lib/api";
import type { ObjectDetail } from "@/lib/types";

import { mergeReview } from "./EvidencePanel";

// After "apply suggestion" or "save box" the panel stored the review response as the object it was
// showing. The response has no frame id and no image url, so the image went blank and the footer threw.

const shown = {
  object_id: "obj-1", frame_id: "frame-1", session_id: "s-1", cam_id: "cam_front", image_url: "/api/frames/frame-1/image",
  width: 1920, height: 1080, class_id: 4, class_name: "sedan", bbox: [10, 10, 50, 40], conf: 0.72,
  state: "review", source: "fused", version: 3, attrs: { occlusion: 1 },
} as unknown as ObjectDetail;

const reviewed: ReviewResult = {
  object_id: "obj-1", class_id: 9, class_name: "suv", bbox: [12, 11, 52, 41], attrs: { occlusion: 2 },
  state: "accepted", source: "human", version: 4, rot_deg: 0, bbox_amodal: null, keypoints: null,
};

describe("mergeReview", () => {
  it("takes what the review changed and keeps everything the response does not carry", () => {
    const next = mergeReview(shown, reviewed)!;
    expect(next).toMatchObject({ class_name: "suv", class_id: 9, state: "accepted", source: "human", version: 4,
                                 bbox: [12, 11, 52, 41], attrs: { occlusion: 2 } });
    // The fields that used to vanish.
    expect(next.frame_id).toBe("frame-1");
    expect(next.image_url).toBe("/api/frames/frame-1/image");
    expect(next.width).toBe(1920);
    expect(next.conf).toBe(0.72);
  });

  it("keeps the attributes already shown when the response has none", () => {
    expect(mergeReview(shown, { ...reviewed, attrs: null })!.attrs).toEqual({ occlusion: 1 });
  });

  it("ignores a response for a different object, which is what a slow reply after moving on is", () => {
    expect(mergeReview(shown, { ...reviewed, object_id: "obj-2" })).toBe(shown);
  });

  it("has nothing to update when nothing is shown", () => {
    expect(mergeReview(null, reviewed)).toBeNull();
  });
});
