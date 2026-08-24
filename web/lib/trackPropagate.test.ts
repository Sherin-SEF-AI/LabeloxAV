import { describe, expect, it } from "vitest";

import { propagationMessage, shouldPropagate } from "./trackPropagate";

// The predicate half of "why did my rename not travel". Both directions matter: not propagating when it
// should is the original bug, and propagating when it should not would relabel a track on a geometry drag.

const obj = (over: Partial<Parameters<typeof shouldPropagate>[0]> = {}) =>
  ({ id: "aaaaaaaa-1111", track_id: "tttttttt-2222", ...over } as NonNullable<Parameters<typeof shouldPropagate>[0]>);

describe("shouldPropagate", () => {
  it("travels when the class actually changed on a tracked object", () => {
    expect(shouldPropagate(obj(), "sedan", "minivan")).toBe(true);
  });

  it("does not relabel a track on a geometry-only edit", () => {
    // The editor's save sends class_name on every dirty object, so the presence of a class in the payload
    // proves nothing. Dragging a box must not rewrite 93 frames.
    expect(shouldPropagate(obj(), "sedan", "sedan")).toBe(false);
  });

  it("does nothing for an untracked object, which is 2% of the corpus", () => {
    expect(shouldPropagate(obj({ track_id: null }), "sedan", "minivan")).toBe(false);
    expect(shouldPropagate(obj({ track_id: undefined }), "sedan", "minivan")).toBe(false);
  });

  it("does nothing for a box drawn in this session", () => {
    // Not saved, so it has no track and no siblings.
    expect(shouldPropagate(obj({ isNew: true }), "sedan", "minivan")).toBe(false);
    expect(shouldPropagate(obj({ id: "tmp-3" }), "sedan", "minivan")).toBe(false);
  });

  it("does nothing when there is no previous class to compare against", () => {
    expect(shouldPropagate(obj(), null, "minivan")).toBe(false);
  });

  it("does nothing for no object", () => {
    expect(shouldPropagate(null, "sedan", "minivan")).toBe(false);
  });
});

describe("propagationMessage", () => {
  it("leads with what changed and where", () => {
    expect(propagationMessage({ relabeled: 92, class_name: "minivan" }))
      .toBe("minivan applied to 92 more frames on this track");
  });

  it("says nothing about clauses that carry no information", () => {
    const m = propagationMessage({ relabeled: 92, class_name: "minivan", skipped_human: [], id_switch_events: 0 });
    expect(m).not.toMatch(/left alone|review|occlusion/);
  });

  it("names the frames it refused to overwrite", () => {
    const m = propagationMessage({ relabeled: 90, class_name: "minivan", skipped_human: ["a", "b"] });
    expect(m).toMatch(/2 left alone \(a person had already labelled them\)/);
  });

  it("tells an annotator their approval went to review rather than counting", () => {
    expect(propagationMessage({ relabeled: 3, class_name: "minivan", clamped: true }))
      .toMatch(/saved for review/);
  });

  it("surfaces an ID switch instead of blocking on it", () => {
    // 9,139 tracks carry one. Blocking would refuse the fix on most of the corpus; the undo is in the
    // same toast, so the operator is told the one fact that would make them look.
    expect(propagationMessage({ relabeled: 92, class_name: "minivan", id_switch_events: 3 }))
      .toMatch(/re-identified after an occlusion at 3 points/);
  });

  it("reads correctly for a single frame", () => {
    const m = propagationMessage({ relabeled: 1, class_name: "bus", skipped_human: ["x"], id_switch_events: 1 });
    expect(m).toContain("1 more frame on this track");
    expect(m).toContain("1 left alone (a person had already labelled it)");
    expect(m).toContain("at 1 point,");
  });
});
