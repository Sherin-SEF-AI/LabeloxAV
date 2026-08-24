// The correction dialog reported "no similar objects above the threshold" for its entire existence.
//
// It was searching the legacy CLIP table, which nothing writes to, while the embeddings it needed sat in
// pgvector. Every empty result rendered as a statement about similarity, when the truth was that nothing had
// been compared at all. These are the display rules that make the three empty cases distinguishable, plus
// the preselection rule that keeps a bulk apply from overwriting somebody's decision.

import { describe, expect, it } from "vitest";

import { applicableCount, byCurrentClass, defaultSelection, emptyReason, onExcludedTrack } from "./correctionCandidates";
import type { CorrectionCandidate, CorrectionSuggestion } from "./types";

const cand = (over: Partial<CorrectionCandidate> = {}): CorrectionCandidate => ({
  object_id: "o1", frame_id: "f1", class_name: "bus", current: "bus", conf: 0.5,
  state: "review", score: 0.9, crop_url: "/api/objects/o1/crop",
  already: false, source: "auto_accept", human: false, track_id: null, ...over,
});

const sug = (over: Partial<CorrectionSuggestion> = {}): CorrectionSuggestion => ({
  kind: "class", change: {}, count: 0, candidates: [], ...over,
});

describe("what gets preselected", () => {
  it("selects the objects that share the mistake", () => {
    const sel = defaultSelection([cand({ object_id: "a" }), cand({ object_id: "b" })]);
    expect(sel).toEqual(new Set(["a", "b"]));
  });

  it("leaves out anything already at the corrected value", () => {
    expect(defaultSelection([cand({ object_id: "a", already: true }), cand({ object_id: "b" })]))
      .toEqual(new Set(["b"]));
  });

  it("never preselects an object a person already ruled on", () => {
    // Overwriting somebody's decision by default is worse than doing nothing. It stays visible and tickable.
    expect(defaultSelection([cand({ object_id: "a", human: true }), cand({ object_id: "b" })]))
      .toEqual(new Set(["b"]));
  });

  it("an empty candidate list selects nothing rather than throwing", () => {
    expect(defaultSelection([])).toEqual(new Set());
  });
});

describe("why the list is empty", () => {
  it("says so when the object has no embedding, instead of blaming similarity", () => {
    // The actual state of this feature for its whole life. "No similar objects above the threshold" is a
    // claim about similarity, and nothing had been compared.
    const r = emptyReason(sug({ reason: "this object has no visual embedding yet, so nothing can be compared against it" }));
    expect(r?.headline).toContain("no visual embedding");
  });

  it("separates a strict threshold from an empty corpus", () => {
    const r = emptyReason(sug({ reason: "no objects above the similarity threshold", examined: 240 }));
    expect(r?.headline).toContain("above the threshold");
    expect(r?.detail).toContain("240 neighbours were examined");
  });

  it("says nothing was found when nothing was examined either", () => {
    const r = emptyReason(sug({ reason: "no objects above the similarity threshold", examined: 0 }));
    expect(r?.headline).toBe("no similar objects found");
    expect(r?.detail).toBeNull();
  });

  it("is silent when there are candidates to show", () => {
    expect(emptyReason(sug({ candidates: [cand()], count: 1 }))).toBeNull();
  });

  it("handles a suggestion that has not arrived yet", () => {
    expect(emptyReason(null)).toBeNull();
  });
});

describe("the header count", () => {
  it("counts what the correction would change, not the rows returned", () => {
    expect(applicableCount([cand(), cand({ already: true }), cand()])).toBe(2);
  });
});

describe("grouping by current class", () => {
  it("groups the lineages a systematic error was spread across", () => {
    // The relabel agent put objects into one class from bus, traffic_sign and hoarding. A flat grid hides
    // that; the grouping is what makes it a statement about which lineages are involved.
    const groups = byCurrentClass([
      cand({ object_id: "a", class_name: "bus" }),
      cand({ object_id: "b", class_name: "traffic_sign" }),
      cand({ object_id: "c", class_name: "bus" }),
    ]);
    expect(groups.map(([name, cs]) => [name, cs.length])).toEqual([["bus", 2], ["traffic_sign", 1]]);
  });

  it("breaks a tie by name so the order does not jump between queries", () => {
    const groups = byCurrentClass([cand({ class_name: "zebra" }), cand({ class_name: "alpha" })]);
    expect(groups.map(([n]) => n)).toEqual(["alpha", "zebra"]);
  });

  it("an empty list groups into nothing", () => {
    expect(byCurrentClass([])).toEqual([]);
  });
});

describe("the track that was just fixed", () => {
  // A class correction now fans across its whole track before this dialog opens. Those frames come back
  // here as the most visually similar things in the corpus to the object that was just corrected, and
  // ticking them would send them through bulkReview, which writes source=human onto frames nobody looked
  // at. That is the corpus lie the propagated source exists to avoid, arriving by another door.
  const on = (id: string, track: string | null, over: Partial<CorrectionCandidate> = {}) =>
    cand({ object_id: id, track_id: track, ...over });

  it("does not preselect frames of the track the correction already covered", () => {
    const list = [on("a", "T1"), on("b", "T1"), on("c", "T2")];
    expect([...defaultSelection(list, "T1")]).toEqual(["c"]);
  });

  it("preselects everything when no track was propagated", () => {
    const list = [on("a", "T1"), on("b", "T2")];
    expect(defaultSelection(list, null).size).toBe(2);
    expect(defaultSelection(list).size).toBe(2);
  });

  it("leaves the count honest about how much is left to do", () => {
    const list = [on("a", "T1"), on("b", "T1"), on("c", "T2")];
    expect(applicableCount(list, "T1")).toBe(1);
    expect(applicableCount(list)).toBe(3);
  });

  it("does not exclude an untracked candidate", () => {
    expect(onExcludedTrack(on("a", null), "T1")).toBe(false);
  });
});
