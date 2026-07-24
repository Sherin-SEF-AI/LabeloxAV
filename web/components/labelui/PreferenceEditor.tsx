"use client";

import { useState } from "react";
import type { AnnotationRow } from "@/lib/types";

// LLM evaluation and RLHF: side-by-side preference, rubric scoring, and full ranking.
//
// All three land on the same annotation spine as a bounding box. Preference records the candidate list it was
// judged against, not just the winning index: candidates get regenerated and reordered, and a bare "chose #1"
// becomes meaningless (or worse, silently wrong) the moment they do.

type Props = {
  candidates: string[];
  prompt: string;
  annotations: AnnotationRow[];
  onPreference: (chosen: number) => void;
  onRubric: (scores: Record<string, number>) => void;
  onRanking: (order: string[]) => void;
};

const DEFAULT_CRITERIA = ["helpfulness", "accuracy", "safety"];

export default function PreferenceEditor({
  candidates, prompt, annotations, onPreference, onRubric, onRanking,
}: Props) {
  const [scores, setScores] = useState<Record<string, number>>({});
  const [order, setOrder] = useState<string[]>(candidates.map((_, i) => String(i)));

  const pref = annotations.find((a) => a.kind === "preference");
  const chosen = pref ? Number(pref.payload.chosen) : null;

  const move = (i: number, dir: -1 | 1) => {
    const j = i + dir;
    if (j < 0 || j >= order.length) return;
    const next = order.slice();
    [next[i], next[j]] = [next[j], next[i]];
    setOrder(next);
  };

  if (!candidates.length) {
    return (
      <div className="p-4 font-mono text-[11px] text-ink-3">
        this asset has no candidates in meta.candidates, so there is nothing to compare
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <div>
        <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">prompt</div>
        <div className="font-sans text-[14px] text-ink whitespace-pre-wrap">{prompt}</div>
      </div>

      {/* side by side preference */}
      <div>
        <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">which response is better</div>
        <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(candidates.length, 3)}, minmax(0,1fr))` }}>
          {candidates.map((c, i) => (
            <button key={i} onClick={() => onPreference(i)}
              className={`text-left panel p-2 border ${chosen === i ? "border-accent" : "border-line hover:border-ink-3"}`}>
              <div className="font-mono text-[10px] text-ink-3 mb-1">
                candidate {i + 1}{chosen === i ? " - chosen" : ""}
              </div>
              <div className="font-sans text-[13px] text-ink whitespace-pre-wrap">{c}</div>
            </button>
          ))}
        </div>
      </div>

      {/* rubric */}
      <div>
        <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">rubric (1 to 5)</div>
        <div className="flex items-center gap-3 flex-wrap font-mono text-[11px]">
          {DEFAULT_CRITERIA.map((k) => (
            <label key={k} className="flex items-center gap-1">
              <span className="text-ink-3">{k}</span>
              <input type="number" min={1} max={5} step={1} value={scores[k] ?? ""}
                onChange={(e) => setScores({ ...scores, [k]: Number(e.target.value) })}
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-14" />
            </label>
          ))}
          <button onClick={() => onRubric(scores)} disabled={!Object.keys(scores).length}
            className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
            save rubric
          </button>
        </div>
      </div>

      {/* ranking */}
      <div>
        <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">full ranking (best first)</div>
        <div className="space-y-1">
          {order.map((idx, i) => (
            <div key={idx} className="flex items-center gap-2 font-mono text-[11px]">
              <span className="text-ink-3 w-5">{i + 1}.</span>
              <span className="text-ink-2 flex-1 truncate">
                candidate {Number(idx) + 1}: {candidates[Number(idx)]?.slice(0, 60)}
              </span>
              <button onClick={() => move(i, -1)} className="border border-line px-1 text-ink-3 hover:border-accent">up</button>
              <button onClick={() => move(i, 1)} className="border border-line px-1 text-ink-3 hover:border-accent">dn</button>
            </div>
          ))}
        </div>
        <button onClick={() => onRanking(order)}
          className="mt-1 border border-line px-2 py-0.5 font-mono text-[11px] text-ink-2 hover:border-accent">
          save ranking
        </button>
      </div>
    </div>
  );
}
