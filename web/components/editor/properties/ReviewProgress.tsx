"use client";

// How much of this frame has been ruled on, as a bar rather than a number nobody reads.
//
// The three segments are who accepted each object, not how sure the model was: see panelStats.ts for why
// the design's "confirmed / reviewed / low conf" cannot be built as drawn. Every dot carries the exact
// predicate in its title, so the bar is auditable instead of decorative, and the ratio is the same
// expression the page footer prints so the two cannot disagree.

import { type ReviewCounts, reviewWidths } from "./panelStats";

export default function ReviewProgress({ counts }: { counts: ReviewCounts }) {
  const w = reviewWidths(counts);
  const seg = [
    { k: "confirmed", n: counts.confirmed, pct: w.confirmed, bar: "bg-pass", dot: "bg-pass",
      tip: "state accepted: a person ruled on it" },
    { k: "auto", n: counts.auto, pct: w.auto, bar: "bg-pass/40", dot: "bg-pass/40",
      tip: "state auto_accept: the gate accepted it and nobody has looked since" },
    { k: "open", n: counts.open, pct: w.open, bar: "bg-warn", dot: "bg-warn",
      tip: "review, annotate, submitted or rejected: still work to do" },
  ];

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[9.5px] uppercase tracking-wider text-ink-3">review progress</span>
        <span className="ml-auto font-mono text-[10px] text-ink-2 tabular-nums">
          {counts.confirmed} / {counts.total}
        </span>
      </div>
      <div className="h-[3px] rounded-sm bg-line-2 overflow-hidden flex" role="img"
        aria-label={`${counts.confirmed} confirmed, ${counts.auto} auto-accepted, ${counts.open} open, of ${counts.total}`}>
        {seg.map((s) => <div key={s.k} className={s.bar} style={{ width: `${s.pct}%` }} />)}
      </div>
      <div className="flex gap-3 font-mono text-[9.5px] text-ink-3">
        {seg.map((s) => (
          <span key={s.k} className="flex items-center gap-1.5" title={s.tip}>
            <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
            {s.n} {s.k}
          </span>
        ))}
      </div>
    </div>
  );
}
