"use client";

import { useEffect, useRef, useState } from "react";

import { apiGet, humanizeError } from "@/lib/api";

// Why the queue is in the order it is, measured rather than asserted (services/queue_order.py). The numbers
// are recomputed on a gold set when the panel opens: how many of the wrong labels a reviewer meets in the
// first 10, 25 and 50% of the queue, for this queue's ordering and the alternatives it was chosen over.

type Ordering = { found: number[]; ap: number; ap_p95?: number };
type Report = {
  gold_id: string; pool: number; wrong: number; wrong_rate: number | null; fractions: number[];
  orderings: Record<string, Ordering>; best?: string; refused?: string;
};

const NAMES: Record<string, string> = {
  queue_formula: "This queue",
  al_value: "Active-learning value",
  lowest_confidence: "Lowest confidence first",
  random: "Random order",
};
const pct = (x: number) => `${Math.round(x * 100)}%`;

export default function QueueOrdering() {
  const [open, setOpen] = useState(false);
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open || report || error) return;
    apiGet<Report>("/api/triage/ordering").then(setReport).catch((e) => setError(humanizeError(e)));
  }, [open, report, error]);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", esc); };
  }, [open]);

  return (
    <div className="relative" ref={box}>
      <button className="text-[12px] text-accent hover:text-accent-2" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        Why this order?
      </button>
      {open && (
        <div role="dialog" aria-label="How the queue is ordered"
             className="absolute right-0 top-full mt-2 z-30 w-[min(92vw,440px)] panel p-4 shadow-lg flex flex-col gap-3 text-[12.5px]">
          <p className="text-ink-2">
            The labels most likely to be wrong come first: how unsure the detector was, doubled for rare classes,
            raised when the mask and box disagree or the models proposed different classes.
          </p>
          {!report && !error && <p className="text-ink-3">Measuring on the newest gold set...</p>}
          {error && <p className="text-block">{error}</p>}
          {report?.refused && <p className="text-ink-3">{report.refused}</p>}
          {report && !report.refused && (
            <>
              <p className="text-ink-3">
                Measured now on gold: {report.pool} machine labels overlap a gold label, and {pct(report.wrong_rate ?? 0)} of
                them are wrong. Share of those wrong labels a reviewer has met after working through:
              </p>
              <div className="overflow-x-auto">
                <table className="w-full tabular-nums">
                  <thead>
                    <tr className="text-ink-3 text-left border-b hairline">
                      <th className="font-medium py-1.5 pr-2">Ordering</th>
                      {report.fractions.map((f) => <th key={f} className="font-medium py-1.5 px-1 text-right">first {pct(f)}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(report.orderings).map(([k, o]) => (
                      <tr key={k} className={`border-b hairline last:border-0 ${k === "queue_formula" ? "text-ink font-medium" : "text-ink-2"}`}>
                        <td className="py-1.5 pr-2">{NAMES[k] ?? k}{k === report.best ? <span className="ml-1.5 text-pass text-[11px]">best</span> : null}</td>
                        {o.found.map((v, i) => <td key={i} className="py-1.5 px-1 text-right">{pct(v)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-ink-3 text-[11.5px]">
                Gold set <span className="font-mono">{report.gold_id}</span>. Only labels that overlap a gold label can be checked, so this is a sample of the queue, not all of it.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
