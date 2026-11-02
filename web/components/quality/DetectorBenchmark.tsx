"use client";

import { useState } from "react";

import { apiPost, humanizeError } from "@/lib/api";

// How good the class-error detectors are, measured by planting errors (services/errordetect/benchmark.py).
// Human verdicts measure a detector only as fast as people judge its candidates; planting answers today. The
// run happens in a transaction that is rolled back, so no label changes.

type Pooled = { caught: number; planted: number; new_flags: number; recall: number | null; precision: number | null };
type Result = { pooled: Record<string, Pooled>; rate: number; sessions: { labels: number; refused?: string }[] };

const NAMES: Record<string, string> = {
  confident_learning: "Confident learning",
  track_consistency: "Track consistency",
  embedding_outlier: "Embedding outlier",
};
const pct = (x: number | null) => (x == null ? "n/a" : `${Math.round(x * 100)}%`);

export default function DetectorBenchmark() {
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    try { setResult(await apiPost<Result>("/api/errordetect/benchmark", {})); }
    catch (e) { setError(humanizeError(e)); }
    finally { setBusy(false); }
  };

  const measured = result?.sessions.filter((s) => !s.refused) ?? [];
  const planted = result ? Object.values(result.pooled)[0]?.planted ?? 0 : 0;

  return (
    <section className="panel px-3 py-2.5 flex flex-col gap-2 text-[12.5px]">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-ink font-medium">Do the detectors find real errors?</span>
        <span className="text-ink-3">Plant wrong classes in {result ? `${measured.length} sessions` : "the busiest sessions"}, see which detectors catch them, then undo.</span>
        <button className="btn ml-auto" onClick={() => void run()} disabled={busy}>{busy ? "Measuring" : result ? "Measure again" : "Measure"}</button>
      </div>
      {error && <p className="text-block">{error}</p>}
      {result && (
        <div className="overflow-x-auto">
          <table className="w-full tabular-nums">
            <thead>
              <tr className="text-left text-ink-3 border-b hairline">
                <th className="font-medium py-1.5 pr-2">Detector</th>
                <th className="font-medium py-1.5 px-2 text-right">Caught</th>
                <th className="font-medium py-1.5 px-2 text-right">Recall</th>
                <th className="font-medium py-1.5 px-2 text-right">Precision of new flags</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.pooled).sort(([, a], [, b]) => (b.recall ?? 0) - (a.recall ?? 0)).map(([k, v]) => (
                <tr key={k} className="border-b hairline last:border-0">
                  <td className="py-1.5 pr-2 text-ink">{NAMES[k] ?? k}</td>
                  <td className="py-1.5 px-2 text-right text-ink-2">{v.caught} of {v.planted}</td>
                  <td className={`py-1.5 px-2 text-right ${(v.recall ?? 0) >= 0.7 ? "text-pass" : (v.recall ?? 0) < 0.2 ? "text-block" : "text-warn"}`}>{pct(v.recall)}</td>
                  <td className="py-1.5 px-2 text-right text-ink-2">{pct(v.precision)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1.5 text-ink-3 text-[11.5px]">
            {planted} labels given a wrong class ({Math.round(result.rate * 100)}% of each session&apos;s accepted labels, a sibling class where one exists). Planted errors are single labels, easier than a wrong class on a whole track. Nothing was saved.
          </p>
        </div>
      )}
    </section>
  );
}
