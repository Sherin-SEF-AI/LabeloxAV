"use client";

// Slice matrix: per-slice champion vs challenger, so aggregate mAP cannot hide the failures that matter
// on Indian roads. Cells are colored ONLY by regression state (green better, red worse, neutral same). A
// challenger that improves aggregate but regresses a protected slice is what the gate rejects. Enter a champion
// and challenger model version to compare their latest evaluations.

import { useCallback, useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";
import { Badge } from "@/components/ui/kit";
import { apiGet } from "@/lib/api";
import { humanize, plural } from "@/lib/format";

type Row = { slice: string; champion: number | null; challenger: number | null; delta: number | null; state: string };
type Pair = { champion: string; challenger: string };
const STATE: Record<string, string> = { better: "text-pass", worse: "text-block", same: "text-ink-3" };
const STATE_WORD: Record<string, string> = { better: "Better", worse: "Worse", same: "Same" };
const STATE_BADGE: Record<string, "pass" | "block" | "muted"> = { better: "pass", worse: "block", same: "muted" };

const fmt = (v: number | null) => (v != null ? v.toFixed(3) : "-");

export default function VerdyxMatrix() {
  const [champion, setChampion] = useState("");
  const [challenger, setChallenger] = useState("");
  const [rows, setRows] = useState<Row[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pairs, setPairs] = useState<Pair[]>([]);

  const load = useCallback(async () => {
    if (!champion || !challenger) return;
    setErr(null);
    const r = await apiGet<{ error?: string; rows?: Row[] }>(`/api/verdyx/matrix?champion=${encodeURIComponent(champion)}&challenger=${encodeURIComponent(challenger)}`);
    if (r.error) { setErr(r.error); setRows(null); } else setRows(r.rows ?? []);
  }, [champion, challenger]);

  // prefill from a real evaluated pair and compare immediately, so the matrix is populated out of the box
  useEffect(() => {
    apiGet<{ pairs?: Pair[] }>("/api/verdyx/pairs").then((d) => {
      setPairs(d.pairs ?? []);
      const pair = (d.pairs ?? [])[0];
      if (pair) { setChampion(pair.champion); setChallenger(pair.challenger); }
    }).catch(() => {});
  }, []);
  useEffect(() => { if (champion && challenger) load(); }, [champion, challenger, load]);

  const regressed = (rows ?? []).filter((r) => r.state === "worse").length;
  const scored = (rows ?? []).filter((r) => r.delta != null).length;
  // Model versions seen in evaluated pairs, offered as suggestions in both fields.
  const versions = Array.from(new Set(pairs.flatMap((p) => [p.champion, p.challenger])));

  return (
    <PageShell active="Slice evaluation" title="Slice matrix"
      subtitle="Compare two model versions slice by slice, so an overall score cannot hide a slice that got worse."
      meta={rows && scored > 0 ? (
        regressed
          ? <Badge state="block">{plural(regressed, "slice")} regressed</Badge>
          : <Badge state="pass">No slice regressed</Badge>
      ) : undefined}>
      <div className="page-wrap space-y-4">
        <form className="panel p-4 flex flex-wrap items-end gap-3"
          onSubmit={(e) => { e.preventDefault(); void load(); }}>
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] text-ink-2">Champion (current model)</span>
            <input value={champion} onChange={(e) => setChampion(e.target.value)} list="verdyx-versions"
              placeholder="Model version" className="field w-72" />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] text-ink-2">Challenger (candidate)</span>
            <input value={challenger} onChange={(e) => setChallenger(e.target.value)} list="verdyx-versions"
              placeholder="Model version" className="field w-72" />
          </label>
          <datalist id="verdyx-versions">
            {versions.map((v) => <option key={v} value={v} />)}
          </datalist>
          <button type="submit" disabled={!champion || !challenger} className="btn btn-primary">Compare</button>
        </form>

        {err && (
          <div role="alert" className="panel border-block/40 bg-block/5 px-3.5 py-3 text-[12.5px]">
            <div className="text-ink font-medium">Could not compare these versions</div>
            <div className="text-ink-2 mt-0.5 break-words">{err}</div>
          </div>
        )}

        {rows && rows.length > 0 && (
          <section className="panel overflow-hidden">
            <div className="panel-head">
              <h2 className="section-title">{plural(rows.length, "slice")}</h2>
              <span className="ml-auto text-[12px] text-ink-3">Colour shows only whether the challenger is better or worse.</span>
            </div>
            {scored === 0 && (
              <p className="px-3.5 py-2.5 border-b hairline text-[12.5px] text-warn">
                The challenger has no scores on these slices yet, so nothing can be compared. Evaluate it on the
                same slices first.
              </p>
            )}
            <table className="data-table">
              <thead><tr>
                <th>Slice</th><th className="num">Champion</th><th className="num">Challenger</th>
                <th className="num">Change</th><th>Result</th>
              </tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.slice}>
                    <td className="text-ink">{humanize(r.slice)}</td>
                    <td className="num">{fmt(r.champion)}</td>
                    <td className="num text-ink">{fmt(r.challenger)}</td>
                    <td className={`num ${STATE[r.state] ?? ""}`}>
                      {r.delta != null ? (r.delta >= 0 ? "+" : "") + r.delta.toFixed(3) : "-"}
                    </td>
                    <td>
                      {r.delta == null
                        ? <span className="text-ink-3">Not scored on both</span>
                        : <Badge state={STATE_BADGE[r.state] ?? "muted"}>{STATE_WORD[r.state] ?? humanize(r.state)}</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {rows && rows.length === 0 && (
          <div className="panel">
            <EmptyState icon="chart" title="No slices in common">
              These two evaluations do not share any slices, so there is nothing to compare side by side.
              Pick versions that were evaluated on the same slice set.
            </EmptyState>
          </div>
        )}

        {!rows && !err && (
          <div className="panel">
            <EmptyState icon="chart" title="Pick two model versions to compare">
              Enter the model in use as the champion and the candidate as the challenger, then compare.
              Each slice shows both scores and whether the candidate got better or worse.
            </EmptyState>
          </div>
        )}
      </div>
    </PageShell>
  );
}
