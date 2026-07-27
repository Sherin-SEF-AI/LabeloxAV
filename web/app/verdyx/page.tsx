"use client";

// VERDYX slice matrix: per-slice champion vs challenger, so aggregate mAP cannot hide the failures that matter
// on Indian roads. Cells are colored ONLY by regression state (green better, red worse, neutral same). A
// challenger that improves aggregate but regresses a protected slice is what the gate rejects. Enter a champion
// and challenger model version to compare their latest evaluations.

import { useCallback, useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { apiGet } from "@/lib/api";

type Row = { slice: string; champion: number | null; challenger: number | null; delta: number | null; state: string };
const STATE: Record<string, string> = { better: "text-pass", worse: "text-block", same: "text-ink-3" };

export default function VerdyxMatrix() {
  const [champion, setChampion] = useState("");
  const [challenger, setChallenger] = useState("");
  const [rows, setRows] = useState<Row[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!champion || !challenger) return;
    setErr(null);
    const r = await apiGet<{ error?: string; rows?: Row[] }>(`/api/verdyx/matrix?champion=${encodeURIComponent(champion)}&challenger=${encodeURIComponent(challenger)}`);
    if (r.error) { setErr(r.error); setRows(null); } else setRows(r.rows ?? []);
  }, [champion, challenger]);

  // prefill from a real evaluated pair and compare immediately, so the matrix is populated out of the box
  useEffect(() => {
    apiGet<{ pairs?: { champion: string; challenger: string }[] }>("/api/verdyx/pairs").then((d) => {
      const pair = (d.pairs ?? [])[0];
      if (pair) { setChampion(pair.champion); setChallenger(pair.challenger); }
    }).catch(() => {});
  }, []);
  useEffect(() => { if (champion && challenger) load(); }, [champion, challenger, load]);

  const regressed = (rows ?? []).filter((r) => r.state === "worse").length;

  return (
    <PageShell active="VERDYX" title="VERDYX slice matrix"
      right={rows ? <span className="font-mono text-[11px]">{regressed ? <span className="text-block">{regressed} slice(s) regressed</span> : <span className="text-pass">no slice regression</span>}</span> : undefined}>
      <div className="p-4 max-w-4xl space-y-3 font-mono text-[11px]">
        <div className="flex items-end gap-2">
          <label className="flex flex-col gap-1"><span className="text-ink-3 text-[10px] uppercase">champion</span>
            <input value={champion} onChange={(e) => setChampion(e.target.value)} className="bg-panel border border-line px-2 py-1 text-ink w-64" /></label>
          <label className="flex flex-col gap-1"><span className="text-ink-3 text-[10px] uppercase">challenger</span>
            <input value={challenger} onChange={(e) => setChallenger(e.target.value)} className="bg-panel border border-line px-2 py-1 text-ink w-64" /></label>
          <button onClick={load} className="border border-line px-3 py-1 hover:border-accent">compare</button>
        </div>

        {err && <div className="text-warn">{err}</div>}

        {rows && (
          <table className="w-full">
            <thead><tr className="text-ink-3 text-left border-b hairline">
              <th className="px-2 py-1">slice</th><th>champion</th><th>challenger</th><th>delta</th><th></th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.slice} className="border-b hairline">
                  <td className="px-2 py-1 text-ink-2">{r.slice}</td>
                  <td className="text-ink-3">{r.champion != null ? r.champion.toFixed(3) : "-"}</td>
                  <td className="text-ink">{r.challenger != null ? r.challenger.toFixed(3) : "-"}</td>
                  <td className={STATE[r.state]}>{r.delta != null ? (r.delta >= 0 ? "+" : "") + r.delta.toFixed(3) : "-"}</td>
                  <td><span className={`inline-block w-3 h-3 ${r.state === "better" ? "bg-pass" : r.state === "worse" ? "bg-block" : "bg-line"}`} /></td>
                </tr>
              ))}
              {!rows.length && <tr><td colSpan={5} className="text-ink-3 text-center py-4">no shared slices between these evaluations</td></tr>}
            </tbody>
          </table>
        )}
        {!rows && !err && <div className="text-ink-3 py-6 text-center">enter a champion and challenger version, then compare</div>}
      </div>
    </PageShell>
  );
}
