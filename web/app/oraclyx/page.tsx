"use client";

// ORACLYX consensus board: the offline-fusion pseudo-GT verdict at a glance. Where fusion and the three
// auto-label paths agree, the pseudo-label is auto-truthed; where they disagree, the sample is routed to the
// human queue. This shows the split (the whole point: humans only touch disagreements) and the distillation
// set size. Operational Materialism, color only by state.

import { useCallback, useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { Stat } from "@/components/ui/kit";
import { apiGet } from "@/lib/api";
import { num, pct } from "@/lib/format";

type Board = { total: number; auto_accepted: number; routed_to_human: number; auto_accept_rate: number };

export default function OraclyxBoard() {
  const [b, setB] = useState<Board | null>(null);
  const [distill, setDistill] = useState<number | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const load = useCallback(() => {
    setErr(null);
    apiGet<Board>("/api/oraclyx/board").then(setB).catch(setErr);
    apiGet<{ n: number }>("/api/oraclyx/distillation").then((d) => setDistill(d.n)).catch(() => {});
  }, []);
  useEffect(() => { load(); }, [load]);

  const auto = b?.auto_accepted ?? 0;
  const human = b?.routed_to_human ?? 0;
  const total = b?.total ?? 0;

  return (
    <PageShell active="ORACLYX" title="Consensus board"
      subtitle="Labels the models all agree on are accepted; only the ones they disagree on go to a person"
      right={<span className="text-[12.5px] text-ink-3">{num(distill ?? 0)} in the training set</span>}>
      <div className="page-wrap-narrow space-y-4">
        {err != null && <LoadState error={err} onRetry={load} />}
        {!b && err == null ? <LoadState loading rows={3} /> : b && (!total ? (
          <div className="panel">
            <EmptyState icon="wand" title="No machine labels compared yet">
              When several models label the same recordings, their answers are compared here. Agreement is
              accepted automatically; disagreement goes to the review queue. Nothing has been compared yet.
            </EmptyState>
          </div>
        ) : (<>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <Stat label="Machine labels" value={num(total)} />
            <Stat label="Accepted" value={num(auto)} tone="pass" sub="All models agreed, no person needed" />
            <Stat label="Sent to a person" value={num(human)} tone="warn" sub="Models disagreed" />
          </div>

          <section className="panel">
            <div className="panel-head">
              <h2 className="section-title">Agreement</h2>
              <span className="text-[12px] text-ink-3">{pct(b.auto_accept_rate, 1)} accepted without review</span>
            </div>
            <div className="p-3.5">
              <div className="h-6 bg-bg-2 rounded-sm overflow-hidden flex text-[11px] font-medium">
                <span className="h-full bg-pass flex items-center justify-center text-bg"
                  style={{ width: `${(auto / total) * 100}%` }}>{auto ? num(auto) : ""}</span>
                <span className="h-full bg-warn flex items-center justify-center text-bg"
                  style={{ width: `${(human / total) * 100}%` }}>{human ? num(human) : ""}</span>
              </div>
              <div className="mt-2 flex gap-4 text-[12px] text-ink-3">
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-pass" />Accepted</span>
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-warn" />Sent to a person</span>
              </div>
            </div>
          </section>
        </>))}
      </div>
    </PageShell>
  );
}
