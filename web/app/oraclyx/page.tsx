"use client";

// ORACLYX consensus board: the offline-fusion pseudo-GT verdict at a glance. Where fusion and the three
// auto-label paths agree, the pseudo-label is auto-truthed; where they disagree, the sample is routed to the
// human queue. This shows the split (the whole point: humans only touch disagreements) and the distillation
// set size. Operational Materialism, color only by state.

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { apiGet } from "@/lib/api";

type Board = { total: number; auto_accepted: number; routed_to_human: number; auto_accept_rate: number };

export default function OraclyxBoard() {
  const [b, setB] = useState<Board | null>(null);
  const [distill, setDistill] = useState<number | null>(null);
  useEffect(() => {
    apiGet<Board>("/api/oraclyx/board").then(setB).catch(() => {});
    apiGet<{ n: number }>("/api/oraclyx/distillation").then((d) => setDistill(d.n)).catch(() => {});
  }, []);

  const auto = b?.auto_accepted ?? 0;
  const human = b?.routed_to_human ?? 0;
  const total = b?.total ?? 0;

  return (
    <PageShell active="ORACLYX" title="ORACLYX consensus board"
      right={<span className="font-mono text-[11px] text-ink-3">{total} pseudo-labels · {distill ?? 0} in distillation set</span>}>
      <div className="p-4 max-w-3xl space-y-4">
        <div className="grid grid-cols-3 gap-3 font-mono">
          <div className="border border-line p-4">
            <div className="text-[10px] uppercase text-ink-3">total</div>
            <div className="text-2xl text-ink mt-1">{total}</div>
          </div>
          <div className="border border-line p-4">
            <div className="text-[10px] uppercase text-ink-3">auto-truthed</div>
            <div className="text-2xl text-pass mt-1">{auto}</div>
            <div className="text-[10px] text-ink-3">consensus, no human touch</div>
          </div>
          <div className="border border-line p-4">
            <div className="text-[10px] uppercase text-ink-3">routed to human</div>
            <div className="text-2xl text-warn mt-1">{human}</div>
            <div className="text-[10px] text-ink-3">disagreement, highest value</div>
          </div>
        </div>

        <div className="border border-line p-3">
          <div className="font-mono text-[10px] uppercase text-ink-3 mb-2">
            consensus split · auto-accept rate {((b?.auto_accept_rate ?? 0) * 100).toFixed(1)}%
          </div>
          <div className="h-5 bg-line/40 relative flex font-mono text-[9px]">
            <span className="h-full bg-pass flex items-center justify-center text-bg"
              style={{ width: `${total ? (auto / total) * 100 : 0}%` }}>{auto || ""}</span>
            <span className="h-full bg-warn flex items-center justify-center text-bg"
              style={{ width: `${total ? (human / total) * 100 : 0}%` }}>{human || ""}</span>
          </div>
          <div className="mt-2 font-mono text-[9px] text-ink-3">
            the fusion path auto-truths the easy majority so humans only ever touch the disagreements
          </div>
        </div>

        {!total && <div className="font-mono text-[11px] text-ink-3 text-center py-6">
          no pseudo-labels yet; ORACLYX records them as fusion runs over sessions
        </div>}
      </div>
    </PageShell>
  );
}
