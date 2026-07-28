"use client";

// SIEVYX queue-composition dashboard: what the label budget is being spent on. The combined priority (model
// uncertainty + embedding rarity) ranks the queue; this shows the class mix, the rarity-band split, and the
// mean per-signal contribution, so a human can see whether the budget is buying rare tail data or common data.
// Operational Materialism: matte, monospace, bars colored only by state.

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { apiGet } from "@/lib/api";

type Composition = {
  n: number;
  by_class: { class_name: string; count: number; share: number; mean_value: number }[];
  by_rarity_band: Record<string, { count: number; share: number }>;
  mean_signals: Record<string, number>;
  mean_value: number;
};

const BAND_COLOR: Record<string, string> = { high: "bg-pass", medium: "bg-warn", low: "bg-ink-3" };

export default function SievyxComposition() {
  const [c, setC] = useState<Composition | null>(null);
  useEffect(() => { apiGet<Composition>("/api/sievyx/composition?top_n=800").then(setC).catch(() => {}); }, []);

  return (
    <PageShell active="SIEVYX" title="SIEVYX queue composition"
      right={<span className="font-mono text-[11px] text-ink-3">{c ? `${c.n} items in the priority window` : "loading"}</span>}>
      <div className="p-4 grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4 max-w-5xl">
        {/* class mix */}
        <section className="border border-line p-3">
          <div className="font-mono text-[10px] uppercase text-ink-3 mb-2">class mix (what the budget buys)</div>
          <div className="space-y-1">
            {(c?.by_class ?? []).slice(0, 18).map((b) => (
              <div key={b.class_name} className="flex items-center gap-2 font-mono text-[10px]">
                <span className="w-28 shrink-0 truncate text-ink-2">{b.class_name}</span>
                <span className="flex-1 h-3 bg-line/40 relative">
                  <span className="absolute left-0 top-0 h-full bg-accent" style={{ width: `${b.share * 100}%` }} />
                </span>
                <span className="w-24 shrink-0 text-right text-ink-3">{(b.share * 100).toFixed(1)}% · {b.count}</span>
              </div>
            ))}
            {!c?.by_class?.length && <div className="text-ink-3 font-mono text-[11px] py-4 text-center">no scored candidates</div>}
          </div>
        </section>

        {/* rarity band + signals */}
        <section className="space-y-4">
          <div className="border border-line p-3">
            <div className="font-mono text-[10px] uppercase text-ink-3 mb-2">rarity band</div>
            {["high", "medium", "low"].map((k) => {
              const v = c?.by_rarity_band?.[k];
              return (
                <div key={k} className="flex items-center gap-2 font-mono text-[10px] mb-1">
                  <span className="w-16 text-ink-2">{k}</span>
                  <span className="flex-1 h-3 bg-line/40 relative">
                    <span className={`absolute left-0 top-0 h-full ${BAND_COLOR[k]}`} style={{ width: `${(v?.share ?? 0) * 100}%` }} />
                  </span>
                  <span className="w-14 text-right text-ink-3">{v ? `${(v.share * 100).toFixed(0)}%` : "-"}</span>
                </div>
              );
            })}
            <div className="mt-2 font-mono text-[9px] text-ink-3">high rarity = rare tail data, the high-value labels</div>
          </div>

          <div className="border border-line p-3">
            <div className="font-mono text-[10px] uppercase text-ink-3 mb-2">mean signal contribution</div>
            {Object.entries(c?.mean_signals ?? {}).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2 font-mono text-[10px] mb-1">
                <span className="w-24 text-ink-2">{k}</span>
                <span className="flex-1 h-3 bg-line/40 relative">
                  <span className="absolute left-0 top-0 h-full bg-info" style={{ width: `${Math.min(1, v) * 100}%` }} />
                </span>
                <span className="w-12 text-right text-ink-3">{v.toFixed(2)}</span>
              </div>
            ))}
            <div className="mt-2 font-mono text-[9px] text-ink-3">mean priority value {c?.mean_value?.toFixed(3) ?? "-"}</div>
          </div>
        </section>
      </div>
    </PageShell>
  );
}
