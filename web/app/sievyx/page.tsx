"use client";

// SIEVYX queue-composition dashboard: what the label budget is being spent on. The combined priority (model
// uncertainty + embedding rarity) ranks the queue; this shows the class mix, the rarity-band split, and the
// mean per-signal contribution, so a human can see whether the budget is buying rare tail data or common data.
// Operational Materialism: matte, monospace, bars colored only by state.

import { useCallback, useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { apiGet } from "@/lib/api";
import { humanize, num, pct } from "@/lib/format";

type Composition = {
  n: number;
  by_class: { class_name: string; count: number; share: number; mean_value: number }[];
  by_rarity_band: Record<string, { count: number; share: number }>;
  mean_signals: Record<string, number>;
  mean_value: number;
};

const BAND_COLOR: Record<string, string> = { high: "bg-pass", medium: "bg-warn", low: "bg-ink-3" };
const BAND_LABEL: Record<string, string> = { high: "Rare", medium: "Uncommon", low: "Common" };
// The ranking's own signal names, as a person would say them.
const SIGNAL_LABEL: Record<string, string> = {
  uncertainty: "Model unsure", diversity: "Unlike the rest", rarity: "Rare class",
  error_prone: "Often mislabelled", flicker: "Flickers between frames", fn: "Likely missed object",
  reasoner_conflict: "Second opinion disagrees",
};

function Row({ label, frac, fill, value, tip, narrow = false }: { label: string; frac: number; fill: string; value: string; tip?: string; narrow?: boolean }) {
  return (
    <div className="flex items-center gap-3 text-[12.5px]" data-tip={tip}>
      <span className={`${narrow ? "w-36" : "w-40"} shrink-0 truncate text-ink-2`}>{label}</span>
      <span className="flex-1 h-2.5 bg-bg-2 rounded-sm relative overflow-hidden">
        <span className={`absolute left-0 top-0 h-full rounded-sm ${fill}`} style={{ width: `${Math.max(0, Math.min(1, frac)) * 100}%` }} />
      </span>
      <span className={`${narrow ? "w-11" : "w-28"} shrink-0 text-right text-ink-3 tabular-nums`}>{value}</span>
    </div>
  );
}

export default function SievyxComposition() {
  const [c, setC] = useState<Composition | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const load = useCallback(() => {
    setErr(null);
    apiGet<Composition>("/api/sievyx/composition?top_n=800").then(setC).catch(setErr);
  }, []);
  useEffect(() => { load(); }, [load]);

  return (
    <PageShell active="SIEVYX" title="Label budget"
      subtitle="What the next labels in the queue would be spent on, by class and by how rare they are"
      right={<span className="text-[12.5px] text-ink-3">{c ? `Next ${num(c.n)} items in the queue` : ""}</span>}>
      <div className="page-wrap space-y-4">
        {err != null && <LoadState error={err} onRetry={load} />}
        {!c && err == null ? <LoadState loading rows={6} /> : c && (
          !c.by_class?.length ? (
            <div className="panel">
              <EmptyState icon="chart" title="Nothing in the labelling queue yet">
                Once frames are auto-labelled and scored, this shows which classes the next labels would go to
                and whether they are rare or common data.
              </EmptyState>
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-4">
              <section className="panel">
                <div className="panel-head">
                  <h2 className="section-title">Where the labels go</h2>
                  <span className="text-[12px] text-ink-3">Share of the queue, by class</span>
                </div>
                <div className="p-3.5 space-y-1.5">
                  {c.by_class.slice(0, 18).map((b) => (
                    <Row key={b.class_name} label={humanize(b.class_name)} frac={b.share} fill="bg-accent"
                      value={`${pct(b.share, 1)} · ${num(b.count)}`} />
                  ))}
                </div>
              </section>

              <div className="space-y-4">
                <section className="panel">
                  <div className="panel-head"><h2 className="section-title">How rare it is</h2></div>
                  <div className="p-3.5 space-y-1.5">
                    {["high", "medium", "low"].map((k) => {
                      const v = c.by_rarity_band?.[k];
                      return <Row key={k} label={BAND_LABEL[k]} frac={v?.share ?? 0} fill={BAND_COLOR[k]} value={v ? pct(v.share) : "-"} narrow />;
                    })}
                    <p className="section-desc pt-1.5">Rare data teaches the model the most per label.</p>
                  </div>
                </section>

                <section className="panel">
                  <div className="panel-head"><h2 className="section-title">Why items rank high</h2></div>
                  <div className="p-3.5 space-y-1.5">
                    {Object.entries(c.mean_signals ?? {}).map(([k, v]) => (
                      <Row key={k} label={SIGNAL_LABEL[k] ?? humanize(k)} frac={Math.min(1, v)} fill="bg-info"
                        value={v.toFixed(2)} tip={`Average contribution of "${humanize(k)}" to the ranking`} narrow />
                    ))}
                    <p className="section-desc pt-1.5">Average priority score {c.mean_value?.toFixed(2) ?? "-"} (0 to 1).</p>
                  </div>
                </section>
              </div>
            </div>
          )
        )}
      </div>
    </PageShell>
  );
}
