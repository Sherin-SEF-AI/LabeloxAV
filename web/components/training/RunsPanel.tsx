"use client";

// Graphical view of recorded model runs (the finetune close-the-loop output). Pick a run to see its
// per-epoch training curves (mAP/precision/recall and box loss), the champion -> candidate metric
// comparison (overall + per class), and the promotion gate's verdict. Read-only; the gate disposes.

import { useEffect, useMemo, useState } from "react";
import { api, type ModelRunRow, type TrainingCurve } from "@/lib/api";
import LineChart from "@/components/charts/LineChart";
import { useQuery } from "@/lib/useQuery";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { Badge, Segmented } from "@/components/ui/kit";
import { humanize, num, plural } from "@/lib/format";

const C = { accent: "#FF7A2F", pass: "#56D364", warn: "#E3B341", block: "#F85149", info: "#58A6FF", ink3: "#6C727A" };

function Delta({ from, to, pct = false }: { from?: number; to?: number; pct?: boolean }) {
  if (from == null || to == null) return <span className="text-ink-3">-</span>;
  const d = to - from;
  const tone = d > 0.0005 ? "text-pass" : d < -0.0005 ? "text-block" : "text-ink-3";
  const fmt = (v: number) => (pct ? `${(v * 100).toFixed(0)}%` : v.toFixed(3));
  return (
    <span className="tabular-nums">
      <span className="text-ink-3">{fmt(from)}</span>
      <span className="text-ink-3"> → </span>
      <span className="text-ink">{fmt(to)}</span>
      <span className={`ml-1 ${tone}`}>{d >= 0 ? "+" : ""}{fmt(d)}</span>
    </span>
  );
}

// A paired baseline(dim) vs candidate(bright) horizontal bar for one class.
function ClassBar({ name, base, cand }: { name: string; base?: number; cand?: number }) {
  const b = Math.max(0, Math.min(1, base ?? 0)), c = Math.max(0, Math.min(1, cand ?? 0));
  return (
    <div className="flex items-center gap-2 text-[12px]">
      <span className="w-28 shrink-0 truncate text-ink-2" title={name}>{humanize(name)}</span>
      <span className="flex-1 relative h-3 bg-line/50">
        <span className="absolute left-0 top-0 h-1.5 opacity-40" style={{ width: `${b * 100}%`, background: C.ink3 }} />
        <span className="absolute left-0 bottom-0 h-1.5" style={{ width: `${c * 100}%`, background: C.accent }} />
      </span>
      <span className="w-32 shrink-0 text-right"><Delta from={base} to={cand} /></span>
    </div>
  );
}

export default function RunsPanel() {
  // Optional panel: a failed read shows "no model runs recorded yet", as it always has.
  const runsQ = useQuery(["trainingRuns"], () => api.trainingRuns());
  const runs: ModelRunRow[] = runsQ.data ?? [];
  const [sel, setSel] = useState<string | null>(null);
  // Keyed by the run, so flipping between two runs redraws from the cache; a run not yet fetched shows
  // "loading curve", and a failed one stays there, as before.
  const curveQ = useQuery(sel ? ["trainingRunCurve", sel] : null, () => api.trainingRunCurve(sel as string));
  const curve: TrainingCurve | null = curveQ.error ? null : curveQ.data ?? null;
  const [metric, setMetric] = useState<"ap" | "recall">("ap");

  // Open on the newest run once the list arrives, unless one is already chosen.
  const firstRunId = runs[0]?.run_id;
  useEffect(() => { if (firstRunId) setSel((cur) => cur ?? firstRunId); }, [firstRunId]);

  const run = runs.find((r) => r.run_id === sel) || null;

  const classRows = useMemo(() => {
    if (!run) return [];
    const bk = metric === "ap" ? run.baseline_metrics.per_class : run.baseline_metrics.per_class_recall;
    const ck = metric === "ap" ? run.metrics.per_class : run.metrics.per_class_recall;
    const names = Array.from(new Set([...Object.keys(bk || {}), ...Object.keys(ck || {})]));
    return names
      .map((nm) => ({ name: nm, base: bk?.[nm], cand: ck?.[nm] }))
      .sort((a, b) => (b.cand ?? 0) - (a.cand ?? 0));
  }, [run, metric]);

  if (runsQ.loading && !runsQ.data) return <LoadState loading rows={4} />;
  if (!runs.length) {
    return (
      <EmptyState compact icon="chart" title="No model runs recorded yet">
        A run appears here once a training job finishes and is evaluated against the champion.
      </EmptyState>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[220px_1fr] gap-4">
      {/* run list */}
      <div className="space-y-1">
        {runs.map((r) => {
          const promoted = r.gate?.promote;
          const active = r.run_id === sel;
          return (
            <button key={r.run_id} onClick={() => setSel(r.run_id)}
              className={`w-full text-left text-[12px] px-2.5 py-2 border rounded ${active ? "border-accent bg-accent/10" : "border-line hover:border-ink-3"}`}>
              <div className="flex items-center justify-between">
                <span className="truncate text-ink-2">{r.dataset_name}</span>
                <span className={`shrink-0 ml-2 ${promoted ? "text-pass" : "text-warn"}`}>{promoted ? "Passed gate" : "Blocked"}</span>
              </div>
              <div className="text-ink-3">
                {plural(r.epochs, "epoch")} · mAP50 {r.metrics.map50?.toFixed(3) ?? "-"}
                {r.promoted && <span className="text-pass"> · Champion</span>}
              </div>
            </button>
          );
        })}
      </div>

      {/* detail */}
      {run && (
        <div className="space-y-4 min-w-0">
          {/* curves */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <div className="border border-line rounded p-3">
              <div className="section-title mb-2">Validation metrics by epoch</div>
              {curve?.available ? (
                <LineChart x={curve.epochs} height={140} yDomain={[0, 1]} series={[
                  { key: "map50", label: "mAP50", color: C.accent, values: curve.series.map50 || [] },
                  { key: "precision", label: "Precision", color: C.info, values: curve.series.precision || [] },
                  { key: "recall", label: "Recall", color: C.pass, values: curve.series.recall || [] },
                ]} />
              ) : <div className="text-[12px] text-ink-3 py-6 text-center">{curveQ.loading ? "Loading curve..." : "No curve recorded for this run."}</div>}
            </div>
            <div className="border border-line rounded p-3">
              <div className="section-title mb-2">Box loss by epoch</div>
              {curve?.available ? (
                <LineChart x={curve.epochs} height={140} series={[
                  { key: "train", label: "Train", color: C.warn, values: curve.series.train_box_loss || [] },
                  { key: "val", label: "Validation", color: C.block, values: curve.series.val_box_loss || [] },
                ]} />
              ) : <div className="text-[12px] text-ink-3 py-6 text-center">{curveQ.loading ? "Loading curve..." : "No curve recorded for this run."}</div>}
            </div>
          </div>

          {/* champion -> candidate overall */}
          <div className="border border-line rounded p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="section-title">Champion → candidate, on held-out validation</span>
              <Badge state={run.gate?.promote ? "pass" : "warn"}>
                {run.gate?.promote ? "Gate: promote" : "Gate: blocked"}
              </Badge>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-2 text-[12px]">
              {([["mAP50", "map50"], ["mAP", "map"], ["Precision", "precision"], ["Recall", "recall"]] as const).map(([lbl, k]) => (
                <div key={k} className="flex flex-col">
                  <span className="label-caps">{lbl}</span>
                  <Delta from={run.baseline_metrics[k]} to={run.metrics[k]} />
                </div>
              ))}
            </div>
            {run.gate?.reasons?.length ? (
              <div className="mt-3 text-[12px] text-ink-3 border-t hairline pt-2 space-y-0.5">
                {run.gate.reasons.map((rs, i) => (
                  <div key={i} className="flex gap-1.5"><span className={run.gate?.promote ? "text-pass" : "text-warn"}>·</span>{rs}</div>
                ))}
              </div>
            ) : null}
          </div>

          {/* per-class comparison */}
          <div className="border border-line rounded p-3">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div>
                <div className="section-title">Per class ({metric === "ap" ? "AP50" : "recall"})</div>
                <div className="text-[11.5px] text-ink-3 mt-0.5">Grey bar: champion. Orange bar: candidate.</div>
              </div>
              <Segmented options={[{ value: "ap", label: "AP50" }, { value: "recall", label: "Recall" }]}
                value={metric} onChange={(m) => setMetric(m as "ap" | "recall")} />
            </div>
            <div className="space-y-1">
              {classRows.length ? classRows.map((r) => <ClassBar key={r.name} name={r.name} base={r.base} cand={r.cand} />)
                : <div className="text-[12px] text-ink-3 text-center py-2">No per-class metrics were recorded for this run.</div>}
            </div>
          </div>

          <div className="text-[11.5px] text-ink-3">
            <span className="font-mono">{run.run_id}</span> · started from {run.base_weights?.split("/").slice(-1)[0]} · {num(run.n_train)} training / {num(run.n_val)} validation images
            {run.notes ? ` · ${run.notes}` : ""}
          </div>
        </div>
      )}
    </div>
  );
}
