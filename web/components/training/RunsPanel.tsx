"use client";

// Graphical view of recorded model runs (the finetune close-the-loop output). Pick a run to see its
// per-epoch training curves (mAP/precision/recall and box loss), the champion -> candidate metric
// comparison (overall + per class), and the promotion gate's verdict. Read-only; the gate disposes.

import { useEffect, useMemo, useState } from "react";
import { api, type ModelRunRow, type TrainingCurve } from "@/lib/api";
import LineChart from "@/components/charts/LineChart";

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
    <div className="flex items-center gap-2 font-mono text-[10px]">
      <span className="w-24 shrink-0 truncate text-ink-2" title={name}>{name}</span>
      <span className="flex-1 relative h-3 bg-line/50">
        <span className="absolute left-0 top-0 h-1.5 opacity-40" style={{ width: `${b * 100}%`, background: C.ink3 }} />
        <span className="absolute left-0 bottom-0 h-1.5" style={{ width: `${c * 100}%`, background: C.accent }} />
      </span>
      <span className="w-24 shrink-0 text-right"><Delta from={base} to={cand} /></span>
    </div>
  );
}

export default function RunsPanel() {
  const [runs, setRuns] = useState<ModelRunRow[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [curve, setCurve] = useState<TrainingCurve | null>(null);
  const [metric, setMetric] = useState<"ap" | "recall">("ap");

  useEffect(() => {
    api.trainingRuns().then((r) => { setRuns(r); if (r.length && !sel) setSel(r[0].run_id); }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (!sel) return;
    setCurve(null);
    api.trainingRunCurve(sel).then(setCurve).catch(() => setCurve(null));
  }, [sel]);

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

  if (!runs.length) return <div className="font-mono text-xs text-ink-3 py-4 text-center">no model runs recorded yet</div>;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[220px_1fr] gap-4">
      {/* run list */}
      <div className="space-y-1">
        {runs.map((r) => {
          const promoted = r.gate?.promote;
          const active = r.run_id === sel;
          return (
            <button key={r.run_id} onClick={() => setSel(r.run_id)}
              className={`w-full text-left font-mono text-[10px] px-2 py-1.5 border ${active ? "border-accent bg-line/40" : "border-line hover:border-ink-3"}`}>
              <div className="flex items-center justify-between">
                <span className="truncate text-ink-2">{r.dataset_name}</span>
                <span className={promoted ? "text-pass" : "text-warn"}>{promoted ? "gate ✓" : "blocked"}</span>
              </div>
              <div className="text-ink-3">
                ep {r.epochs} · map50 {r.metrics.map50?.toFixed(3) ?? "-"}
                {r.promoted && <span className="text-pass"> · champion</span>}
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
            <div className="border border-line p-2">
              <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">val metrics / epoch</div>
              {curve?.available ? (
                <LineChart x={curve.epochs} height={140} yDomain={[0, 1]} series={[
                  { key: "map50", label: "mAP50", color: C.accent, values: curve.series.map50 || [] },
                  { key: "precision", label: "precision", color: C.info, values: curve.series.precision || [] },
                  { key: "recall", label: "recall", color: C.pass, values: curve.series.recall || [] },
                ]} />
              ) : <div className="font-mono text-[10px] text-ink-3 py-6 text-center">loading curve…</div>}
            </div>
            <div className="border border-line p-2">
              <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">box loss / epoch</div>
              {curve?.available ? (
                <LineChart x={curve.epochs} height={140} series={[
                  { key: "train", label: "train", color: C.warn, values: curve.series.train_box_loss || [] },
                  { key: "val", label: "val", color: C.block, values: curve.series.val_box_loss || [] },
                ]} />
              ) : <div className="font-mono text-[10px] text-ink-3 py-6 text-center">loading curve…</div>}
            </div>
          </div>

          {/* champion -> candidate overall */}
          <div className="border border-line p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[10px] uppercase text-ink-3">champion → candidate (held-out val)</span>
              <span className={`font-mono text-[10px] px-1.5 py-0.5 border rounded ${run.gate?.promote ? "border-pass text-pass" : "border-warn text-warn"}`}>
                gate: {run.gate?.promote ? "promote" : "blocked"}
              </span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-1 font-mono text-[10px]">
              {([["mAP50", "map50"], ["mAP", "map"], ["precision", "precision"], ["recall", "recall"]] as const).map(([lbl, k]) => (
                <div key={k} className="flex flex-col">
                  <span className="text-ink-3 uppercase text-[9px]">{lbl}</span>
                  <Delta from={run.baseline_metrics[k]} to={run.metrics[k]} />
                </div>
              ))}
            </div>
            {run.gate?.reasons?.length ? (
              <div className="mt-2 font-mono text-[10px] text-ink-3 border-t hairline pt-1.5">
                {run.gate.reasons.map((rs, i) => (
                  <div key={i} className="flex gap-1.5"><span className={run.gate?.promote ? "text-pass" : "text-warn"}>·</span>{rs}</div>
                ))}
              </div>
            ) : null}
          </div>

          {/* per-class comparison */}
          <div className="border border-line p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[10px] uppercase text-ink-3">per-class ({metric === "ap" ? "AP50" : "recall"}) · dim=champion, orange=candidate</span>
              <div className="flex gap-1 font-mono text-[9px]">
                {(["ap", "recall"] as const).map((m) => (
                  <button key={m} onClick={() => setMetric(m)}
                    className={`px-1.5 border ${metric === m ? "border-accent text-accent" : "border-line text-ink-3"}`}>{m === "ap" ? "AP50" : "recall"}</button>
                ))}
              </div>
            </div>
            <div className="space-y-1">
              {classRows.length ? classRows.map((r) => <ClassBar key={r.name} name={r.name} base={r.base} cand={r.cand} />)
                : <div className="font-mono text-[10px] text-ink-3 text-center py-2">no per-class metrics recorded</div>}
            </div>
          </div>

          <div className="font-mono text-[9px] text-ink-3">
            {run.run_id} · base {run.base_weights?.split("/").slice(-1)[0]} · {run.n_train} train / {run.n_val} val imgs
            {run.notes ? ` · ${run.notes}` : ""}
          </div>
        </div>
      )}
    </div>
  );
}
