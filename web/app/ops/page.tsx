"use client";

// M19 hardening + scale: the operability surface. A per-plane SLO board (which plane is the bottleneck), a
// byte-stable reproducibility checker (a release must rebuild to the same bytes), and the label-budget
// efficiency report (what each slice's labels actually bought). Operational Materialism: a plane row is green
// only when it meets its SLO; a breach is amber/red; efficiency bars are neutral, ranked.

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, Bar, Verdict, RunButton, useRun, ErrLine } from "@/components/engine/prim";
import { getJSON, runJSON } from "@/lib/engine";

type PlaneSlo = { plane: string; met: boolean;
  breaches: { metric: string; value: number | null; threshold: number; op: string; reason: string }[];
  metrics: Record<string, number>; created_at: string | null };
type Board = { planes: PlaneSlo[]; all_met: boolean };
type Repro = { reproducible: boolean; hash_a: string; hash_b: string; first_divergence: string | null };
type EffRow = { slice: string; labels: number; gain: number; gain_per_1k: number; negative_roi: boolean };
type EffReport = { per_slice: EffRow[]; total_labels: number; total_gain: number; overall_gain_per_1k: number;
  best_slice: string | null; wasted_spend_slices: string[] };

const DEMO_EFF = JSON.stringify([
  { slice: "vru_night", labels: 500, map_before: 0.40, map_after: 0.55 },
  { slice: "car_day", labels: 2000, map_before: 0.80, map_after: 0.805 },
  { slice: "sign_rain", labels: 300, map_before: 0.60, map_after: 0.58 },
], null, 2);
const DEMO_A = JSON.stringify({ commit: "c1", samples: 1000, metrics: { map50: 0.612 } }, null, 2);

export default function OpsPage() {
  const [board, setBoard] = useState<Board | null>(null);
  const [effIn, setEffIn] = useState(DEMO_EFF);
  const [runA, setRunA] = useState(DEMO_A);
  const [runB, setRunB] = useState(DEMO_A);
  const eff = useRun<EffReport>();
  const repro = useRun<Repro>();

  useEffect(() => { getJSON<Board>("/api/hardening/slo/board").then(setBoard).catch(() => {}); }, []);

  return (
    <PageShell active="Operations" title="Operations: SLOs, reproducibility, efficiency"
      subtitle="per-plane service objectives, byte-stable rebuilds, and label-budget return"
      right={board && <Verdict ok={board.all_met} yes="all planes met" no="slo breach" />}>
      <div className="p-4 grid grid-cols-1 xl:grid-cols-2 gap-4 max-w-6xl">
        {/* SLO board */}
        <Panel title="per-plane SLO board" hint="unobserved counts as a breach" className="xl:col-span-2">
          {!board?.planes.length && <div className="text-ink-3 font-mono text-[11px] py-4 text-center">
            no SLO ticks recorded yet (POST /api/hardening/slo to feed the board)</div>}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {board?.planes.map((p) => (
              <div key={p.plane} className={`border p-2 ${p.met ? "border-pass/30" : "border-block/40"}`}>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-[11px] uppercase text-ink">{p.plane}</span>
                  <Verdict ok={p.met} yes="met" no="breach" />
                </div>
                {p.breaches.map((b) => (
                  <div key={b.metric} className="font-mono text-[10px] text-block flex justify-between">
                    <span>{b.metric}</span>
                    <span>{b.reason === "unobserved" ? "unobserved" : `${b.value} ${b.op} ${b.threshold}`}</span>
                  </div>
                ))}
                {p.met && <div className="font-mono text-[10px] text-ink-3">all objectives within budget</div>}
              </div>
            ))}
          </div>
        </Panel>

        {/* efficiency */}
        <Panel title="label-budget efficiency">
          <textarea value={effIn} onChange={(e) => setEffIn(e.target.value)} rows={8}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <div className="mt-2"><RunButton busy={eff.busy}
            onClick={() => eff.run(() => runJSON<EffReport>("/api/hardening/efficiency", { entries: JSON.parse(effIn) }))}
            label="report" /></div>
          <ErrLine err={eff.err} />
          {eff.out && (
            <div className="mt-3">
              <div className="font-mono text-[10px] text-ink-3 mb-1">
                best {eff.out.best_slice ?? "-"} · overall {eff.out.overall_gain_per_1k}/1k · {eff.out.total_labels} labels
              </div>
              <div className="space-y-1">
                {eff.out.per_slice.map((r) => {
                  const max = Math.max(0.0001, ...eff.out!.per_slice.map((x) => Math.abs(x.gain_per_1k)));
                  return (
                    <div key={r.slice} className="flex items-center gap-2 font-mono text-[10px]">
                      <span className={`w-24 shrink-0 truncate ${r.negative_roi ? "text-block" : "text-ink-2"}`}>{r.slice}</span>
                      <Bar frac={Math.abs(r.gain_per_1k) / max} tone={r.negative_roi ? "block" : "pass"} />
                      <span className="w-24 shrink-0 text-right text-ink-3">{r.gain_per_1k}/1k · {r.labels}</span>
                    </div>
                  );
                })}
              </div>
              {eff.out.wasted_spend_slices.length > 0 && (
                <div className="font-mono text-[10px] text-block mt-2">wasted spend: {eff.out.wasted_spend_slices.join(", ")}</div>)}
            </div>
          )}
        </Panel>

        {/* reproducibility */}
        <Panel title="byte-stable reproducibility">
          <div className="font-mono text-[10px] text-ink-3 mb-1">build A</div>
          <textarea value={runA} onChange={(e) => setRunA(e.target.value)} rows={4}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <div className="font-mono text-[10px] text-ink-3 mt-2 mb-1">build B (rebuild)</div>
          <textarea value={runB} onChange={(e) => setRunB(e.target.value)} rows={4}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <div className="mt-2"><RunButton busy={repro.busy}
            onClick={() => repro.run(() => runJSON<Repro>("/api/hardening/reproducible", { run_a: JSON.parse(runA), run_b: JSON.parse(runB) }))}
            label="compare" /></div>
          <ErrLine err={repro.err} />
          {repro.out && (
            <div className="mt-3 space-y-1">
              <Verdict ok={repro.out.reproducible} yes="byte-stable" no="diverged" />
              <div className="font-mono text-[10px] text-ink-3 break-all">A {repro.out.hash_a.slice(0, 24)}</div>
              <div className="font-mono text-[10px] text-ink-3 break-all">B {repro.out.hash_b.slice(0, 24)}</div>
              {repro.out.first_divergence && <div className="font-mono text-[10px] text-block">first divergence: {repro.out.first_divergence}</div>}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
