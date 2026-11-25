"use client";

// Operations: the operability surface. A per-plane SLO board (which plane is the bottleneck), a byte-stable
// reproducibility checker (a release must rebuild to the same bytes), and the label-budget efficiency report
// (what each slice's labels actually bought). A plane row is green only when it meets its SLO; a breach is
// red; efficiency bars are neutral, ranked.

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";
import { Panel, Bar, Verdict, useRun, ErrLine } from "@/components/ui/kit";
import { getJSON, runJSON, parseInput } from "@/lib/engine";
import { humanize, num } from "@/lib/format";

type PlaneSlo = { plane: string; met: boolean;
  breaches: { metric: string; value: number | null; threshold: number; op: string; reason: string }[];
  metrics: Record<string, number>; created_at: string | null };
type Board = { planes: PlaneSlo[]; all_met: boolean };
type Repro = { reproducible: boolean; hash_a: string; hash_b: string; first_divergence: string | null };
type EffRow = { slice: string; labels: number; gain: number; gain_per_1k: number; negative_roi: boolean };
type EffReport = { per_slice: EffRow[]; total_labels: number; total_gain: number; overall_gain_per_1k: number;
  best_slice: string | null; wasted_spend_slices: string[] };

const EXAMPLE_EFF = JSON.stringify([
  { slice: "vru_night", labels: 500, map_before: 0.40, map_after: 0.55 },
  { slice: "car_day", labels: 2000, map_before: 0.80, map_after: 0.805 },
  { slice: "sign_rain", labels: 300, map_before: 0.60, map_after: 0.58 },
], null, 2);
const EXAMPLE_A = JSON.stringify({ commit: "c1", samples: 1000, metrics: { map50: 0.612 } }, null, 2);

// Planes are stored under their retired code names; show the names the sidebar uses.
const PLANE_NAME: Record<string, string> = {
  labelox: "Labelling", sanyx: "Ingest health", calyx: "Calibration", sievyx: "Curation",
  oraclyx: "Pseudo ground truth", verdyx: "Slice evaluation", forgyx: "Edge deployment",
};
const planeName = (p: string) => PLANE_NAME[p.toLowerCase()] ?? humanize(p);

function JsonField({ id, label, value, onChange, example, rows }: {
  id: string; label: string; value: string; onChange: (s: string) => void; example: string; rows: number;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label htmlFor={id} className="text-[12px] text-ink-2">{label}</label>
        <button type="button" onClick={() => onChange(example)} className="link text-[12px]">Use example</button>
      </div>
      <textarea id={id} value={value} placeholder={example} onChange={(e) => onChange(e.target.value)} rows={rows}
        className="field w-full font-mono text-[12px]" />
    </div>
  );
}

function RunButton({ busy, onClick, label }: { busy: boolean; onClick: () => void; label: string }) {
  return (
    <button type="button" onClick={onClick} disabled={busy} className="btn btn-primary mt-3">
      {busy ? "Running..." : label}
    </button>
  );
}

export default function OpsPage() {
  const [board, setBoard] = useState<Board | null>(null);
  const [effIn, setEffIn] = useState("");
  const [runA, setRunA] = useState("");
  const [runB, setRunB] = useState("");
  const eff = useRun<EffReport>();
  const repro = useRun<Repro>();

  useEffect(() => { getJSON<Board>("/api/hardening/slo/board").then(setBoard).catch(() => {}); }, []);
  const hasPlanes = Boolean(board?.planes.length);

  return (
    <PageShell active="Operations" title="Operations"
      subtitle="Service levels per part of the system, rebuild checks, and what labelling spend bought."
      meta={board && hasPlanes ? <Verdict ok={board.all_met} yes="All service levels met" no="Service level breached" /> : undefined}>
      <div className="page-wrap grid grid-cols-1 xl:grid-cols-2 gap-4">
        {/* SLO board */}
        <Panel title="Service levels" className="xl:col-span-2"
          desc={hasPlanes ? "A part with no recent measurement counts as a breach." : undefined}>
          {!hasPlanes ? (
            <EmptyState compact icon="server" title="No service-level measurements yet">
              Each part of the system (ingest, calibration, curation, evaluation and so on) reports whether it
              is meeting its targets. Their status appears here once the first reports come in.
              <details className="mt-3 text-left">
                <summary className="cursor-pointer text-ink-2">For administrators</summary>
                <p className="mt-1.5">Services report by posting to <code className="font-mono">/api/hardening/slo</code>.</p>
              </details>
            </EmptyState>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {board?.planes.map((p) => (
                <div key={p.plane} className={`rounded border p-3 ${p.met ? "border-pass/30" : "border-block/40"}`}>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[13px] font-medium text-ink">{planeName(p.plane)}</span>
                    <Verdict ok={p.met} yes="Met" no="Breached" />
                  </div>
                  {p.breaches.map((b) => (
                    <div key={b.metric} className="text-[12px] text-block flex justify-between gap-2">
                      <span>{humanize(b.metric)}</span>
                      <span className="tabular-nums">{b.reason === "unobserved" ? "No measurement" : `${b.value} ${b.op} ${b.threshold}`}</span>
                    </div>
                  ))}
                  {p.met && <div className="text-[12px] text-ink-3">All targets within budget</div>}
                </div>
              ))}
            </div>
          )}
        </Panel>

        {/* efficiency */}
        <Panel title="What the labelling budget bought"
          desc="For each slice, how many labels went in and how much accuracy came out. Slices where accuracy fell are flagged.">
          <JsonField id="eff-input" label="Labels and accuracy per slice (JSON)" value={effIn} onChange={setEffIn}
            example={EXAMPLE_EFF} rows={8} />
          <RunButton busy={eff.busy} label="Make report"
            onClick={() => eff.run(() => runJSON<EffReport>("/api/hardening/efficiency", { entries: parseInput(effIn, "the labels per slice") }))} />
          <ErrLine err={eff.err} />
          {eff.out && (
            <div className="mt-3">
              <div className="text-[12px] text-ink-3 mb-2">
                Best: {eff.out.best_slice ? humanize(eff.out.best_slice) : "-"} · overall {eff.out.overall_gain_per_1k} per 1,000 labels · {num(eff.out.total_labels)} labels
              </div>
              <div className="space-y-1.5">
                {eff.out.per_slice.map((r) => {
                  const max = Math.max(0.0001, ...eff.out!.per_slice.map((x) => Math.abs(x.gain_per_1k)));
                  return (
                    <div key={r.slice} className="flex items-center gap-2 text-[12px]">
                      <span className={`w-32 shrink-0 truncate ${r.negative_roi ? "text-block" : "text-ink-2"}`}>{humanize(r.slice)}</span>
                      <Bar frac={Math.abs(r.gain_per_1k) / max} state={r.negative_roi ? "block" : "pass"} />
                      <span className="w-36 shrink-0 text-right text-ink-3 tabular-nums">{r.gain_per_1k} per 1k · {num(r.labels)}</span>
                    </div>
                  );
                })}
              </div>
              {eff.out.wasted_spend_slices.length > 0 && (
                <div className="text-[12px] text-block mt-2">No return on: {eff.out.wasted_spend_slices.map(humanize).join(", ")}</div>)}
            </div>
          )}
        </Panel>

        {/* reproducibility */}
        <Panel title="Does a rebuild match?"
          desc="Paste the manifest of a release and of its rebuild. A release must rebuild to exactly the same bytes.">
          <div className="space-y-3">
            <JsonField id="repro-a" label="Original build" value={runA} onChange={setRunA} example={EXAMPLE_A} rows={4} />
            <JsonField id="repro-b" label="Rebuild" value={runB} onChange={setRunB} example={EXAMPLE_A} rows={4} />
          </div>
          <RunButton busy={repro.busy} label="Compare"
            onClick={() => repro.run(() => runJSON<Repro>("/api/hardening/reproducible", { run_a: parseInput(runA, "the original build"), run_b: parseInput(runB, "the rebuild") }))} />
          <ErrLine err={repro.err} />
          {repro.out && (
            <div className="mt-3 space-y-1">
              <Verdict ok={repro.out.reproducible} yes="Identical" no="Different" />
              <div className="text-[12px] text-ink-3 break-all font-mono">Original {repro.out.hash_a.slice(0, 24)}</div>
              <div className="text-[12px] text-ink-3 break-all font-mono">Rebuild {repro.out.hash_b.slice(0, 24)}</div>
              {repro.out.first_divergence && <div className="text-[12px] text-block">First difference: {repro.out.first_divergence}</div>}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
