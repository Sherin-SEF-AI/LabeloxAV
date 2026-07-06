"use client";

// M17 adaptive flywheel controller: the closed loop. VERDYX safety regressions and SIEVYX ODD coverage gaps
// drive a label-budget allocation and a set of collection tasks; each cycle is recorded to the ledger. This
// page shows the ledger and lets an operator run a cycle from the two signal sets. Operational Materialism:
// matte, monospace, allocation bars colored only by whether the slice is protected.

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Bar, RunButton, useRun, ErrLine, NumField } from "@/components/engine/prim";
import { getJSON, runJSON } from "@/lib/engine";

type Alloc = { slice: string; labels: number; weight: number; reason: string };
type Task = { cell: string; priority: number; target_count: number; missing: boolean; reason: string };
type Cycle = { cycle_id: string; label_budget: number; allocation: Alloc[]; collection_tasks: Task[];
  rationale: string; created_at: string | null };
type Plan = Cycle & { allocated: number; held: number };

const DEMO_REGRESSIONS = JSON.stringify([
  { slice: "vru_night", delta: -0.10, protected: true },
  { slice: "cattle_dusk", delta: -0.06, protected: true },
  { slice: "car_day", delta: -0.03, protected: false },
], null, 2);
const DEMO_GAPS = JSON.stringify([
  { cell: "night_rain_junction", gap: 0.04, missing: true },
  { cell: "day_clear_highway", gap: 0.05, missing: false },
], null, 2);

export default function AdaptiveFlywheel() {
  const [cycles, setCycles] = useState<Cycle[]>([]);
  const [reg, setReg] = useState(DEMO_REGRESSIONS);
  const [gaps, setGaps] = useState(DEMO_GAPS);
  const [budget, setBudget] = useState(2000);
  const [samples, setSamples] = useState(100000);
  const [floor, setFloor] = useState(200);
  const { out, err, busy, run } = useRun<Plan>();

  const load = () => getJSON<{ cycles: Cycle[] }>("/api/flywheel/adaptive/cycles?limit=20")
    .then((d) => setCycles(d.cycles)).catch(() => {});
  useEffect(() => { load(); }, []);

  const plan = out;
  const runCycle = () => run(async () => {
    const body = {
      regressions: JSON.parse(reg), odd_gaps: JSON.parse(gaps),
      total_label_budget: budget, total_fleet_samples: samples,
      safety_slices: JSON.parse(reg).filter((r: { protected?: boolean }) => r.protected)
        .map((r: { slice: string }) => r.slice),
      safety_floor: floor,
    };
    const p = await runJSON<Plan>("/api/flywheel/adaptive/cycle", body);
    await load();
    return p;
  });
  const maxLabels = Math.max(1, ...(plan?.allocation ?? []).map((a) => a.labels));

  return (
    <PageShell active="Flywheel" title="Adaptive flywheel controller"
      subtitle="VERDYX failures and SIEVYX gaps steer the next cycle of label budget and collection"
      right={<span className="font-mono text-[11px] text-ink-3">{cycles.length} cycles on record</span>}>
      <div className="p-4 grid grid-cols-1 xl:grid-cols-[380px_1fr] gap-4 max-w-6xl">
        {/* run panel */}
        <div className="space-y-4">
          <Panel title="signals in" hint="VERDYX + SIEVYX">
            <div className="font-mono text-[10px] text-ink-3 mb-1">regressions (slice, delta, protected)</div>
            <textarea value={reg} onChange={(e) => setReg(e.target.value)} rows={7}
              className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
            <div className="font-mono text-[10px] text-ink-3 mt-2 mb-1">odd gaps (cell, gap, missing)</div>
            <textarea value={gaps} onChange={(e) => setGaps(e.target.value)} rows={6}
              className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          </Panel>
          <Panel title="budget">
            <div className="space-y-2">
              <NumField label="label budget" value={budget} onChange={setBudget} w="w-28" />
              <NumField label="fleet samples" value={samples} onChange={setSamples} w="w-28" />
              <NumField label="safety floor" value={floor} onChange={setFloor} w="w-28" />
              <div className="pt-1"><RunButton busy={busy} onClick={runCycle} label="run cycle" /></div>
              <ErrLine err={err} />
            </div>
          </Panel>
        </div>

        {/* plan + ledger */}
        <div className="space-y-4">
          {plan && (
            <Panel title="cycle plan" hint={`allocated ${plan.allocated} / held ${plan.held}`}>
              <div className="font-mono text-[11px] text-ink-2 mb-2">{plan.rationale}</div>
              <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">label allocation</div>
              <div className="space-y-1 mb-3">
                {plan.allocation.slice(0, 20).map((a) => {
                  const prot = a.reason.includes("protected");
                  return (
                    <div key={a.slice} className="flex items-center gap-2 font-mono text-[10px]">
                      <span className={`w-32 shrink-0 truncate ${prot ? "text-warn" : "text-ink-2"}`}>{a.slice}</span>
                      <Bar frac={a.labels / maxLabels} tone={prot ? "warn" : "accent"} />
                      <span className="w-16 shrink-0 text-right text-ink-3">{a.labels}</span>
                    </div>
                  );
                })}
                {!plan.allocation.length && <div className="text-ink-3 text-[11px]">no regressions: budget held</div>}
              </div>
              <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">collection tasks</div>
              <div className="space-y-0.5">
                {plan.collection_tasks.slice(0, 12).map((t) => (
                  <div key={t.cell} className="flex items-center justify-between font-mono text-[10px]">
                    <span className={t.missing ? "text-block" : "text-ink-2"}>{t.cell}{t.missing ? " (missing)" : ""}</span>
                    <span className="text-ink-3">prio {t.priority} · need {t.target_count}</span>
                  </div>
                ))}
              </div>
            </Panel>
          )}

          <Panel title="cycle ledger" hint="newest first">
            <div className="space-y-2">
              {cycles.map((c) => (
                <div key={c.cycle_id} className="border border-line/60 p-2">
                  <div className="flex items-center justify-between font-mono text-[10px] mb-1">
                    <span className="text-ink-3">{c.created_at?.slice(0, 19).replace("T", " ") ?? c.cycle_id.slice(0, 8)}</span>
                    <span className="text-ink-2">budget {c.label_budget}</span>
                  </div>
                  <div className="font-mono text-[11px] text-ink-2">{c.rationale}</div>
                  <KV k="slices allocated" v={c.allocation.length} tone="ink-3" />
                  <KV k="collection tasks" v={c.collection_tasks.length} tone="ink-3" />
                </div>
              ))}
              {!cycles.length && <div className="text-ink-3 font-mono text-[11px] py-4 text-center">no cycles yet: run one</div>}
            </div>
          </Panel>
        </div>
      </div>
    </PageShell>
  );
}
