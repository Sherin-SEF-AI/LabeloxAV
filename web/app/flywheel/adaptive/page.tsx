"use client";

// M17 adaptive flywheel controller, wired to live corpus signals. The headline runs one cycle from the real
// corpus: class starvation (VRU/animal/two-wheeler classes below their floor) drives the label budget, ODD
// scene gaps + empty classes drive collection, and a per-class work order lists the real candidate objects a
// reviewer would open. A manual tool and the cycle ledger sit below. Blender-style: rounded panels, blue
// primary run, protected classes in amber, shortfall (collect) in red.

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Bar, RunButton, useRun, ErrLine, NumField } from "@/components/engine/prim";
import { HintBar } from "@/components/ui/kit";
import { getJSON, runJSON } from "@/lib/engine";

type Alloc = { slice: string; labels: number; weight: number; reason: string };
type Task = { cell: string; priority: number; target_count: number; missing: boolean; reason: string };
type Cycle = { cycle_id: string; label_budget: number; allocation: Alloc[]; collection_tasks: Task[];
  rationale: string; created_at: string | null };
type Plan = Cycle & { allocated: number; held: number };
type WorkItem = { slice: string; class_id: number; labels_requested: number; candidates_available: number;
  shortfall: number };
type Starved = { slice: string; count: number; share: number; delta: number; protected: boolean };
type AutoPlan = Plan & {
  signals: { n_starved: number; total_objects: number; n_labelable: number; n_empty: number;
    n_odd_gaps: number; frames: number; labelable: Starved[]; empty_classes: Starved[]; odd_gaps: Task[] };
  work_order: WorkItem[];
};

const DEMO_REGRESSIONS = JSON.stringify([
  { slice: "vru_night", delta: -0.10, protected: true },
  { slice: "car_day", delta: -0.03, protected: false },
], null, 2);
const DEMO_GAPS = JSON.stringify([{ cell: "night_rain_junction", gap: 0.04, missing: true }], null, 2);

export default function AdaptiveFlywheel() {
  const [cycles, setCycles] = useState<Cycle[]>([]);
  const auto = useRun<AutoPlan>();
  const [budget, setBudget] = useState(2000);
  const [floor, setFloor] = useState(150);
  const dispatch = useRun<{ run_id: string; cycle_id: string; dispatched: number; by_slice: Record<string, number> }>();
  const orders = useRun<{ cycle_id: string; orders: number; scene: number; classes: number }>();

  // manual tool
  const [reg, setReg] = useState(DEMO_REGRESSIONS);
  const [gaps, setGaps] = useState(DEMO_GAPS);
  const manual = useRun<Plan>();

  const load = () => getJSON<{ cycles: Cycle[] }>("/api/flywheel/adaptive/cycles?limit=15")
    .then((d) => setCycles(d.cycles)).catch(() => {});
  useEffect(() => { load(); }, []);

  const runAuto = () => auto.run(async () => {
    const p = await runJSON<AutoPlan>(
      `/api/flywheel/adaptive/auto?total_label_budget=${budget}&safety_floor=${floor}`, {});
    await load();
    return p;
  });
  const runManual = () => manual.run(async () => {
    const rg = JSON.parse(reg);
    const p = await runJSON<Plan>("/api/flywheel/adaptive/cycle", {
      regressions: rg, odd_gaps: JSON.parse(gaps), total_label_budget: budget, total_fleet_samples: 100000,
      safety_slices: rg.filter((r: { protected?: boolean }) => r.protected).map((r: { slice: string }) => r.slice),
      safety_floor: floor,
    });
    await load();
    return p;
  });

  const p = auto.out;
  const maxLabels = Math.max(1, ...(p?.allocation ?? []).map((a) => a.labels));
  const workBySlice = new Map((p?.work_order ?? []).map((w) => [w.slice, w]));

  return (
    <PageShell active="Flywheel" title="Adaptive flywheel controller"
      subtitle="one cycle from the real corpus: what to label, what to collect"
      right={<span className="text-[12px] text-ink-3">{cycles.length} cycles on record</span>}>
      <div className="p-4 max-w-6xl space-y-4">
        <HintBar>
          The controller reads the live corpus and decides the next cycle: safety classes below their coverage
          floor become label demands (only where labelable instances still exist), and starved scene cells plus
          empty classes become collection tasks. It proposes work; it never writes labels.
        </HintBar>

        {/* live run bar */}
        <div className="flex items-center gap-3 flex-wrap bg-panel border border-line rounded p-3">
          <NumField label="label budget" value={budget} onChange={setBudget} w="w-24" />
          <NumField label="safety floor" value={floor} onChange={setFloor} w="w-20" />
          <RunButton busy={auto.busy} onClick={runAuto} label="run cycle from live corpus" />
          {p && <span className="text-[12px] text-ink-3">
            corpus {p.signals.total_objects.toLocaleString()} objects / {p.signals.frames.toLocaleString()} frames ·
            {" "}{p.signals.n_starved} starved = {p.signals.n_labelable} labelable + {p.signals.n_empty} empty ·
            {" "}{p.signals.n_odd_gaps} scene gaps</span>}
          <ErrLine err={auto.err} />
        </div>

        {p && (
          <>
            <div className="text-[12px] text-ink-2">{p.rationale}</div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* label allocation + work order */}
              <Panel title="label allocation" hint={`${p.allocated} labels across classes labeling can improve`}
                right={<button onClick={() => dispatch.run(() => runJSON("/api/flywheel/adaptive/dispatch",
                  { cycle_id: p.cycle_id, per_slice_cap: 300 }))} disabled={dispatch.busy}
                  className="btn btn-primary text-[11px]"
                  data-tip="bump the real labelable candidates into the review queue as one reversible run">
                  {dispatch.busy ? "sending..." : "send to review"}</button>}>
                {dispatch.out && (
                  <div className="text-[11px] text-pass border border-pass/30 rounded px-2 py-1.5 mb-2 bg-pass/5">
                    dispatched {dispatch.out.dispatched} candidates to the review queue (run {dispatch.out.run_id.slice(0, 8)}, revertible).{" "}
                    <a href={`/?flywheel=${dispatch.out.cycle_id}`} className="text-accent-2 underline">open worklist &rarr;</a>
                  </div>
                )}
                <ErrLine err={dispatch.err} />
                <div className="space-y-1.5">
                  {p.allocation.map((a) => {
                    const w = workBySlice.get(a.slice);
                    const prot = a.reason.includes("protected");
                    return (
                      <div key={a.slice} className="flex items-center gap-2 text-[11px] font-mono">
                        <span className={`w-40 shrink-0 truncate ${prot ? "text-warn" : "text-ink-2"}`}
                          data-tip={prot ? "safety-critical (VRU/animal), floored" : undefined}>{a.slice}</span>
                        <Bar frac={a.labels / maxLabels} tone={prot ? "warn" : "accent"} />
                        <span className="w-12 shrink-0 text-right text-ink">{a.labels}</span>
                        {w && <span className="w-28 shrink-0 text-right text-ink-3"
                          data-tip="real candidate objects available to label now vs the shortfall that needs collection">
                          {w.candidates_available} have{w.shortfall ? ` · ${w.shortfall} short` : ""}</span>}
                      </div>
                    );
                  })}
                </div>
              </Panel>

              {/* collection tasks */}
              <Panel title="collection tasks" hint={`${p.collection_tasks.length} to collect, not label`}
                right={<button onClick={() => orders.run(() =>
                  runJSON(`/api/flywheel/adaptive/collection-orders?cycle_id=${p.cycle_id}`, {}))}
                  disabled={orders.busy} className="btn btn-primary text-[11px]"
                  data-tip="turn these into per-vehicle fleet collection orders (scene drives + missing species)">
                  {orders.busy ? "planning..." : "make collection orders"}</button>}>
                {orders.out && (
                  <div className="text-[11px] text-pass border border-pass/30 rounded px-2 py-1.5 mb-2 bg-pass/5">
                    {orders.out.orders} collection orders proposed ({orders.out.scene} scene drives, {orders.out.classes} species).{" "}
                    <a href="/agent" className="text-accent-2 underline">open fleet board &rarr;</a>
                  </div>
                )}
                <ErrLine err={orders.err} />
                <div className="space-y-1">
                  {p.collection_tasks.slice(0, 18).map((t) => (
                    <div key={t.cell} className="flex items-center justify-between text-[11px] font-mono">
                      <span className={t.missing ? "text-block" : "text-ink-2"}>
                        {t.cell}{t.missing ? " (missing)" : ""}</span>
                      <span className="text-ink-3">prio {t.priority.toFixed(3)} · need {t.target_count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </Panel>
            </div>
          </>
        )}

        {/* manual tool + ledger */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Panel title="manual cycle" hint="hand-fed signals">
            <div className="text-[10px] text-ink-3 mb-1">regressions</div>
            <textarea value={reg} onChange={(e) => setReg(e.target.value)} rows={5}
              className="w-full field font-mono text-[10px]" />
            <div className="text-[10px] text-ink-3 mt-2 mb-1">odd gaps</div>
            <textarea value={gaps} onChange={(e) => setGaps(e.target.value)} rows={3}
              className="w-full field font-mono text-[10px]" />
            <div className="mt-2"><RunButton busy={manual.busy} onClick={runManual} label="run manual" /></div>
            <ErrLine err={manual.err} />
            {manual.out && <div className="text-[11px] text-ink-2 mt-2">{manual.out.rationale}</div>}
          </Panel>

          <Panel title="cycle ledger" hint="newest first">
            <div className="space-y-2">
              {cycles.map((c) => (
                <div key={c.cycle_id} className="border border-line/60 rounded p-2">
                  <div className="flex items-center justify-between text-[10px] font-mono mb-1">
                    <span className="text-ink-3">{c.created_at?.slice(0, 19).replace("T", " ") ?? c.cycle_id.slice(0, 8)}</span>
                    <span className="text-ink-2">budget {c.label_budget}</span>
                  </div>
                  <div className="text-[11px] text-ink-2">{c.rationale}</div>
                  <KV k="label slices" v={c.allocation.length} tone="ink-3" />
                  <KV k="collection tasks" v={c.collection_tasks.length} tone="ink-3" />
                </div>
              ))}
              {!cycles.length && <div className="text-ink-3 text-[12px] py-4 text-center">no cycles yet: run one</div>}
            </div>
          </Panel>
        </div>
      </div>
    </PageShell>
  );
}
