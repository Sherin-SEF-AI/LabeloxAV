"use client";

// Adaptive flywheel controller, wired to live corpus signals. The headline runs one cycle from the real
// corpus: class starvation (VRU/animal/two-wheeler classes below their floor) drives the label budget, ODD
// scene gaps + empty classes drive collection, and a per-class work order lists the real candidate objects a
// reviewer would open. A manual tool and the cycle ledger sit below. Blender-style: rounded panels, blue
// primary run, protected classes in amber, shortfall (collect) in red.

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";
import { Panel, Bar, Field, NumberInput, Stat, useRun, ErrLine } from "@/components/ui/kit";
import { ago, humanize, num, plural } from "@/lib/format";
import { getJSON, runJSON, parseInput } from "@/lib/engine";

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

const EXAMPLE_REGRESSIONS = JSON.stringify([
  { slice: "vru_night", delta: -0.10, protected: true },
  { slice: "car_day", delta: -0.03, protected: false },
], null, 2);
const EXAMPLE_GAPS = JSON.stringify([{ cell: "night_rain_junction", gap: 0.04, missing: true }], null, 2);

export default function AdaptiveFlywheel() {
  const [cycles, setCycles] = useState<Cycle[]>([]);
  const auto = useRun<AutoPlan>();
  const [budget, setBudget] = useState(2000);
  const [floor, setFloor] = useState(150);
  const dispatch = useRun<{ run_id: string; cycle_id: string; dispatched: number; by_slice: Record<string, number> }>();
  const orders = useRun<{ cycle_id: string; orders: number; scene: number; classes: number }>();

  // manual tool
  const [reg, setReg] = useState("");
  const [gaps, setGaps] = useState("");
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
    const rg = parseInput<{ slice: string; protected?: boolean }[]>(reg, "the regressions");
    const p = await runJSON<Plan>("/api/flywheel/adaptive/cycle", {
      regressions: rg, odd_gaps: parseInput(gaps, "the coverage gaps"), total_label_budget: budget, total_fleet_samples: 100000,
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
    <PageShell active="Operations" title="Adaptive flywheel"
      subtitle="Reads the corpus and proposes the next round: what to label and what to go and record. It never writes labels.">
      <div className="page-wrap space-y-4">
        {/* live run bar */}
        <div className="panel p-4 flex items-end gap-5 flex-wrap">
          <div className="w-56"><Field label="Labels to spend" tip="How many labels this round may ask for"><NumberInput value={budget} onChange={setBudget} /></Field></div>
          <div className="w-56"><Field label="Safety floor per class" tip="Safety-critical classes below this many labelled objects are topped up first"><NumberInput value={floor} onChange={setFloor} w="w-20" /></Field></div>
          <button type="button" onClick={runAuto} disabled={auto.busy} className="btn btn-primary">
            {auto.busy ? "Planning..." : "Plan the next round"}
          </button>
          <ErrLine err={auto.err} />
        </div>

        {p && (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <Stat label="Objects in the corpus" value={num(p.signals.total_objects)} sub={`${num(p.signals.frames)} frames`} />
              <Stat label="Classes short of labels" value={num(p.signals.n_starved)} tone={p.signals.n_starved ? "warn" : "pass"}
                sub={`${num(p.signals.n_labelable)} can be labelled, ${num(p.signals.n_empty)} have none`} />
              <Stat label="Scene gaps" value={num(p.signals.n_odd_gaps)} sub="conditions the corpus barely covers" />
              <Stat label="Labels allocated" value={num(p.allocated)} sub={p.held ? `${num(p.held)} held back` : undefined} />
            </div>
            {p.rationale && <p className="section-desc">{p.rationale}</p>}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* label allocation + work order */}
              <Panel title="What to label"
                desc={`${num(p.allocated)} labels across the classes more labelling can improve. Amber classes are safety-critical.`}
                right={<button type="button" onClick={() => dispatch.run(() => runJSON("/api/flywheel/adaptive/dispatch",
                  { cycle_id: p.cycle_id, per_slice_cap: 300 }))} disabled={dispatch.busy}
                  className="btn btn-primary"
                  data-tip="Puts the real candidate objects into the review queue as one run that can be undone">
                  {dispatch.busy ? "Sending..." : "Send to review"}</button>}>
                {dispatch.out && (
                  <div className="text-[12px] text-pass border border-pass/30 rounded px-2.5 py-2 mb-3 bg-pass/5">
                    Sent {plural(dispatch.out.dispatched, "candidate")} to the review queue. This can be undone.{" "}
                    <a href={`/?flywheel=${dispatch.out.cycle_id}`} className="link">Open the worklist &rarr;</a>
                  </div>
                )}
                <ErrLine err={dispatch.err} />
                <div className="space-y-1.5">
                  {p.allocation.map((a) => {
                    const w = workBySlice.get(a.slice);
                    const prot = a.reason.includes("protected");
                    return (
                      <div key={a.slice} className="flex items-center gap-2 text-[12px]">
                        <span className={`w-44 shrink-0 truncate ${prot ? "text-warn" : "text-ink-2"}`} title={a.slice}
                          data-tip={prot ? "Safety-critical (people and animals), topped up to the floor" : undefined}>{humanize(a.slice)}</span>
                        <Bar frac={a.labels / maxLabels} state={prot ? "warn" : "accent"} />
                        <span className="w-12 shrink-0 text-right text-ink tabular-nums">{num(a.labels)}</span>
                        {w && <span className="w-32 shrink-0 text-right text-ink-3 tabular-nums"
                          data-tip="Objects available to label now, and how many more must be recorded">
                          {num(w.candidates_available)} available{w.shortfall ? `, ${num(w.shortfall)} short` : ""}</span>}
                      </div>
                    );
                  })}
                </div>
              </Panel>

              {/* collection tasks */}
              <Panel title="What to record"
                desc={`${plural(p.collection_tasks.length, "condition")} the corpus lacks; these need new driving, not labelling. Red ones have no data at all.`}
                right={<button type="button" onClick={() => orders.run(() =>
                  runJSON(`/api/flywheel/adaptive/collection-orders?cycle_id=${p.cycle_id}`, {}))}
                  disabled={orders.busy} className="btn"
                  data-tip="Turns these into per-vehicle recording orders for the fleet">
                  {orders.busy ? "Planning..." : "Make recording orders"}</button>}>
                {orders.out && (
                  <div className="text-[12px] text-pass border border-pass/30 rounded px-2.5 py-2 mb-3 bg-pass/5">
                    {plural(orders.out.orders, "recording order")} proposed ({num(orders.out.scene)} scene drives, {num(orders.out.classes)} classes).{" "}
                    <a href="/agent" className="link">Open the fleet board &rarr;</a>
                  </div>
                )}
                <ErrLine err={orders.err} />
                <table className="data-table">
                  <thead><tr><th>Condition</th><th className="num">Priority</th><th className="num">Needed</th></tr></thead>
                  <tbody>
                    {p.collection_tasks.slice(0, 18).map((t) => (
                      <tr key={t.cell}>
                        <td className={t.missing ? "text-block" : "text-ink-2"}>
                          {humanize(t.cell)}{t.missing ? " (none recorded)" : ""}</td>
                        <td className="num">{t.priority.toFixed(3)}</td>
                        <td className="num">{num(t.target_count)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Panel>
            </div>
          </>
        )}

        <Panel title="Past rounds"
          right={<span className="text-[12px] text-ink-3">{plural(cycles.length, "round")}, newest first</span>}>
          {cycles.length ? (
            <div className="-m-3.5">
              <table className="data-table">
                <thead><tr><th>When</th><th>Summary</th><th className="num">Label budget</th><th className="num">Classes to label</th><th className="num">Conditions to record</th></tr></thead>
                <tbody>
                  {cycles.map((c) => (
                    <tr key={c.cycle_id}>
                      <td className="whitespace-nowrap" title={c.created_at ?? undefined}>{c.created_at ? ago(c.created_at) : "-"}</td>
                      <td className="text-ink-2">{c.rationale}</td>
                      <td className="num">{num(c.label_budget)}</td>
                      <td className="num">{num(c.allocation.length)}</td>
                      <td className="num">{num(c.collection_tasks.length)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState compact icon="activity" title="No rounds planned yet">
              Plan the next round above. Each plan is kept here so you can see how the corpus&apos;s needs changed.
            </EmptyState>
          )}
        </Panel>

        {/* manual tool */}
        <details className="panel">
          <summary className="panel-head cursor-pointer section-title">Plan a round from hand-entered signals</summary>
          <div className="p-3.5 grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label htmlFor="fw-regressions" className="text-[12px] text-ink-2">Slices that got worse</label>
                <button type="button" onClick={() => setReg(EXAMPLE_REGRESSIONS)} className="link text-[12px]">Use example</button>
              </div>
              <textarea id="fw-regressions" value={reg} placeholder={EXAMPLE_REGRESSIONS} onChange={(e) => setReg(e.target.value)} rows={6}
                className="w-full field font-mono text-[12px]" />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label htmlFor="fw-gaps" className="text-[12px] text-ink-2">Scene gaps</label>
                <button type="button" onClick={() => setGaps(EXAMPLE_GAPS)} className="link text-[12px]">Use example</button>
              </div>
              <textarea id="fw-gaps" value={gaps} placeholder={EXAMPLE_GAPS} onChange={(e) => setGaps(e.target.value)} rows={6}
                className="w-full field font-mono text-[12px]" />
            </div>
            <div className="lg:col-span-2">
              <button type="button" onClick={runManual} disabled={manual.busy} className="btn">
                {manual.busy ? "Planning..." : "Plan from these"}
              </button>
              <ErrLine err={manual.err} />
              {manual.out && <div className="text-[12.5px] text-ink-2 mt-2">{manual.out.rationale}</div>}
            </div>
          </div>
        </details>
      </div>
    </PageShell>
  );
}
