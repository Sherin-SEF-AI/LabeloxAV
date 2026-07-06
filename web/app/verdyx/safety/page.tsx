"use client";

// M15 VERDYX safety + statistical eval: the metrics a driving safety case needs, beyond frame-averaged mAP.
// Critical-object recall over the India VRU + cattle classes, TTC-weighted recall (an imminent-collision miss
// dominates), the near-miss slice a reviewer signs off on, a paired significance test for a challenger, and
// shadow regression triage. Operational Materialism: recall is green above the floor, red below; a missed
// near-miss object is an escalation.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Verdict, RunButton, useRun, ErrLine } from "@/components/engine/prim";
import { runJSON } from "@/lib/engine";

type Safety = {
  critical: { n_critical: number; detected: number; recall: number | null };
  ttc_weighted: { ttc_weighted_recall: number | null; weight_mass: number };
  near_miss: { n_near_miss: number; detected: number; recall: number | null; missed_object_ids: string[] };
};
type Sig = { delta: number | null; p_value: number | null; significant: boolean; n: number };
type Triage = { regressions: { slice: string; baseline: number; current: number; delta: number; protected: boolean }[];
  protected_regression: boolean; alarm: boolean };

const DEMO_OBJ = JSON.stringify([
  { object_id: "p1", class_id: 0, detected: true, ttc_s: 1.0 },
  { object_id: "p2", class_id: 0, detected: false, ttc_s: 1.5 },
  { object_id: "c1", class_id: 8, detected: true, ttc_s: 4.0 },
  { object_id: "v1", class_id: 4, detected: true, ttc_s: 5.0 },
], null, 2);

export default function SafetyEval() {
  const [objs, setObjs] = useState(DEMO_OBJ);
  const safety = useRun<Safety>();

  const [champ, setChamp] = useState("[0,0,0,0,0,1,0,0,1,0]");
  const [chall, setChall] = useState("[1,1,0,1,1,1,0,1,1,0]");
  const sig = useRun<Sig>();

  const [baseline, setBaseline] = useState('{"vru_night":0.90,"car_day":0.90}');
  const [current, setCurrent] = useState('{"vru_night":0.80,"car_day":0.89}');
  const triage = useRun<Triage>();

  const pct = (v: number | null) => v === null ? "-" : `${(v * 100).toFixed(1)}%`;

  return (
    <PageShell active="VERDYX" title="Safety and statistical evaluation"
      subtitle="critical-object and TTC-weighted recall, significance, and shadow regression triage">
      <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-4 max-w-6xl">
        {/* safety recall */}
        <Panel title="safety-weighted recall" hint="VRU + cattle, TTC-weighted" className="lg:col-span-2">
          <div className="grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-3">
            <div>
              <textarea value={objs} onChange={(e) => setObjs(e.target.value)} rows={9}
                className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
              <div className="mt-2"><RunButton busy={safety.busy}
                onClick={() => safety.run(() => runJSON<Safety>("/api/verdyx/safety/recall", { objects: JSON.parse(objs) }))}
                label="evaluate" /></div>
              <ErrLine err={safety.err} />
            </div>
            {safety.out && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                <div className="border border-line p-2">
                  <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">critical-object</div>
                  <div className={`font-mono text-2xl ${(safety.out.critical.recall ?? 0) >= 0.5 ? "text-pass" : "text-block"}`}>
                    {pct(safety.out.critical.recall)}</div>
                  <div className="font-mono text-[10px] text-ink-3">{safety.out.critical.detected}/{safety.out.critical.n_critical} VRU+cattle</div>
                </div>
                <div className="border border-line p-2">
                  <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">TTC-weighted</div>
                  <div className="font-mono text-2xl text-ink">{pct(safety.out.ttc_weighted.ttc_weighted_recall)}</div>
                  <div className="font-mono text-[10px] text-ink-3">imminent misses weigh most</div>
                </div>
                <div className="border border-line p-2">
                  <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">near-miss slice</div>
                  <div className={`font-mono text-2xl ${safety.out.near_miss.missed_object_ids.length ? "text-block" : "text-pass"}`}>
                    {pct(safety.out.near_miss.recall)}</div>
                  <div className="font-mono text-[10px] text-ink-3">
                    {safety.out.near_miss.missed_object_ids.length
                      ? `missed: ${safety.out.near_miss.missed_object_ids.join(", ")}`
                      : `${safety.out.near_miss.n_near_miss} within TTC, all caught`}</div>
                </div>
              </div>
            )}
          </div>
        </Panel>

        {/* significance */}
        <Panel title="paired significance" hint="challenger vs champion, per-object">
          <label className="block font-mono text-[10px] text-ink-3 mb-1">champion per-object scores</label>
          <input value={champ} onChange={(e) => setChamp(e.target.value)}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <label className="block font-mono text-[10px] text-ink-3 mt-2 mb-1">challenger per-object scores</label>
          <input value={chall} onChange={(e) => setChall(e.target.value)}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <div className="mt-2"><RunButton busy={sig.busy}
            onClick={() => sig.run(() => runJSON<Sig>("/api/verdyx/stats/significance",
              { champion: JSON.parse(champ), challenger: JSON.parse(chall) }))} label="test" /></div>
          <ErrLine err={sig.err} />
          {sig.out && (
            <div className="mt-3">
              <Verdict ok={sig.out.significant} yes="significant improvement" no="not significant" />
              <KV k="delta" v={sig.out.delta} tone={((sig.out.delta ?? 0) > 0) ? "pass" : "block"} />
              <KV k="p-value" v={sig.out.p_value} tone="ink-3" />
              <KV k="n" v={sig.out.n} tone="ink-3" />
            </div>
          )}
        </Panel>

        {/* shadow triage */}
        <Panel title="shadow regression triage" hint="protected slice alarms">
          <label className="block font-mono text-[10px] text-ink-3 mb-1">baseline per-slice recall</label>
          <input value={baseline} onChange={(e) => setBaseline(e.target.value)}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <label className="block font-mono text-[10px] text-ink-3 mt-2 mb-1">current per-slice recall</label>
          <input value={current} onChange={(e) => setCurrent(e.target.value)}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <div className="mt-2"><RunButton busy={triage.busy}
            onClick={() => triage.run(() => runJSON<Triage>("/api/verdyx/shadow/triage",
              { baseline: JSON.parse(baseline), current: JSON.parse(current), protected: ["vru_night"] }))} label="triage" /></div>
          <ErrLine err={triage.err} />
          {triage.out && (
            <div className="mt-3">
              <Verdict ok={!triage.out.alarm} yes="no regression" no="regression alarm" />
              {triage.out.regressions.map((r) => (
                <div key={r.slice} className="flex justify-between font-mono text-[10px] mt-1">
                  <span className={r.protected ? "text-block" : "text-warn"}>{r.slice}{r.protected ? " (protected)" : ""}</span>
                  <span className="text-ink-3">{r.baseline} to {r.current} ({r.delta})</span>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
