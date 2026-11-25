"use client";

// Safety + statistical eval: the metrics a driving safety case needs, beyond frame-averaged mAP.
// Critical-object recall over the India VRU + cattle classes, TTC-weighted recall (an imminent-collision miss
// dominates), the near-miss slice a reviewer signs off on, a paired significance test for a challenger, and
// shadow regression triage. Operational Materialism: recall is green above the floor, red below; a missed
// near-miss object is an escalation.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Verdict, Stat, useRun, ErrLine } from "@/components/ui/kit";
import { runJSON, parseInput } from "@/lib/engine";
import { humanize, num, pct } from "@/lib/format";

type Safety = {
  critical: { n_critical: number; detected: number; recall: number | null };
  ttc_weighted: { ttc_weighted_recall: number | null; weight_mass: number };
  near_miss: { n_near_miss: number; detected: number; recall: number | null; missed_object_ids: string[] };
};
type Sig = { delta: number | null; p_value: number | null; significant: boolean; n: number };
type Triage = { regressions: { slice: string; baseline: number; current: number; delta: number; protected: boolean }[];
  protected_regression: boolean; alarm: boolean };

const EXAMPLE_OBJ = JSON.stringify([
  { object_id: "p1", class_id: 0, detected: true, ttc_s: 1.0 },
  { object_id: "p2", class_id: 0, detected: false, ttc_s: 1.5 },
  { object_id: "c1", class_id: 8, detected: true, ttc_s: 4.0 },
  { object_id: "v1", class_id: 4, detected: true, ttc_s: 5.0 },
], null, 2);
const EXAMPLE_CHAMP = "[0,1,1,0,1]";
const EXAMPLE_CHALL = "[1,1,0,1,1]";
const EXAMPLE_BASE = '{"vru_night": 0.9}';
const EXAMPLE_CUR = '{"vru_night": 0.8}';

// One labelled input with an "example" filler, so the expected shape is one click away rather than a
// placeholder that vanishes on the first keystroke.
function JsonField({ id, label, value, onChange, example, rows }: {
  id: string; label: string; value: string; onChange: (s: string) => void; example: string; rows?: number;
}) {
  const cls = "field w-full font-mono text-[12px]";
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label htmlFor={id} className="text-[12px] text-ink-2">{label}</label>
        <button type="button" onClick={() => onChange(example)} className="link text-[12px]">Use example</button>
      </div>
      {rows
        ? <textarea id={id} value={value} placeholder={example} onChange={(e) => onChange(e.target.value)} rows={rows} className={cls} />
        : <input id={id} value={value} placeholder={example} onChange={(e) => onChange(e.target.value)} className={cls} />}
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

const pct1 = (v: number | null) => pct(v, 1);

export default function SafetyEval() {
  const [objs, setObjs] = useState("");
  const safety = useRun<Safety>();

  const [champ, setChamp] = useState("");
  const [chall, setChall] = useState("");
  const sig = useRun<Sig>();

  const [baseline, setBaseline] = useState("");
  const [current, setCurrent] = useState("");
  const triage = useRun<Triage>();

  return (
    <PageShell active="Slice evaluation" title="Safety and statistical evaluation"
      subtitle="Recall on the objects that matter most, whether a new model is really better, and slice regressions.">
      <div className="page-wrap grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* safety recall */}
        <Panel title="Recall on safety-critical objects" className="lg:col-span-2"
          desc="Pedestrians, riders and cattle, with misses close to a collision weighted most. Paste one row per object: whether it was detected and its time to collision in seconds.">
          <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-4">
            <div>
              <JsonField id="safety-objects" label="Objects (JSON)" value={objs} onChange={setObjs}
                example={EXAMPLE_OBJ} rows={9} />
              <RunButton busy={safety.busy} label="Evaluate"
                onClick={() => safety.run(() => runJSON<Safety>("/api/verdyx/safety/recall", { objects: parseInput(objs, "the objects") }))} />
              <ErrLine err={safety.err} />
            </div>
            {safety.out ? (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 content-start">
                <Stat label="Critical-object recall" value={pct1(safety.out.critical.recall)}
                  tone={(safety.out.critical.recall ?? 0) >= 0.5 ? "pass" : "block"}
                  sub={`${num(safety.out.critical.detected)} of ${num(safety.out.critical.n_critical)} people and cattle found`} />
                <Stat label="Weighted by time to collision" value={pct1(safety.out.ttc_weighted.ttc_weighted_recall)}
                  sub="Imminent misses weigh most" />
                <Stat label="Near-miss recall" value={pct1(safety.out.near_miss.recall)}
                  tone={safety.out.near_miss.missed_object_ids.length ? "block" : "pass"}
                  sub={safety.out.near_miss.missed_object_ids.length
                    ? `Missed: ${safety.out.near_miss.missed_object_ids.join(", ")}`
                    : `${num(safety.out.near_miss.n_near_miss)} close calls, all caught`} />
              </div>
            ) : (
              <div className="flex items-center justify-center rounded border border-dashed border-line text-[12.5px] text-ink-3 p-6 text-center">
                Results appear here: recall on critical objects, recall weighted by time to collision, and any near-miss the model failed to see.
              </div>
            )}
          </div>
        </Panel>

        {/* significance */}
        <Panel title="Is the challenger really better?"
          desc="A paired test on the same objects: did the new model find more of them than chance would explain? Use 1 for found and 0 for missed, in the same order for both.">
          <div className="space-y-3">
            <JsonField id="champ-scores" label="Champion, per object" value={champ} onChange={setChamp} example={EXAMPLE_CHAMP} />
            <JsonField id="chall-scores" label="Challenger, per object" value={chall} onChange={setChall} example={EXAMPLE_CHALL} />
          </div>
          <RunButton busy={sig.busy} label="Run test"
            onClick={() => sig.run(() => runJSON<Sig>("/api/verdyx/stats/significance",
              { champion: parseInput(champ, "the champion scores"), challenger: parseInput(chall, "the challenger scores") }))} />
          <ErrLine err={sig.err} />
          {sig.out && (
            <div className="mt-3 space-y-1">
              <Verdict ok={sig.out.significant} yes="Significant improvement" no="Not significant" />
              <KV k="Difference" v={sig.out.delta ?? "-"} state={((sig.out.delta ?? 0) > 0) ? "pass" : "block"} />
              <KV k="p-value" v={sig.out.p_value ?? "-"} state="muted" />
              <KV k="Objects compared" v={num(sig.out.n)} state="muted" />
            </div>
          )}
        </Panel>

        {/* shadow triage */}
        <Panel title="Slice regression check"
          desc="Compare recall per slice before and after. A drop on a protected slice (pedestrians at night) raises an alarm.">
          <div className="space-y-3">
            <JsonField id="baseline-recall" label="Before, recall per slice" value={baseline} onChange={setBaseline} example={EXAMPLE_BASE} />
            <JsonField id="current-recall" label="After, recall per slice" value={current} onChange={setCurrent} example={EXAMPLE_CUR} />
          </div>
          <RunButton busy={triage.busy} label="Check"
            onClick={() => triage.run(() => runJSON<Triage>("/api/verdyx/shadow/triage",
              { baseline: parseInput(baseline, "the before recall"), current: parseInput(current, "the after recall"), protected: ["vru_night"] }))} />
          <ErrLine err={triage.err} />
          {triage.out && (
            <div className="mt-3 space-y-1">
              <Verdict ok={!triage.out.alarm} yes="No regression" no="Regression alarm" />
              {triage.out.regressions.map((r) => (
                <div key={r.slice} className="flex justify-between text-[12px] mt-1">
                  <span className={r.protected ? "text-block" : "text-warn"}>{humanize(r.slice)}{r.protected ? " (protected)" : ""}</span>
                  <span className="text-ink-3 tabular-nums">{r.baseline} to {r.current} ({r.delta})</span>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
