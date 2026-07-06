"use client";

// M13 LabeloxAV label quality: gold-set audit that catches bad annotations, the learned-reconciler parity gate
// (the learned fusion head may only replace the heuristic once it beats it on a held-out set), and 4D
// propagation of a keyframe box across a clip with one stable identity and healed gaps. Operational
// Materialism: a passed annotation is green, a failed one red; the parity gate promotes only when earned.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Verdict, RunButton, useRun, ErrLine, NumField } from "@/components/engine/prim";
import { runJSON } from "@/lib/engine";

type Audit = { n: number; n_pass: number; n_fail: number; missed_gold: number;
  verdicts: { object_id: string; verdict: string; iou: number }[] };
type Parity = { promote: boolean; delta: number; learned: number; heuristic: number; reason: string };
type Prop = { track_id: string; n_frames: number; identities: number; healed_gaps: number;
  boxes: { frame: number; source: string }[] };

const DEMO_PRED = JSON.stringify([
  { object_id: "a", class_id: 6, bbox: [10, 10, 50, 50] },
  { object_id: "b", class_id: 6, bbox: [500, 500, 540, 540] },
], null, 2);
const DEMO_GOLD = JSON.stringify([{ class_id: 6, bbox: [11, 11, 51, 51] }], null, 2);

export default function QualityPage() {
  const [pred, setPred] = useState(DEMO_PRED);
  const [gold, setGold] = useState(DEMO_GOLD);
  const audit = useRun<Audit>();

  const [learned, setLearned] = useState(0.56);
  const [heur, setHeur] = useState(0.55);
  const [margin, setMargin] = useState(0.0);
  const parity = useRun<Parity>();

  const [nframes, setNframes] = useState(6);
  const prop = useRun<Prop>();

  return (
    <PageShell active="Labelox" title="Label quality and 4D propagation"
      subtitle="gold audit, the learned-reconciler parity gate, and single-identity propagation">
      <div className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4 max-w-6xl">
        {/* gold audit */}
        <Panel title="gold-set audit" hint="catches bad labels">
          <div className="font-mono text-[10px] text-ink-3 mb-1">predicted annotations</div>
          <textarea value={pred} onChange={(e) => setPred(e.target.value)} rows={5}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <div className="font-mono text-[10px] text-ink-3 mt-2 mb-1">gold</div>
          <textarea value={gold} onChange={(e) => setGold(e.target.value)} rows={3}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <div className="mt-2"><RunButton busy={audit.busy}
            onClick={() => audit.run(() => runJSON<Audit>("/api/labelox/quality/audit",
              { predicted: JSON.parse(pred), gold: JSON.parse(gold) }))} label="audit" /></div>
          <ErrLine err={audit.err} />
          {audit.out && (
            <div className="mt-3">
              <KV k="pass / fail" v={`${audit.out.n_pass} / ${audit.out.n_fail}`}
                tone={audit.out.n_fail ? "block" : "pass"} />
              <KV k="missed gold" v={audit.out.missed_gold} tone="ink-3" />
              <div className="space-y-0.5 mt-1">
                {audit.out.verdicts.map((v) => (
                  <div key={v.object_id} className="flex justify-between font-mono text-[10px]">
                    <span className="text-ink-2">{v.object_id}</span>
                    <span className={v.verdict === "pass" ? "text-pass" : "text-block"}>{v.verdict} · iou {v.iou}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Panel>

        {/* parity gate */}
        <Panel title="reconciler parity gate" hint="learned must beat heuristic">
          <div className="space-y-2 mb-2">
            <NumField label="learned map50" value={learned} onChange={setLearned} w="w-24" />
            <NumField label="heuristic map50" value={heur} onChange={setHeur} w="w-24" />
            <NumField label="margin" value={margin} onChange={setMargin} w="w-24" />
          </div>
          <RunButton busy={parity.busy}
            onClick={() => parity.run(() => runJSON<Parity>("/api/labelox/reconcile/parity",
              { learned_metrics: { map50: learned }, heuristic_metrics: { map50: heur }, margin }))} label="check gate" />
          <ErrLine err={parity.err} />
          {parity.out && (
            <div className="mt-3">
              <Verdict ok={parity.out.promote} yes="promote learned" no="keep heuristic" />
              <KV k="delta" v={parity.out.delta} tone={parity.out.delta > 0 ? "pass" : "block"} />
              <div className="font-mono text-[10px] text-ink-3 mt-1">{parity.out.reason}</div>
            </div>
          )}
        </Panel>

        {/* 4D propagation */}
        <Panel title="4D propagation" hint="one identity, healed gaps">
          <NumField label="frames" value={nframes} onChange={setNframes} w="w-20" />
          <div className="mt-2"><RunButton busy={prop.busy}
            onClick={() => prop.run(() => runJSON<Prop>("/api/labelox/propagate4d",
              { keyframe_box: [10, 10, 50, 50], n_frames: nframes, velocity: [1, 0, 1, 0],
                known: { [Math.max(1, nframes - 2)]: [30, 10, 70, 50] } }))} label="propagate" /></div>
          <ErrLine err={prop.err} />
          {prop.out && (
            <div className="mt-3">
              <KV k="identities" v={prop.out.identities} tone={prop.out.identities === 1 ? "pass" : "block"} />
              <KV k="healed gaps" v={prop.out.healed_gaps} tone="ink" />
              <KV k="track" v={prop.out.track_id.slice(0, 8)} tone="ink-3" />
              <div className="flex flex-wrap gap-0.5 mt-2">
                {prop.out.boxes.map((b) => (
                  <span key={b.frame} title={b.source}
                    className={`w-4 h-4 ${b.source === "keyframe" ? "bg-accent" :
                      b.source === "observed" ? "bg-info" : b.source === "interpolated" ? "bg-pass" : "bg-ink-3"}`} />
                ))}
              </div>
              <div className="font-mono text-[9px] text-ink-3 mt-1">keyframe / observed / interpolated / propagated</div>
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
