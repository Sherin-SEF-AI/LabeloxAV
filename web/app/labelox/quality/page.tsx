"use client";

// M13 LabeloxAV label quality: gold-set audit that catches bad annotations, the learned-reconciler parity gate
// (the learned fusion head may only replace the heuristic once it beats it on a held-out set), and 4D
// propagation of a keyframe box across a clip with one stable identity and healed gaps. Operational
// Materialism: a passed annotation is green, a failed one red; the parity gate promotes only when earned.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import OntologyRepair from "@/components/labelui/OntologyRepair";
import { Panel, KV, Verdict, RunButton, useRun, ErrLine, NumField } from "@/components/engine/prim";
import { runJSON, parseInput } from "@/lib/engine";
import { humanize, shortId } from "@/lib/format";

type Audit = { n: number; n_pass: number; n_fail: number; missed_gold: number;
  verdicts: { object_id: string; verdict: string; iou: number }[] };
type Parity = { promote: boolean; delta: number; learned: number; heuristic: number; reason: string };
type Prop = { track_id: string; n_frames: number; identities: number; healed_gaps: number;
  boxes: { frame: number; source: string }[] };

const EXAMPLE_PRED = JSON.stringify([
  { object_id: "a", class_id: 6, bbox: [10, 10, 50, 50] },
  { object_id: "b", class_id: 6, bbox: [500, 500, 540, 540] },
], null, 2);
const EXAMPLE_GOLD = JSON.stringify([{ class_id: 6, bbox: [11, 11, 51, 51] }], null, 2);

export default function QualityPage() {
  const [pred, setPred] = useState("");
  const [gold, setGold] = useState("");
  const audit = useRun<Audit>();

  const [learned, setLearned] = useState(0.56);
  const [heur, setHeur] = useState(0.55);
  const [margin, setMargin] = useState(0.0);
  const parity = useRun<Parity>();

  const [nframes, setNframes] = useState(6);
  const prop = useRun<Prop>();

  return (
    <PageShell active="Labelox" title="Label quality and 4D propagation"
      subtitle="Repair the class vocabulary, audit labels against gold, and check tools that decide which labels are trusted.">
      <div className="page-wrap space-y-4">
        {/* The vocabulary could be added to and never repaired: merge, rename and retire existed on the
            server and the interface offered only "add a class". A mistake in the ontology is a label
            quality problem, which is why it belongs on this page. */}
        <OntologyRepair />

        <div>
          <h2 className="section-title">Try the quality checks</h2>
          <p className="section-desc mt-0.5">Paste or adjust inputs and run each check to see what it would decide.</p>
        </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* gold audit */}
        <Panel title="Gold-set audit" hint="Catches bad labels">
          <label className="block text-[12px] text-ink-2 mb-1" htmlFor="audit-pred">Predicted annotations (JSON)</label>
          <textarea id="audit-pred" value={pred} placeholder={EXAMPLE_PRED} onChange={(e) => setPred(e.target.value)} rows={5}
            className="field w-full font-mono text-[11.5px]" />
          <label className="block text-[12px] text-ink-2 mt-3 mb-1" htmlFor="audit-gold">Gold annotations (JSON)</label>
          <textarea id="audit-gold" value={gold} placeholder={EXAMPLE_GOLD} onChange={(e) => setGold(e.target.value)} rows={3}
            className="field w-full font-mono text-[11.5px]" />
          <div className="mt-3"><RunButton busy={audit.busy}
            onClick={() => audit.run(() => runJSON<Audit>("/api/labelox/quality/audit",
              { predicted: parseInput(pred, "pred"), gold: parseInput(gold, "gold") }))} label="Run audit" /></div>
          <ErrLine err={audit.err} />
          {audit.out && (
            <div className="mt-3">
              <KV k="Passed / failed" v={`${audit.out.n_pass} / ${audit.out.n_fail}`}
                tone={audit.out.n_fail ? "block" : "pass"} />
              <KV k="Gold objects missed" v={audit.out.missed_gold} tone="ink-3" />
              <div className="space-y-0.5 mt-1">
                {audit.out.verdicts.map((v) => (
                  <div key={v.object_id} className="flex justify-between text-[12px]">
                    <span className="text-ink-2 font-mono">{v.object_id}</span>
                    <span className={v.verdict === "pass" ? "text-pass" : "text-block"}>{humanize(v.verdict)} · IoU {v.iou}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Panel>

        {/* parity gate */}
        <Panel title="Reconciler parity gate" hint="Learned must beat heuristic">
          <div className="space-y-2 mb-3">
            <NumField label="Learned mAP50" value={learned} onChange={setLearned} w="w-24" />
            <NumField label="Heuristic mAP50" value={heur} onChange={setHeur} w="w-24" />
            <NumField label="Required margin" value={margin} onChange={setMargin} w="w-24" />
          </div>
          <RunButton busy={parity.busy}
            onClick={() => parity.run(() => runJSON<Parity>("/api/labelox/reconcile/parity",
              { learned_metrics: { map50: learned }, heuristic_metrics: { map50: heur }, margin }))} label="Check gate" />
          <ErrLine err={parity.err} />
          {parity.out && (
            <div className="mt-3">
              <Verdict ok={parity.out.promote} yes="Promote learned" no="Keep heuristic" />
              <KV k="Difference" v={parity.out.delta} tone={parity.out.delta > 0 ? "pass" : "block"} />
              <div className="text-[12px] text-ink-3 mt-1">{parity.out.reason}</div>
            </div>
          )}
        </Panel>

        {/* 4D propagation */}
        <Panel title="4D propagation" hint="One identity, gaps filled">
          <NumField label="Frames" value={nframes} onChange={setNframes} w="w-20" />
          <div className="mt-3"><RunButton busy={prop.busy}
            onClick={() => prop.run(() => runJSON<Prop>("/api/labelox/propagate4d",
              { keyframe_box: [10, 10, 50, 50], n_frames: nframes, velocity: [1, 0, 1, 0],
                known: { [Math.max(1, nframes - 2)]: [30, 10, 70, 50] } }))} label="Propagate" /></div>
          <ErrLine err={prop.err} />
          {prop.out && (
            <div className="mt-3">
              <KV k="Identities" v={prop.out.identities} tone={prop.out.identities === 1 ? "pass" : "block"} />
              <KV k="Gaps filled" v={prop.out.healed_gaps} tone="ink" />
              <KV k="Track" v={shortId(prop.out.track_id)} tone="ink-3" />
              <div className="flex flex-wrap gap-0.5 mt-2">
                {prop.out.boxes.map((b) => (
                  <span key={b.frame} title={`Frame ${b.frame}: ${b.source}`}
                    className={`w-4 h-4 rounded-sm ${b.source === "keyframe" ? "bg-accent" :
                      b.source === "observed" ? "bg-info" : b.source === "interpolated" ? "bg-pass" : "bg-ink-3"}`} />
                ))}
              </div>
              <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11.5px] text-ink-3 mt-2">
                <span className="inline-flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-accent" />Keyframe</span>
                <span className="inline-flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-info" />Observed</span>
                <span className="inline-flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-pass" />Interpolated</span>
                <span className="inline-flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-ink-3" />Propagated</span>
              </div>
            </div>
          )}
        </Panel>
      </div>
      </div>
    </PageShell>
  );
}
