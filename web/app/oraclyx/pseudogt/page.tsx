"use client";

// M14 ORACLYX 4D uncertainty-aware pseudo-GT: the auto-truth path trains on tracks, not frames, and on labels
// weighted by how much ORACLYX trusts them. This surfaces the pieces: calibrated uncertainty + soft target for
// one label, monocular metric-depth recovery (refused without calibration or a size prior), and the
// disagreement queue ranked by expected information gain (where the human review budget teaches the most).
// Operational Materialism: uncertainty is amber-to-red, soft target is the training weight, ranking is neutral.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Bar, RunButton, useRun, ErrLine, NumField } from "@/components/engine/prim";
import { runJSON } from "@/lib/engine";

type Unc = { uncertainty: number; soft_target: number };
type Depth = { depth_m: number | null; uncertainty?: number; priors?: string[]; reason?: string };
type Ranked = { ranked: { id?: string; uncertainty: number; info_gain: number; rarity?: number; disagreement?: number }[] };

const DEMO_QUEUE = JSON.stringify([
  { id: "obj-a", uncertainty: 0.8, rarity: 0.9, disagreement: 0.7, safety_weight: 2.0 },
  { id: "obj-b", uncertainty: 0.4, rarity: 0.2, disagreement: 0.3 },
  { id: "obj-c", uncertainty: 0.05, rarity: 0.0, disagreement: 0.0 },
], null, 2);

export default function PseudoGtPage() {
  const [cons, setCons] = useState(0.7);
  const [views, setViews] = useState(2);
  const [calib, setCalib] = useState(0.9);
  const [depthU, setDepthU] = useState(0.1);
  const unc = useRun<Unc>();

  const [boxH, setBoxH] = useState(340);
  const [cls, setCls] = useState(0);
  const [focal, setFocal] = useState(1000);
  const [camH, setCamH] = useState(1.4);
  const depth = useRun<Depth>();

  const [queue, setQueue] = useState(DEMO_QUEUE);
  const rank = useRun<Ranked>();

  return (
    <PageShell active="ORACLYX" title="4D uncertainty-aware pseudo-GT"
      subtitle="calibrated soft targets, monocular depth priors, and the info-gain review queue">
      <div className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4 max-w-6xl">
        {/* uncertainty */}
        <Panel title="pseudo-label uncertainty" hint="consensus to soft target">
          <div className="space-y-2 mb-2">
            <NumField label="consensus" value={cons} onChange={setCons} w="w-20" />
            <NumField label="n views" value={views} onChange={setViews} w="w-20" />
            <NumField label="calib conf" value={calib} onChange={setCalib} w="w-20" />
            <NumField label="depth unc" value={depthU} onChange={setDepthU} w="w-20" />
          </div>
          <RunButton busy={unc.busy}
            onClick={() => unc.run(() => runJSON<Unc>("/api/oraclyx/uncertainty",
              { consensus_score: cons, n_views: views, calib_confidence: calib, depth_uncertainty: depthU }))} label="calibrate" />
          <ErrLine err={unc.err} />
          {unc.out && (
            <div className="mt-3">
              <div className="flex items-center gap-2 font-mono text-[10px] mb-1">
                <span className="w-20 text-ink-3">uncertainty</span>
                <Bar frac={unc.out.uncertainty} tone={unc.out.uncertainty > 0.5 ? "block" : "warn"} />
                <span className="w-10 text-right text-ink-2">{unc.out.uncertainty}</span>
              </div>
              <div className="flex items-center gap-2 font-mono text-[10px]">
                <span className="w-20 text-ink-3">soft target</span>
                <Bar frac={unc.out.soft_target} tone="pass" />
                <span className="w-10 text-right text-ink-2">{unc.out.soft_target}</span>
              </div>
            </div>
          )}
        </Panel>

        {/* mono depth */}
        <Panel title="monocular metric depth" hint="refuses without a prior">
          <div className="space-y-2 mb-2">
            <NumField label="box height px" value={boxH} onChange={setBoxH} w="w-20" />
            <NumField label="class id" value={cls} onChange={setCls} w="w-20" />
            <NumField label="focal px" value={focal} onChange={setFocal} w="w-20" />
            <NumField label="cam height m" value={camH} onChange={setCamH} w="w-20" />
          </div>
          <RunButton busy={depth.busy}
            onClick={() => depth.run(() => runJSON<Depth>("/api/oraclyx/mono-depth",
              { box: [0, 720 - boxH, 40, 720], class_id: cls, focal_px: focal, cy: 360, cam_height_m: camH }))} label="recover" />
          <ErrLine err={depth.err} />
          {depth.out && (
            <div className="mt-3">
              {depth.out.depth_m === null
                ? <div className="font-mono text-[11px] text-block">refused: {depth.out.reason}</div>
                : (<>
                    <KV k="depth" v={`${depth.out.depth_m} m`} tone="pass" />
                    <KV k="uncertainty" v={depth.out.uncertainty} tone="ink-3" />
                    <KV k="priors" v={depth.out.priors?.join(" + ")} tone="ink-3" />
                  </>)}
            </div>
          )}
        </Panel>

        {/* info-gain ranking */}
        <Panel title="disagreement review queue" hint="ranked by info gain">
          <textarea value={queue} onChange={(e) => setQueue(e.target.value)} rows={8}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <div className="mt-2"><RunButton busy={rank.busy}
            onClick={() => rank.run(() => runJSON<Ranked>("/api/oraclyx/disagreements/rank", { items: JSON.parse(queue) }))}
            label="rank" /></div>
          <ErrLine err={rank.err} />
          {rank.out && (
            <div className="mt-3 space-y-1">
              {rank.out.ranked.map((r, i) => {
                const max = Math.max(0.0001, ...rank.out!.ranked.map((x) => x.info_gain));
                return (
                  <div key={r.id ?? i} className="flex items-center gap-2 font-mono text-[10px]">
                    <span className="w-16 shrink-0 truncate text-ink-2">{r.id ?? `#${i}`}</span>
                    <Bar frac={r.info_gain / max} tone="accent" />
                    <span className="w-12 shrink-0 text-right text-ink-3">{r.info_gain}</span>
                  </div>
                );
              })}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
