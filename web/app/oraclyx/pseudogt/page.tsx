"use client";

// M14 ORACLYX 4D uncertainty-aware pseudo-GT: the auto-truth path trains on tracks, not frames, and on labels
// weighted by how much ORACLYX trusts them. This surfaces the pieces: calibrated uncertainty + soft target for
// one label, monocular metric-depth recovery (refused without calibration or a size prior), and the
// disagreement queue ranked by expected information gain (where the human review budget teaches the most).
// Operational Materialism: uncertainty is amber-to-red, soft target is the training weight, ranking is neutral.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Bar, RunButton, useRun, ErrLine, NumField } from "@/components/engine/prim";
import { runJSON, parseInput } from "@/lib/engine";
import { humanize } from "@/lib/format";

type Unc = { uncertainty: number; soft_target: number };
type Depth = { depth_m: number | null; uncertainty?: number; priors?: string[]; reason?: string };
type Ranked = { ranked: { id?: string; uncertainty: number; info_gain: number; rarity?: number; disagreement?: number }[] };

const EXAMPLE_QUEUE = JSON.stringify([
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

  const [queue, setQueue] = useState("");
  const rank = useRun<Ranked>();

  return (
    <PageShell active="ORACLYX" title="4D pseudo ground truth"
      subtitle="Try out how machine labels are weighted for training, before any person checks them">
      <div className="page-wrap grid grid-cols-1 lg:grid-cols-3 gap-4 items-start">
        {/* uncertainty */}
        <Panel title="How much to trust a label" hint="From model agreement to training weight">
          <p className="section-desc mb-3">A label the models agree on, seen from several cameras, counts for more in training.</p>
          <div className="space-y-2 mb-3">
            <NumField label="Model agreement (0 to 1)" value={cons} onChange={setCons} w="w-20" />
            <NumField label="Cameras that saw it" value={views} onChange={setViews} w="w-20" />
            <NumField label="Calibration trust (0 to 1)" value={calib} onChange={setCalib} w="w-20" />
            <NumField label="Depth doubt (0 to 1)" value={depthU} onChange={setDepthU} w="w-20" />
          </div>
          <RunButton busy={unc.busy}
            onClick={() => unc.run(() => runJSON<Unc>("/api/oraclyx/uncertainty",
              { consensus_score: cons, n_views: views, calib_confidence: calib, depth_uncertainty: depthU }))} label="Work it out" />
          <ErrLine err={unc.err} />
          {unc.out && (
            <div className="mt-3 border-t hairline pt-3 space-y-1.5">
              <div className="flex items-center gap-2 text-[12px]">
                <span className="w-28 text-ink-3">Doubt</span>
                <Bar frac={unc.out.uncertainty} tone={unc.out.uncertainty > 0.5 ? "block" : "warn"} />
                <span className="w-12 text-right text-ink-2 tabular-nums">{unc.out.uncertainty}</span>
              </div>
              <div className="flex items-center gap-2 text-[12px]">
                <span className="w-28 text-ink-3">Training weight</span>
                <Bar frac={unc.out.soft_target} tone="pass" />
                <span className="w-12 text-right text-ink-2 tabular-nums">{unc.out.soft_target}</span>
              </div>
            </div>
          )}
        </Panel>

        {/* mono depth */}
        <Panel title="Distance from one camera" hint="Needs a known object size or camera height">
          <p className="section-desc mb-3">Estimates how far away an object is from its box height in a single image. It declines rather than guess when it lacks a reference.</p>
          <div className="space-y-2 mb-3">
            <NumField label="Box height (px)" value={boxH} onChange={setBoxH} w="w-20" />
            <NumField label="Class number" value={cls} onChange={setCls} w="w-20" />
            <NumField label="Focal length (px)" value={focal} onChange={setFocal} w="w-20" />
            <NumField label="Camera height (m)" value={camH} onChange={setCamH} w="w-20" />
          </div>
          <RunButton busy={depth.busy}
            onClick={() => depth.run(() => runJSON<Depth>("/api/oraclyx/mono-depth",
              { box: [0, 720 - boxH, 40, 720], class_id: cls, focal_px: focal, cy: 360, cam_height_m: camH }))} label="Estimate distance" />
          <ErrLine err={depth.err} />
          {depth.out && (
            <div className="mt-3 border-t hairline pt-3">
              {depth.out.depth_m === null
                ? <div className="text-[12.5px] text-block">Could not estimate: {depth.out.reason}</div>
                : (<>
                    <KV k="Distance" v={`${depth.out.depth_m} m`} tone="pass" />
                    <KV k="Doubt" v={depth.out.uncertainty} tone="ink-3" />
                    <KV k="Based on" v={depth.out.priors?.map(humanize).join(" + ")} tone="ink-3" />
                  </>)}
            </div>
          )}
        </Panel>

        {/* info-gain ranking */}
        <Panel title="What to review first" hint="Where a person's check teaches the model most">
          <label className="block">
            <span className="block text-[12px] text-ink-2 mb-1">Disagreements as JSON</span>
            <textarea value={queue} placeholder={EXAMPLE_QUEUE} onChange={(e) => setQueue(e.target.value)} rows={8}
              className="field w-full font-mono text-[12px]" />
          </label>
          <div className="mt-2.5"><RunButton busy={rank.busy}
            onClick={() => rank.run(() => runJSON<Ranked>("/api/oraclyx/disagreements/rank", { items: parseInput(queue, "the disagreements") }))}
            label="Rank them" /></div>
          <ErrLine err={rank.err} />
          {rank.out && (
            <div className="mt-3 border-t hairline pt-3 space-y-1.5">
              {rank.out.ranked.map((r, i) => {
                const max = Math.max(0.0001, ...rank.out!.ranked.map((x) => x.info_gain));
                return (
                  <div key={r.id ?? i} className="flex items-center gap-2 text-[12px]" data-tip={`Value of reviewing: ${r.info_gain}`}>
                    <span className="w-6 shrink-0 text-ink-3 tabular-nums">{i + 1}.</span>
                    <span className="w-20 shrink-0 truncate text-ink-2">{r.id ?? `Item ${i + 1}`}</span>
                    <Bar frac={r.info_gain / max} tone="accent" />
                    <span className="w-12 shrink-0 text-right text-ink-3 tabular-nums">{r.info_gain}</span>
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
