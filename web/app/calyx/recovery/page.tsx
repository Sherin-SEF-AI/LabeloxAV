"use client";

// M11 CALYX calibration recovery: instead of only flagging drift, recover from it. Targetless calibration
// recovers focal + pitch from natural-scene cues (vanishing points + horizon), so a field session calibrates
// from the scene it recorded; rig consensus fuses a vehicle's per-session overrides into a stronger prior with
// the residual spread made visible. Operational Materialism: a recovered estimate is green, a refused one red,
// a wide consensus spread is amber.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, RunButton, useRun, ErrLine, NumField } from "@/components/engine/prim";
import { getJSON, runJSON } from "@/lib/engine";

type Targetless = { ok: boolean; reason?: string; focal: number | null; pitch_deg: number | null; confidence?: number };
type Consensus = { vehicle_id: string; n_overrides: number;
  prior: { n: number; rpy_deg: number[] | null; xyz_m: number[] | null; confidence: number; spread?: unknown } };

export default function RecoveryPage() {
  const [vp1x, setVp1x] = useState(-200); const [vp1y, setVp1y] = useState(360);
  const [vp2x, setVp2x] = useState(2100); const [vp2y, setVp2y] = useState(360);
  const [horizon, setHorizon] = useState(360);
  const [cx, setCx] = useState(960); const [cy, setCy] = useState(540);
  const tl = useRun<Targetless>();

  const [vehicle, setVehicle] = useState("TIGOR-07");
  const cons = useRun<Consensus>();

  return (
    <PageShell active="CALYX" title="Recovery"
      subtitle="Recalibrate a camera from what it recorded, without a calibration target">
      <div className="page-wrap grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
        {/* targetless */}
        <Panel title="From the scene" hint="Two vanishing points and the horizon, in image pixels">
          <p className="section-desc mb-3">
            Mark where two sets of parallel lines meet (for example lane edges and building edges) and the
            horizon height. The camera&apos;s focal length and pitch are worked out from them.
          </p>
          <fieldset className="grid grid-cols-2 gap-x-5 gap-y-2 mb-3">
            <legend className="label-caps mb-1.5">First vanishing point</legend>
            <NumField label="Across (x)" value={vp1x} onChange={setVp1x} w="w-24" />
            <NumField label="Down (y)" value={vp1y} onChange={setVp1y} w="w-24" />
          </fieldset>
          <fieldset className="grid grid-cols-2 gap-x-5 gap-y-2 mb-3">
            <legend className="label-caps mb-1.5">Second vanishing point</legend>
            <NumField label="Across (x)" value={vp2x} onChange={setVp2x} w="w-24" />
            <NumField label="Down (y)" value={vp2y} onChange={setVp2y} w="w-24" />
          </fieldset>
          <fieldset className="grid grid-cols-2 gap-x-5 gap-y-2 mb-3">
            <legend className="label-caps mb-1.5">Horizon and image centre</legend>
            <NumField label="Horizon height (y)" value={horizon} onChange={setHorizon} w="w-24" />
            <span aria-hidden />
            <NumField label="Centre across (x)" value={cx} onChange={setCx} w="w-24" />
            <NumField label="Centre down (y)" value={cy} onChange={setCy} w="w-24" />
          </fieldset>
          <RunButton busy={tl.busy}
            onClick={() => tl.run(() => runJSON<Targetless>("/api/calyx/targetless",
              { vp1: [vp1x, vp1y], vp2: [vp2x, vp2y], horizon_y: horizon, cx, cy }))} label="Recalibrate" />
          <ErrLine err={tl.err} />
          {tl.out && (
            <div className="mt-3 border-t hairline pt-3">
              {tl.out.ok
                ? (<>
                    <KV k="Focal length" v={tl.out.focal != null ? `${tl.out.focal.toFixed(1)} px` : "-"} tone="pass" />
                    <KV k="Pitch" v={tl.out.pitch_deg != null ? `${tl.out.pitch_deg.toFixed(2)}°` : "-"} tone="ink" />
                    <KV k="Confidence" v={tl.out.confidence?.toFixed(2)} tone="ink-3" />
                  </>)
                : <div className="text-[12.5px] text-block">Could not recalibrate: {tl.out.reason}</div>}
            </div>
          )}
        </Panel>

        {/* rig consensus */}
        <Panel title="From past corrections" hint="Combine one vehicle's manual fixes into one best guess">
          <p className="section-desc mb-3">
            Each time someone corrected this vehicle&apos;s calibration by hand, it was recorded. Combining them
            gives a starting point for new drives, and shows how much they disagree.
          </p>
          <label className="flex items-center gap-3 text-[12px] mb-3">
            <span className="text-ink-2">Vehicle</span>
            <input value={vehicle} onChange={(e) => setVehicle(e.target.value)} className="field flex-1" />
          </label>
          <RunButton busy={cons.busy}
            onClick={() => cons.run(() => getJSON<Consensus>(`/api/calyx/rig/${encodeURIComponent(vehicle)}/consensus`))} label="Combine" />
          <ErrLine err={cons.err} />
          {cons.out && (
            <div className="mt-3 border-t hairline pt-3">
              {cons.out.n_overrides === 0 ? (
                <div className="text-[12.5px] text-ink-3">No manual corrections recorded for {cons.out.vehicle_id || vehicle} yet.</div>
              ) : (<>
                <KV k="Corrections combined" v={cons.out.n_overrides} tone="ink" />
                <KV k="Effective sample size" v={cons.out.prior.n} tone="ink-3" />
                <KV k="Roll, pitch, yaw" v={cons.out.prior.rpy_deg ? cons.out.prior.rpy_deg.map((x) => `${x.toFixed(2)}°`).join(", ") : "-"} tone="ink" />
                <KV k="Position x, y, z" v={cons.out.prior.xyz_m ? cons.out.prior.xyz_m.map((x) => `${x.toFixed(2)} m`).join(", ") : "-"} tone="ink" />
                <KV k="Confidence" v={cons.out.prior.confidence.toFixed(2)}
                  tone={cons.out.prior.confidence >= 0.7 ? "pass" : cons.out.prior.confidence > 0 ? "warn" : "ink-3"} />
              </>)}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
