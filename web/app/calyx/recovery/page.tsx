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
    <PageShell active="CALYX" title="Calibration recovery"
      subtitle="targetless recovery from scene cues, and rig-consensus priors">
      <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-4 max-w-5xl">
        {/* targetless */}
        <Panel title="targetless calibration" hint="vanishing points + horizon">
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 mb-2">
            <NumField label="vp1 x" value={vp1x} onChange={setVp1x} w="w-20" />
            <NumField label="vp1 y" value={vp1y} onChange={setVp1y} w="w-20" />
            <NumField label="vp2 x" value={vp2x} onChange={setVp2x} w="w-20" />
            <NumField label="vp2 y" value={vp2y} onChange={setVp2y} w="w-20" />
            <NumField label="horizon y" value={horizon} onChange={setHorizon} w="w-20" />
            <NumField label="cx" value={cx} onChange={setCx} w="w-20" />
            <NumField label="cy" value={cy} onChange={setCy} w="w-20" />
          </div>
          <RunButton busy={tl.busy}
            onClick={() => tl.run(() => runJSON<Targetless>("/api/calyx/targetless",
              { vp1: [vp1x, vp1y], vp2: [vp2x, vp2y], horizon_y: horizon, cx, cy }))} label="recover" />
          <ErrLine err={tl.err} />
          {tl.out && (
            <div className="mt-3">
              {tl.out.ok
                ? (<>
                    <KV k="focal px" v={tl.out.focal?.toFixed(1)} tone="pass" />
                    <KV k="pitch deg" v={tl.out.pitch_deg?.toFixed(2)} tone="ink" />
                    <KV k="confidence" v={tl.out.confidence?.toFixed(3)} tone="ink-3" />
                  </>)
                : <div className="font-mono text-[11px] text-block">refused: {tl.out.reason}</div>}
            </div>
          )}
        </Panel>

        {/* rig consensus */}
        <Panel title="rig consensus prior" hint="fuse per-session overrides">
          <label className="flex items-center gap-2 font-mono text-[11px] mb-2">
            <span className="text-ink-3">vehicle</span>
            <input value={vehicle} onChange={(e) => setVehicle(e.target.value)}
              className="flex-1 bg-bg-2 border border-line px-1.5 py-0.5 text-ink outline-none focus:border-accent" />
          </label>
          <RunButton busy={cons.busy}
            onClick={() => cons.run(() => getJSON<Consensus>(`/api/calyx/rig/${encodeURIComponent(vehicle)}/consensus`))} label="fuse" />
          <ErrLine err={cons.err} />
          {cons.out && (
            <div className="mt-3">
              <KV k="overrides fused" v={cons.out.n_overrides} tone="ink" />
              <KV k="effective n" v={cons.out.prior.n} tone="ink-3" />
              <KV k="rpy deg" v={cons.out.prior.rpy_deg ? cons.out.prior.rpy_deg.map((x) => x.toFixed(2)).join(", ") : "-"} tone="ink" />
              <KV k="xyz m" v={cons.out.prior.xyz_m ? cons.out.prior.xyz_m.map((x) => x.toFixed(2)).join(", ") : "-"} tone="ink" />
              <KV k="confidence" v={cons.out.prior.confidence.toFixed(3)}
                tone={cons.out.prior.confidence >= 0.7 ? "pass" : cons.out.prior.confidence > 0 ? "warn" : "ink-3"} />
              {cons.out.n_overrides === 0 &&
                <div className="font-mono text-[10px] text-ink-3 mt-1">no overrides recorded for this vehicle yet</div>}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
