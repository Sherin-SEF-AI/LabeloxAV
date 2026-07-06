"use client";

// M16 FORGYX hardware-in-the-loop: co-optimize a model to its target, verify its thermal envelope from a real
// device-farm run, and stage the rollout. Co-optimization ranks (prune, int8) configs against the target
// latency budget; the thermal check refuses without real device readings; rollout refuses skipping canary to
// the whole fleet. Operational Materialism: feasible/passed states are green, refused states red.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Verdict, RunButton, useRun, ErrLine, NumField } from "@/components/engine/prim";
import { runJSON } from "@/lib/engine";

const TARGETS = ["sentrixai_litert", "agx_orin_trt", "orin_nano_trt", "pi_hailo"];

type Coopt = { target: string; budget_ms: number; feasible: boolean;
  chosen: { prune: number; int8: boolean; est_latency_ms: number; est_map50: number } | null;
  ranked: { prune: number; int8: boolean; est_latency_ms: number; est_map50: number }[] };
type Thermal = { passed: boolean; throttled: boolean; over_power: boolean; sustained_fps: number;
  cold_fps: number; headroom_c: number; power_w: number; power_ceiling_w: number; throttle_temp_c: number };
type Rollout = { from: string; action: string; allowed: boolean; new_state: string; reason: string | null };

export default function DeployPage() {
  const [target, setTarget] = useState("agx_orin_trt");
  const [lat, setLat] = useState(50);
  const [map50, setMap50] = useState(0.6);
  const coopt = useRun<Coopt>();

  const [peak, setPeak] = useState(78);
  const [power, setPower] = useState(12);
  const [fps, setFps] = useState(29.8);
  const [cold, setCold] = useState(30);
  const thermal = useRun<Thermal>();

  const [state, setState] = useState("none");
  const rollout = useRun<Rollout>();

  return (
    <PageShell active="FORGYX" title="Hardware-in-the-loop deployment"
      subtitle="co-optimize to target, verify thermal envelope, stage rollout">
      <div className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4 max-w-6xl">
        {/* co-optimization */}
        <Panel title="model-target co-optimization" hint="vs latency budget">
          <label className="flex items-center gap-2 font-mono text-[11px] mb-2">
            <span className="text-ink-3">target</span>
            <select value={target} onChange={(e) => setTarget(e.target.value)}
              className="flex-1 bg-bg-2 border border-line px-1 py-0.5 text-ink outline-none focus:border-accent">
              {TARGETS.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <div className="space-y-2 mb-2">
            <NumField label="base latency ms" value={lat} onChange={setLat} w="w-24" />
            <NumField label="base map50" value={map50} onChange={setMap50} w="w-24" />
          </div>
          <RunButton busy={coopt.busy}
            onClick={() => coopt.run(() => runJSON<Coopt>("/api/forgyx/cooptimize",
              { target, base_latency_ms: lat, base_map50: map50 }))} label="plan" />
          <ErrLine err={coopt.err} />
          {coopt.out && (
            <div className="mt-3">
              <Verdict ok={coopt.out.feasible} yes="feasible" no="no config fits" />
              <KV k="budget" v={`${coopt.out.budget_ms} ms`} tone="ink-3" />
              {coopt.out.chosen && (
                <div className="border border-pass/30 p-2 mt-2 font-mono text-[10px]">
                  <div className="text-ink-2">chosen config</div>
                  <KV k="prune" v={coopt.out.chosen.prune} tone="ink" />
                  <KV k="int8" v={coopt.out.chosen.int8 ? "yes" : "no"} tone="ink" />
                  <KV k="est latency" v={`${coopt.out.chosen.est_latency_ms} ms`} tone="pass" />
                  <KV k="est map50" v={coopt.out.chosen.est_map50} tone="ink" />
                </div>
              )}
            </div>
          )}
        </Panel>

        {/* thermal */}
        <Panel title="thermal / power envelope" hint="real device-farm run">
          <div className="space-y-2 mb-2">
            <NumField label="peak temp C" value={peak} onChange={setPeak} w="w-24" />
            <NumField label="power W" value={power} onChange={setPower} w="w-24" />
            <NumField label="throttled fps" value={fps} onChange={setFps} w="w-24" />
            <NumField label="cold fps" value={cold} onChange={setCold} w="w-24" />
          </div>
          <RunButton busy={thermal.busy}
            onClick={() => thermal.run(() => runJSON<Thermal>("/api/forgyx/thermal",
              { target, readings: { peak_temp_c: peak, power_w: power, throttled_fps: fps }, cold_fps: cold }))} label="check" />
          <ErrLine err={thermal.err} />
          {thermal.out && (
            <div className="mt-3">
              <Verdict ok={thermal.out.passed} yes="within envelope" no="throttled / over power" />
              <KV k="headroom" v={`${thermal.out.headroom_c} C`} tone={thermal.out.headroom_c > 0 ? "pass" : "block"} />
              <KV k="sustained fps" v={`${thermal.out.sustained_fps} / ${thermal.out.cold_fps}`} tone="ink-3" />
              <KV k="power" v={`${thermal.out.power_w} / ${thermal.out.power_ceiling_w} W`}
                tone={thermal.out.over_power ? "block" : "ink-3"} />
            </div>
          )}
        </Panel>

        {/* rollout */}
        <Panel title="rollout / rollback" hint="canary before fleet">
          <div className="flex items-center gap-2 font-mono text-[11px] mb-3">
            <span className="text-ink-3">state</span>
            <span className="px-2 py-0.5 border border-line text-ink">{rollout.out?.new_state ?? state}</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {["canary", "full", "rolled_back"].map((action) => (
              <button key={action} disabled={rollout.busy}
                onClick={() => rollout.run(async () => {
                  const r = await runJSON<Rollout>("/api/forgyx/rollout",
                    { current_state: rollout.out?.new_state ?? state, action });
                  setState(r.new_state);
                  return r;
                })}
                className="font-mono text-[11px] px-2 py-1 border border-line text-ink-2 hover:text-accent hover:border-accent disabled:opacity-40">
                {action}
              </button>
            ))}
          </div>
          <ErrLine err={rollout.err} />
          {rollout.out && (
            <div className="mt-3">
              <Verdict ok={rollout.out.allowed} yes={`${rollout.out.from} to ${rollout.out.action}`} no="transition refused" />
              {rollout.out.reason && <div className="font-mono text-[10px] text-block mt-1">{rollout.out.reason}</div>}
            </div>
          )}
          <button onClick={() => { setState("none"); rollout.setOut(null); }}
            className="mt-3 font-mono text-[10px] text-ink-3 hover:text-accent">reset to none</button>
        </Panel>
      </div>
    </PageShell>
  );
}
