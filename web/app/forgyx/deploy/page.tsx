"use client";

// Hardware-in-the-loop deployment: co-optimize a model to its target, verify its thermal envelope from a real
// device-farm run, and stage the rollout. Co-optimization ranks (prune, int8) configs against the target
// latency budget; the thermal check refuses without real device readings; rollout refuses skipping canary to
// the whole fleet. Feasible/passed states are green, refused states red.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Verdict, Field, NumberInput, useRun, ErrLine, Badge } from "@/components/ui/kit";
import { runJSON } from "@/lib/engine";
import { humanize } from "@/lib/format";

const TARGETS: [string, string][] = [
  ["sentrixai_litert", "SentrixAI (LiteRT)"],
  ["agx_orin_trt", "Jetson AGX Orin (TensorRT)"],
  ["orin_nano_trt", "Jetson Orin Nano (TensorRT)"],
  ["pi_hailo", "Raspberry Pi + Hailo"],
];

// The rollout actions the server accepts, with what each means to the person pressing it.
const ACTIONS: { action: string; label: string; danger?: boolean }[] = [
  { action: "canary", label: "Start canary" },
  { action: "full", label: "Roll out to the fleet" },
  { action: "rolled_back", label: "Roll back", danger: true },
];
const STATE_NAME: Record<string, string> = {
  none: "Not rolled out", canary: "Canary (a few vehicles)", full: "Whole fleet", rolled_back: "Rolled back",
};
const stateName = (s: string) => STATE_NAME[s] ?? humanize(s);

type Coopt = { target: string; budget_ms: number; feasible: boolean;
  chosen: { prune: number; int8: boolean; est_latency_ms: number; est_map50: number } | null;
  ranked: { prune: number; int8: boolean; est_latency_ms: number; est_map50: number }[] };
type Thermal = { passed: boolean; throttled: boolean; over_power: boolean; sustained_fps: number;
  cold_fps: number; headroom_c: number; power_w: number; power_ceiling_w: number; throttle_temp_c: number };
type Rollout = { from: string; action: string; allowed: boolean; new_state: string; reason: string | null };

function RunButton({ busy, onClick, label }: { busy: boolean; onClick: () => void; label: string }) {
  return (
    <button type="button" onClick={onClick} disabled={busy} className="btn btn-primary mt-3">
      {busy ? "Running..." : label}
    </button>
  );
}

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
  const current = rollout.out?.new_state ?? state;

  return (
    <PageShell active="Edge deployment" title="Hardware-in-the-loop deployment"
      subtitle="Fit a model to a vehicle computer, check it stays cool under load, then roll it out in stages.">
      <div className="page-wrap space-y-4">
        <div className="panel p-4 flex flex-wrap items-center gap-3">
          <label htmlFor="deploy-target" className="text-[12.5px] text-ink-2">Vehicle computer</label>
          <select id="deploy-target" value={target} onChange={(e) => setTarget(e.target.value)} className="field w-72">
            {TARGETS.map(([t, name]) => <option key={t} value={t}>{name}</option>)}
          </select>
          <span className="text-[12px] text-ink-3">Used by all three steps below.</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* co-optimization */}
          <Panel title="1. Fit the model to the device"
            desc="Finds the pruning and 8-bit settings that meet the device's latency budget while losing the least accuracy.">
            <div className="space-y-1">
              <Field label="Current latency (ms)"><NumberInput value={lat} onChange={setLat} /></Field>
              <Field label="Current mAP50"><NumberInput value={map50} onChange={setMap50} /></Field>
            </div>
            <RunButton busy={coopt.busy} label="Find settings"
              onClick={() => coopt.run(() => runJSON<Coopt>("/api/forgyx/cooptimize",
                { target, base_latency_ms: lat, base_map50: map50 }))} />
            <ErrLine err={coopt.err} />
            {coopt.out && (
              <div className="mt-3 space-y-1">
                <Verdict ok={coopt.out.feasible} yes="Fits the budget" no="No setting fits" />
                <KV k="Latency budget" v={`${coopt.out.budget_ms} ms`} state="muted" />
                {coopt.out.chosen && (
                  <div className="rounded border border-pass/30 bg-pass/5 p-2.5 mt-2">
                    <div className="text-[12px] text-ink-2 mb-1">Recommended settings</div>
                    <KV k="Pruning" v={coopt.out.chosen.prune} />
                    <KV k="8-bit (int8)" v={coopt.out.chosen.int8 ? "Yes" : "No"} />
                    <KV k="Estimated latency" v={`${coopt.out.chosen.est_latency_ms} ms`} state="pass" />
                    <KV k="Estimated mAP50" v={coopt.out.chosen.est_map50} />
                  </div>
                )}
              </div>
            )}
          </Panel>

          {/* thermal */}
          <Panel title="2. Check heat and power"
            desc="Enter readings from a real run on the device. A model that throttles when hot is slower in the vehicle than on the bench.">
            <div className="space-y-1">
              <Field label="Peak temperature (°C)"><NumberInput value={peak} onChange={setPeak} /></Field>
              <Field label="Power draw (W)"><NumberInput value={power} onChange={setPower} /></Field>
              <Field label="Frames per second when hot"><NumberInput value={fps} onChange={setFps} /></Field>
              <Field label="Frames per second when cold"><NumberInput value={cold} onChange={setCold} /></Field>
            </div>
            <RunButton busy={thermal.busy} label="Check"
              onClick={() => thermal.run(() => runJSON<Thermal>("/api/forgyx/thermal",
                { target, readings: { peak_temp_c: peak, power_w: power, throttled_fps: fps }, cold_fps: cold }))} />
            <ErrLine err={thermal.err} />
            {thermal.out && (
              <div className="mt-3 space-y-1">
                <Verdict ok={thermal.out.passed} yes="Within limits" no="Throttles or draws too much power" />
                <KV k="Temperature headroom" v={`${thermal.out.headroom_c} °C`} state={thermal.out.headroom_c > 0 ? "pass" : "block"} />
                <KV k="Sustained frame rate" v={`${thermal.out.sustained_fps} of ${thermal.out.cold_fps} fps`} state="muted" />
                <KV k="Power" v={`${thermal.out.power_w} of ${thermal.out.power_ceiling_w} W`}
                  state={thermal.out.over_power ? "block" : "muted"} />
              </div>
            )}
          </Panel>

          {/* rollout */}
          <Panel title="3. Roll out in stages"
            desc="A canary on a few vehicles comes before the whole fleet; skipping it is refused.">
            <div className="flex items-center gap-2 text-[12.5px] mb-3">
              <span className="text-ink-3">Current stage</span>
              <Badge state={current === "full" ? "pass" : current === "rolled_back" ? "block" : current === "canary" ? "warn" : "muted"}>
                {stateName(current)}
              </Badge>
            </div>
            <div className="flex flex-wrap gap-2">
              {ACTIONS.map(({ action, label, danger }) => (
                <button key={action} type="button" disabled={rollout.busy}
                  onClick={() => rollout.run(async () => {
                    const r = await runJSON<Rollout>("/api/forgyx/rollout",
                      { current_state: rollout.out?.new_state ?? state, action });
                    setState(r.new_state);
                    return r;
                  })}
                  className={`btn ${danger ? "btn-danger ml-auto" : ""}`}>
                  {label}
                </button>
              ))}
            </div>
            <ErrLine err={rollout.err} />
            {rollout.out && (
              <div className="mt-3 space-y-1">
                <Verdict ok={rollout.out.allowed}
                  yes={`${stateName(rollout.out.from)} to ${stateName(rollout.out.action)}`} no="Step refused" />
                {rollout.out.reason && <div className="text-[12px] text-block mt-1">{rollout.out.reason}</div>}
              </div>
            )}
            <button type="button" onClick={() => { setState("none"); rollout.setOut(null); }}
              className="link mt-4 text-[12px] block">Start over</button>
          </Panel>
        </div>
      </div>
    </PageShell>
  );
}
