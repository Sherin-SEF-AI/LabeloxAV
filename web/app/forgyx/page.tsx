"use client";

// Latency and accuracy (Pareto explorer): how each (model, target) trades latency against accuracy on the
// target silicon. The Pareto front (rank 0) is what you actually deploy. Also shows which targets this machine
// can build (capability-gated) and the deployment artifact registry. Hand-rolled SVG scatter.

import { useEffect, useState } from "react";
import Link from "next/link";
import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";
import { Badge, Stat } from "@/components/ui/kit";
import { apiGet } from "@/lib/api";
import { humanize, num } from "@/lib/format";

type Bench = {
  benchmark_id: string; model_version: string; target: string; latency_ms: Record<string, number>;
  throughput_fps: number | null; power_w: number | null; pareto_rank: number; map50?: number;
};
type Deploy = { deployment_id: string; model_version: string; target: string; export_format: string; status: string };

const STATUS: Record<string, "pass" | "muted" | "block"> = { verified: "pass", deployed: "pass", built: "muted", blocked: "block", retired: "muted" };

// The build targets, by the hardware a person would recognise rather than the toolchain key.
const TARGET_NAME: Record<string, string> = {
  onnx: "ONNX export",
  onnxruntime: "ONNX Runtime",
  onnx_cpu: "ONNX on CPU",
  sentrixai_litert: "SentrixAI (LiteRT)",
  agx_orin_trt: "Jetson AGX Orin (TensorRT)",
  orin_nano_trt: "Jetson Orin Nano (TensorRT)",
  pi_hailo: "Raspberry Pi + Hailo",
  diffusion: "Diffusion",
  esmini: "esmini",
};
const targetName = (t: string) => TARGET_NAME[t] ?? humanize(t);

export default function ForgyxPareto() {
  const [caps, setCaps] = useState<Record<string, boolean>>({});
  const [benches, setBenches] = useState<Bench[]>([]);
  const [deploys, setDeploys] = useState<Deploy[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    apiGet<{ targets?: Record<string, boolean> }>("/api/forgyx/capabilities").then((d) => setCaps(d.targets ?? {})).catch(() => {});
    apiGet<{ benchmarks?: Bench[] }>("/api/forgyx/benchmarks").then((d) => setBenches(d.benchmarks ?? [])).catch(() => {}).finally(() => setLoaded(true));
    apiGet<{ deployments?: Deploy[] }>("/api/forgyx/deployments").then((d) => setDeploys(d.deployments ?? [])).catch(() => {});
  }, []);

  // scatter geometry: x = p95 latency (lower better), y = accuracy (higher better)
  const W = 880, H = 320, L = 56, R = 150, T = 16, B = 44;
  const lat = (b: Bench) => b.latency_ms?.p95 ?? 0;
  const acc = (b: Bench) => b.map50 ?? 0;
  const lats = benches.map(lat), accs = benches.map(acc);
  const lmax = Math.max(...lats, 1), amax = Math.max(...accs, 0.1);
  const px = (v: number) => L + (v / lmax) * (W - L - R);
  const py = (v: number) => H - B - (v / amax) * (H - B - T);
  const front = benches.filter((b) => b.pareto_rank === 0).length;
  const ready = Object.values(caps).filter(Boolean).length;

  return (
    <PageShell active="Edge deployment" title="Latency and accuracy"
      subtitle="How fast and how accurate each model is on each vehicle computer, measured on the device."
      primaryAction={<Link href="/forgyx/deploy" className="btn btn-primary">Plan a deployment</Link>}>
      <div className="page-wrap space-y-4">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <Stat label="Benchmarks" value={num(benches.length)} sub="model and device pairs measured" />
          <Stat label="Best trade-offs" value={num(front)} tone={front ? "pass" : undefined}
            sub="nothing else is both faster and more accurate" />
          <Stat label="Deployment packages" value={num(deploys.length)} sub="built for a vehicle" />
          <Stat label="Targets this server can build" value={`${ready} of ${Object.keys(caps).length}`} />
        </div>

        {/* Pareto scatter */}
        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Latency against accuracy</h2>
            <span className="ml-auto flex items-center gap-3 text-[12px] text-ink-3">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-pass" />Best trade-off</span>
              <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-ink-3" />Other</span>
            </span>
          </div>
          <div className="p-4">
            {benches.length ? (
              <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto max-w-[960px]" role="img"
                aria-label="Scatter of p95 latency against accuracy for each benchmark">
                {[0, 0.5, 1].map((f) => (
                  <g key={`y${f}`}>
                    <line x1={L} x2={W - R} y1={py(amax * f)} y2={py(amax * f)} className="stroke-line" strokeDasharray={f ? "3 4" : undefined} />
                    <text x={L - 8} y={py(amax * f) + 4} textAnchor="end" fontSize={11} className="fill-ink-3">{(amax * f).toFixed(2)}</text>
                  </g>
                ))}
                {[0, 0.5, 1].map((f) => (
                  <text key={`x${f}`} x={px(lmax * f)} y={H - B + 18} textAnchor="middle" fontSize={11} className="fill-ink-3">
                    {(lmax * f).toFixed(1)} ms
                  </text>
                ))}
                <line x1={L} y1={T} x2={L} y2={H - B} className="stroke-line" />
                <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" fontSize={11} className="fill-ink-3">
                  p95 latency (slower to the right)
                </text>
                <text x={14} y={(T + H - B) / 2} textAnchor="middle" fontSize={11} className="fill-ink-3"
                  transform={`rotate(-90 14 ${(T + H - B) / 2})`}>mAP50 (higher is better)</text>
                {benches.map((b) => (
                  <g key={b.benchmark_id}>
                    <circle cx={px(lat(b))} cy={py(acc(b))} r={b.pareto_rank === 0 ? 6 : 4.5}
                      className={b.pareto_rank === 0 ? "fill-pass" : "fill-ink-3"}>
                      <title>{`${targetName(b.target)}: ${lat(b)} ms p95, mAP50 ${acc(b).toFixed(2)}`}</title>
                    </circle>
                    <text x={px(lat(b)) + 10} y={py(acc(b)) + 4} fontSize={11} className="fill-ink-2">{targetName(b.target)}</text>
                  </g>
                ))}
              </svg>
            ) : loaded ? (
              <EmptyState compact icon="chart" title="No benchmarks yet">
                Each point is a model measured on a real vehicle computer. They appear once a device reports
                its latency and accuracy for a model.
              </EmptyState>
            ) : (
              <div className="skeleton h-[240px] w-full" aria-label="loading" />
            )}
          </div>
          {benches.length > 0 && (
            <table className="data-table border-t hairline">
              <thead><tr>
                <th>Device</th><th>Model</th><th className="num">p95 latency</th><th className="num">mAP50</th>
                <th className="num">Frames per second</th><th className="num">Power</th><th></th>
              </tr></thead>
              <tbody>
                {benches.map((b) => (
                  <tr key={b.benchmark_id}>
                    <td className="text-ink">{targetName(b.target)}</td>
                    <td className="font-mono text-[12px]">{b.model_version}</td>
                    <td className="num">{lat(b)} ms</td>
                    <td className="num">{acc(b).toFixed(2)}</td>
                    <td className="num">{b.throughput_fps != null ? num(b.throughput_fps) : "-"}</td>
                    <td className="num">{b.power_w != null ? `${b.power_w} W` : "-"}</td>
                    <td>{b.pareto_rank === 0 && <Badge state="pass">Best trade-off</Badge>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {/* deployment registry */}
        <section className="panel">
          <div className="panel-head"><h2 className="section-title">Deployment packages</h2></div>
          {deploys.length ? (
            <table className="data-table">
              <thead><tr><th>Model</th><th>Device</th><th>Format</th><th>Status</th></tr></thead>
              <tbody>
                {deploys.map((d) => (
                  <tr key={d.deployment_id}>
                    <td className="text-ink truncate max-w-[260px]">{d.model_version}</td>
                    <td>{targetName(d.target)}</td>
                    <td>{d.export_format}</td>
                    <td><Badge state={STATUS[d.status] ?? "muted"}>{humanize(d.status)}</Badge></td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <EmptyState compact icon="server" title="No deployment packages yet"
              action={<Link href="/forgyx/deploy" className="btn">Plan a deployment</Link>}>
              A package is a model compiled for one vehicle computer. Plan one to check it fits the latency
              and thermal budget before it is built.
            </EmptyState>
          )}
        </section>

        {/* capabilities */}
        {Object.keys(caps).length > 0 && (
          <section className="panel">
            <div className="panel-head">
              <h2 className="section-title">Build targets on this server</h2>
              <span className="ml-auto text-[12px] text-ink-3">A target needs its toolchain installed before models can be built for it.</span>
            </div>
            <div className="p-3.5 flex flex-wrap gap-2">
              {Object.entries(caps).map(([t, ok]) => (
                <Badge key={t} state={ok ? "pass" : "muted"}>{targetName(t)}: {ok ? "ready" : "not installed"}</Badge>
              ))}
            </div>
          </section>
        )}
      </div>
    </PageShell>
  );
}
