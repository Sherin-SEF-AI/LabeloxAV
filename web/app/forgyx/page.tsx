"use client";

// FORGYX Pareto explorer: how each (model, target) trades latency against accuracy on the target silicon.
// The Pareto front (rank 0) is what you actually deploy. Also shows which targets this machine can build
// (capability-gated) and the deployment artifact registry. Hand-rolled SVG scatter, Operational Materialism.

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";

type Bench = {
  benchmark_id: string; model_version: string; target: string; latency_ms: Record<string, number>;
  throughput_fps: number | null; power_w: number | null; pareto_rank: number; map50?: number;
};
type Deploy = { deployment_id: string; model_version: string; target: string; export_format: string; status: string };

const STATUS: Record<string, string> = { verified: "text-pass", deployed: "text-pass", built: "text-ink-3", blocked: "text-block", retired: "text-ink-3" };

export default function ForgyxPareto() {
  const [caps, setCaps] = useState<Record<string, boolean>>({});
  const [benches, setBenches] = useState<Bench[]>([]);
  const [deploys, setDeploys] = useState<Deploy[]>([]);

  useEffect(() => {
    fetch("/api/forgyx/capabilities").then((r) => r.json()).then((d) => setCaps(d.targets ?? {})).catch(() => {});
    fetch("/api/forgyx/benchmarks").then((r) => r.json()).then((d) => setBenches(d.benchmarks ?? [])).catch(() => {});
    fetch("/api/forgyx/deployments").then((r) => r.json()).then((d) => setDeploys(d.deployments ?? [])).catch(() => {});
  }, []);

  // scatter geometry: x = p95 latency (lower better), y = accuracy (higher better)
  const W = 560, H = 260, PAD = 34;
  const lat = (b: Bench) => b.latency_ms?.p95 ?? 0;
  const acc = (b: Bench) => b.map50 ?? 0;
  const lats = benches.map(lat), accs = benches.map(acc);
  const lmax = Math.max(...lats, 1), amax = Math.max(...accs, 0.1);
  const px = (v: number) => PAD + (v / lmax) * (W - PAD - 8);
  const py = (v: number) => H - PAD - (v / amax) * (H - PAD - 8);

  return (
    <PageShell active="FORGYX" title="FORGYX edge optimization"
      right={<span className="font-mono text-[11px] text-ink-3">{benches.length} benchmarks · {deploys.length} artifacts</span>}>
      <div className="p-4 max-w-5xl space-y-4 font-mono text-[11px]">
        {/* capabilities */}
        <div className="flex flex-wrap gap-2">
          {Object.entries(caps).map(([t, ok]) => (
            <span key={t} className={`border px-1.5 py-0.5 rounded text-[10px] ${ok ? "border-pass text-pass" : "border-line text-ink-3"}`}>
              {t} {ok ? "ready" : "not installed"}
            </span>
          ))}
        </div>

        {/* Pareto scatter */}
        <section className="border border-line p-3">
          <div className="text-[10px] uppercase text-ink-3 mb-1">latency (p95, ms) vs accuracy (mAP50) · Pareto front is rank 0</div>
          {benches.length ? (
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
              <line x1={PAD} y1={H - PAD} x2={W - 8} y2={H - PAD} stroke="#23262B" />
              <line x1={PAD} y1={8} x2={PAD} y2={H - PAD} stroke="#23262B" />
              {benches.map((b) => (
                <g key={b.benchmark_id}>
                  <circle cx={px(lat(b))} cy={py(acc(b))} r={b.pareto_rank === 0 ? 5 : 3.5}
                    fill={b.pareto_rank === 0 ? "#56D364" : "#6C727A"} />
                  <text x={px(lat(b)) + 7} y={py(acc(b)) + 3} fontSize={8} fill="#A0A6AD" fontFamily="monospace">{b.target}</text>
                </g>
              ))}
              <text x={W - 8} y={H - PAD + 12} textAnchor="end" fontSize={8} fill="#6C727A" fontFamily="monospace">slower →</text>
              <text x={PAD} y={14} fontSize={8} fill="#6C727A" fontFamily="monospace">↑ more accurate</text>
            </svg>
          ) : <div className="text-ink-3 py-6 text-center">no benchmarks yet; devices POST measured latency to /forgyx/benchmark</div>}
        </section>

        {/* deployment registry */}
        <section className="border border-line p-3">
          <div className="text-[10px] uppercase text-ink-3 mb-2">deployment artifact registry</div>
          <table className="w-full">
            <thead><tr className="text-ink-3 text-left border-b hairline"><th className="px-2 py-1">model</th><th>target</th><th>format</th><th>status</th></tr></thead>
            <tbody>
              {deploys.map((d) => (
                <tr key={d.deployment_id} className="border-b hairline">
                  <td className="px-2 py-1 text-ink-2 truncate max-w-[220px]">{d.model_version}</td>
                  <td className="text-ink-3">{d.target}</td><td className="text-ink-3">{d.export_format}</td>
                  <td className={STATUS[d.status] ?? "text-ink-3"}>{d.status}</td>
                </tr>
              ))}
              {!deploys.length && <tr><td colSpan={4} className="text-ink-3 text-center py-3">no deployments yet</td></tr>}
            </tbody>
          </table>
        </section>
      </div>
    </PageShell>
  );
}
