"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type ModelLine, type TrainingJob } from "@/lib/api";
import TopNav from "@/components/TopNav";

// In-app training platform: submit a training job (a "purpose" = dataset filters + hparams), watch it
// run on the GPU worker (status / stage / epoch / map50), and browse the model registry. The API only
// enqueues; the worker (make train-worker) executes, so the UI never blocks training.

function Section({ title, right, children }: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="flex items-center justify-between border-b hairline px-3 py-2">
        <div className="font-mono text-[11px] uppercase text-ink-3">{title}</div>
        {right}
      </div>
      <div className="p-3">{children}</div>
    </section>
  );
}

const STATUS_COLOR: Record<string, string> = {
  done: "text-pass",
  error: "text-block",
  canceled: "text-ink-3",
  running: "text-warn",
  pending: "text-info",
};

export default function TrainingPage() {
  const router = useRouter();
  const [tasks, setTasks] = useState<{ task_type: string; default_base_weights: string }[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [registry, setRegistry] = useState<ModelLine[]>([]);
  const [err, setErr] = useState<string | null>(null);

  // form
  const [purpose, setPurpose] = useState("vru-detector");
  const [taskType, setTaskType] = useState("detection");
  const [computeTarget, setComputeTarget] = useState("local");
  const [includeClasses, setIncludeClasses] = useState("pedestrian, rider, cycle, motorcycle");
  const [cities, setCities] = useState("");
  const [routePrefix, setRoutePrefix] = useState("");
  const [limit, setLimit] = useState("");
  const [epochs, setEpochs] = useState("20");
  const [promote, setPromote] = useState(false);

  async function refresh() {
    try {
      const [j, r] = await Promise.all([api.listTraining(), api.trainingRegistry()]);
      setJobs(j);
      setRegistry(r);
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    api.trainingTasks().then(setTasks).catch(() => {});
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, []);

  function listFrom(s: string): string[] {
    return s.split(",").map((x) => x.trim()).filter(Boolean);
  }

  async function onSubmit() {
    setErr(null);
    const dataset_spec: Record<string, unknown> = {};
    if (includeClasses.trim()) dataset_spec.include_classes = listFrom(includeClasses);
    if (cities.trim()) dataset_spec.cities = listFrom(cities);
    if (routePrefix.trim()) dataset_spec.route_prefix = routePrefix.trim();
    if (limit.trim()) dataset_spec.limit = Number(limit);
    try {
      await api.startTraining({
        purpose,
        task_type: taskType,
        compute_target: computeTarget,
        dataset_spec,
        hparams: { epochs: Number(epochs) },
        promote,
      });
      refresh();
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="TRAINING" right={<span className="text-ink-3">worker required: <span className="text-ink-2">make train-worker</span></span>} />

      <main className="flex-1 overflow-auto p-4 space-y-4 max-w-5xl w-full mx-auto">
        <Section title="new training job">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-xs">
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">purpose (model line)</span>
              <input value={purpose} onChange={(e) => setPurpose(e.target.value)}
                className="bg-panel border border-line px-2 py-1 text-ink" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">task</span>
              <select value={taskType} onChange={(e) => setTaskType(e.target.value)}
                className="bg-panel border border-line px-2 py-1 text-ink">
                {tasks.map((t) => <option key={t.task_type} value={t.task_type}>{t.task_type}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">compute</span>
              <select value={computeTarget} onChange={(e) => setComputeTarget(e.target.value)}
                className="bg-panel border border-line px-2 py-1 text-ink">
                <option value="local">local (5080, light)</option>
                <option value="cloud">cloud (A100, heavy)</option>
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">epochs</span>
              <input value={epochs} onChange={(e) => setEpochs(e.target.value)} type="number"
                className="bg-panel border border-line px-2 py-1 text-ink" />
            </label>
            <label className="flex items-center gap-2 mt-5">
              <input type="checkbox" checked={promote} onChange={(e) => setPromote(e.target.checked)} />
              <span className="text-ink-2">promote if gate passes</span>
            </label>
            <label className="flex flex-col gap-1 md:col-span-2">
              <span className="text-ink-3 uppercase text-[11px]">include classes (specialized; blank = all)</span>
              <input value={includeClasses} onChange={(e) => setIncludeClasses(e.target.value)}
                placeholder="pedestrian, rider, cycle" className="bg-panel border border-line px-2 py-1 text-ink" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">cities (per-domain)</span>
              <input value={cities} onChange={(e) => setCities(e.target.value)} placeholder="BLR"
                className="bg-panel border border-line px-2 py-1 text-ink" />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 uppercase text-[11px]">route prefix / limit</span>
              <div className="flex gap-1">
                <input value={routePrefix} onChange={(e) => setRoutePrefix(e.target.value)} placeholder="202606"
                  className="bg-panel border border-line px-2 py-1 text-ink w-1/2" />
                <input value={limit} onChange={(e) => setLimit(e.target.value)} placeholder="limit" type="number"
                  className="bg-panel border border-line px-2 py-1 text-ink w-1/2" />
              </div>
            </label>
          </div>
          <div className="flex items-center gap-3 mt-3">
            <button onClick={onSubmit}
              className="font-mono text-xs border border-line px-3 py-1 hover:border-accent">
              queue training job
            </button>
            {err && <span className="font-mono text-[11px] text-block">{err}</span>}
            <span className="font-mono text-[11px] text-ink-3 ml-auto">
              worker required: <span className="text-ink-2">make train-worker</span>
            </span>
          </div>
        </Section>

        <Section title={`jobs (${jobs.length})`}>
          {jobs.length ? (
            <div className="space-y-1">
              {jobs.map((j) => {
                const live = (j.metrics?.live as { map50?: number } | undefined) ?? undefined;
                const ep = j.counts?.epoch ?? 0;
                const tot = j.counts?.total_epochs ?? 0;
                return (
                  <div key={j.job_id} className="flex items-center gap-2 font-mono text-[11px]">
                    <span className="w-28 truncate text-ink-2" title={j.purpose}>{j.purpose}</span>
                    <span className={`w-12 ${j.compute_target === "cloud" ? "text-info" : "text-ink-3"}`}>
                      {j.compute_target}
                    </span>
                    <span className={`w-16 ${STATUS_COLOR[j.status] ?? "text-ink-3"}`}>{j.status}</span>
                    <span className="w-16 text-ink-3">{j.stage ?? "-"}</span>
                    <div className="flex-1 h-2 bg-line relative">
                      <div className="absolute left-0 top-0 h-full bg-accent" style={{ width: `${(j.progress || 0) * 100}%` }} />
                    </div>
                    <span className="w-28 text-right text-ink-3">
                      {tot ? `ep ${ep}/${tot}` : ""} {live?.map50 != null ? `· map50 ${live.map50}` : ""}
                    </span>
                    {(j.status === "pending" || j.status === "running") && (
                      <button onClick={() => api.cancelTraining(j.job_id).then(refresh)}
                        className="border border-line px-1.5 hover:border-block">cancel</button>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="font-mono text-xs text-ink-3 py-4 text-center">no training jobs yet</div>
          )}
        </Section>

        <Section title={`model registry (${registry.length} lines)`}>
          {registry.length ? (
            <div className="space-y-2">
              {registry.map((line) => (
                <div key={line.purpose} className="border-b hairline pb-2">
                  <div className="flex items-center gap-2 font-mono text-xs">
                    <span className="text-ink">{line.purpose}</span>
                    <span className="text-ink-3">{line.task_type}</span>
                    {line.promoted && (
                      <span className="text-pass">promoted map50 {line.promoted.map50 ?? "-"}</span>
                    )}
                    <span className="ml-auto text-ink-3">{line.runs.length} runs</span>
                  </div>
                  <div className="font-mono text-[11px] text-ink-3 mt-1 space-y-0.5">
                    {line.runs.slice(0, 4).map((r) => (
                      <div key={r.run_id} className="flex justify-between">
                        <span className="truncate">{r.run_id}</span>
                        <span>map50 {r.map50 ?? "-"} · safe {r.safe_miou ?? "-"} · ep {r.epochs}{r.promoted ? " · *" : ""}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="font-mono text-xs text-ink-3 py-4 text-center">no models trained yet</div>
          )}
        </Section>
      </main>
    </div>
  );
}
