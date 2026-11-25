"use client";

import { useEffect, useMemo, useState } from "react";
import { T } from "@/lib/i18n";
import { useRouter } from "next/navigation";
import { useJobStream } from "@/lib/useEventStream";
import { api, apiGet, type ModelLine, type TrainingJob, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { StateBadge } from "@/components/StateBadge";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { Badge } from "@/components/ui/kit";
import { ago, humanize, pct, plural, shortId } from "@/lib/format";
import RunsPanel from "@/components/training/RunsPanel";
import { useFeatures } from "@/lib/features";
import { invalidate, useQuery } from "@/lib/useQuery";

// In-app training platform: submit a training job (a "purpose" = dataset filters + hparams), watch it
// run on the GPU worker (status / stage / epoch / map50), and browse the model registry. The API only
// enqueues; the GPU worker (services/gpu_worker.py) executes, so the UI never blocks training.

function Section({ title, desc, right, children, flush = false }: {
  title: string; desc?: string; right?: React.ReactNode; children: React.ReactNode; flush?: boolean;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="section-title">{title}</h2>
        {right && <div className="ml-auto flex items-center gap-2">{right}</div>}
      </div>
      <div className={flush ? "" : "p-4"}>
        {desc && <p className="section-desc mb-3">{desc}</p>}
        {children}
      </div>
    </section>
  );
}

// A stage name the worker records ("queued-cloud", "evaluate") read as words.
const stageLabel = (st: string | null, status: string) =>
  !st || st === status ? "" : humanize(st);

// A metric stored as 0..1 shown to two places, as it is everywhere else in the registry.
const m2 = (v: number | null | undefined) => (v == null ? "-" : Number(v).toFixed(2));

export default function TrainingPage() {
  const features = useFeatures();
  const router = useRouter();
  // Optional: without it the task picker shows only the default, which is what a failure looked like.
  const tasksQ = useQuery(["trainingTasks"], () => api.trainingTasks());
  const tasks: { task_type: string; default_base_weights: string }[] = tasksQ.data ?? [];
  // The job list and the registry are drawn together and refreshed together, so they stay one request
  // pair. A failed refresh leaves what is on screen, as before.
  const trainingQ = useQuery(["listTraining+trainingRegistry"],
    () => Promise.all([api.listTraining(), api.trainingRegistry()]));
  const jobs: TrainingJob[] = trainingQ.data?.[0] ?? [];
  const registry: ModelLine[] = trainingQ.data?.[1] ?? [];
  const refresh = trainingQ.reload;
  const [err, setErr] = useState<string | null>(null);

  // form
  const [purpose, setPurpose] = useState("");
  const [taskType, setTaskType] = useState("detection");
  const [computeTarget, setComputeTarget] = useState("local");
  const [includeClasses, setIncludeClasses] = useState("");
  const [projectId, setProjectId] = useState("");
  type ProjectOption = { project_id: string; name: string; sessions: number; class_set: string; class_count: number };
  const projectsQ = useQuery(["/api/projects"], () => apiGet<{ projects: ProjectOption[] }>("/api/projects"));
  const projects: ProjectOption[] = useMemo(
    () => (projectsQ.data?.projects ?? []).filter((p) => p.sessions > 0), [projectsQ.data]);
  const [cities, setCities] = useState("");
  const [routePrefix, setRoutePrefix] = useState("");
  const [limit, setLimit] = useState("");
  const [epochs, setEpochs] = useState("20");
  const [promote, setPromote] = useState(false);

  // Pushed, not polled. This page used to re-fetch the whole job list every two seconds whether or not
  // anything had changed, with no pause when the tab was hidden. The stream sends only on change, and the
  // one-shot refresh below covers the first paint and any deployment where the stream cannot connect.
  const jobStream = useJobStream();
  useEffect(() => {
    if (projectsQ.error) {
      setErr(`Could not load projects, so training covers every project: ${humanizeError(projectsQ.error)}`);
    }
  }, [projectsQ.error]);
  useEffect(() => { if (jobStream.data) void refresh(); }, [jobStream.data, refresh]);

  function listFrom(s: string): string[] {
    return s.split(",").map((x) => x.trim()).filter(Boolean);
  }

  async function onSubmit() {
    setErr(null);
    const dataset_spec: Record<string, unknown> = {};
    if (includeClasses.trim()) dataset_spec.include_classes = listFrom(includeClasses);
    if (projectId) dataset_spec.project_id = projectId;
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
      void refresh();
      // A training run is a job: the jobs table and home's running count both change.
      void invalidate("jobs", "/api/home");
    } catch (e) {
      setErr(humanizeError(e));
    }
  }

  const fieldCls = "field w-full text-[12.5px]";
  const active = jobs.filter((j) => j.status === "pending" || j.status === "running").length;

  return (
    <PageShell
      active="TRAINING"
      title="Training"
      subtitle="Train a model on your labelled data, watch it run, and compare it with the current champion."
      right={features.gpu_worker === undefined ? undefined : features.gpu_worker
        ? <span className="inline-flex items-center gap-1.5 text-[12px] text-pass"><span className="w-2 h-2 rounded-full bg-pass" aria-hidden />GPU worker running</span>
        : <span className="inline-flex items-center gap-1.5 text-[12px] text-warn" data-tip="Queued jobs wait until an administrator starts the GPU worker"><span className="w-2 h-2 rounded-full bg-warn" aria-hidden />GPU worker stopped: queued jobs wait</span>}
    >
      <div className="page-wrap space-y-4">
        <Section title="New training job">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-x-4 gap-y-3 text-[12.5px]">
            <label className="flex flex-col gap-1.5 sm:col-span-2">
              <span className="label-caps">Model line name</span>
              <input value={purpose} placeholder="e.g. vehicle-detector" onChange={(e) => setPurpose(e.target.value)}
                className={fieldCls} />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="label-caps">Task</span>
              <select value={taskType} onChange={(e) => setTaskType(e.target.value)} className={fieldCls}>
                {tasks.map((t) => <option key={t.task_type} value={t.task_type}>{humanize(t.task_type)}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="label-caps">Compute</span>
              <select value={computeTarget} onChange={(e) => setComputeTarget(e.target.value)} className={fieldCls}>
                <option value="local">Local GPU (RTX 5080, lighter runs)</option>
                <option value="cloud">Cloud GPU (A100, heavier runs)</option>
              </select>
            </label>
            <label className="flex flex-col gap-1.5 sm:col-span-2">
              <span className="label-caps">Project</span>
              <select value={projectId} onChange={(e) => setProjectId(e.target.value)} className={fieldCls}>
                <option value="">Every project</option>
                {projects.map((p) => (
                  <option key={p.project_id} value={p.project_id}>
                    {p.name} ({plural(p.sessions, "session")}{p.class_set !== "india" ? `, its ${p.class_count} classes` : ""})
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="label-caps">Epochs</span>
              <input value={epochs} onChange={(e) => setEpochs(e.target.value)} type="number" className={fieldCls} />
            </label>
            <label className="flex items-center gap-2 lg:mt-6" data-tip="Make this the champion automatically if it beats the current one on the promotion gate">
              <input type="checkbox" checked={promote} onChange={(e) => setPromote(e.target.checked)} />
              <span className="text-ink-2">Promote if it passes the gate</span>
            </label>
          </div>
          <details className="mt-4 group">
            <summary className="cursor-pointer select-none text-[12.5px] text-ink-2">Narrow the training data (optional)</summary>
            <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-x-4 gap-y-3 text-[12.5px]">
              <label className="flex flex-col gap-1.5 sm:col-span-2">
                <span className="label-caps">Only these classes</span>
                <input value={includeClasses} placeholder="Blank trains on every class; e.g. pedestrian, rider, motorcycle" onChange={(e) => setIncludeClasses(e.target.value)}
                  className={fieldCls} />
              </label>
              <label className="flex flex-col gap-1.5">
                <span className="label-caps">Cities</span>
                <input value={cities} onChange={(e) => setCities(e.target.value)} placeholder="e.g. BLR" className={fieldCls} />
              </label>
              <label className="flex flex-col gap-1.5">
                <span className="label-caps">Route prefix and frame limit</span>
                <div className="flex gap-2">
                  <input value={routePrefix} onChange={(e) => setRoutePrefix(e.target.value)} placeholder="e.g. 202606"
                    aria-label="Route prefix" className="field text-[12.5px] w-1/2 min-w-0" />
                  <input value={limit} onChange={(e) => setLimit(e.target.value)} placeholder="Max frames" type="number"
                    aria-label="Frame limit" className="field text-[12.5px] w-1/2 min-w-0" />
                </div>
              </label>
            </div>
          </details>
          <div className="flex items-center gap-3 mt-4">
            <button onClick={onSubmit} disabled={!purpose.trim()}
              data-tip={purpose.trim() ? undefined : "Name the model line first"}
              className="btn btn-primary">
              Queue training job
            </button>
            {!purpose.trim() && <span className="text-[12px] text-ink-3">Give the model line a name to queue it.</span>}
            {err && <span className="text-[12px] text-block">{err}</span>}
          </div>
        </Section>

        <Section title="Jobs" flush
          right={<span className="text-[12px] text-ink-3 font-normal">{plural(jobs.length, "job")}{active ? `, ${active} active` : ""}</span>}>
          {trainingQ.error && !trainingQ.data ? (
            <div className="p-3"><LoadState error={trainingQ.error} onRetry={() => void refresh()} /></div>
          ) : trainingQ.loading && !trainingQ.data ? (
            <div className="p-3"><LoadState loading rows={5} /></div>
          ) : jobs.length ? (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model line</th>
                    <th>Status</th>
                    <th className="w-[220px]">Progress</th>
                    <th className="num">Epoch</th>
                    <th className="num">mAP50</th>
                    <th>Compute</th>
                    <th>Queued</th>
                    <th><span className="sr-only">Actions</span></th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((j) => {
                    const live = (j.metrics?.live as { map50?: number } | undefined) ?? undefined;
                    const ep = j.counts?.epoch ?? 0;
                    const tot = j.counts?.total_epochs ?? 0;
                    const failed = j.status === "error" || j.status === "failed";
                    const stage = stageLabel(j.stage, j.status);
                    return (
                      <tr key={j.job_id} className="align-top">
                        <td>
                          <div className="text-ink font-medium">{j.purpose}</div>
                          <div className="text-[11px] text-ink-3 font-mono">{shortId(j.job_id)}</div>
                        </td>
                        <td>
                          <StateBadge state={j.status} />
                          {/* Why it failed, on the row. The API has returned this field all along and the row
                              never drew it, so a failed run and a stalled one looked identical. */}
                          {failed && (
                            <div className={`text-[12px] mt-1 max-w-[46ch] leading-snug ${j.error ? "text-block" : "text-ink-3"}`}>
                              {j.error ?? "Failed without recording a reason, which is itself worth reporting."}
                            </div>
                          )}
                        </td>
                        <td>
                          <div className="flex items-center gap-2">
                            <span className="flex-1 h-1.5 rounded-full bg-line overflow-hidden min-w-[80px]">
                              <span className={`block h-full rounded-full ${failed ? "bg-block" : j.status === "done" ? "bg-pass" : "bg-accent"}`}
                                style={{ width: `${Math.max(0, Math.min(1, j.progress || 0)) * 100}%` }} />
                            </span>
                            <span className="text-[12px] text-ink-3 tabular-nums w-10 text-right">{pct(j.progress || 0)}</span>
                          </div>
                          {stage && <div className="text-[11.5px] text-ink-3 mt-1">{stage}</div>}
                        </td>
                        <td className="num">{tot ? `${ep} / ${tot}` : "-"}</td>
                        <td className="num">{m2(live?.map50)}</td>
                        <td className={j.compute_target === "cloud" ? "text-info" : ""}>{j.compute_target === "cloud" ? "Cloud" : "Local"}</td>
                        <td className="text-ink-3 whitespace-nowrap" title={j.created_at ?? undefined}>{ago(j.created_at)}</td>
                        <td className="text-right">
                          {(j.status === "pending" || j.status === "running") && (
                            <button onClick={() => api.cancelTraining(j.job_id).then(() => { void refresh(); void invalidate("jobs", "/api/home"); })}
                              className="btn"><T k="action.cancel">Cancel</T></button>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState compact icon="chip" title="No training jobs yet">
              Queue one above. It runs on the GPU worker and shows its progress here.
            </EmptyState>
          )}
        </Section>

        <Section title="Model runs" desc="Training curves for a run, and how the candidate compares with the champion on the promotion gate.">
          <RunsPanel />
        </Section>

        <Section title="Model lines" flush
          right={<span className="text-[12px] text-ink-3 font-normal">{plural(registry.length, "line")}</span>}>
          {registry.length ? (
            <div className="divide-y divide-line">
              {registry.map((line) => (
                <div key={line.purpose} className="px-4 py-3">
                  <div className="flex items-center gap-3 text-[13px]">
                    <span className="text-ink font-medium">{line.purpose}</span>
                    <span className="text-ink-3 text-[12px]">{humanize(line.task_type)}</span>
                    {line.promoted && (
                      <Badge state="pass">Champion mAP50 {m2(line.promoted.map50)}</Badge>
                    )}
                    <span className="ml-auto text-[12px] text-ink-3">{plural(line.runs.length, "run")}</span>
                  </div>
                  <table className="w-full mt-2 text-[12px]">
                    <thead>
                      <tr className="text-ink-3 text-left">
                        <th className="font-normal py-1">Run</th>
                        <th className="font-normal py-1 text-right w-20">mAP50</th>
                        <th className="font-normal py-1 text-right w-24" data-tip="Mean IoU on safety-critical classes">Safety mIoU</th>
                        <th className="font-normal py-1 text-right w-20">Epochs</th>
                        <th className="font-normal py-1 text-right w-28">Trained</th>
                      </tr>
                    </thead>
                    <tbody>
                      {line.runs.slice(0, 4).map((r) => (
                        <tr key={r.run_id} className="text-ink-2">
                          <td className="py-0.5 truncate max-w-0 w-full">
                            <span className="font-mono text-[11.5px]">{r.run_id}</span>
                            {r.promoted && <span className="ml-2 text-pass">Champion</span>}
                          </td>
                          <td className="py-0.5 text-right tabular-nums">{m2(r.map50)}</td>
                          <td className="py-0.5 text-right tabular-nums">{m2(r.safe_miou)}</td>
                          <td className="py-0.5 text-right tabular-nums">{r.epochs}</td>
                          <td className="py-0.5 text-right text-ink-3 whitespace-nowrap">{ago(r.created_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {line.runs.length > 4 && (
                    <div className="text-[11.5px] text-ink-3 mt-1">and {plural(line.runs.length - 4, "older run")}</div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <EmptyState compact icon="chip" title="No models trained yet">
              A model line appears here after its first training job finishes.
            </EmptyState>
          )}
        </Section>
      </div>
    </PageShell>
  );
}
