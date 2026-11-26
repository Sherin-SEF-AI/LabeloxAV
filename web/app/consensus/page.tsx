"use client";

// Consensus labelling (tool #11). Several annotators label the same frames blind; the model's own labels can vote
// too. Where a quorum of voters drew the same thing with the same class, one consensus label is written
// (source "consensus", never editing anyone's own label); where they differ, an issue opens on each disputed box
// and lands in the queue below, which opens the frame.
//
// Server: services/labelops/agreement.py (merge_replica_group, consensus_board, task_disagreements) through
// services/api/routers/tools_quality.py.

import Link from "next/link";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import EvidenceImage from "@/components/quality/EvidenceImage";
import VlmReason from "@/components/quality/VlmReason";
import { Badge, Button, Panel, Stat } from "@/components/ui/kit";
import { humanizeError } from "@/lib/api";
import { ago, humanize, num, pct, shortId } from "@/lib/format";
import { toast } from "@/lib/toast";
import { useCurrentUser } from "@/lib/user";
import { classTotals, heatStyle, mergeSummary } from "@/lib/tools/quality";
import {
  DEFAULT_MERGE, MAX_REPLICAS, createReplicatedTask, getConsensusBoard, listConsensusTasks, listDisagreements,
  listProjects, previewMerge, runMerge,
  type ConsensusBoard, type ConsensusTask, type Disagreement, type MergeParams, type MergePreview,
} from "@/lib/tools/qualityApi";
import { useToolJob } from "@/lib/tools/toolJobs";

const rank = (role?: string) => (role === "admin" ? 3 : role === "reviewer" ? 2 : role === "annotator" ? 1 : 0);

export default function ConsensusPage() {
  return (
    <Suspense fallback={<PageShell active="Consensus" title="Consensus"><div className="page-wrap text-[13px] text-ink-3" role="status">Loading...</div></PageShell>}>
      <Consensus />
    </Suspense>
  );
}

function Consensus() {
  const me = useCurrentUser();
  const canWrite = rank(me?.role) >= 2;
  const params = useSearchParams();
  const router = useRouter();
  const taskId = params.get("task");

  const [tasks, setTasks] = useState<ConsensusTask[] | null>(null);
  const [tasksErr, setTasksErr] = useState<unknown>(null);
  const [reloadTasks, setReloadTasks] = useState(0);
  useEffect(() => {
    let stop = false;
    setTasksErr(null);
    listConsensusTasks().then((r) => { if (!stop) setTasks(r.tasks); }).catch((e) => { if (!stop) setTasksErr(e); });
    return () => { stop = true; };
  }, [reloadTasks]);

  // Open the newest task with something to merge when none is chosen.
  useEffect(() => {
    if (!taskId && tasks && tasks.length) {
      const pick = tasks.find((t) => t.ready_groups > 0) ?? tasks[0];
      router.replace(`/consensus?task=${pick.task_id}`);
    }
  }, [taskId, tasks, router]);

  return (
    <PageShell active="Consensus" title="Consensus"
      subtitle="Where independent annotators (and the model) agree, one label is written; where they differ, a person decides">
      <div className="page-wrap space-y-4">
        <Panel title="Tasks labelled by more than one person"
          desc="Each chunk of frames went to several annotators blind. A group can be merged once every replica is submitted."
          right={canWrite ? <NewTaskButton onMade={(id) => { setReloadTasks((n) => n + 1); router.push(`/consensus?task=${id}`); }} /> : null}>
          <LoadState loading={tasks === null && !tasksErr} error={tasksErr} empty={tasks !== null && tasks.length === 0}
            emptyTitle="No replicated tasks yet"
            emptyText="Send frames to two or more annotators (New replicated task) to measure agreement and merge it."
            onRetry={() => setReloadTasks((n) => n + 1)}>
            <div className="overflow-x-auto">
              <table className="data-table w-full">
                <thead><tr><th>Task</th><th>Project</th><th>Replicas</th><th>Groups ready</th><th>Consensus labels</th><th>Created</th></tr></thead>
                <tbody>
                  {(tasks ?? []).map((t) => (
                    <tr key={t.task_id} className={t.task_id === taskId ? "bg-head" : "cursor-pointer hover:bg-head/60"}
                      onClick={() => router.push(`/consensus?task=${t.task_id}`)}>
                      <td><Link href={`/consensus?task=${t.task_id}`} className="link">{t.name}</Link></td>
                      <td className="text-ink-2">{t.project}</td>
                      <td className="tabular-nums">{t.replicas}</td>
                      <td className="tabular-nums">{t.ready_groups} of {t.n_groups}</td>
                      <td className="tabular-nums">{num(t.consensus_objects)}</td>
                      <td className="text-ink-3">{ago(t.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </LoadState>
        </Panel>
        {taskId && <TaskBoard key={taskId} taskId={taskId} canWrite={canWrite} onMerged={() => setReloadTasks((n) => n + 1)} />}
      </div>
    </PageShell>
  );
}

function NewTaskButton({ onMade }: { onMade: (taskId: string) => void }) {
  const [open, setOpen] = useState(false);
  const [projects, setProjects] = useState<{ project_id: string; name: string }[]>([]);
  const [f, setF] = useState({ project_id: "", name: "", session_id: "", replicas: 2, jobs_of: 25 });
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!open || projects.length) return;
    listProjects().then((r) => {
      setProjects(r.projects);
      setF((s) => ({ ...s, project_id: s.project_id || r.projects[0]?.project_id || "" }));
    }).catch((e) => toast(humanizeError(e), "error"));
  }, [open, projects.length]);
  const make = async () => {
    setBusy(true);
    try {
      const r = await createReplicatedTask({ project_id: f.project_id, name: f.name.trim(), replicas: f.replicas,
        jobs_of: f.jobs_of, session_id: f.session_id.trim() || undefined });
      toast(`${r.n_frames} frames in ${r.n_jobs} jobs, each frame to ${r.replicas} annotators`, "success");
      setOpen(false);
      onMade(r.task_id);
    } catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };
  if (!open) return <Button onClick={() => setOpen(true)} tip="Send a session's frames to several annotators blind">New replicated task</Button>;
  return (
    <div className="flex flex-wrap items-end gap-2 text-[12px]">
      <label className="flex flex-col gap-0.5"><span className="text-ink-3">Project</span>
        <select className="field" value={f.project_id} onChange={(e) => setF({ ...f, project_id: e.target.value })}>
          {projects.map((p) => <option key={p.project_id} value={p.project_id}>{p.name}</option>)}
        </select></label>
      <label className="flex flex-col gap-0.5"><span className="text-ink-3">Name</span>
        <input className="field w-40" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} /></label>
      <label className="flex flex-col gap-0.5"><span className="text-ink-3">Session id</span>
        <input className="field w-72 font-mono" value={f.session_id} placeholder="the frames to label"
          onChange={(e) => setF({ ...f, session_id: e.target.value })} /></label>
      <label className="flex flex-col gap-0.5"><span className="text-ink-3">Annotators per frame</span>
        <input type="number" min={2} max={MAX_REPLICAS} className="field w-20" value={f.replicas}
          onChange={(e) => setF({ ...f, replicas: Math.max(2, Math.min(MAX_REPLICAS, Number(e.target.value) || 2)) })} /></label>
      <label className="flex flex-col gap-0.5"><span className="text-ink-3">Frames per job</span>
        <input type="number" min={1} max={500} className="field w-20" value={f.jobs_of}
          onChange={(e) => setF({ ...f, jobs_of: Math.max(1, Number(e.target.value) || 25) })} /></label>
      <Button variant="primary" onClick={make} disabled={busy || !f.project_id || !f.name.trim() || !f.session_id.trim()}>
        {busy ? "Creating..." : "Create"}</Button>
      <Button variant="ghost" onClick={() => setOpen(false)}>Cancel</Button>
    </div>
  );
}

function Settings({ p, onChange }: { p: MergeParams; onChange: (p: MergeParams) => void }) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[12px]">
      <label className="flex items-center gap-2" data-tip="How much two boxes must overlap to count as the same object">
        <span className="text-ink-2">Overlap (IoU)</span>
        <input type="range" min={0.3} max={0.9} step={0.05} value={p.iou_thresh}
          onChange={(e) => onChange({ ...p, iou_thresh: Number(e.target.value) })} aria-label="IoU threshold" />
        <span className="tabular-nums w-8">{p.iou_thresh.toFixed(2)}</span>
      </label>
      <label className="flex items-center gap-2" data-tip="How many voters must draw it with the same class">
        <span className="text-ink-2">Quorum</span>
        <input type="number" min={2} max={12} className="field w-16" value={p.quorum}
          onChange={(e) => onChange({ ...p, quorum: Math.max(2, Number(e.target.value) || 2) })} />
      </label>
      <label className="flex items-center gap-1.5" data-tip="The frame's machine labels count as one more voter">
        <input type="checkbox" checked={p.include_model} onChange={(e) => onChange({ ...p, include_model: e.target.checked })} />
        <span className="text-ink-2">Model votes</span>
      </label>
      <label className="flex items-center gap-1.5" data-tip="Accept only when at least one person is in the agreeing set">
        <input type="checkbox" checked={p.require_human} onChange={(e) => onChange({ ...p, require_human: e.target.checked })} />
        <span className="text-ink-2">A person must agree</span>
      </label>
    </div>
  );
}

function TaskBoard({ taskId, canWrite, onMerged }: { taskId: string; canWrite: boolean; onMerged: () => void }) {
  const [p, setP] = useState<MergeParams>(DEFAULT_MERGE);
  const [board, setBoard] = useState<ConsensusBoard | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [reload, setReload] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [preview, setPreview] = useState<MergePreview | null>(null);

  useEffect(() => {
    let stop = false;
    setLoading(true); setErr(null);
    const t = setTimeout(() => {
      getConsensusBoard(taskId, p)
        .then((b) => { if (!stop) setBoard(b); })
        .catch((e) => { if (!stop) setErr(e); })
        .finally(() => { if (!stop) setLoading(false); });
    }, 250);
    return () => { stop = true; clearTimeout(t); };
  }, [taskId, p, reload]);

  const { job } = useToolJob(jobId, {
    onDone: (j) => {
      if (j.status === "done") toast(`Merged: ${mergeSummary((j.result as { counts?: Record<string, number> })?.counts)}`, "success");
      else toast(j.error ?? `Merge ${j.status}`, "error");
      setJobId(null);
      setReload((n) => n + 1);
      onMerged();
    },
  });

  const merge = async (target: { task_id?: string; replica_group?: string }) => {
    try { setJobId((await runMerge(target, p)).job_id); } catch (e) { toast(humanizeError(e), "error"); }
  };
  const doPreview = async (group: string) => {
    try { setPreview(await previewMerge(group, p)); } catch (e) { toast(humanizeError(e), "error"); }
  };

  const totals = useMemo(() => (board ? classTotals(board.heatmap.cells) : []), [board]);
  const ready = board?.groups.filter((g) => g.ready).length ?? 0;
  const sum = (k: string) => board?.groups.reduce((a, g) => a + (g.counts?.[k] ?? 0), 0) ?? 0;

  return (
    <>
      <Panel title={board ? `Agreement: ${board.name}` : "Agreement"}
        desc="Per class and per group of frames: the share of objects a merge with these settings would accept. Tint strength grows with the number of objects behind a cell."
        right={canWrite && board ? (
          <Button variant="primary" disabled={!ready || !!jobId} onClick={() => merge({ task_id: taskId })}
            tip="Score every ready group, write consensus labels where a quorum agrees, open issues where not">
            {jobId ? `Merging ${pct(job?.progress ?? 0)}` : `Merge ${ready} ready group${ready === 1 ? "" : "s"}`}
          </Button>
        ) : null}>
        <div className="space-y-3">
          <Settings p={p} onChange={setP} />
          <LoadState loading={loading && !board} error={err} onRetry={() => setReload((n) => n + 1)}>
            {board && (
              <>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                  <Stat label="Groups ready" value={`${ready} / ${board.groups.length}`} />
                  <Stat label="Would agree" value={num(sum("accepted"))} tone="pass" />
                  <Stat label="Would go to review" value={num(sum("disputed"))} tone={sum("disputed") ? "warn" : undefined} />
                  <Stat label="Model-only, dropped" value={num(sum("dropped"))} />
                  <Stat label="Consensus labels written" value={num(board.groups.reduce((a, g) => a + g.consensus_objects, 0))} />
                </div>
                {board.heatmap.classes.length === 0 ? (
                  <p className="text-[12.5px] text-ink-3">
                    {ready ? "The submitted replicas hold no labels on their shared frames." :
                      "No group is ready: every replica of a group must be submitted before its agreement is measured."}
                  </p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Class</th>
                          {board.groups.map((g, i) => (
                            <th key={g.replica_group} className="text-center"
                              title={g.jobs.map((j) => `replica ${j.replica_index}: ${j.state}`).join(", ")}>
                              G{i + 1}
                            </th>
                          ))}
                          <th className="text-center">All</th>
                        </tr>
                      </thead>
                      <tbody>
                        {board.heatmap.classes.map((c, ri) => (
                          <tr key={c}>
                            <td className="text-ink">{humanize(c)}</td>
                            {board.heatmap.cells[ri].map((cell, ci) => (
                              <td key={ci} className="text-center tabular-nums text-[12px]" style={{ background: heatStyle(cell).background }}
                                title={cell ? `${cell.accepted} agreed, ${cell.disputed} disputed, ${cell.dropped} model-only` : "none"}>
                                {cell ? (cell.rate == null ? `${cell.dropped} dropped` : `${cell.accepted}/${cell.accepted + cell.disputed}`) : ""}
                              </td>
                            ))}
                            <td className="text-center tabular-nums text-[12px] font-medium" style={{ background: heatStyle(totals[ri]).background }}>
                              {totals[ri].rate == null ? "-" : pct(totals[ri].rate)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <div className="overflow-x-auto">
                  <table className="data-table w-full">
                    <thead><tr><th>Group</th><th>Replicas</th><th>Pairwise detection agreement</th><th>Merge preview</th><th /></tr></thead>
                    <tbody>
                      {board.groups.map((g, i) => (
                        <tr key={g.replica_group}>
                          <td className="font-mono text-[12px]">G{i + 1} · {shortId(g.replica_group)}</td>
                          <td className="text-[12px]">
                            {g.jobs.map((j) => (
                              <Badge key={j.job_id} state={["submitted", "completed", "accepted"].includes(j.state) ? "pass" : "muted"}>
                                r{j.replica_index} {j.state}
                              </Badge>
                            ))}
                          </td>
                          <td className="tabular-nums">{g.detection_agreement == null ? "not scored" : pct(g.detection_agreement)}</td>
                          <td className="text-[12px] text-ink-2">{g.ready ? mergeSummary(g.counts) : "waiting for replicas"}</td>
                          <td className="text-right whitespace-nowrap">
                            {g.ready && <Button variant="ghost" onClick={() => doPreview(g.replica_group)}>Preview</Button>}
                            {g.ready && canWrite && <Button variant="ghost" disabled={!!jobId}
                              onClick={() => merge({ replica_group: g.replica_group })}>Merge</Button>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </LoadState>
        </div>
      </Panel>
      {preview && <PreviewPanel preview={preview} onClose={() => setPreview(null)} />}
      <DisagreementQueue taskId={taskId} reloadKey={reload} />
    </>
  );
}

function PreviewPanel({ preview, onClose }: { preview: MergePreview; onClose: () => void }) {
  const clusters = preview.frames.flatMap((f) => f.clusters.map((c) => ({ ...c, frame_id: f.frame_id })));
  const order = { disputed: 0, accepted: 1, dropped: 2 } as const;
  clusters.sort((a, b) => order[a.status] - order[b.status]);
  return (
    <Panel title={`Preview: group ${shortId(preview.replica_group)}`}
      desc={`Voters: ${preview.voters.join(", ")}. ${mergeSummary(preview.counts)}. Nothing has been written.`}
      right={<Button variant="ghost" onClick={onClose}>Close</Button>}>
      <div className="overflow-x-auto max-h-[420px]">
        <table className="data-table w-full">
          <thead><tr><th>Outcome</th><th>Class votes</th><th>Why</th><th>Frame</th></tr></thead>
          <tbody>
            {clusters.slice(0, 300).map((c, i) => (
              <tr key={i}>
                <td><Badge state={c.status === "accepted" ? "pass" : c.status === "disputed" ? "warn" : "muted"}>{c.status}</Badge></td>
                <td className="text-[12px]">{Object.entries(c.class_votes).map(([k, n]) => `${humanize(k)} ${n}`).join(", ")}</td>
                <td className="text-[12px] text-ink-2">{c.reason}</td>
                <td><Link className="link text-[12px]" href={`/frame/${c.frame_id}`}>open</Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function DisagreementQueue({ taskId, reloadKey }: { taskId: string; reloadKey: number }) {
  const [status, setStatus] = useState<"open" | "resolved">("open");
  const [items, setItems] = useState<Disagreement[] | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [sel, setSel] = useState<string | null>(null);
  const load = useCallback(() => {
    setErr(null);
    listDisagreements(taskId, status).then((r) => setItems(r.items)).catch((e) => setErr(e));
  }, [taskId, status]);
  useEffect(() => { setItems(null); load(); }, [load, reloadKey]);
  const cur = items?.find((d) => d.issue_id === sel) ?? null;

  return (
    <Panel title="Disagreements" desc="Each disputed box, with the reason the merge gave. Open the frame to settle it; resolving the issue there clears it here."
      right={
        <div className="flex gap-1">
          {(["open", "resolved"] as const).map((s) => (
            <button key={s} type="button" className={`chip ${status === s ? "chip-on" : ""}`} onClick={() => setStatus(s)}>{humanize(s)}</button>
          ))}
        </div>
      }>
      <LoadState loading={items === null && !err} error={err} empty={items !== null && items.length === 0}
        emptyTitle={status === "open" ? "Nothing in dispute" : "Nothing resolved yet"}
        emptyText={status === "open" ? "Every disputed box has been settled, or no group has been merged yet." : "Resolved disagreements appear here."}
        onRetry={load}>
        <div className="flex flex-col lg:flex-row gap-3 items-start">
          <div className="overflow-x-auto flex-1 max-h-[520px]">
            <table className="data-table w-full">
              <thead><tr><th>Class</th><th>Drawn by</th><th>Reason</th><th>Frame</th></tr></thead>
              <tbody>
                {(items ?? []).map((d) => (
                  <tr key={d.issue_id} onClick={() => setSel(d.issue_id)}
                    className={d.issue_id === sel ? "bg-head" : "cursor-pointer hover:bg-head/60"}>
                    <td>{d.class_name ? humanize(d.class_name) : "-"}</td>
                    <td className="text-[12px] text-ink-2">{d.replica_index != null ? `annotator ${d.replica_index + 1}` : d.source === "fused" ? "model" : (d.source ?? "-")}</td>
                    <td className="text-[12px] text-ink-2">{d.reason.replace(/^annotators disagree: /, "")}</td>
                    <td><Link className="link text-[12px]" href={`/frame/${d.frame_id}${d.job_id ? `?job=${d.job_id}` : ""}`}
                      onClick={(e) => e.stopPropagation()}>Open frame</Link></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {cur && cur.bbox && cur.bbox.length >= 4 && (
            <div className="w-full lg:w-[380px] shrink-0 space-y-2">
              <FrameBox frameId={cur.frame_id} bbox={cur.bbox} label={cur.class_name ?? ""} />
              {cur.object_id && <VlmReason objectId={cur.object_id} />}
            </div>
          )}
        </div>
      </LoadState>
    </Panel>
  );
}

/** The disputed box on its frame. The frame size is read from the image itself, so no extra request is needed. */
function FrameBox({ frameId, bbox, label }: { frameId: string; bbox: number[]; label: string }) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const src = `/api/frames/${frameId}/image`;
  useEffect(() => {
    setSize(null);
    const img = new Image();
    img.onload = () => setSize({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = src;
  }, [src]);
  if (!size) return <div className="h-[200px] rounded border hairline bg-bg-2 grid place-items-center text-[12px] text-ink-3">Loading the frame...</div>;
  return <EvidenceImage src={src} width={size.w} height={size.h} maxW={380} maxH={260}
    boxes={[{ bbox, tone: "warn", label: humanize(label) }]} alt={`Disputed ${humanize(label)}`} />;
}
