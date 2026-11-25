"use client";

import { Suspense, useEffect, useState } from "react";
import { T } from "@/lib/i18n";
import { useRouter } from "next/navigation";
import { api , humanizeError } from "@/lib/api";
import type { AssetRow, BoardCell, LabelJobRow, LabelProjectRow, ScorecardRow, UserRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import SessionSelect from "@/components/shell/SessionSelect";
import LoadState from "@/components/shell/LoadState";
import { useQueryFlag } from "@/lib/useQueryParam";
import { getUser } from "@/lib/user";
import { honeypotVerdict, nextAction } from "@/lib/labelJobActions";
import { invalidate, useQuery } from "@/lib/useQuery";
import EmptyState from "@/components/ui/EmptyState";
import { duration, humanize, pct, shortId } from "@/lib/format";

// The labeling-operations board: who is doing what, where it is stuck, and whether the work is any good.
// stage runs left to right; state is the column within a stage, so "in validation, not yet started" is a
// cell you can actually see rather than an inference.

const STAGES = ["annotation", "validation", "acceptance"] as const;
const STATES = ["new", "in_progress", "completed", "rejected"] as const;

const STATE_TONE: Record<string, string> = {
  new: "text-ink-3",
  in_progress: "text-info",
  completed: "text-pass",
  rejected: "text-block",
};

function Section({ title, desc, children, right }: { title: string; desc?: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="section-title">{title}</h2>
        {right && <span className="ml-auto">{right}</span>}
      </div>
      <div className="p-3.5">
        {desc && <p className="section-desc mb-3">{desc}</p>}
        {children}
      </div>
    </section>
  );
}

export default function ProjectsPage() {
  // useSearchParams (via useQueryParam) forces a CSR bailout Next requires be under Suspense.
  return <Suspense fallback={null}><ProjectsBody /></Suspense>;
}

function ProjectsBody() {
  const router = useRouter();
  // The project list is this page's primary data. Swallowed, a failed fetch left `projects` empty and the
  // board rendered as though a manager had no projects at all - the same shape as the triage page telling
  // an annotator their queue was clear because a request dropped. So its error is rendered (bootErr below).
  const projectsQ = useQuery(["lopProjects"], () => api.lopProjects());
  const projects: LabelProjectRow[] = projectsQ.data?.projects ?? [];
  const bootErr = projectsQ.error ?? null;
  const [projectId, setProjectId] = useState("");
  // Label > My jobs deep-links ?mine=1; filter the board to the signed-in user instead of showing all
  const mineOnly = useQueryFlag("mine");
  // These two fill controls rather than the board: an empty assignee dropdown or gold-set picker is
  // survivable, but it must say why rather than looking like an installation with no users in it.
  const usersQ = useQuery(["users"], () => api.users());
  const users: UserRow[] = usersQ.data ?? [];
  const goldsQ = useQuery(["goldSets"], () => api.goldSets());
  const golds: { gold_id: string; name: string; n_objects: number }[] = goldsQ.data ?? [];
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [showNew, setShowNew] = useState(false);

  // new project form
  const [pName, setPName] = useState("");
  const [hpFrac, setHpFrac] = useState(0.1);
  const [goldId, setGoldId] = useState("");
  // new task form
  const [tName, setTName] = useState("");
  const [tSession, setTSession] = useState("");
  const [jobsOf, setJobsOf] = useState(50);
  // multi-modal assets
  const [newMedia, setNewMedia] = useState("text");
  const [newBody, setNewBody] = useState("");

  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(null), 5000); };

  // The board is five reads that only mean something together, so they stay one all-or-nothing request,
  // keyed by the project and by whether "My jobs" is on. The previous project's board stays up while the
  // next one loads, and after a failed refresh, exactly as the state it replaces did.
  const boardQ = useQuery(
    projectId ? ["lopBoard+lopJobs+lopScorecards+projectAssets+projectStats", projectId, mineOnly] : null,
    () => Promise.all([
      api.lopBoard(projectId),
      // "My jobs" asks the server, which answers across every project. Filtering `lopJobs` by assignee on
      // the client only ever showed the jobs inside whichever project happened to be selected, and showed
      // nothing at all when none was, which is not what the menu entry promises.
      mineOnly ? api.lopMyJobs() : api.lopJobs({ project_id: projectId }),
      api.lopScorecards(projectId), api.projectAssets(projectId), api.projectStats(projectId),
    ]),
    { keepPrevious: true });
  const [b, j, sc, as, st] = boardQ.data ?? [];
  const cells: BoardCell[] = projectId ? b?.cells ?? [] : [];
  const load: { assignee: string; open_jobs: number }[] = projectId ? b?.open_load ?? [] : [];
  const jobs: LabelJobRow[] = projectId ? j?.jobs ?? [] : [];
  const cards: ScorecardRow[] = sc?.scorecards ?? [];
  const assets: { total: number; assets: AssetRow[] } = as ?? { total: 0, assets: [] };
  const annKinds: Record<string, number> = st?.annotations_by_kind ?? {};
  const refreshProject = boardQ.reload;

  // Take a job, and hand it back. `set_state` and `submit_job` are the two calls that make a label job
  // finishable at all: submit is the only writer of `honeypot_accuracy` and the only thing that advances
  // annotation to validation to acceptance.
  const act = async (j: LabelJobRow, kind: "start" | "submit") => {
    setBusy(true);
    try {
      if (kind === "start") {
        await api.lopSetState(j.job_id, "in_progress", j.version);
        flash(`Started job ${shortId(j.job_id)}`);
      } else {
        const r = await api.lopSubmit(j.job_id, j.version);
        flash(r.honeypot_failed
          ? `Sent back: check-question accuracy ${pct(r.honeypot_accuracy ?? 0)} is below the `
            + `project minimum of ${pct(r.min_honeypot_accuracy)}`
          : `Submitted; now ${humanize(r.stage).toLowerCase()}, ${humanize(r.state).toLowerCase()}`);
      }
      await refreshProject();
      // A submit is the only writer of honeypot accuracy, which is what the scorecards page reads.
      void invalidate("lopScorecardsFull");
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  // Open on the first project once the list arrives, unless one is already chosen.
  const firstProjectId = projects[0]?.project_id;
  useEffect(() => {
    if (firstProjectId) setProjectId((cur) => cur || firstProjectId);
  }, [firstProjectId]);

  // Failures that were a flash stay a flash.
  /* eslint-disable react-hooks/exhaustive-deps */
  useEffect(() => { if (boardQ.error) flash(humanizeError(boardQ.error)); }, [boardQ.error]);
  useEffect(() => {
    if (usersQ.error) flash("Could not load the user list, so jobs cannot be assigned");
  }, [usersQ.error]);
  useEffect(() => { if (goldsQ.error) flash("Could not load gold sets"); }, [goldsQ.error]);
  /* eslint-enable react-hooks/exhaustive-deps */

  const createProject = async () => {
    if (!pName.trim()) { flash("Give the project a name"); return; }
    setBusy(true);
    try {
      const p = await api.lopCreateProject({
        name: pName.trim(), honeypot_frac: hpFrac, gold_id: goldId || null,
        min_honeypot_accuracy: 0.9,
      });
      projectsQ.setData((cur) => ({ ...cur, projects: [p, ...(cur?.projects ?? [])] }));
      setProjectId(p.project_id);
      setPName("");
      setShowNew(false);
      flash(`Created "${p.name}"`);
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const createTask = async () => {
    if (!projectId) { flash("Pick a project first"); return; }
    if (!tName.trim()) { flash("Give the task a name"); return; }
    setBusy(true);
    try {
      const t = await api.lopCreateTask({
        project_id: projectId, name: tName.trim(),
        session_id: tSession.trim() || null, jobs_of: jobsOf,
      });
      flash(`${t.n_frames} frames split into ${t.n_jobs} jobs, with ${t.honeypots_seeded} check questions`);
      setTName("");
      await refreshProject();
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const addAsset = async () => {
    if (!projectId || !newBody.trim()) { flash("Add the text, or a link to the file"); return; }
    setBusy(true);
    try {
      const inline = newMedia === "text" || newMedia === "dialogue";
      await api.createAssets(projectId, [inline
        ? { media_type: newMedia, text: newBody.trim() }
        : { media_type: newMedia, uri: newBody.trim() }]);
      setNewBody("");
      await refreshProject();
      flash("Asset added");
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const assign = async (job: LabelJobRow, assignee: string) => {
    try {
      await api.lopAssign(job.job_id, assignee || null, job.version);
      await refreshProject();
    } catch (e) { flash(humanizeError(e)); }
  };

  const countAt = (stage: string, state: string) =>
    cells.find((c) => c.stage === stage && c.state === state)?.count ?? 0;

  const project = projects.find((p) => p.project_id === projectId) ?? null;

  return (
    <PageShell
      active="PROJECTS"
      title="Projects"
      subtitle="Assign labelling work, follow it through each stage, and see whose labels hold up"
      meta={project ? (
        <span className="text-[12px] text-ink-3"
          title="Check questions are frames with a known answer, mixed into jobs to measure accuracy">
          {project.honeypot_frac > 0
            ? `${Math.round(project.honeypot_frac * 100)}% check questions, ${pct(project.min_honeypot_accuracy)} needed to pass`
            : "No check questions set up"}
        </span>
      ) : null}
      primaryAction={
        <button onClick={() => setShowNew((v) => !v)} className="btn btn-primary" aria-expanded={showNew}>New project</button>
      }
      right={msg ? <span role="status" className="text-[12px] text-ink-2">{msg}</span> : null}
      filters={
        <>
          <label htmlFor="project-pick" className="text-ink-3">Project</label>
          <select id="project-pick" value={projectId} onChange={(e) => setProjectId(e.target.value)}
            className="field min-w-[240px]">
            <option value="">None</option>
            {projects.map((p) => <option key={p.project_id} value={p.project_id}>{p.name}</option>)}
          </select>
          {mineOnly && <span className="chip chip-on">Only my jobs</span>}
        </>
      }
    >
      <div className="page-wrap space-y-4">
        {/* Ahead of the selector, because an empty project list and an unreachable one look identical in it. */}
        {bootErr != null && <LoadState error={bootErr} onRetry={() => window.location.reload()} />}

        {showNew && (
          <Section title="New project" desc="Check questions are frames with a known answer mixed into each job; a job whose accuracy on them is too low is sent back.">
            <div className="flex items-end gap-3 text-[12.5px] flex-wrap">
              <label className="flex flex-col gap-1">
                <span className="text-ink-3 text-[12px]">Name</span>
                <input value={pName} onChange={(e) => setPName(e.target.value)} placeholder="Project name"
                  className="field min-w-[240px]" />
              </label>
              <label htmlFor="hp-frac" className="flex flex-col gap-1">
                <span className="text-ink-3 text-[12px]">Share of check questions</span>
                <input id="hp-frac" type="number" step="0.05" min="0" max="0.5" value={hpFrac}
                  onChange={(e) => setHpFrac(Number(e.target.value))}
                  className="field w-24" />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-ink-3 text-[12px]">Answers from gold set</span>
                <select value={goldId} onChange={(e) => setGoldId(e.target.value)} className="field max-w-[220px]">
                  <option value="">No gold set</option>
                  {golds.map((g) => <option key={g.gold_id} value={g.gold_id}>{g.name} ({g.n_objects})</option>)}
                </select>
              </label>
              <button onClick={createProject} disabled={busy} className="btn btn-primary"><T k="action.create">Create</T></button>
              <button onClick={() => setShowNew(false)} className="btn">Cancel</button>
            </div>
          </Section>
        )}

        {!projectsQ.loading && !bootErr && projects.length === 0 && !showNew && (
          <div className="panel">
            <EmptyState icon="clipboard" title="No projects yet"
              action={<button onClick={() => setShowNew(true)} className="btn btn-primary">New project</button>}>
              A project splits sessions into jobs, assigns them to people, and tracks each job through
              annotation, validation and acceptance.
            </EmptyState>
          </div>
        )}

        <div className="grid gap-4 xl:grid-cols-2">
        {/* board */}
        <Section title="Jobs by stage" desc="Each job moves left to right through the stages; the columns show where it is within one.">
          <table className="data-table">
            <thead>
              <tr>
                <th>Stage</th>
                {STATES.map((s) => <th key={s} className="num">{humanize(s)}</th>)}
              </tr>
            </thead>
            <tbody>
              {STAGES.map((stage) => (
                <tr key={stage}>
                  <td className="text-ink-2">{humanize(stage)}</td>
                  {STATES.map((state) => {
                    const n = countAt(stage, state);
                    return (
                      <td key={state} className="num">
                        <span className={n ? STATE_TONE[state] : "text-ink-3/50"}>{n}</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          {load.length > 0 && (
            <div className="mt-3 text-[12.5px] text-ink-3">
              Open jobs per person: {load.map((l) => `${l.assignee} (${l.open_jobs})`).join(", ")}
            </div>
          )}
        </Section>

        {/* task creation */}
        <Section title="New task" desc="Split a session's frames into jobs of a fixed size, ready to assign.">
          <div className="flex items-end gap-3 text-[12.5px] flex-wrap">
            <label className="flex flex-col gap-1">
              <span className="text-ink-3 text-[12px]">Task name</span>
              <input value={tName} onChange={(e) => setTName(e.target.value)} placeholder="e.g. Koramangala evening"
                className="field min-w-[200px]" />
            </label>
            <div className="flex flex-col gap-1">
              <span className="text-ink-3 text-[12px]">Session</span>
              <SessionSelect value={tSession} onChange={setTSession} className="min-w-[280px]" />
            </div>
            <label htmlFor="jobs-of" className="flex flex-col gap-1">
              <span className="text-ink-3 text-[12px]">Frames per job</span>
              <input id="jobs-of" type="number" min="1" value={jobsOf} onChange={(e) => setJobsOf(Number(e.target.value))}
                className="field w-24" />
            </label>
            <button onClick={createTask} disabled={busy || !projectId} className="btn">
              Create task
            </button>
          </div>
        </Section>
        </div>

        {/* jobs */}
        <Section title={`Jobs (${jobs.length})`}>
          {jobs.length === 0 ? (
            <EmptyState compact icon="list" title={projectId ? "No jobs yet" : "Pick a project"}>
              {projectId ? "Create a task above to split a session into jobs." : "Choose a project at the top to see its jobs."}
            </EmptyState>
          ) : (
            <div className="overflow-auto max-h-[28rem] -mx-3.5 -mb-3.5">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Job</th><th>Stage</th><th>State</th>
                    <th className="num">Frames</th>
                    <th className="num">Check questions</th><th className="num">Accuracy</th>
                    <th>Assigned to</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((j) => (
                    <tr key={j.job_id}>
                      <td className="font-mono text-[12px] text-ink-3" title={j.job_id}>{shortId(j.job_id)}</td>
                      <td className="text-ink-2">{humanize(j.stage)}</td>
                      <td><span className={STATE_TONE[j.state] ?? "text-ink-3"}>{humanize(j.state)}</span></td>
                      <td className="num">{j.n_frames}</td>
                      <td className="num">{j.n_honeypots || "-"}</td>
                      {/* Unmeasured is not failing. Until submit existed every job in the system was
                          unmeasured, and colouring those red would be a board that means nothing. */}
                      <td className="num"
                        title={j.honeypot_accuracy == null ? "Not scored yet; a job is scored when it is submitted" : ""}>
                        <span className={{ unmeasured: "text-ink-3", pass: "text-pass", fail: "text-block" }[
                          honeypotVerdict(j.honeypot_accuracy, project?.min_honeypot_accuracy)]}>
                          {j.honeypot_accuracy == null ? "-" : pct(j.honeypot_accuracy)}
                        </span>
                      </td>
                      <td>
                        <select value={j.assignee_id ?? ""} onChange={(e) => assign(j, e.target.value)}
                          aria-label={`Assign job ${shortId(j.job_id)}`}
                          className="field text-[12px] max-w-[160px]">
                          <option value="">Unassigned</option>
                          {users.map((u) => <option key={u.user_id} value={u.user_id}>{u.name}</option>)}
                        </select>
                      </td>
                      <td className="text-right whitespace-nowrap">
                        {j.frame_ids[0] && (
                          <button onClick={() => router.push(`/frame/${j.frame_ids[0]}?job=${j.job_id}`)}
                            title="Open the first frame of this job"
                            className="btn"><T k="action.open">Open</T></button>
                        )}
                        {(() => {
                          const a = nextAction(j, getUser()?.user_id ?? null);
                          return a ? (
                            <button onClick={() => act(j, a.kind)} disabled={busy} title={a.hint}
                              className="btn ml-2">
                              {humanize(a.label)}
                            </button>
                          ) : null;
                        })()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Section>

        {/* multi-modal assets */}
        <Section title={`Other assets (${assets.total})`}
          desc="Text, audio, documents and other non-camera items labelled in this project."
          right={<span className="text-[12px] text-ink-3">
            {Object.entries(annKinds).map(([k, v]) => `${humanize(k)} ${v}`).join(" · ") || "No annotations yet"}
          </span>}>
          <div className="flex items-center gap-2 text-[12.5px] flex-wrap mb-3">
            <select value={newMedia} onChange={(e) => setNewMedia(e.target.value)} aria-label="Asset type"
              className="field">
              {["text", "dialogue", "audio", "timeseries", "document", "image"].map((m) =>
                <option key={m} value={m}>{m === "timeseries" ? "Time series" : humanize(m)}</option>)}
            </select>
            <input value={newBody} onChange={(e) => setNewBody(e.target.value)}
              aria-label={newMedia === "text" || newMedia === "dialogue" ? "Text" : "Link to the file"}
              placeholder={newMedia === "text" || newMedia === "dialogue" ? "The text to label" : "Link to the file"}
              className="field flex-1 min-w-[240px]" />
            <button onClick={addAsset} disabled={busy || !projectId} className="btn">
              Add asset
            </button>
          </div>
          {assets.assets.length === 0 ? (
            <>
              <EmptyState compact icon="folder" title="No assets in this project">
                Add one above. Large batches can be imported in one go.
              </EmptyState>
              <details className="text-[12px] text-ink-3">
                <summary className="cursor-pointer hover:text-ink-2">For administrators</summary>
                <p className="mt-1.5">Bulk import: POST a list of assets to <code className="font-mono">/api/assets</code>.</p>
              </details>
            </>
          ) : (
            <div className="overflow-auto max-h-72 -mx-3.5 -mb-3.5">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Asset</th><th>Type</th><th>State</th>
                    <th>Preview</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {assets.assets.map((a) => (
                    <tr key={a.asset_id}>
                      <td className="text-ink-2" title={a.asset_id}>{a.external_id ?? <span className="font-mono text-[12px] text-ink-3">{shortId(a.asset_id)}</span>}</td>
                      <td className="text-ink-2">{humanize(a.media_type)}</td>
                      <td><span className={a.state === "labeled" ? "text-pass" : "text-ink-3"}>{humanize(a.state)}</span></td>
                      <td className="text-ink-3 truncate max-w-[320px]">{a.text ?? a.uri ?? "-"}</td>
                      <td className="text-right">
                        <button onClick={() => router.push(`/annotate/asset/${a.asset_id}`)} className="btn">Annotate</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Section>

        {/* scorecards */}
        <Section title="Annotator scorecards" desc="Reviews done, time per review, and accuracy on check questions, per person.">
          {cards.length === 0 ? (
            <EmptyState compact icon="user" title="No review activity yet">
              Scorecards fill in as people review and submit jobs.
            </EmptyState>
          ) : (
            <div className="overflow-auto -mx-3.5 -mb-3.5">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Person</th><th>Role</th>
                  <th className="num">Reviews</th>
                  <th className="num">Median per review</th>
                  <th className="num">Mean per review</th>
                  <th className="num">Total time</th>
                  <th className="num">Jobs</th>
                  <th className="num">Check accuracy</th>
                </tr>
              </thead>
              <tbody>
                {cards.map((c) => (
                  <tr key={c.user_id}>
                    <td><span className="text-ink">{c.name}</span></td>
                    <td className="text-ink-3">{humanize(c.role)}</td>
                    <td className="num"><span className={c.reviews ? "text-ink" : "text-ink-3"}>{c.reviews.toLocaleString()}</span></td>
                    {/* No reviews, or no time recorded on them, is not "0.0 s" of work. */}
                    <td className="num">{c.reviews && c.median_time_ms ? duration(c.median_time_ms / 1000) : "-"}</td>
                    <td className="num">{c.reviews && c.mean_time_ms ? duration(c.mean_time_ms / 1000) : "-"}</td>
                    <td className="num">{c.total_time_min ? duration(c.total_time_min * 60) : "-"}</td>
                    <td className="num">{c.jobs || "-"}</td>
                    <td className="num">
                      <span className={c.honeypot_accuracy == null ? "text-ink-3"
                        : c.honeypot_accuracy >= 0.9 ? "text-pass" : "text-block"}>
                        {c.honeypot_accuracy == null ? "-" : pct(c.honeypot_accuracy)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          )}
        </Section>
      </div>
    </PageShell>
  );
}
