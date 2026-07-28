"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api , humanizeError } from "@/lib/api";
import type { AssetRow, BoardCell, LabelJobRow, LabelProjectRow, ScorecardRow, UserRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import { useQueryFlag } from "@/lib/useQueryParam";
import { getUser } from "@/lib/user";

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

function Section({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="flex items-center gap-2 font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
        <span>{title}</span>
        {right && <span className="ml-auto normal-case">{right}</span>}
      </div>
      <div className="p-3">{children}</div>
    </section>
  );
}

export default function ProjectsPage() {
  // useSearchParams (via useQueryParam) forces a CSR bailout Next requires be under Suspense.
  return <Suspense fallback={null}><ProjectsBody /></Suspense>;
}

function ProjectsBody() {
  const router = useRouter();
  const [projects, setProjects] = useState<LabelProjectRow[]>([]);
  const [projectId, setProjectId] = useState("");
  const [cells, setCells] = useState<BoardCell[]>([]);
  const [load, setLoad] = useState<{ assignee: string; open_jobs: number }[]>([]);
  const [jobs, setJobs] = useState<LabelJobRow[]>([]);
  // Label > My jobs deep-links ?mine=1; filter the board to the signed-in user instead of showing all
  const mineOnly = useQueryFlag("mine");
  const [users, setUsers] = useState<UserRow[]>([]);
  const [cards, setCards] = useState<ScorecardRow[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // new project form
  const [pName, setPName] = useState("");
  const [hpFrac, setHpFrac] = useState(0.1);
  const [goldId, setGoldId] = useState("");
  const [golds, setGolds] = useState<{ gold_id: string; name: string; n_objects: number }[]>([]);
  // new task form
  const [tName, setTName] = useState("");
  const [tSession, setTSession] = useState("");
  const [jobsOf, setJobsOf] = useState(50);
  // multi-modal assets
  const [assets, setAssets] = useState<{ total: number; assets: AssetRow[] }>({ total: 0, assets: [] });
  const [annKinds, setAnnKinds] = useState<Record<string, number>>({});
  const [newMedia, setNewMedia] = useState("text");
  const [newBody, setNewBody] = useState("");

  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(null), 5000); };

  const refreshProject = useCallback(async (pid: string) => {
    if (!pid) { setCells([]); setJobs([]); setLoad([]); return; }
    try {
      const [b, j, s, as, st] = await Promise.all([
        api.lopBoard(pid), api.lopJobs({ project_id: pid }), api.lopScorecards(pid),
        api.projectAssets(pid), api.projectStats(pid),
      ]);
      setCells(b.cells); setLoad(b.open_load); setCards(s.scorecards);
      // ?mine=1 (Label > My jobs) narrows the board to the signed-in user's assignments
      const me = getUser()?.user_id;
      setJobs(mineOnly && me ? j.jobs.filter((x) => x.assignee_id === me) : j.jobs);
      setAssets(as); setAnnKinds(st.annotations_by_kind ?? {});
    } catch (e) { flash(humanizeError(e)); }
  }, [mineOnly]);

  useEffect(() => {
    api.lopProjects().then((r) => {
      setProjects(r.projects);
      if (r.projects.length && !projectId) setProjectId(r.projects[0].project_id);
    }).catch(() => {});
    api.users().then(setUsers).catch(() => {});
    api.goldSets().then((g) => setGolds(g)).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { refreshProject(projectId); }, [projectId, refreshProject]);

  const createProject = async () => {
    if (!pName.trim()) { flash("name the project"); return; }
    setBusy(true);
    try {
      const p = await api.lopCreateProject({
        name: pName.trim(), honeypot_frac: hpFrac, gold_id: goldId || null,
        min_honeypot_accuracy: 0.9,
      });
      setProjects((ps) => [p, ...ps]);
      setProjectId(p.project_id);
      setPName("");
      flash(`created "${p.name}"`);
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const createTask = async () => {
    if (!projectId) { flash("pick a project"); return; }
    if (!tName.trim()) { flash("name the task"); return; }
    setBusy(true);
    try {
      const t = await api.lopCreateTask({
        project_id: projectId, name: tName.trim(),
        session_id: tSession.trim() || null, jobs_of: jobsOf,
      });
      flash(`${t.n_frames} frames split into ${t.n_jobs} jobs, ${t.honeypots_seeded} honeypots seeded`);
      setTName("");
      await refreshProject(projectId);
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const addAsset = async () => {
    if (!projectId || !newBody.trim()) { flash("give the asset a body or uri"); return; }
    setBusy(true);
    try {
      const inline = newMedia === "text" || newMedia === "dialogue";
      await api.createAssets(projectId, [inline
        ? { media_type: newMedia, text: newBody.trim() }
        : { media_type: newMedia, uri: newBody.trim() }]);
      setNewBody("");
      await refreshProject(projectId);
      flash("asset added");
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const assign = async (job: LabelJobRow, assignee: string) => {
    try {
      await api.lopAssign(job.job_id, assignee || null, job.version);
      await refreshProject(projectId);
    } catch (e) { flash(humanizeError(e)); }
  };

  const countAt = (stage: string, state: string) =>
    cells.find((c) => c.stage === stage && c.state === state)?.count ?? 0;

  const project = projects.find((p) => p.project_id === projectId) ?? null;

  return (
    <PageShell
      active="PROJECTS"
      title="Projects"
      subtitle="assign work, watch it move through the stages, and measure who is producing good labels"
      meta={project ? (
        <span className="font-mono text-[11px] text-ink-3">
          {project.honeypot_frac > 0
            ? `${Math.round(project.honeypot_frac * 100)}% honeypots, floor ${project.min_honeypot_accuracy}`
            : "no honeypot QA configured"}
        </span>
      ) : null}
      right={msg ? <span className="font-mono text-[11px] text-accent">{msg}</span> : null}
    >
      <div className="p-4 space-y-4">
        <div className="flex items-center gap-2 font-mono text-[11px] flex-wrap">
          <span className="text-ink-3">project</span>
          <select value={projectId} onChange={(e) => setProjectId(e.target.value)}
            className="bg-bg border border-line px-1.5 py-0.5 text-ink min-w-[200px]">
            <option value="">none</option>
            {projects.map((p) => <option key={p.project_id} value={p.project_id}>{p.name}</option>)}
          </select>
          <span className="text-ink-3 ml-3">new:</span>
          <input value={pName} onChange={(e) => setPName(e.target.value)} placeholder="project name"
            className="bg-bg border border-line px-1.5 py-0.5 text-ink" />
          <label className="text-ink-3">honeypot</label>
          <input type="number" step="0.05" min="0" max="0.5" value={hpFrac}
            onChange={(e) => setHpFrac(Number(e.target.value))}
            className="bg-bg border border-line px-1.5 py-0.5 text-ink w-16" />
          <select value={goldId} onChange={(e) => setGoldId(e.target.value)}
            className="bg-bg border border-line px-1.5 py-0.5 text-ink max-w-[180px]">
            <option value="">no gold set</option>
            {golds.map((g) => <option key={g.gold_id} value={g.gold_id}>{g.name} ({g.n_objects})</option>)}
          </select>
          <button onClick={createProject} disabled={busy}
            className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">create</button>
        </div>

        {/* board */}
        <Section title="board - jobs by stage and state">
          <table className="font-mono text-[11px] w-full max-w-2xl">
            <thead>
              <tr className="text-ink-3 text-left border-b hairline">
                <th className="py-1">stage</th>
                {STATES.map((s) => <th key={s} className="text-right px-2">{s.replace("_", " ")}</th>)}
              </tr>
            </thead>
            <tbody>
              {STAGES.map((stage) => (
                <tr key={stage} className="border-b hairline">
                  <td className="py-1 text-ink-2">{stage}</td>
                  {STATES.map((state) => {
                    const n = countAt(stage, state);
                    return (
                      <td key={state} className={`text-right px-2 ${n ? STATE_TONE[state] : "text-ink-3/40"}`}>
                        {n}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          {load.length > 0 && (
            <div className="mt-3 font-mono text-[11px] text-ink-3">
              open load: {load.map((l) => `${l.assignee} (${l.open_jobs})`).join(", ")}
            </div>
          )}
        </Section>

        {/* task creation */}
        <Section title="new task - split a session into jobs">
          <div className="flex items-center gap-2 font-mono text-[11px] flex-wrap">
            <input value={tName} onChange={(e) => setTName(e.target.value)} placeholder="task name"
              className="bg-bg border border-line px-1.5 py-0.5 text-ink" />
            <input value={tSession} onChange={(e) => setTSession(e.target.value)} placeholder="session id"
              className="bg-bg border border-line px-1.5 py-0.5 text-ink min-w-[280px]" />
            <label className="text-ink-3">frames per job</label>
            <input type="number" min="1" value={jobsOf} onChange={(e) => setJobsOf(Number(e.target.value))}
              className="bg-bg border border-line px-1.5 py-0.5 text-ink w-20" />
            <button onClick={createTask} disabled={busy || !projectId}
              className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
              create task
            </button>
          </div>
        </Section>

        {/* jobs */}
        <Section title={`jobs (${jobs.length})`}>
          {jobs.length === 0 ? (
            <div className="font-mono text-[11px] text-ink-3 py-3">no jobs yet. create a task above.</div>
          ) : (
            <div className="overflow-auto max-h-96">
              <table className="w-full font-mono text-[11px]">
                <thead>
                  <tr className="text-ink-3 text-left border-b hairline">
                    <th className="py-1">job</th><th>stage</th><th>state</th><th>frames</th>
                    <th>honeypots</th><th>accuracy</th><th>assignee</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((j) => (
                    <tr key={j.job_id} className="border-b hairline">
                      <td className="py-1 text-ink-3">{j.job_id.slice(0, 8)}</td>
                      <td className="text-ink-2">{j.stage}</td>
                      <td className={STATE_TONE[j.state] ?? "text-ink-3"}>{j.state}</td>
                      <td className="text-ink-3">{j.n_frames}</td>
                      <td className="text-ink-3">{j.n_honeypots || "-"}</td>
                      <td className={j.honeypot_accuracy == null ? "text-ink-3"
                        : j.honeypot_accuracy >= (project?.min_honeypot_accuracy ?? 0.9) ? "text-pass" : "text-block"}>
                        {j.honeypot_accuracy == null ? "-" : j.honeypot_accuracy.toFixed(2)}
                      </td>
                      <td>
                        <select value={j.assignee_id ?? ""} onChange={(e) => assign(j, e.target.value)}
                          className="bg-bg border border-line px-1 py-0.5 text-ink-2 max-w-[140px]">
                          <option value="">unassigned</option>
                          {users.map((u) => <option key={u.user_id} value={u.user_id}>{u.name}</option>)}
                        </select>
                      </td>
                      <td className="text-right">
                        {j.frame_ids[0] && (
                          <button onClick={() => router.push(`/frame/${j.frame_ids[0]}`)}
                            title="open the first frame of this job"
                            className="text-ink-3 hover:text-accent">open</button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Section>

        {/* multi-modal assets */}
        <Section title={`assets (${assets.total})`}
          right={<span className="font-mono text-[11px] text-ink-3">
            {Object.entries(annKinds).map(([k, v]) => `${k} ${v}`).join("  ") || "no annotations yet"}
          </span>}>
          <div className="flex items-center gap-2 font-mono text-[11px] flex-wrap mb-2">
            <select value={newMedia} onChange={(e) => setNewMedia(e.target.value)}
              className="bg-bg border border-line px-1.5 py-0.5 text-ink">
              {["text", "dialogue", "audio", "timeseries", "document", "image"].map((m) =>
                <option key={m} value={m}>{m}</option>)}
            </select>
            <input value={newBody} onChange={(e) => setNewBody(e.target.value)}
              placeholder={newMedia === "text" || newMedia === "dialogue" ? "inline text" : "uri"}
              className="flex-1 min-w-[240px] bg-bg border border-line px-1.5 py-0.5 text-ink" />
            <button onClick={addAsset} disabled={busy || !projectId}
              className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
              add asset
            </button>
          </div>
          {assets.assets.length === 0 ? (
            <div className="font-mono text-[11px] text-ink-3 py-2">
              no assets. add one above, or POST to /api/assets to bulk import.
            </div>
          ) : (
            <div className="overflow-auto max-h-72">
              <table className="w-full font-mono text-[11px]">
                <thead>
                  <tr className="text-ink-3 text-left border-b hairline">
                    <th className="py-1">asset</th><th>media</th><th>state</th><th>preview</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {assets.assets.map((a) => (
                    <tr key={a.asset_id} className="border-b hairline">
                      <td className="py-1 text-ink-3">{(a.external_id ?? a.asset_id).slice(0, 12)}</td>
                      <td className="text-ink-2">{a.media_type}</td>
                      <td className={a.state === "labeled" ? "text-pass" : "text-ink-3"}>{a.state}</td>
                      <td className="text-ink-3 truncate max-w-[280px]">{a.text ?? a.uri ?? "-"}</td>
                      <td className="text-right">
                        <button onClick={() => router.push(`/annotate/asset/${a.asset_id}`)}
                          className="text-ink-3 hover:text-accent">annotate</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Section>

        {/* scorecards */}
        <Section title="annotator scorecards">
          {cards.length === 0 ? (
            <div className="font-mono text-[11px] text-ink-3 py-3">no review activity yet</div>
          ) : (
            <table className="w-full max-w-3xl font-mono text-[11px]">
              <thead>
                <tr className="text-ink-3 text-left border-b hairline">
                  <th className="py-1">annotator</th><th>role</th>
                  <th className="text-right">reviews</th><th className="text-right">median</th>
                  <th className="text-right">mean</th><th className="text-right">time</th>
                  <th className="text-right">jobs</th><th className="text-right">honeypot</th>
                </tr>
              </thead>
              <tbody>
                {cards.map((c) => (
                  <tr key={c.user_id} className="border-b hairline">
                    <td className="py-1 text-ink-2">{c.name}</td>
                    <td className="text-ink-3">{c.role}</td>
                    <td className="text-right text-ink">{c.reviews}</td>
                    <td className="text-right text-ink-3">{(c.median_time_ms / 1000).toFixed(1)}s</td>
                    <td className="text-right text-ink-3">{(c.mean_time_ms / 1000).toFixed(1)}s</td>
                    <td className="text-right text-ink-3">{c.total_time_min}m</td>
                    <td className="text-right text-ink-3">{c.jobs}</td>
                    <td className={`text-right ${c.honeypot_accuracy == null ? "text-ink-3"
                      : c.honeypot_accuracy >= 0.9 ? "text-pass" : "text-block"}`}>
                      {c.honeypot_accuracy == null ? "-" : c.honeypot_accuracy.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>
      </div>
    </PageShell>
  );
}
