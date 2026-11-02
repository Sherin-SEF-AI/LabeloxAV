"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import Icon from "@/components/shell/Icon";
import LoadState from "@/components/shell/LoadState";
import SessionSelect from "@/components/shell/SessionSelect";
import ProjectClasses, { type ClassSet } from "@/components/settings/ProjectClasses";
import { useConfirm } from "@/components/ConfirmProvider";
import { api, apiDelete, apiGet, apiPatch, apiPost, apiPut, humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import { useCurrentUser } from "@/lib/user";

// Projects decide who sees which sessions. Every project starts visible to the whole team, which is how a
// small lab usually works; making one "members only" hides its sessions, frames and labels from everyone
// except its members and administrators, in every list and by direct link. The rule lives in
// services/project_access.py; this page only changes it, and every change lands in the activity log.

type Project = {
  project_id: string; name: string; description: string | null; visibility: "team" | "members";
  sessions: number; members: number; is_member: boolean;
  class_set: ClassSet; class_count: number; of: number;
};
type Member = { user_id: string; name: string; role: string; added_at: string | null };
type User = { user_id: string; name: string; role: string };
type SessionRow = { session_id: string; vehicle_id: string | null; city: string | null; start_ts_ns: number };

const SESSION_PAGE = 50;

function VisibilityPill({ v }: { v: Project["visibility"] }) {
  return v === "members"
    ? <span className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] bg-warn/10 text-warn border border-warn/30"><Icon name="shield" size={11} />Members only</span>
    : <span className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] bg-bg-2 text-ink-2 border hairline"><Icon name="person" size={11} />Whole team</span>;
}

function sessionLabel(s: SessionRow): string {
  const when = s.start_ts_ns ? new Date(s.start_ts_ns / 1e6).toLocaleDateString() : "";
  return [s.vehicle_id, s.city, when].filter(Boolean).join(" · ") || s.session_id.slice(0, 8);
}

function ProjectDetail({ project, users, onChanged }: { project: Project; users: User[]; onChanged: () => void }) {
  const confirm = useConfirm();
  const router = useRouter();
  const [members, setMembers] = useState<Member[] | null>(null);
  const [sessions, setSessions] = useState<{ total: number; rows: SessionRow[] } | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [adding, setAdding] = useState("");
  const [moving, setMoving] = useState("");
  const [description, setDescription] = useState(project.description ?? "");

  const load = useCallback(async () => {
    setError(null);
    try {
      const [m, s] = await Promise.all([
        apiGet<{ members: Member[] }>(`/api/projects/${project.project_id}/members`),
        apiGet<{ total: number; sessions: SessionRow[] }>(`/api/sessions/page?limit=${SESSION_PAGE}&project_id=${project.project_id}`),
      ]);
      setMembers(m.members);
      setSessions({ total: s.total, rows: s.sessions });
    } catch (e) { setError(e); }
  }, [project.project_id]);
  useEffect(() => { void load(); setDescription(project.description ?? ""); }, [load, project.description]);

  const setVisibility = async (v: Project["visibility"]) => {
    if (v === project.visibility) return;
    if (v === "members" && !(await confirm({
      title: `Make ${project.name} members only?`,
      body: `Its ${project.sessions.toLocaleString()} session(s) disappear for everyone who is not an administrator or one of its ${project.members} member(s), in every list and by direct link. Work already assigned to other people in it stops being reachable for them.`,
      confirmLabel: "Make members only",
    }))) return;
    try {
      await apiPatch(`/api/projects/${project.project_id}`, { visibility: v });
      toast(v === "members" ? `${project.name} is now members only` : `${project.name} is visible to the whole team`, "success");
      onChanged();
    } catch (e) { toast(humanizeError(e), "error"); }
  };
  const saveDescription = async () => {
    if (description === (project.description ?? "")) return;
    try { await apiPatch(`/api/projects/${project.project_id}`, { description }); toast("Description saved", "success"); onChanged(); }
    catch (e) { toast(humanizeError(e), "error"); }
  };
  const addMember = async () => {
    if (!adding) return;
    try { await apiPut(`/api/projects/${project.project_id}/members/${adding}`, {}); setAdding(""); await load(); onChanged(); }
    catch (e) { toast(humanizeError(e), "error"); }
  };
  const removeMember = async (m: Member) => {
    if (project.visibility === "members" && m.role !== "admin" && !(await confirm({
      title: `Remove ${m.name} from ${project.name}?`,
      body: `${m.name} stops seeing this project's sessions straight away.`,
      confirmLabel: "Remove", danger: true,
    }))) return;
    try { await apiDelete(`/api/projects/${project.project_id}/members/${m.user_id}`); await load(); onChanged(); }
    catch (e) { toast(humanizeError(e), "error"); }
  };
  const moveSession = async () => {
    if (!moving) return;
    try {
      const r = await apiPost<{ moved: number }>(`/api/projects/${project.project_id}/sessions`, { session_ids: [moving] });
      toast(r.moved ? `Moved into ${project.name}` : "That session was not found", r.moved ? "success" : "error");
      setMoving("");
      await load(); onChanged();
    } catch (e) { toast(humanizeError(e), "error"); }
  };

  const openSession = async (sessionId: string) => {
    try { router.push(`/frame/${(await api.firstFrame(sessionId)).frame_id}`); }
    catch (e) { toast(humanizeError(e), "error"); }
  };

  const memberIds = new Set((members ?? []).map((m) => m.user_id));
  const candidates = users.filter((u) => !memberIds.has(u.user_id));

  return (
    <section className="panel p-5 flex flex-col gap-5">
      <header className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-[15px] font-semibold text-ink">{project.name}</h2>
        <VisibilityPill v={project.visibility} />
      </header>

      <fieldset className="flex flex-col gap-2">
        <legend className="text-[12px] text-ink-2 mb-1.5">Who sees its sessions</legend>
        {([
          ["team", "The whole team", "Everyone signed in. The default, and right for most labs."],
          ["members", "Members only", "Its members and administrators. Everyone else does not see it exists."],
        ] as const).map(([v, label, hint]) => (
          <label key={v} className={`flex items-start gap-2.5 rounded border px-3 py-2.5 cursor-pointer ${project.visibility === v ? "border-accent/60 bg-accent/5" : "hairline"}`}>
            <input type="radio" name={`vis-${project.project_id}`} className="mt-0.5" checked={project.visibility === v}
                   onChange={() => void setVisibility(v)} />
            <span className="flex flex-col">
              <span className="text-[13px] text-ink">{label}</span>
              <span className="text-[12px] text-ink-3">{hint}</span>
            </span>
          </label>
        ))}
      </fieldset>

      <label className="flex flex-col gap-1.5">
        <span className="text-[12px] text-ink-2">Description</span>
        <textarea className="field min-h-[60px]" value={description} maxLength={2000}
                  onChange={(e) => setDescription(e.target.value)} onBlur={() => void saveDescription()}
                  placeholder="What this project holds, and who it is for" />
      </label>

      <ProjectClasses projectId={project.project_id} projectName={project.name} classSet={project.class_set}
                      classCount={project.class_count} of={project.of} onChanged={onChanged} />

      <LoadState loading={!members && !error} error={error} onRetry={() => void load()} />

      {members && (
        <div className="flex flex-col gap-2">
          <h3 className="text-[13px] font-semibold text-ink">Members ({members.length})</h3>
          {project.visibility === "team" && (
            <p className="text-[12px] text-ink-3">The project is visible to the whole team, so membership changes nothing yet. It takes effect when you make it members only.</p>
          )}
          {members.length > 0 && (
            <ul className="flex flex-col divide-y divide-line border hairline rounded">
              {members.map((m) => (
                <li key={m.user_id} className="flex items-center gap-3 px-3 py-2 text-[12.5px]">
                  <span className="text-ink">{m.name}</span>
                  <span className="text-ink-3 capitalize">{m.role}</span>
                  <button className="btn ml-auto" onClick={() => void removeMember(m)}>Remove</button>
                </li>
              ))}
            </ul>
          )}
          <div className="flex gap-2">
            <select className="field flex-1 min-w-0" value={adding} onChange={(e) => setAdding(e.target.value)}
                    aria-label="Teammate to add">
              <option value="">{candidates.length ? "Choose a teammate" : "Everyone is already a member"}</option>
              {candidates.map((u) => <option key={u.user_id} value={u.user_id}>{u.name} ({u.role})</option>)}
            </select>
            <button className="btn btn-primary shrink-0" disabled={!adding} onClick={() => void addMember()}>
              <Icon name="plus" size={13} />Add
            </button>
          </div>
        </div>
      )}

      {sessions && (
        <div className="flex flex-col gap-2">
          <h3 className="text-[13px] font-semibold text-ink">Sessions ({sessions.total.toLocaleString()})</h3>
          {sessions.rows.length > 0 && (
            <ul className="flex flex-col divide-y divide-line border hairline rounded max-h-64 overflow-y-auto">
              {sessions.rows.map((s) => (
                <li key={s.session_id} className="flex items-center gap-3 px-3 py-1.5 text-[12.5px]">
                  <button className="text-left text-ink hover:text-accent truncate" onClick={() => void openSession(s.session_id)}>{sessionLabel(s)}</button>
                  <span className="ml-auto font-mono text-[11px] text-ink-3">{s.session_id.slice(0, 8)}</span>
                </li>
              ))}
            </ul>
          )}
          {sessions.total > sessions.rows.length && (
            <p className="text-[12px] text-ink-3">Showing the newest {sessions.rows.length} of {sessions.total.toLocaleString()}.</p>
          )}
          <div className="flex gap-2 items-stretch">
            <SessionSelect value={moving} onChange={setMoving} placeholder="Move a session into this project" className="flex-1 min-w-0" />
            <button className="btn shrink-0" disabled={!moving} onClick={() => void moveSession()}>Move here</button>
          </div>
          <p className="text-[12px] text-ink-3">A session belongs to one project, so moving it here takes it out of its old one, along with who could see it there.</p>
        </div>
      )}
    </section>
  );
}

export default function ProjectSettingsPage() {
  const me = useCurrentUser();
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [visibility, setVisibility] = useState<Project["visibility"]>("team");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [p, u] = await Promise.all([
        apiGet<{ projects: Project[] }>("/api/projects"),
        apiGet<{ users: User[] }>("/api/users?limit=500"),
      ]);
      // The projects holding data first: a lab accumulates empty ones (trials, checks) that would otherwise
      // bury the one everything is in.
      setProjects([...p.projects].sort((a, b) => b.sessions - a.sessions || a.name.localeCompare(b.name)));
      setUsers(u.users);
      setSelected((cur) => cur ?? [...p.projects].sort((a, b) => b.sessions - a.sessions)[0]?.project_id ?? null);
    } catch (e) { setError(e); }
  }, []);
  useEffect(() => { if (me?.role === "admin") void load(); }, [load, me?.role]);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      const p = await apiPost<{ project_id: string }>("/api/projects", { name: name.trim(), visibility });
      setName("");
      setSelected(p.project_id);
      await load();
      toast("Project created", "success");
    } catch (err) { toast(humanizeError(err), "error"); }
    finally { setBusy(false); }
  };

  if (me && me.role !== "admin") {
    return (
      <PageShell active="Projects" title="Projects">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
          <section className="panel p-5 text-[13px] text-ink-2">Projects and who can see them are managed by administrators.</section>
        </div>
      </PageShell>
    );
  }

  const current = projects?.find((p) => p.project_id === selected) ?? null;

  return (
    <PageShell active="Projects" title="Projects" subtitle="who sees which sessions">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6 flex flex-col gap-5">
        <form onSubmit={create} className="panel p-5 grid gap-4 sm:grid-cols-[1fr_1fr_auto] items-end">
          <div className="sm:col-span-3">
            <h2 className="text-[14px] font-semibold text-ink">New project</h2>
            <p className="mt-0.5 text-[12.5px] text-ink-3">Group sessions that belong together, for a client, a study or a partner whose data not everyone should see.</p>
          </div>
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] text-ink-2">Name</span>
            <input className="field" value={name} onChange={(e) => setName(e.target.value)} required maxLength={120} placeholder="e.g. Pune pilot" />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] text-ink-2">Visible to</span>
            <select className="field" value={visibility} onChange={(e) => setVisibility(e.target.value as Project["visibility"])}>
              <option value="team">The whole team</option>
              <option value="members">Members only</option>
            </select>
          </label>
          <button className="btn btn-primary" type="submit" disabled={busy || !name.trim()}>
            <Icon name="plus" size={13} />{busy ? "Creating" : "Create"}
          </button>
        </form>

        <LoadState loading={!projects && !error} error={error} onRetry={() => void load()} />

        {projects && projects.length === 0 && (
          <section className="panel p-5 text-[13px] text-ink-2">No projects yet. Sessions you import go into the default project once one exists; create one above.</section>
        )}

        {projects && projects.length > 0 && (
          <div className="grid gap-5 grid-cols-[minmax(0,1fr)] lg:grid-cols-[minmax(0,320px)_minmax(0,1fr)] items-start">
            <nav className="panel p-1.5 flex flex-col min-w-0 max-h-[70vh] overflow-y-auto" aria-label="Projects">
              {projects.map((p) => (
                <button key={p.project_id} onClick={() => setSelected(p.project_id)}
                        aria-current={p.project_id === selected ? "true" : undefined}
                        className={`text-left rounded px-3 py-2.5 flex flex-col gap-1 min-w-0 ${p.project_id === selected ? "bg-accent/10" : "hover:bg-bg-2"}`}>
                  <span className="flex items-center gap-2 min-w-0">
                    <span className="text-[13px] text-ink truncate min-w-0">{p.name}</span>
                    <span className="ml-auto shrink-0"><VisibilityPill v={p.visibility} /></span>
                  </span>
                  <span className="text-[11.5px] text-ink-3 tabular-nums">
                    {p.sessions.toLocaleString()} session{p.sessions === 1 ? "" : "s"} · {p.members} member{p.members === 1 ? "" : "s"}
                  </span>
                </button>
              ))}
            </nav>
            {current && <ProjectDetail key={current.project_id} project={current} users={users} onChanged={() => void load()} />}
          </div>
        )}
      </div>
    </PageShell>
  );
}
