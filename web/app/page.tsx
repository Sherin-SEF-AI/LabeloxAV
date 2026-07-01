"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { DatasetRow, SessionRow, TriageRow } from "@/lib/types";
import { ConfBar, StateBadge } from "@/components/StateBadge";
import PageShell from "@/components/shell/PageShell";
import CorrectionModal from "@/components/CorrectionModal";
import { SkeletonRows, Spinner } from "@/components/Spinner";
import { acceptState, getUser } from "@/lib/user";

const BANDS = [
  { key: "review,annotate", label: "My queue", hint: "everything assigned to you" },
  { key: "review", label: "To review", hint: "model proposals awaiting a decision" },
  { key: "annotate", label: "To annotate", hint: "frames that still need labels" },
  { key: "submitted", label: "QA queue", hint: "submitted work awaiting approval" },
];

// A compact overview number (sessions, queue size, datasets ...).
function Stat({ label, value, sub, loading }: { label: string; value: number | string; sub?: string; loading?: boolean }) {
  return (
    <div className="panel px-4 py-3">
      <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3">{label}</div>
      <div className="font-mono text-2xl text-ink mt-1 tabular-nums">
        {loading ? <span className="text-ink-3 animate-pulse">···</span> : value}
      </div>
      {sub ? <div className="font-mono text-[11px] text-ink-3 mt-0.5">{sub}</div> : null}
    </div>
  );
}

// A big, obvious entry point into one of the platform's workflows.
function ActionCard({ title, desc, onClick }: { title: string; desc: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="panel text-left px-4 py-3 hover:border-accent transition-colors group focus:outline-none focus:border-accent"
    >
      <div className="flex items-center justify-between">
        <span className="text-ink font-medium">{title}</span>
        <span className="text-ink-3 group-hover:text-accent transition-colors">&rarr;</span>
      </div>
      <div className="text-ink-3 text-xs mt-1 leading-snug">{desc}</div>
    </button>
  );
}

export default function HomePage() {
  const router = useRouter();
  const [rows, setRows] = useState<TriageRow[]>([]);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [session, setSession] = useState<string>("");
  const [states, setStates] = useState<string>("review,annotate");
  const [cursor, setCursor] = useState(0);
  const [loading, setLoading] = useState(true);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [bulkClass, setBulkClass] = useState("");
  const [classes, setClasses] = useState<string[]>([]);
  const [corr, setCorr] = useState<{ objectId: string; old: string; to: string } | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [role, setRole] = useState<string | undefined>(undefined);
  const [showActions, setShowActions] = useState(false);
  const [ingest, setIngest] = useState<{ active: boolean; finished: boolean; done: number; total: number; current: string | null; frames: number } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params: Record<string, string> = { states, limit: "200" };
      if (session) params.session_id = session;
      setRows(await api.triage(params));
      setCursor(0);
    } finally {
      setLoading(false);
    }
  }, [states, session]);

  useEffect(() => {
    api.sessions().then(setSessions).catch(() => {});
    api.datasets().then(setDatasets).catch(() => {});
    api.ontology().then((o) => setClasses(o.classes.map((c) => c.name))).catch(() => {});
    setRole(getUser()?.role);
  }, []);
  useEffect(() => {
    const f = () => api.ingestProgress().then(setIngest).catch(() => {});
    f();
    const t = setInterval(f, 3000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    load();
    setSel(new Set());
  }, [load]);

  const toggle = (id: string) =>
    setSel((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const allSelected = rows.length > 0 && rows.every((r) => sel.has(r.object_id));
  const selectAll = () => setSel(allSelected ? new Set() : new Set(rows.map((r) => r.object_id)));

  const bulk = useCallback(
    async (action: string, className?: string, state?: string) => {
      if (!sel.size) return;
      setMsg(null);
      try {
        const r = await api.bulkReview([...sel], action, className, state);
        setMsg(`${action}: ${r.updated} objects`);
        setSel(new Set());
        load();
      } catch (e) {
        setMsg(String(e));
      }
    },
    [sel, load],
  );

  const autolabel = useCallback(async () => {
    if (!session) return;
    try {
      const r = await api.startAutolabel(session, undefined, "local");
      setMsg(r.status === "queued-cloud" ? "queued for the cloud A100 - see Jobs" : "autolabel queued - watch it on Jobs");
    } catch (e) {
      setMsg(String(e).includes("503") ? "GPU busy (training). Try after it finishes." : String(e));
    }
  }, [session]);

  const vlmQa = useCallback(async () => {
    if (!session) return;
    try {
      await api.startVlmQa(session);
      setMsg("VLM auto-QA running - it flags likely-wrong labels into the QA queue + fills attributes");
    } catch (e) {
      setMsg(String(e).includes("503") ? "GPU busy (training). Try after it finishes." : String(e));
    }
  }, [session]);

  const recognizeSigns = useCallback(async () => {
    if (!session) return;
    try {
      const r = await api.recognizeSigns(session);
      setMsg(`typed ${r.recognized} signs (Indian taxonomy), ${r.text_bearing} text-bearing routed to OCR`);
    } catch (e) {
      setMsg(String(e));
    }
  }, [session]);

  const open = useCallback(
    (r?: TriageRow) => {
      const row = r ?? rows[cursor];
      if (row) router.push(`/frame/${row.frame_id}?focus=${row.object_id}`);
    },
    [rows, cursor, router],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "j") setCursor((c) => Math.min(c + 1, rows.length - 1));
      else if (e.key === "k") setCursor((c) => Math.max(c - 1, 0));
      else if (e.key === "Enter") open();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows.length, open]);

  const cities = useMemo(() => new Set(sessions.map((s) => s.city).filter(Boolean)).size, [sessions]);
  const activeBand = BANDS.find((b) => b.key === states);

  return (
    <PageShell
      active="TRIAGE"
      subtitle="HOME"
      right={loading ? <Spinner label="loading" /> : <span className="border border-line px-2 py-0.5">{rows.length} in queue</span>}
    >
      <div className="h-full overflow-auto">
        <div className="max-w-6xl mx-auto p-6 space-y-6">
          {/* Live ingest progress (dashcam batch) */}
          {ingest && (ingest.active || (ingest.done > 0 && ingest.done < ingest.total)) ? (
            <div className="panel px-4 py-3 border-accent/40">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="inline-block w-3.5 h-3.5 rounded-full border-2 border-line border-t-accent animate-spin" aria-hidden />
                  <span className="text-ink font-medium">Ingesting dashcam videos</span>
                </div>
                <span className="font-mono text-[11px] text-ink-3 tabular-nums">
                  {ingest.done}/{ingest.total} videos &middot; {ingest.frames.toLocaleString()} frames
                </span>
              </div>
              <div className="h-1.5 bg-bg-2 rounded overflow-hidden">
                <div className="h-full bg-accent transition-all duration-500"
                  style={{ width: `${ingest.total ? Math.round((ingest.done / ingest.total) * 100) : 0}%` }} />
              </div>
              {ingest.current ? (
                <div className="font-mono text-[10px] text-ink-3 mt-1 truncate">now: {ingest.current}</div>
              ) : null}
            </div>
          ) : ingest?.finished && ingest.frames > 0 ? (
            <div className="panel px-4 py-2.5 border-pass/40 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-pass" />
              <span className="text-ink text-sm">Ingested {ingest.done} dashcam videos &middot; {ingest.frames.toLocaleString()} frames ready</span>
            </div>
          ) : null}

          {/* Welcome + orientation */}
          <div>
            <h1 className="text-xl text-ink font-semibold">Welcome to LabeloxAV</h1>
            <p className="text-ink-3 text-sm mt-1">
              Multimodal annotation for autonomous driving. Start from your review queue below, or jump into a workflow.
            </p>
          </div>

          {/* At-a-glance stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Sessions" value={sessions.length} sub={`${cities} cit${cities === 1 ? "y" : "ies"}`} loading={!sessions.length && loading} />
            <Stat label="In your queue" value={rows.length} sub={activeBand?.label.toLowerCase()} loading={loading} />
            <Stat label="Datasets" value={datasets.length} sub={datasets.length ? "sealed exports" : "none yet"} />
            <Stat label="Your role" value={role ?? "annotator"} sub="what you can approve" />
          </div>

          {/* Quick actions */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3">Jump to a workflow</h2>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <ActionCard title="Review queue" desc="Work through model proposals ranked by what matters most." onClick={() => document.getElementById("queue")?.scrollIntoView({ behavior: "smooth" })} />
              <ActionCard title="Import data" desc="Bring in dashcam video or sensor logs to annotate." onClick={() => router.push("/import")} />
              <ActionCard title="Analytics" desc="See labeling progress, agreement, and coverage." onClick={() => router.push("/analytics")} />
              <ActionCard title="Datasets" desc="Seal and export a versioned dataset (COCO, YOLO, ...)." onClick={() => router.push("/datasets")} />
              <ActionCard title="Jobs" desc="Watch autolabel, training, and import jobs live." onClick={() => router.push("/jobs")} />
              <ActionCard title="Curation" desc="Find the highest-value frames to label next." onClick={() => router.push("/curation")} />
            </div>
          </div>

          {/* Review queue */}
          <div id="queue" className="panel">
            <div className="flex flex-wrap items-center gap-3 px-4 py-3 border-b hairline">
              <div>
                <div className="text-ink font-medium">Your review queue</div>
                <div className="text-ink-3 text-xs">{activeBand?.hint} &middot; ranked by uncertainty and rarity</div>
              </div>
              <div className="flex items-center gap-1 ml-auto font-mono text-[11px]">
                {BANDS.map((b) => (
                  <button key={b.key} onClick={() => setStates(b.key)} title={b.hint}
                    className={`px-2 py-1 border ${states === b.key ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink"}`}>
                    {b.label}
                  </button>
                ))}
              </div>
              <select value={session} onChange={(e) => setSession(e.target.value)}
                className="bg-panel border hairline text-ink text-xs px-2 py-1 max-w-[240px]">
                <option value="">all sessions ({sessions.length})</option>
                {[...sessions]
                  .sort((a, b) => (b.route ?? "").localeCompare(a.route ?? "") || b.start_ts_ns - a.start_ts_ns)
                  .map((s) => (
                    <option key={s.session_id} value={s.session_id}>
                      {s.route
                        ? `${s.route}${s.city ? ` · ${s.city}` : ""}`
                        : `${s.vehicle_id} · ${s.city ?? "?"} · #${s.session_id.slice(0, 4)}`}
                    </option>
                  ))}
              </select>
            </div>

            {/* AI helpers for the selected session (secondary, tucked away) */}
            {session ? (
              <div className="px-4 py-2 border-b hairline flex flex-wrap items-center gap-2 font-mono text-[11px]">
                <button onClick={() => setShowActions((v) => !v)} className="text-ink-3 hover:text-ink">
                  {showActions ? "▾" : "▸"} AI helpers for this session
                </button>
                {showActions ? (
                  <>
                    <button onClick={autolabel} className="border border-line text-ink-2 px-2 py-1 hover:border-accent">autolabel</button>
                    <button onClick={recognizeSigns} className="border border-line text-ink-2 px-2 py-1 hover:border-accent">recognize signs</button>
                    <button onClick={vlmQa} className="border border-line text-ink-2 px-2 py-1 hover:border-accent">VLM auto-QA</button>
                  </>
                ) : null}
              </div>
            ) : null}

            {msg ? <div className="px-4 py-1.5 font-mono text-[11px] text-warn border-b hairline">{msg}</div> : null}

            {/* Bulk action bar */}
            {sel.size > 0 && (
              <div className="sticky top-0 z-20 flex flex-wrap items-center gap-2 px-4 py-1.5 bg-panel border-b hairline font-mono text-[11px]">
                <span className="text-ink">{sel.size} selected</span>
                <button onClick={() => bulk("accept", undefined, acceptState(role))} className="border border-pass text-pass px-2 py-0.5"
                  title={role === "annotator" ? "submit for QA" : "approve (accept)"}>
                  {role === "annotator" ? "submit" : "approve"}
                </button>
                <button onClick={() => bulk("reject")} className="border border-block text-block px-2 py-0.5">reject</button>
                <input list="cls" value={bulkClass} onChange={(e) => setBulkClass(e.target.value)} placeholder="relabel to class"
                  className="bg-bg border border-line px-2 py-0.5 w-40 text-ink" />
                <datalist id="cls">{classes.map((c) => <option key={c} value={c} />)}</datalist>
                <button onClick={() => bulk("reclassify", bulkClass)} disabled={!classes.includes(bulkClass)}
                  className="border border-line px-2 py-0.5 hover:border-accent disabled:opacity-40">relabel</button>
                {sel.size === 1 && classes.includes(bulkClass) && (
                  <button onClick={() => { const row = rows.find((r) => sel.has(r.object_id)); if (row) setCorr({ objectId: row.object_id, old: row.class_name, to: bulkClass }); }}
                    title="find visually-similar objects of the old class and bulk-fix them too"
                    className="border border-accent text-accent px-2 py-0.5 hover:bg-accent/10">correct similar &rarr;</button>
                )}
                <button onClick={() => setSel(new Set())} className="text-ink-3 hover:text-ink ml-auto">clear</button>
              </div>
            )}

            {/* Table */}
            {loading && !rows.length ? (
              <SkeletonRows rows={8} cols="grid-cols-[32px_40px_1fr_1fr_130px_110px]" />
            ) : rows.length ? (
              <table className="w-full text-sm">
                <thead className="text-ink-3 font-mono text-[11px] uppercase border-b hairline">
                  <tr>
                    <th className="px-3 py-2 w-8"><input type="checkbox" checked={allSelected} onChange={selectAll} /></th>
                    <th className="text-left font-normal px-3 py-2 w-10">#</th>
                    <th className="text-left font-normal px-3 py-2">object &middot; why it needs you</th>
                    <th className="text-left font-normal px-3 py-2">proposed class</th>
                    <th className="text-left font-normal px-3 py-2 w-32">confidence</th>
                    <th className="text-left font-normal px-3 py-2 w-28">state</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={r.object_id} onClick={() => { setCursor(i); open(r); }} onMouseEnter={() => setCursor(i)}
                      className={`border-b hairline cursor-pointer ${sel.has(r.object_id) ? "bg-bg-2" : i === cursor ? "bg-panel" : "hover:bg-bg-2"}`}>
                      <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                        <input type="checkbox" checked={sel.has(r.object_id)} onChange={() => toggle(r.object_id)} />
                      </td>
                      <td className="px-3 py-2 font-mono text-ink-3">{String(i + 1).padStart(2, "0")}</td>
                      <td className="px-3 py-2">
                        <div className="text-ink-2">{r.class_name}</div>
                        <div className="text-ink-3 text-xs">{r.why}</div>
                      </td>
                      <td className="px-3 py-2 font-mono">{r.class_name}</td>
                      <td className="px-3 py-2"><ConfBar conf={r.conf} /></td>
                      <td className="px-3 py-2"><StateBadge state={r.state} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="px-4 py-12 text-center">
                <div className="text-ink font-medium">Your queue is clear</div>
                <div className="text-ink-3 text-sm mt-1">
                  Nothing to review in <span className="text-ink-2">{activeBand?.label.toLowerCase()}</span> right now.
                  Try another band, pick a session, or import more data.
                </div>
                <button onClick={() => router.push("/import")} className="mt-3 border border-line px-3 py-1.5 text-sm text-ink-2 hover:border-accent">
                  Import data &rarr;
                </button>
              </div>
            )}
          </div>

          <div className="font-mono text-[11px] text-ink-3 text-center">
            J / K to move &middot; Enter to open &middot; queue ranked by uncertainty x rarity
          </div>
        </div>
      </div>

      {corr && (
        <CorrectionModal objectId={corr.objectId} kind="class" change={{ old: corr.old, new: corr.to }}
          onClose={() => setCorr(null)}
          onApplied={(n) => { setMsg(`corrected ${n} similar objects`); setCorr(null); setSel(new Set()); load(); }} />
      )}
    </PageShell>
  );
}
