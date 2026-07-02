"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { DatasetRow, SessionRow, TriageRow } from "@/lib/types";
import { ConfBar, StateBadge } from "@/components/StateBadge";
import PageShell from "@/components/shell/PageShell";
import CorrectionModal from "@/components/CorrectionModal";
import { SkeletonRows, Spinner } from "@/components/Spinner";
import { ObjectSourceBadge } from "@/components/SourceBadge";
import { objectSource, sessionOrigin } from "@/lib/source";
import { acceptState, getUser } from "@/lib/user";
import Icon from "@/components/shell/Icon";

const BANDS = [
  { key: "review,annotate", label: "My queue", hint: "everything assigned to you" },
  { key: "review", label: "To review", hint: "model proposals awaiting a decision" },
  { key: "annotate", label: "To annotate", hint: "frames that still need labels" },
  { key: "submitted", label: "QA queue", hint: "submitted work awaiting approval" },
];

// A compact overview number (sessions, queue size, datasets ...). accent draws the eye to the one that
// represents work waiting on you.
function Stat({ label, value, sub, loading, accent }: { label: string; value: number | string; sub?: string; loading?: boolean; accent?: boolean }) {
  return (
    <div className={`panel px-4 py-3 ${accent ? "border-accent/40" : ""}`}>
      <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3">{label}</div>
      <div className={`font-mono text-2xl mt-1 tabular-nums ${accent ? "text-accent" : "text-ink"}`}>
        {loading ? <span className="text-ink-3 animate-pulse">···</span> : value}
      </div>
      {sub ? <div className="font-mono text-[11px] text-ink-3 mt-0.5">{sub}</div> : null}
    </div>
  );
}

// A compact launcher tile into one of the platform's workflows. Single row so the six of them form a slim
// strip (the full description rides in the tooltip); primary marks the recommended next step.
function ActionCard({ title, desc, icon, onClick, primary }: { title: string; desc: string; icon: string; onClick: () => void; primary?: boolean }) {
  return (
    <button
      onClick={onClick}
      title={desc}
      className={`panel flex items-center gap-2 px-3 py-2.5 transition-colors group focus:outline-none focus:border-accent ${primary ? "border-accent/50 hover:border-accent" : "hover:border-accent"}`}
    >
      <span className={`flex shrink-0 transition-colors ${primary ? "text-accent" : "text-ink-3 group-hover:text-accent"}`}><Icon name={icon} size={16} /></span>
      <span className="text-ink text-[13px] font-medium truncate">{title}</span>
      <span className="ml-auto text-ink-3 group-hover:text-accent transition-colors text-xs">&rarr;</span>
    </button>
  );
}

// The "why it needs you" reasons as scannable colored chips instead of one run-on line: conflicts read red,
// low confidence amber, rare classes blue, mask/box disagreement accent.
function WhyChips({ why }: { why: string }) {
  if (!why) return null;
  const tone = (t: string) =>
    /conflict/.test(t) ? "text-block border-block/40"
      : /low conf/.test(t) ? "text-warn border-warn/40"
      : /rare/.test(t) ? "text-info border-info/40"
      : /mask/.test(t) ? "text-accent border-accent/40"
      : "text-ink-3 border-line";
  return (
    <div className="flex flex-wrap gap-1 mt-1">
      {why.split(/,\s*/).filter(Boolean).map((t, i) => (
        <span key={i} className={`font-mono text-[9.5px] leading-none px-1.5 py-0.5 rounded border ${tone(t)}`}>{t}</span>
      ))}
    </div>
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
  const [srcFilter, setSrcFilter] = useState<"all" | "app" | "imported">("all");

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
  const shownRows = useMemo(
    () => (srcFilter === "all" ? rows : rows.filter((r) => objectSource(r.source, r.import_format).kind === srcFilter)),
    [rows, srcFilter],
  );
  const importedCount = useMemo(() => rows.filter((r) => r.source === "imported").length, [rows]);
  const SRC_FILTERS: { key: "all" | "app" | "imported"; label: string }[] = [
    { key: "all", label: "all" },
    { key: "app", label: "your work" },
    { key: "imported", label: "imported" },
  ];

  return (
    <PageShell
      active="TRIAGE"
      subtitle="HOME"
      right={loading ? <Spinner label="loading" /> : <span className="border border-line px-2 py-0.5">{rows.length} in queue</span>}
    >
      <div className="h-full flex flex-col">
        <div className="max-w-6xl mx-auto w-full px-6 py-4 flex flex-col gap-3 flex-1 min-h-0">
          {/* Live ingest progress (dashcam batch) */}
          {ingest && (ingest.active || (ingest.done > 0 && ingest.done < ingest.total)) ? (
            <div className="panel px-4 py-3 border-accent/40 shrink-0">
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
            <div className="panel px-4 py-2 border-pass/40 flex items-center gap-2 shrink-0">
              <span className="w-2 h-2 rounded-full bg-pass" />
              <span className="text-ink text-sm">Ingested {ingest.done} dashcam videos &middot; {ingest.frames.toLocaleString()} frames ready</span>
            </div>
          ) : null}

          {/* Welcome + the one action that matters: pick up where the queue left off */}
          <div className="flex flex-wrap items-end justify-between gap-4 shrink-0">
            <div>
              <h1 className="text-xl text-ink font-semibold">Welcome back{role ? `, ${role}` : ""}</h1>
              <p className="text-ink-3 text-sm mt-1 max-w-xl">
                Multimodal annotation for autonomous driving. Pick up your review queue, or jump into a workflow.
              </p>
            </div>
            <button
              onClick={() => open(shownRows[0] ?? rows[0])}
              disabled={!rows.length}
              className="flex items-center gap-2 bg-accent text-bg font-medium text-sm px-4 py-2.5 rounded hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
            >
              <Icon name="confirm" size={16} />
              {rows.length ? "Continue reviewing" : "Queue is clear"}
              {rows.length ? <span className="font-mono text-[11px] opacity-80 tabular-nums">{rows.length} queued</span> : null}
            </button>
          </div>

          {/* At-a-glance stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 shrink-0">
            <Stat label="In your queue" value={rows.length} sub={activeBand?.label.toLowerCase()} loading={loading} accent />
            <Stat label="Sessions" value={sessions.length} sub={`${cities} cit${cities === 1 ? "y" : "ies"}`} loading={!sessions.length && loading} />
            <Stat label="Datasets" value={datasets.length} sub={datasets.length ? "sealed exports" : "none yet"} />
            <Stat label="Your role" value={role ?? "annotator"} sub="what you can approve" />
          </div>

          {/* Workflow launcher: a slim single-row strip */}
          <div className="shrink-0">
            <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3 mb-1.5">Jump to a workflow</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2.5">
              <ActionCard primary icon="review" title="Review" desc="Work through model proposals ranked by what matters most." onClick={() => document.getElementById("queue")?.scrollIntoView({ behavior: "smooth" })} />
              <ActionCard icon="plus" title="Import" desc="Bring in dashcam video or sensor logs to annotate." onClick={() => router.push("/import")} />
              <ActionCard icon="activity" title="Analytics" desc="See labeling progress, agreement, and coverage." onClick={() => router.push("/analytics")} />
              <ActionCard icon="layers" title="Datasets" desc="Seal and export a versioned dataset (COCO, YOLO, ...)." onClick={() => router.push("/datasets")} />
              <ActionCard icon="route" title="Jobs" desc="Watch autolabel, training, and import jobs live." onClick={() => router.push("/jobs")} />
              <ActionCard icon="target" title="Curation" desc="Find the highest-value frames to label next." onClick={() => router.push("/curation")} />
            </div>
          </div>

          {/* Review queue */}
          <div id="queue" className="panel flex-1 min-h-0 flex flex-col overflow-hidden">
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
                      {s.origin === "imported" ? "[imported] " : ""}
                      {s.route
                        ? `${s.route}${s.city ? ` · ${s.city}` : ""}`
                        : `${s.vehicle_id} · ${s.city ?? "?"} · #${s.session_id.slice(0, 4)}`}
                    </option>
                  ))}
              </select>
            </div>

            {/* Source filter: separate imported public-dataset labels from your own work */}
            <div className="px-4 py-2 border-b hairline flex flex-wrap items-center gap-2 font-mono text-[11px]">
              <span className="text-ink-3 uppercase text-[10px]">source</span>
              {SRC_FILTERS.map((sf) => (
                <button key={sf.key} onClick={() => setSrcFilter(sf.key)}
                  className={`px-2 py-0.5 border ${srcFilter === sf.key ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink"}`}>
                  {sf.label}
                </button>
              ))}
              {importedCount > 0 ? (
                <span className="text-ink-3 ml-1">{importedCount} of {rows.length} in view are imported (Mapillary / IDD / BDD), not your annotations</span>
              ) : null}
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

            {/* Table (scrolls inside the panel so the page itself stays fit-to-screen) */}
            <div className="flex-1 min-h-0 overflow-auto">
            {loading && !rows.length ? (
              <SkeletonRows rows={8} cols="grid-cols-[32px_40px_1fr_150px_110px]" />
            ) : rows.length ? (
              <table className="w-full text-sm">
                <thead className="text-ink-3 font-mono text-[11px] uppercase border-b hairline sticky top-0 bg-panel z-10">
                  <tr>
                    <th className="px-3 py-2 w-8"><input type="checkbox" checked={allSelected} onChange={selectAll} /></th>
                    <th className="text-left font-normal px-3 py-2 w-10">#</th>
                    <th className="text-left font-normal px-3 py-2">object &middot; why it needs you</th>
                    <th className="text-left font-normal px-3 py-2 w-36">confidence</th>
                    <th className="text-left font-normal px-3 py-2 w-28">state</th>
                  </tr>
                </thead>
                <tbody>
                  {shownRows.map((r, i) => (
                    <tr key={r.object_id} onClick={() => { setCursor(i); open(r); }} onMouseEnter={() => setCursor(i)}
                      className={`border-b hairline cursor-pointer ${sel.has(r.object_id) ? "bg-bg-2" : i === cursor ? "bg-panel" : "hover:bg-bg-2"}`}>
                      <td className="px-3 py-2 align-top" onClick={(e) => e.stopPropagation()}>
                        <input type="checkbox" checked={sel.has(r.object_id)} onChange={() => toggle(r.object_id)} />
                      </td>
                      <td className="px-3 py-2 font-mono text-ink-3 align-top">{String(i + 1).padStart(2, "0")}</td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-ink">{r.class_name}</span>
                          <ObjectSourceBadge source={r.source} importFormat={r.import_format} />
                        </div>
                        <WhyChips why={r.why} />
                      </td>
                      <td className="px-3 py-2 align-top"><ConfBar conf={r.conf} /></td>
                      <td className="px-3 py-2 align-top"><StateBadge state={r.state} /></td>
                    </tr>
                  ))}
                  {!shownRows.length ? (
                    <tr><td colSpan={5} className="px-3 py-8 text-center text-ink-3 text-sm">
                      No {srcFilter === "imported" ? "imported" : srcFilter === "app" ? "in-app" : ""} items in this band. Try another source or band.
                    </td></tr>
                  ) : null}
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

            {/* Thin footer with the keyboard hint, pinned inside the panel */}
            <div className="shrink-0 border-t hairline px-4 py-1.5 font-mono text-[10px] text-ink-3 flex items-center gap-3">
              <span><span className="text-ink-2">J / K</span> move</span>
              <span><span className="text-ink-2">Enter</span> open</span>
              <span className="ml-auto text-ink-3/70">ranked by uncertainty &times; rarity</span>
            </div>
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
