"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useJobStream } from "@/lib/useEventStream";
import { api , humanizeError } from "@/lib/api";
import type { SessionRow, TriageFlag, TriageRow, TriageSeverity } from "@/lib/types";
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

// A flag reads as loudly as it deserves. Three tiers rather than four equally weighted chips, because a
// defect and a piece of context are not the same news, and a screen of identical chips cannot be sorted by
// eye. High is something probably wrong, medium is worth a look, low is true but not a call to act.
const SEVERITY_TONE: Record<TriageSeverity, string> = {
  high: "text-block border-block/45 bg-block/10",
  medium: "text-warn border-warn/40 bg-warn/10",
  low: "text-ink-3 border-line",
};

// The colour of the row's left edge: the worst thing about it, readable before any word is. Drawn as a
// border on the first cell rather than as its own narrow column, because a table distributes spare width
// into a fixed-width column and turns a 3px rule into a slab.
const SEVERITY_STRIPE: Record<TriageSeverity, string> = {
  high: "border-l-block", medium: "border-l-warn", low: "border-l-line",
};

function worstSeverity(row: TriageRow): TriageSeverity {
  const s = row.flags?.map((f) => f.severity) ?? [];
  return s.includes("high") ? "high" : s.includes("medium") ? "medium" : "low";
}

// Rows from an older API carry only the joined prose. Rendering them at low severity is honest: unknown is
// not urgent, and inventing a tier from a regex here is the coupling this change removed.
function flagsOf(row: TriageRow): TriageFlag[] {
  if (row.flags?.length) return row.flags;
  return (row.why || "")
    .split(/,\s*/)
    .filter(Boolean)
    .map((label) => ({ code: label, label, severity: "low" as TriageSeverity }));
}

function WhyChips({ row }: { row: TriageRow }) {
  const flags = flagsOf(row);
  if (!flags.length) return null;
  return (
    <div className="flex flex-wrap gap-1 mt-1.5">
      {flags.map((f, i) => (
        <span key={`${f.code}-${i}`}
          className={`text-[10px] leading-none px-1.5 py-1 rounded border ${SEVERITY_TONE[f.severity]}`}>
          {f.label}
        </span>
      ))}
    </div>
  );
}

export default function HomePage() {
  const router = useRouter();
  const [rows, setRows] = useState<TriageRow[]>([]);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
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
      // a flywheel worklist deep-link (?flywheel=<cycle>) scopes the queue to that cycle's dispatched objects
      const fw = typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("flywheel") : null;
      if (fw) { params.flywheel = fw; params.states = "review,annotate"; }
      setRows(await api.triage(params));
      setCursor(0);
    } finally {
      setLoading(false);
    }
  }, [states, session]);

  useEffect(() => {
    // The dropdown needs the session rows; nothing on this page shows a fleet total any more.
    api.sessions().then(setSessions).catch(() => {});
    api.ontology().then((o) => setClasses(o.classes.map((c) => c.name))).catch(() => {});
    setRole(getUser()?.role);
  }, []);
  // Ingest progress rides the shared job stream. It used to poll every three seconds on every open tab,
  // and the endpoint behind it scraped a log file with a regex, which reports a quiet system rather than
  // an error the moment the API runs in a different container from the ingest script.
  const jobStream = useJobStream();
  useEffect(() => { api.ingestProgress().then(setIngest).catch(() => {}); }, []);
  useEffect(() => {
    const live = jobStream.data?.ingest?.[0];
    if (live) setIngest(live as typeof ingest);
  }, [jobStream.data]);
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
        setMsg(humanizeError(e));
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
      setMsg(humanizeError(e).includes("503") ? "GPU busy (training). Try after it finishes." : humanizeError(e));
    }
  }, [session]);

  const vlmQa = useCallback(async () => {
    if (!session) return;
    try {
      await api.startVlmQa(session);
      setMsg("VLM auto-QA running - it flags likely-wrong labels into the QA queue + fills attributes");
    } catch (e) {
      setMsg(humanizeError(e).includes("503") ? "GPU busy (training). Try after it finishes." : humanizeError(e));
    }
  }, [session]);

  const recognizeSigns = useCallback(async () => {
    if (!session) return;
    try {
      const r = await api.recognizeSigns(session);
      setMsg(`typed ${r.recognized} signs (Indian taxonomy), ${r.text_bearing} text-bearing routed to OCR`);
    } catch (e) {
      setMsg(humanizeError(e));
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
      // Ignore keys aimed at a field. The handler is on window, so without this, typing a class name into
      // the reclassify box moved the cursor on every "j" and submitted on Enter. Adding accept and reject
      // to the same handler would have turned that from an annoyance into a wrong decision on real data.
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key === "j") setCursor((c) => Math.min(c + 1, rows.length - 1));
      else if (e.key === "k") setCursor((c) => Math.max(c - 1, 0));
      else if (e.key === "Enter") open();
      // Accept and reject act on the selection when there is one, otherwise on the row under the cursor,
      // which is what makes the queue workable without the mouse.
      else if (e.key === "a" || e.key === "r") {
        const row = rows[cursor];
        if (!sel.size && !row) return;
        const targets = sel.size ? sel : new Set([row.object_id]);
        e.preventDefault();
        void (async () => {
          try {
            const res = await api.bulkReview([...targets], e.key === "a" ? "accept" : "reject",
              undefined, e.key === "a" ? acceptState(role) : undefined);
            setMsg(`${e.key === "a" ? "accepted" : "rejected"} ${res.updated}`);
            setSel(new Set());
            load();
          } catch (err) {
            setMsg(humanizeError(err));
          }
        })();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows, cursor, sel, role, open, load]);

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

          {/* One command row where four stacked blocks used to be.
              What went, and why:
                - the greeting, which said the same thing on every one of the day's forty visits;
                - six workflow cards, each duplicating a menu entry one click away, for about 180px of the
                  most valuable space on the screen;
                - "your role", which is identity and already sits in the top right corner;
                - "datasets", a lifetime total that cannot change during a shift.
              A dashboard number has to change what you do next. These did not, and they were pushing the
              queue, which is the entire point of this page, below the fold. */}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2 shrink-0">
            <h1 className="font-display text-lg text-ink font-semibold tracking-tight">
              {activeBand?.label ?? "Review queue"}
            </h1>
            {/* Frames and drives rather than a session total: sessions count LiDAR-only records with no
                frames in them, so one number would mean two different things. */}
            <div className="font-mono text-[11px] text-ink-3 tabular-nums flex items-center gap-2">
              <span>{loading ? "..." : rows.length.toLocaleString()} objects</span>
              {ingest?.frames ? <><span className="text-line">/</span>
                <span>{ingest.frames.toLocaleString()} frames</span></> : null}
              {cities > 0 && <><span className="text-line">/</span><span>{cities} cit{cities === 1 ? "y" : "ies"}</span></>}
            </div>
            <div className="ml-auto flex items-center gap-2">
              <button onClick={() => setShowActions((v) => !v)}
                className="text-[12px] text-ink-2 border border-line rounded px-2.5 py-1.5 hover:bg-btn">
                Session actions
              </button>
              <button
                onClick={() => open(shownRows[0] ?? rows[0])}
                disabled={!rows.length}
                className="flex items-center gap-2 bg-accent hover:bg-accent-2 text-white font-medium text-[13px] px-4 py-1.5 rounded disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Icon name="confirm" size={14} />
                {rows.length ? "Review next" : "Queue is clear"}
              </button>
            </div>
          </div>

          {/* Review queue */}
          <div id="queue" className="fade-up panel flex-1 min-h-0 flex flex-col overflow-hidden" style={{ "--d": "300ms" } as React.CSSProperties}>
            <div className="flex flex-wrap items-center gap-3 px-4 py-3 border-b hairline">
              {/* The title moved to the command row above. Repeating it here is the same duplication this
                  change set out to remove, so what stays is only the band's own description. */}
              <div className="text-ink-3 text-xs">{activeBand?.hint}</div>
              <div className="flex items-center gap-0.5 ml-auto font-mono text-[11px] bg-bg-2 rounded p-0.5">
                {BANDS.map((b) => (
                  <button key={b.key} onClick={() => setStates(b.key)} title={b.hint}
                    className={`px-2.5 py-1 rounded transition-colors ${states === b.key ? "bg-accent text-white" : "text-ink-3 hover:text-ink"}`}>
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
              <SkeletonRows rows={8} cols="grid-cols-[32px_76px_1fr_150px_110px]" />
            ) : rows.length ? (
              <table className="w-full text-sm table-fixed">
                {/* Explicit columns. Auto layout was resolving the header and the body to different widths
                    once the thumbnail column arrived, leaving every heading sitting over the wrong column.
                    Stating the widths once removes the inference entirely. */}
                <colgroup>
                  <col style={{ width: 44 }} />
                  <col style={{ width: 84 }} />
                  <col />
                  <col style={{ width: 150 }} />
                  <col style={{ width: 120 }} />
                </colgroup>
                <thead className="text-ink-3 font-mono text-[11px] uppercase border-b hairline sticky top-0 bg-panel z-10">
                  <tr>
                    <th className="px-3 py-2 w-8"><input type="checkbox" checked={allSelected} onChange={selectAll} /></th>
                    <th className="text-left font-normal px-2 py-2 w-[76px]">frame</th>
                    <th className="text-left font-normal px-3 py-2">object &middot; why it needs you</th>
                    <th className="text-left font-normal px-3 py-2 w-36">confidence</th>
                    <th className="text-left font-normal px-3 py-2 w-28">state</th>
                  </tr>
                </thead>
                <tbody>
                  {shownRows.map((r, i) => (
                    <tr key={r.object_id} onClick={() => { setCursor(i); open(r); }} onMouseEnter={() => setCursor(i)}
                      data-active={i === cursor}
                      className={`border-b hairline cursor-pointer transition-colors ${sel.has(r.object_id) ? "bg-accent/5" : i === cursor ? "bg-bg-2" : "hover:bg-bg-2/60"}`}>
                      {/* The worst thing about this row, on its edge, readable before any word is. */}
                      <td className={`px-3 py-2 align-top border-l-[3px] ${SEVERITY_STRIPE[worstSeverity(r)]}`}
                        onClick={(e) => e.stopPropagation()}>
                        <input type="checkbox" checked={sel.has(r.object_id)} onChange={() => toggle(r.object_id)} />
                      </td>
                      {/* A crop decides a good share of these without opening anything, and the endpoint
                          behind it is already serving. Lazy so a 200-row queue does not fetch 200 images
                          before the first one is on screen. */}
                      <td className="px-2 py-2 align-top">
                        <img src={`/api/objects/${r.object_id}/crop`} alt="" loading="lazy" decoding="async"
                          className="w-[64px] h-[40px] object-cover rounded border border-line bg-bg-2"
                          onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = "hidden"; }} />
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-ink">{r.class_name}</span>
                          <ObjectSourceBadge source={r.source} importFormat={r.import_format} />
                        </div>
                        <WhyChips row={r} />
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
              <span><span className="text-pass">A</span> accept</span>
              <span><span className="text-block">R</span> reject</span>
              {sel.size ? <span className="text-accent">{sel.size} selected</span> : null}
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
