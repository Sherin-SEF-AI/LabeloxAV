"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { T } from "@/lib/i18n";
import { useRouter } from "next/navigation";
import { useJobStream } from "@/lib/useEventStream";
import { api , humanizeError } from "@/lib/api";
import { toast, toastError, toastSuccess } from "@/lib/toast";
import type { SessionRow, TriageFlag, TriageRow, TriageSeverity } from "@/lib/types";
import { StateBadge } from "@/components/StateBadge";
import PageShell from "@/components/shell/PageShell";
import QueueOrdering from "@/components/queue/QueueOrdering";
import LoadState from "@/components/shell/LoadState";
import CorrectionModal from "@/components/CorrectionModal";
import { SkeletonRows, Spinner } from "@/components/Spinner";
import { ObjectSourceBadge } from "@/components/SourceBadge";
import { objectSource, sessionOrigin } from "@/lib/source";
import { acceptState, getUser } from "@/lib/user";
import Icon from "@/components/shell/Icon";
import { invalidate, useQuery } from "@/lib/useQuery";
import EmptyState from "@/components/ui/EmptyState";
import { humanize, num } from "@/lib/format";

const BANDS = [
  { key: "review,annotate", label: "My queue", hint: "Everything assigned to you" },
  { key: "review", label: "To review", hint: "Model proposals waiting for a decision" },
  { key: "annotate", label: "To annotate", hint: "Frames that still need labels" },
  { key: "submitted", label: "QA queue", hint: "Submitted work waiting for approval" },
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

// A key cap for the shortcut hints in the footer.
const KBD = "inline-block min-w-[18px] text-center px-1 py-px rounded border hairline bg-head text-[11px] text-ink-2 font-mono";

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
  // Only what is probably wrong or worth a look gets a chip. Low-severity flags are context ("rare class"),
  // and a chip per fact made every row a strip of four identical boxes; they read as one quiet line instead.
  const loud = flags.filter((f) => f.severity !== "low");
  const quiet = flags.filter((f) => f.severity === "low");
  return (
    <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
      {loud.map((f, i) => (
        <span key={`${f.code}-${i}`}
          className={`text-[11px] leading-none px-1.5 py-1 rounded border ${SEVERITY_TONE[f.severity]}`}>
          {f.label}
        </span>
      ))}
      {quiet.length > 0 && (
        <span className="text-[11.5px] text-ink-3">{quiet.map((f) => f.label).join(" · ")}</span>
      )}
    </div>
  );
}

// The detector's score as a percentage. A score of zero on a row is not "0% sure": it is an object no model
// proposed (a person drew it, or it came from an import), so it says that rather than drawing an empty bar.
function Confidence({ conf }: { conf: number }) {
  if (conf == null || !Number.isFinite(conf) || conf <= 0) {
    return <span className="text-[12px] text-ink-3" title="The detector gave this object no confidence (0)">No score</span>;
  }
  const color = conf >= 0.95 ? "bg-pass" : conf >= 0.6 ? "bg-warn" : "bg-accent";
  return (
    <span className="inline-flex items-center gap-2 text-[12.5px] tabular-nums text-ink-2" title={`Detector confidence ${conf.toFixed(2)}`}>
      <span className="w-14 h-1.5 rounded-full overflow-hidden bg-line relative">
        <span className={`absolute left-0 top-0 h-full ${color}`} style={{ width: `${Math.min(1, conf) * 100}%` }} />
      </span>
      {Math.round(conf * 100)}%
    </span>
  );
}

export default function HomePage() {
  const router = useRouter();
  const [session, setSession] = useState<string>("");
  const [states, setStates] = useState<string>("review,annotate");
  const [cursor, setCursor] = useState(0);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [bulkClass, setBulkClass] = useState("");
  const [corr, setCorr] = useState<{ objectId: string; old: string; to: string } | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [role, setRole] = useState<string | undefined>(undefined);
  const [showActions, setShowActions] = useState(false);
  type Ingest = { active: boolean; finished: boolean; done: number; total: number; current: string | null; frames: number };
  const [liveIngest, setLiveIngest] = useState<Ingest | null>(null);
  const [srcFilter, setSrcFilter] = useState<"all" | "app" | "imported">("all");

  // a flywheel worklist deep-link (?flywheel=<cycle>) scopes the queue to that cycle's dispatched objects
  const fw = useMemo(
    () => (typeof window !== "undefined" ? new URLSearchParams(window.location.search).get("flywheel") : null), []);
  const params = useMemo(() => {
    const p: Record<string, string> = { states, limit: "200" };
    if (session) p.session_id = session;
    if (fw) { p.flywheel = fw; p.states = "review,annotate"; }
    return p;
  }, [states, session, fw]);
  // Keyed by the filter, so coming back from a frame shows the queue that was here while it refreshes,
  // and the rows on screen stay up while another band or session loads.
  const triageQ = useQuery(["triage", params], () => api.triage(params), { keepPrevious: true });
  const rows: TriageRow[] = useMemo(() => triageQ.data ?? [], [triageQ.data]);
  const loading = triageQ.validating;
  // There was a finally and no catch, so a dropped request left rows empty and the page said
  // "Queue is clear" - it told an annotator their shift was over because a fetch failed. An empty
  // queue is a real and meaningful state here, which is exactly why it must not be what a failure
  // looks like. A request in flight supersedes the last failure.
  const err = triageQ.validating ? null : triageQ.error ?? null;
  const load = triageQ.reload;
  useEffect(() => { setCursor(0); }, [triageQ.data]);
  // The first answer after arriving on the page. Until it lands the rows may be the cached ones from
  // the last visit, which is fine to look at and not to rule on from the keyboard: the fresh list
  // replaces them a moment later and the row under the cursor would not be the one that was read.
  const [fresh, setFresh] = useState(false);
  useEffect(() => { if (!triageQ.validating) setFresh(true); }, [triageQ.validating]);

  // The dropdown needs the session rows; nothing on this page shows a fleet total any more. Both of
  // these fill controls, and an empty control is what their failure has always looked like here.
  const sessionsQ = useQuery(["sessions"], () => api.sessions());
  const sessions: SessionRow[] = useMemo(() => sessionsQ.data ?? [], [sessionsQ.data]);
  const ontologyQ = useQuery(["ontology"], () => api.ontology());
  const classes: string[] = useMemo(() => (ontologyQ.data?.classes ?? []).map((c) => c.name), [ontologyQ.data]);
  useEffect(() => { setRole(getUser()?.role); }, []);
  // Ingest progress rides the shared job stream. It used to poll every three seconds on every open tab,
  // and the endpoint behind it scraped a log file with a regex, which reports a quiet system rather than
  // an error the moment the API runs in a different container from the ingest script.
  const jobStream = useJobStream();
  const ingestQ = useQuery(["ingestProgress"], () => api.ingestProgress());
  useEffect(() => {
    const live = jobStream.data?.ingest?.[0];
    if (live) setLiveIngest(live as Ingest);
  }, [jobStream.data]);
  const ingest: Ingest | null = liveIngest ?? (ingestQ.data as Ingest | undefined) ?? null;
  useEffect(() => { setSel(new Set()); }, [states, session]);

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
        setSel(new Set());
        void load();
        // Home counts what is waiting for review.
        void invalidate("/api/home");
        // The server returns a handle that takes the whole batch back in one move, and ToastAction was
        // built for exactly this - its comment says a batch that changed fifty objects and offers no way
        // back asks the operator to be certain before acting, where one that can be undone from the
        // confirmation asks them only to look. It had zero call sites. This is the surface that needed
        // it: a mis-picked class across sixty selected objects was sixty manual corrections.
        const changed = `${humanize(action)}: ${r.updated} object${r.updated === 1 ? "" : "s"}`;
        if (r.run_id) {
          toast(changed, "success", 5000, {
            label: "Undo",
            run: async () => {
              try {
                const back = await api.agentRevert(r.run_id as string);
                toastSuccess(`Undone: ${back.reverted} object${back.reverted === 1 ? "" : "s"} restored`);
              } catch (e) {
                // Toaster dismisses before awaiting this, so a failure has nowhere else to surface.
                toastError(`Undo failed: ${humanizeError(e)}`);
              }
              void load();
              void invalidate("/api/home");
            },
          });
        } else {
          toast(changed, "success");
        }
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
      setMsg(r.status === "queued-cloud" ? "Auto-label queued on the cloud GPU. Follow it in Jobs." : "Auto-label queued. Follow it in Jobs.");
      void invalidate("jobs", "/api/home");
    } catch (e) {
      setMsg(humanizeError(e).includes("503") ? "The GPU is busy training. Try again when it finishes." : humanizeError(e));
    }
  }, [session]);

  const vlmQa = useCallback(async () => {
    if (!session) return;
    try {
      await api.startVlmQa(session);
      setMsg("AI check running. It sends labels that look wrong to the QA queue and fills in attributes.");
      void invalidate("jobs", "/api/home");
    } catch (e) {
      setMsg(humanizeError(e).includes("503") ? "The GPU is busy training. Try again when it finishes." : humanizeError(e));
    }
  }, [session]);

  const recognizeSigns = useCallback(async () => {
    if (!session) return;
    try {
      const r = await api.recognizeSigns(session);
      setMsg(`Recognised ${r.recognized} signs; ${r.text_bearing} with text were sent for reading.`);
      void load();
    } catch (e) {
      setMsg(humanizeError(e));
    }
  }, [session, load]);

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
        if (!fresh) return;
        const row = rows[cursor];
        if (!sel.size && !row) return;
        const targets = sel.size ? sel : new Set([row.object_id]);
        e.preventDefault();
        void (async () => {
          try {
            const res = await api.bulkReview([...targets], e.key === "a" ? "accept" : "reject",
              undefined, e.key === "a" ? acceptState(role) : undefined);
            setMsg(`${e.key === "a" ? "Accepted" : "Rejected"} ${res.updated}`);
            setSel(new Set());
            void load();
            void invalidate("/api/home");
          } catch (err) {
            setMsg(humanizeError(err));
          }
        })();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows, cursor, sel, role, open, load, fresh]);

  const cities = useMemo(() => new Set(sessions.map((s) => s.city).filter(Boolean)).size, [sessions]);
  const activeBand = BANDS.find((b) => b.key === states);
  const shownRows = useMemo(
    () => (srcFilter === "all" ? rows : rows.filter((r) => objectSource(r.source, r.import_format).kind === srcFilter)),
    [rows, srcFilter],
  );
  const importedCount = useMemo(() => rows.filter((r) => r.source === "imported").length, [rows]);
  const SRC_FILTERS: { key: "all" | "app" | "imported"; label: string }[] = [
    { key: "all", label: "All" },
    { key: "app", label: "Your work" },
    { key: "imported", label: "Imported" },
  ];

  return (
    <PageShell
      active="My queue"
      subtitle="Objects waiting for a decision, the likeliest mistakes first"
      right={<span className="flex items-center gap-3">
        <QueueOrdering />
        {loading ? <Spinner label="loading" /> : <span className="text-[12px] text-ink-2 tabular-nums">{num(rows.length)} in queue</span>}
      </span>}
    >
      <div className="h-full flex flex-col">
        <div className="page-wrap !pb-4 flex flex-col gap-3 flex-1 min-h-0">
          {/* Live ingest progress (dashcam batch) */}
          {ingest && (ingest.active || (ingest.done > 0 && ingest.done < ingest.total)) ? (
            <div className="panel px-4 py-3 border-accent/40 shrink-0">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="inline-block w-3.5 h-3.5 rounded-full border-2 border-line border-t-accent animate-spin" aria-hidden />
                  <span className="text-ink font-medium">Bringing in dashcam videos</span>
                </div>
                <span className="text-[12px] text-ink-3 tabular-nums">
                  {ingest.done}/{ingest.total} videos &middot; {ingest.frames.toLocaleString()} frames
                </span>
              </div>
              <div className="h-1.5 bg-bg-2 rounded overflow-hidden">
                <div className="h-full bg-accent transition-all duration-500"
                  style={{ width: `${ingest.total ? Math.round((ingest.done / ingest.total) * 100) : 0}%` }} />
              </div>
              {ingest.current ? (
                <div className="text-[12px] text-ink-3 mt-1 truncate">Now: {ingest.current}</div>
              ) : null}
            </div>
          ) : ingest?.finished && ingest.frames > 0 ? (
            <div className="panel px-4 py-2 border-pass/40 flex items-center gap-2 shrink-0">
              <span className="w-2 h-2 rounded-full bg-pass" />
              <span className="text-ink text-[13px]">Brought in {ingest.done} dashcam videos &middot; {ingest.frames.toLocaleString()} frames ready</span>
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
            <div className="text-[12px] text-ink-3 tabular-nums flex items-center gap-2">
              <span>{loading ? "..." : rows.length.toLocaleString()} objects</span>
              {ingest?.frames ? <><span className="text-line">/</span>
                <span>{ingest.frames.toLocaleString()} frames</span></> : null}
              {cities > 0 && <><span className="text-line">/</span><span>{cities} cit{cities === 1 ? "y" : "ies"}</span></>}
            </div>
            <div className="ml-auto flex items-center gap-2">
              <button onClick={() => setShowActions((v) => !v)} className="btn" aria-expanded={showActions}>
                Session actions
              </button>
              <button
                onClick={() => open(shownRows[0] ?? rows[0])}
                disabled={!rows.length}
                className="btn btn-primary disabled:cursor-not-allowed"
              >
                <Icon name="confirm" size={14} />
                {rows.length ? "Review next" : err != null ? "Queue unavailable" : loading ? "Loading" : "Queue is clear"}
              </button>
            </div>
          </div>

          {/* Review queue */}
          <div id="queue" className="fade-up panel flex-1 min-h-0 flex flex-col overflow-hidden" style={{ "--d": "300ms" } as React.CSSProperties}>
            <div className="flex flex-wrap items-center gap-3 px-4 py-3 border-b hairline">
              {/* The title moved to the command row above. Repeating it here is the same duplication this
                  change set out to remove, so what stays is only the band's own description. */}
              <div className="text-ink-3 text-[12.5px]">{activeBand?.hint}</div>
              <div className="flex items-center gap-0.5 ml-auto text-[12px] bg-bg-2 rounded p-0.5">
                {BANDS.map((b) => (
                  <button key={b.key} onClick={() => setStates(b.key)} title={b.hint}
                    className={`px-2.5 py-1 rounded transition-colors ${states === b.key ? "bg-accent text-bg font-medium" : "text-ink-3 hover:text-ink"}`}>
                    {b.label}
                  </button>
                ))}
              </div>
              <select value={session} onChange={(e) => setSession(e.target.value)}
                aria-label="Session" className="field text-[12.5px] max-w-[260px]">
                <option value="">All sessions ({sessions.length})</option>
                {[...sessions]
                  .sort((a, b) => (b.route ?? "").localeCompare(a.route ?? "") || b.start_ts_ns - a.start_ts_ns)
                  .map((s) => (
                    <option key={s.session_id} value={s.session_id}>
                      {s.origin === "imported" ? "Imported: " : ""}
                      {s.route
                        ? `${s.route}${s.city ? ` · ${s.city}` : ""}`
                        : `${s.vehicle_id} · ${s.city ?? "?"} · #${s.session_id.slice(0, 4)}`}
                    </option>
                  ))}
              </select>
            </div>

            {/* Source filter: separate imported public-dataset labels from your own work */}
            <div className="px-4 py-2 border-b hairline flex flex-wrap items-center gap-2 text-[12px]">
              <span className="text-ink-3 text-[12px] mr-0.5">Source</span>
              {SRC_FILTERS.map((sf) => (
                <button key={sf.key} onClick={() => setSrcFilter(sf.key)} aria-pressed={srcFilter === sf.key}
                  className={`chip ${srcFilter === sf.key ? "chip-on" : ""}`}>
                  {sf.label}
                </button>
              ))}
              {importedCount > 0 ? (
                <span className="text-ink-3 ml-1">{importedCount} of {rows.length} shown came from imported datasets, not your team&apos;s labels</span>
              ) : null}
            </div>

            {/* AI helpers for the selected session (secondary, tucked away) */}
            {session ? (
              <div className="px-4 py-2 border-b hairline flex flex-wrap items-center gap-2 text-[12px]">
                <button onClick={() => setShowActions((v) => !v)} className="text-ink-2 hover:text-ink" aria-expanded={showActions}>
                  <span aria-hidden>{showActions ? "▾" : "▸"}</span> AI helpers for this session
                </button>
                {showActions ? (
                  <>
                    <button onClick={autolabel} className="btn" data-tip="Run the detectors on this session">Auto-label</button>
                    <button onClick={recognizeSigns} className="btn" data-tip="Give each traffic sign its specific type">Recognise signs</button>
                    <button onClick={vlmQa} className="btn" data-tip="An AI model looks for labels that are probably wrong">AI check</button>
                  </>
                ) : null}
              </div>
            ) : null}

            {msg ? <div role="status" className="px-4 py-2 text-[12.5px] text-ink-2 bg-head/50 border-b hairline">{msg}</div> : null}

            {/* Bulk action bar */}
            {sel.size > 0 && (
              <div className="sticky top-0 z-20 flex flex-wrap items-center gap-2 px-4 py-2 bg-panel border-b hairline text-[12.5px]">
                <span className="text-ink font-medium mr-1">{sel.size} selected</span>
                <button onClick={() => bulk("accept", undefined, acceptState(role))} className="btn text-pass border-pass/45"
                  title={role === "annotator" ? "Submit for QA" : "Approve (accept)"}>
                  {role === "annotator" ? "Submit" : "Approve"}
                </button>
                <button onClick={() => bulk("reject")} className="btn btn-danger"><T k="action.reject">Reject</T></button>
                <input list="cls" value={bulkClass} onChange={(e) => setBulkClass(e.target.value)} placeholder="Relabel to class"
                  aria-label="Relabel to class" className="field w-44" />
                <datalist id="cls">{classes.map((c) => <option key={c} value={c} />)}</datalist>
                <button onClick={() => bulk("reclassify", bulkClass)} disabled={!classes.includes(bulkClass)}
                  className="btn">Relabel</button>
                {sel.size === 1 && classes.includes(bulkClass) && (
                  <button onClick={() => { const row = rows.find((r) => sel.has(r.object_id)); if (row) setCorr({ objectId: row.object_id, old: row.class_name, to: bulkClass }); }}
                    title="Find similar-looking objects of the old class and fix them too"
                    className="btn">Correct similar &rarr;</button>
                )}
                <button onClick={() => setSel(new Set())} className="link ml-auto"><T k="action.clear">Clear</T></button>
              </div>
            )}

            {/* Table (scrolls inside the panel so the page itself stays fit-to-screen) */}
            <div className="flex-1 min-h-0 overflow-auto">
            {/* The error branch comes first, ahead of both the skeleton and the empty state, because
                either of those reads as "nothing left to do" and a failed fetch is not that. */}
            {err != null ? (
              <LoadState error={err} onRetry={() => void load()} />
            ) : loading && !rows.length ? (
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
                <thead className="border-b hairline sticky top-0 bg-panel z-10">
                  <tr>
                    <th className="px-3 py-2 w-8"><input type="checkbox" checked={allSelected} onChange={selectAll} aria-label="Select all" /></th>
                    <th className="label-caps text-left font-medium px-2 py-2 w-[76px]">Frame</th>
                    <th className="label-caps text-left font-medium px-3 py-2">Object and why it needs you</th>
                    <th className="label-caps text-left font-medium px-3 py-2 w-36">Confidence</th>
                    <th className="label-caps text-left font-medium px-3 py-2 w-28">State</th>
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
                        <input type="checkbox" checked={sel.has(r.object_id)} onChange={() => toggle(r.object_id)}
                          aria-label={`Select ${humanize(r.class_name)}`} />
                      </td>
                      {/* A crop decides a good share of these without opening anything, and the endpoint
                          behind it is already serving. Lazy so a 200-row queue does not fetch 200 images
                          before the first one is on screen. */}
                      <td className="px-2 py-2 align-top">
                        <img src={`/api/objects/${r.object_id}/crop`} alt="" loading="lazy" decoding="async"
                          className="w-[64px] h-[40px] object-cover rounded border border-line skeleton"
                          onLoad={(e) => { e.currentTarget.classList.remove("skeleton"); }}
                          onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = "hidden"; }} />
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex items-center gap-2">
                          <span className="text-[13.5px] text-ink font-medium" title={r.class_name}>{humanize(r.class_name)}</span>
                          <ObjectSourceBadge source={r.source} importFormat={r.import_format} />
                        </div>
                        <WhyChips row={r} />
                      </td>
                      <td className="px-3 py-2 align-top"><Confidence conf={r.conf} /></td>
                      <td className="px-3 py-2 align-top"><StateBadge state={r.state} source={r.source} conf={r.conf} /></td>
                    </tr>
                  ))}
                  {!shownRows.length ? (
                    <tr><td colSpan={5}>
                      <EmptyState compact icon="list"
                        title={srcFilter === "imported" ? "No imported objects here" : srcFilter === "app" ? "None of your team's objects here" : "Nothing in this band"}
                        action={<button className="btn" onClick={() => setSrcFilter("all")}>Show all sources</button>}>
                        The {activeBand?.label.toLowerCase() ?? "queue"} has objects, but none from this source.
                      </EmptyState>
                    </td></tr>
                  ) : null}
                </tbody>
              </table>
            ) : (
              <EmptyState icon="check" title="Your queue is clear"
                action={<button onClick={() => router.push("/import")} className="btn">Import data</button>}>
                Nothing waiting in <span className="text-ink-2">{activeBand?.label.toLowerCase()}</span> right now.
                Try another band or session, or bring in more data.
              </EmptyState>
            )}
            </div>

            {/* Thin footer with the keyboard hint, pinned inside the panel */}
            <div className="shrink-0 border-t hairline px-4 py-2 text-[12px] text-ink-3 flex items-center gap-4">
              <span><kbd className={KBD}>J</kbd> <kbd className={KBD}>K</kbd> move</span>
              <span><kbd className={KBD}>Enter</kbd> open</span>
              <span><kbd className={`${KBD} text-pass`}>A</kbd> accept</span>
              <span><kbd className={`${KBD} text-block`}>R</kbd> reject</span>
              {sel.size ? <span className="text-accent">{sel.size} selected</span> : null}
              <span className="ml-auto">Most uncertain and rarest first</span>
            </div>
          </div>
        </div>
      </div>

      {corr && (
        <CorrectionModal objectId={corr.objectId} kind="class" change={{ old: corr.old, new: corr.to }}
          onClose={() => setCorr(null)}
          onApplied={(n) => { setMsg(`corrected ${n} similar objects`); setCorr(null); setSel(new Set()); void load(); void invalidate("/api/home"); }} />
      )}
    </PageShell>
  );
}
