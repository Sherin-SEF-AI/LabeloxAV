"use client";

export const dynamic = "force-dynamic";
import { Suspense, useCallback, useMemo, useState } from "react";
import { T } from "@/lib/i18n";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, humanizeError } from "@/lib/api";
import { enqueueSessions, startAutolabel } from "@/lib/uploadManager";
import { watchAutolabelJob, watchImportJob } from "@/lib/useEventStream";
import type { SessionRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import Pager from "@/components/shell/Pager";
import { usePager } from "@/lib/usePager";
import EmptyState from "@/components/ui/EmptyState";
import LoadState from "@/components/shell/LoadState";
import { fromNs, num, pct, plural, shortId } from "@/lib/format";
import { useQueryFlag } from "@/lib/useQueryParam";
import { invalidate, useQuery } from "@/lib/useQuery";

// The "open annotation" browser: every capture session as a card with review progress and a
// state breakdown. "Open" jumps to the first frame; "Resume queue" jumps to the highest-priority
// unreviewed object. Color only encodes state (pass/warn/accent/block).

type SessionStats = {
  session_id: string;
  frames: number;
  objects: number;
  by_state: Record<string, number>;
  done: number;
  progress: number;
};

// Ordered by_state keys shown as the card's state breakdown. Each state keeps the one signal colour StateBadge
// gives it, drawn as a dot beside a readable label and count rather than an 11px outlined chip.
const CHIPS: { key: string; label: string; dot: string }[] = [
  { key: "review", label: "Review", dot: "bg-warn" },
  { key: "annotate", label: "Annotate", dot: "bg-accent" },
  { key: "accepted", label: "Accepted", dot: "bg-pass" },
  { key: "auto_accept", label: "Auto accepted", dot: "bg-pass/60" },
  { key: "rejected", label: "Rejected", dot: "bg-block" },
];

function ProgressBar({ progress }: { progress: number }) {
  const p = Math.round(progress * 100);
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between text-[12px]">
        <span className="text-ink-3">Reviewed</span>
        <span className={`tabular-nums font-medium ${p >= 100 ? "text-pass" : "text-ink-2"}`}>{pct(progress)}</span>
      </div>
      <div className="h-1.5 bg-bg-2 rounded-full overflow-hidden" role="progressbar" aria-valuenow={p}
        aria-valuemin={0} aria-valuemax={100} aria-label="reviewed">
        <div className="h-full bg-pass rounded-full" style={{ width: `${Math.min(100, progress * 100)}%` }} />
      </div>
    </div>
  );
}

function SessionCard({
  session,
  stats,
  onOpen,
  onResume,
  rigMode,
  selected,
  onSelect,
}: {
  session: SessionRow;
  stats: SessionStats | undefined;
  onOpen: (s: SessionRow) => void;
  onResume: (s: SessionRow) => void;
  rigMode: boolean;
  selected: boolean;
  onSelect: (id: string, on: boolean) => void;
}) {
  const name = session.vehicle_id || shortId(session.session_id);
  const when = session.start_ts_ns ? fromNs(session.start_ts_ns) : null;
  return (
    <div className={`panel p-4 flex flex-col gap-3 transition-colors ${selected ? "border-accent/60 bg-accent/[0.04]" : ""}`}>
      <div className="flex items-start gap-2.5 min-w-0">
        {/* Ticking a card is what makes the batch autolabel reachable. Before this the only way to label a
            session already imported was one at a time, through a dropdown on the home page. */}
        <input
          type="checkbox"
          checked={selected}
          onChange={(e) => onSelect(session.session_id, e.target.checked)}
          title="select for batch autolabel"
          aria-label={`Select ${name} for batch autolabel`}
          className="shrink-0 mt-1 accent-accent"
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline justify-between gap-2 min-w-0">
            <div className="text-[14px] font-semibold text-ink truncate" title={session.vehicle_id}>{name}</div>
            {session.city && <div className="text-[12px] text-ink-3 truncate shrink-0">{session.city}</div>}
          </div>
          <div className="mt-0.5 flex items-center gap-2 text-[11.5px] text-ink-3 min-w-0">
            {when && when !== "-" && <span className="truncate">{when}</span>}
            <span className="font-mono shrink-0" title={session.session_id}>{shortId(session.session_id)}</span>
          </div>
        </div>
      </div>

      <div className="text-[12.5px] text-ink-2 flex items-center gap-1.5">
        {stats ? (
          <>
            <span className="tabular-nums">{plural(stats.frames, "frame")}</span>
            <span className="text-ink-3" aria-hidden>·</span>
            <span className="tabular-nums">{plural(stats.objects, "object")}</span>
          </>
        ) : <span className="skeleton h-3 w-32" />}
      </div>

      {stats ? (
        <ProgressBar progress={stats.progress} />
      ) : (
        <span className="skeleton h-1.5 w-full" />
      )}

      {stats && CHIPS.some((c) => (stats.by_state[c.key] ?? 0) > 0) && (
        <div className="flex flex-wrap items-center gap-x-3.5 gap-y-1 text-[12px]">
          {CHIPS.filter((c) => (stats.by_state[c.key] ?? 0) > 0).map((c) => (
            <span key={c.key} className="inline-flex items-center gap-1.5 whitespace-nowrap">
              <span className={`w-2 h-2 rounded-full ${c.dot}`} aria-hidden />
              <span className="text-ink-3">{c.label}</span>
              <span className="text-ink tabular-nums font-medium">{num(stats.by_state[c.key])}</span>
            </span>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2 pt-1 mt-auto">
        <button
          onClick={() => onOpen(session)}
          className="btn"
        >
          <T k="action.open">Open</T>
        </button>
        <button
          onClick={() => onResume(session)}
          className="btn"
          data-tip="Jump to the highest-priority object still waiting for review"
        >
          Resume queue
        </button>
        {rigMode && (
          <Link
            href={`/annotate/multicam/${session.session_id}`}
            className="btn btn-on"
          >
            Rig view
          </Link>
        )}
      </div>
    </div>
  );
}

export default function AnnotationsPage() {
  // usePager reads the query string via useSearchParams, which forces a client-side-render bailout that Next
  // requires be under a Suspense boundary or the static prerender of this route fails the build.
  return <Suspense fallback={null}><AnnotationsBody /></Suspense>;
}

function AnnotationsBody() {
  // Label > Multi-camera deep-links ?rig=1; surface the rig workspace entry for the listed sessions
  const rigMode = useQueryFlag("rig");
  const router = useRouter();
  const { offset, limit, setOffset } = usePager(24);
  // Paginated: the corpus has 2000+ sessions, so the list pages through them with a real total instead
  // of silently showing only the first window. Keyed by the window, so paging back, or returning from a
  // session, shows the page that was there while it refreshes. A failed load says so, with a retry, rather
  // than reading as a corpus with no sessions in it.
  const pageQ = useQuery(["sessionsPage", { limit, offset }], () => api.sessionsPage({ limit, offset }),
    { keepPrevious: true });
  const sessions: SessionRow[] = useMemo(() => pageQ.data?.sessions ?? [], [pageQ.data]);
  const total = pageQ.data?.total ?? 0;
  // One stats read per listed session, after the page itself. A session whose stats fail simply has none.
  const ids = useMemo(() => sessions.map((s) => s.session_id), [sessions]);
  const statsQ = useQuery(pageQ.data ? ["sessionStats", ids] : null, async () => {
    const results = await Promise.all(ids.map((id) => api.sessionStats(id).catch(() => null)));
    const map: Record<string, SessionStats> = {};
    results.forEach((st) => {
      if (st) map[st.session_id] = st;
    });
    return map;
  }, { keepPrevious: true });
  const stats: Record<string, SessionStats> = statsQ.data ?? {};
  // Selection for the batch autolabel. Kept here rather than in the manager: which sessions somebody has
  // ticked is a property of this screen, and only the ones they confirm are handed over.
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [labelBusy, setLabelBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  function flash(text: string) {
    setMsg(text);
    setTimeout(() => setMsg(null), 2500);
  }

  const toggle = useCallback((id: string, on: boolean) => {
    setSel((cur) => {
      const next = new Set(cur);
      if (on) next.add(id); else next.delete(id);
      return next;
    });
  }, []);

  async function onAutolabelSelected() {
    if (!sel.size || labelBusy) return;
    setLabelBusy(true);
    try {
      const picked = sessions.filter((s) => sel.has(s.session_id));
      const n = enqueueSessions(picked.map((s) => ({
        sessionId: s.session_id,
        name: s.vehicle_id ? `${s.vehicle_id} ${s.session_id.slice(0, 8)}` : s.session_id.slice(0, 8),
      })));
      if (!n) { setMsg("A batch is already running; watch it in the top bar"); return; }
      // Runs through the same module-scoped queue as an upload batch, so it survives leaving this page and
      // reports in the same top-bar indicator rather than inventing a second progress surface.
      await startAutolabel({
        upload: api.uploadMultipart,
        startImport: api.startImport,
        watchImport: watchImportJob,
        firstFrame: api.firstFrame,
        humanizeError,
        startAutolabel: (sessionId: string) => api.startAutolabel(sessionId),
        watchAutolabel: watchAutolabelJob,
      });
      setSel(new Set());
      // The batch has written objects: the cards' counts, the jobs table and home are all stale.
      void invalidate("sessionStats", "jobs", "/api/home");
    } finally {
      setLabelBusy(false);
    }
  }

  async function onOpen(s: SessionRow) {
    try {
      const { frame_id } = await api.firstFrame(s.session_id);
      router.push("/frame/" + frame_id);
    } catch {
      flash("This session has no frames");
    }
  }

  async function onResume(s: SessionRow) {
    try {
      const rows = await api.triage({
        session_id: s.session_id,
        states: "review,annotate",
        limit: "1",
      });
      if (rows[0]) {
        router.push("/frame/" + rows[0].frame_id + "?focus=" + rows[0].object_id);
      } else {
        flash("Nothing left to review in this session");
      }
    } catch {
      flash("Could not load the review queue");
    }
  }

  return (
    <PageShell
      active="ANNOTATIONS"
      title="Sessions"
      meta={total > 0 ? <Pager offset={offset} limit={limit} total={total} onOffset={setOffset} /> : undefined}
      right={
        msg ? (
          <span className="panel px-3 py-1.5 text-[12px] text-warn">{msg}</span>
        ) : undefined
      }
      primaryAction={
        <div className="flex items-center gap-2">
          {sel.size > 0 && (
            <>
              <button type="button" onClick={() => setSel(new Set())} className="btn reveal">
                Clear selection
              </button>
              <button
                onClick={onAutolabelSelected}
                disabled={labelBusy}
                className="btn btn-on reveal"
              >
                {labelBusy && <span className="running-dot mr-1.5 align-middle" />}
                Autolabel {plural(sel.size, "session")}
              </button>
            </>
          )}
          <Link
            href="/annotate/new"
            className="btn btn-primary"
          >
            New annotation
          </Link>
        </div>
      }
    >
      <div className="page-wrap space-y-4">
        {sessions.length === 0 ? (
          <LoadState {...pageQ.loadState} rows={6}>
            <div className="panel">
              <EmptyState
                icon="folder"
                title="No sessions yet"
                action={<Link href="/annotate/new" className="btn btn-primary">Upload a drive</Link>}
              >
                A session is one recorded drive. Upload a video, image folder or MCAP file and it appears here,
                ready to autolabel and review.
              </EmptyState>
            </div>
          </LoadState>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {sessions.map((s) => (
              <SessionCard
                key={s.session_id}
                session={s}
                stats={stats[s.session_id]}
                onOpen={onOpen}
                onResume={onResume}
                rigMode={rigMode}
                selected={sel.has(s.session_id)}
                onSelect={toggle}
              />
            ))}
          </div>
        )}
      </div>
    </PageShell>
  );
}
