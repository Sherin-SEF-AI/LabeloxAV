"use client";

export const dynamic = "force-dynamic";
import { Suspense, useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, humanizeError } from "@/lib/api";
import { enqueueSessions, startAutolabel } from "@/lib/uploadManager";
import { watchAutolabelJob, watchImportJob } from "@/lib/useEventStream";
import type { SessionRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import Pager from "@/components/shell/Pager";
import { usePager } from "@/lib/usePager";
import { StateBadge } from "@/components/StateBadge";
import { useQueryFlag } from "@/lib/useQueryParam";

// The "open annotation" browser: every capture session as a card with review progress and a
// state breakdown. "open" jumps to the first frame; "resume queue" jumps to the highest-priority
// unreviewed object. Color only encodes state (pass/warn/accent/block).

type SessionStats = {
  session_id: string;
  frames: number;
  objects: number;
  by_state: Record<string, number>;
  done: number;
  progress: number;
};

// Ordered by_state keys to surface as StateBadge chips. StateBadge owns each state's signal color.
const CHIPS: { key: string }[] = [
  { key: "review" },
  { key: "annotate" },
  { key: "accepted" },
  { key: "auto_accept" },
  { key: "rejected" },
];

function ProgressBar({ progress }: { progress: number }) {
  return (
    <div className="space-y-1">
      <div className="h-1.5 bg-line rounded">
        <div className="h-1.5 bg-pass rounded" style={{ width: progress * 100 + "%" }} />
      </div>
      <div className="font-mono text-[11px] text-ink-3">{Math.round(progress * 100)}% reviewed</div>
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
  return (
    <div className={`panel p-3 space-y-2 transition-colors ${selected ? "border-accent/60" : ""}`}>
      <div className="flex items-baseline justify-between gap-2 min-w-0">
        {/* Ticking a card is what makes the batch autolabel reachable. Before this the only way to label a
            session already imported was one at a time, through a dropdown on the home page. */}
        <input
          type="checkbox"
          checked={selected}
          onChange={(e) => onSelect(session.session_id, e.target.checked)}
          title="select for batch autolabel"
          className="shrink-0 accent-accent"
        />
        <div className="font-mono text-sm text-ink truncate flex-1 min-w-0" title={session.vehicle_id}>
          {session.vehicle_id}
        </div>
        <div className="font-mono text-xs text-ink-3 truncate">{session.city ?? ""}</div>
      </div>

      <div className="font-mono text-[11px] text-ink-3 flex items-center gap-3">
        <span>{stats ? `${stats.frames} frames` : "..."}</span>
        <span>{stats ? `${stats.objects} objects` : ""}</span>
      </div>

      {stats ? (
        <ProgressBar progress={stats.progress} />
      ) : (
        <div className="font-mono text-[11px] text-ink-3">...</div>
      )}

      <div className="flex flex-wrap items-center gap-1.5">
        {stats &&
          CHIPS.filter((c) => (stats.by_state[c.key] ?? 0) > 0).map((c) => (
            <span key={c.key} className="inline-flex items-center gap-1">
              <StateBadge state={c.key} />
              <span className="font-mono text-[10px] text-ink-3">{stats.by_state[c.key]}</span>
            </span>
          ))}
      </div>

      <div className="flex items-center gap-2 pt-1">
        <button
          onClick={() => onOpen(session)}
          className="font-mono text-xs border border-line px-2 py-0.5 hover:border-accent"
        >
          open
        </button>
        <button
          onClick={() => onResume(session)}
          className="font-mono text-xs border border-line px-2 py-0.5 hover:border-accent"
        >
          resume queue
        </button>
        {rigMode && (
          <Link
            href={`/annotate/multicam/${session.session_id}`}
            className="font-mono text-xs border border-accent text-accent px-2 py-0.5 hover:bg-accent/10"
          >
            rig view
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
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  // Selection for the batch autolabel. Kept here rather than in the manager: which sessions somebody has
  // ticked is a property of this screen, and only the ones they confirm are handed over.
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [labelBusy, setLabelBusy] = useState(false);
  const [total, setTotal] = useState(0);
  const [stats, setStats] = useState<Record<string, SessionStats>>({});
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        // Paginated: the corpus has 2000+ sessions, so the list pages through them with a real total instead
        // of silently showing only the first window.
        const page = await api.sessionsPage({ limit, offset });
        setSessions(page.sessions);
        setTotal(page.total);
        const results = await Promise.all(
          page.sessions.map((s) => api.sessionStats(s.session_id).catch(() => null)),
        );
        const map: Record<string, SessionStats> = {};
        results.forEach((st) => {
          if (st) map[st.session_id] = st;
        });
        setStats(map);
      } catch {
        /* ignore: leave the grid empty rather than crash */
      } finally {
        setLoading(false);
      }
    })();
  }, [offset, limit]);

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
      if (!n) { setMsg("a batch is already running; watch it in the top bar"); return; }
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
    } finally {
      setLabelBusy(false);
    }
  }

  async function onOpen(s: SessionRow) {
    try {
      const { frame_id } = await api.firstFrame(s.session_id);
      router.push("/frame/" + frame_id);
    } catch {
      flash("no frames in this session");
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
        flash("queue empty");
      }
    } catch {
      flash("could not load queue");
    }
  }

  return (
    <PageShell
      active="ANNOTATIONS"
      title="annotations"
      meta={total > 0 ? <Pager offset={offset} limit={limit} total={total} onOffset={setOffset} /> : undefined}
      right={
        msg ? (
          <span className="panel px-3 py-1.5 font-mono text-[11px] text-warn">{msg}</span>
        ) : undefined
      }
      primaryAction={
        <div className="flex items-center gap-2">
          {sel.size > 0 && (
            <button
              onClick={onAutolabelSelected}
              disabled={labelBusy}
              className="reveal font-mono text-xs border border-accent/50 text-accent px-3 py-1 rounded hover:bg-accent/10 disabled:opacity-50 transition-colors"
            >
              {labelBusy && <span className="running-dot mr-1.5 align-middle" />}
              autolabel {sel.size} session{sel.size === 1 ? "" : "s"}
            </button>
          )}
          <Link
            href="/annotate/new"
            className="font-mono text-xs border border-accent text-accent px-3 py-1 hover:bg-accent/10"
          >
            + new annotation
          </Link>
        </div>
      }
    >
      <div className="p-4 space-y-4">
        {sessions.length === 0 ? (
          <div className="panel px-3 py-10 text-center space-y-3">
            <div className="font-mono text-xs text-ink-3">
              {loading ? "loading sessions..." : "no annotation sessions yet"}
            </div>
            {!loading && (
              <Link
                href="/annotate/new"
                className="inline-block font-mono text-xs border border-accent text-accent px-3 py-1 hover:bg-accent/10"
              >
                + new annotation
              </Link>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
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
