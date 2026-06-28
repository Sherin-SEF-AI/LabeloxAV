"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api } from "@/lib/api";
import type { SessionRow } from "@/lib/types";
import TopNav from "@/components/TopNav";

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

// Each by_state key maps to a chip color/label. accepted and auto_accept both read as "done" (pass).
const CHIPS: { key: string; label: string; color: string }[] = [
  { key: "review", label: "review", color: "#E3B341" },
  { key: "annotate", label: "annotate", color: "#FF7A2F" },
  { key: "accepted", label: "accepted", color: "#56D364" },
  { key: "auto_accept", label: "auto", color: "#56D364" },
  { key: "rejected", label: "rejected", color: "#F85149" },
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
}: {
  session: SessionRow;
  stats: SessionStats | undefined;
  onOpen: (s: SessionRow) => void;
  onResume: (s: SessionRow) => void;
}) {
  return (
    <div className="panel p-3 space-y-2">
      <div className="flex items-baseline justify-between gap-2 min-w-0">
        <div className="font-mono text-sm text-ink truncate" title={session.vehicle_id}>
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

      <div className="flex flex-wrap gap-1.5">
        {stats &&
          CHIPS.filter((c) => (stats.by_state[c.key] ?? 0) > 0).map((c) => (
            <span
              key={c.key}
              className="font-mono text-[10px] px-1.5 py-0.5 rounded border border-line"
              style={{ color: c.color }}
            >
              {c.label} {stats.by_state[c.key]}
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
      </div>
    </div>
  );
}

export default function AnnotationsPage() {
  const router = useRouter();
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [stats, setStats] = useState<Record<string, SessionStats>>({});
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const rows = await api.sessions();
        setSessions(rows);
        const results = await Promise.all(
          rows.map((s) => api.sessionStats(s.session_id).catch(() => null)),
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
  }, []);

  function flash(text: string) {
    setMsg(text);
    setTimeout(() => setMsg(null), 2500);
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
    <div className="min-h-screen flex flex-col">
      <TopNav active="OPEN" />
      <main className="flex-1 overflow-auto p-4 space-y-4">
        {msg && (
          <div className="panel px-3 py-1.5 font-mono text-[11px] text-warn">{msg}</div>
        )}

        <div className="flex items-center justify-between gap-4">
          <h1 className="font-display text-lg text-ink">annotations</h1>
          <Link
            href="/annotate/new"
            className="font-mono text-xs border border-accent text-accent px-3 py-1 hover:bg-accent/10"
          >
            + new annotation
          </Link>
        </div>

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
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
