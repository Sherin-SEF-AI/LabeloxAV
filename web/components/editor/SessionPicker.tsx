"use client";

// Which drive this frame belongs to, and a way to move to another one.
//
// The editor's top bar printed a frame id and an object count. Across a fleet of 377 sessions that told an
// annotator which frame they were on and not which drive it came from, and the only way to reach a
// different one was to leave the editor for triage and come back in. The session is the unit people
// actually work in, so it belongs in the bar that names the work.
//
// Positioned with useAnchoredDropdown rather than `absolute`, because the top bar is `overflow-x-auto` so
// it can be scrolled on a narrow viewport, and CSS will not let one axis be visible while the other is
// not: overflow-y computes to auto as well and the bar clips its own children to 46 pixels. That is the
// bug the notification panel had, measured as a 132px panel arriving as a five-pixel sliver.
//
// The list is fetched once per page load and shared, so stepping through frames does not refetch 377 rows
// on every keypress.

import { useCallback, useEffect, useRef, useState } from "react";

import { api, humanizeError } from "@/lib/api";
import {
  AUTOLABEL_BATCH, DRIVE_STATUS, canAutolabel, canOpen, driveStatus, jobForSession, matchesSession,
  orderByVisit, previousSession, recentSessions, recordVisit, sessionDetail, sessionLabel,
} from "@/lib/sessionPicker";
import { useJobStream } from "@/lib/useEventStream";
import { toast } from "@/lib/toast";
import type { SessionState } from "@/lib/sessionPicker";
import type { SessionRow } from "@/lib/types";
import Icon from "@/components/shell/Icon";
import { useAnchoredDropdown } from "@/components/shell/useAnchoredDropdown";

// Paged, not `api.sessions()`. That endpoint defaults to 200 and the picker reported "200 of 200 drives"
// against a corpus of 377, so 177 were unreachable and nothing said so. `/api/sessions/page` exists for
// exactly this and its own docstring says why: "so the browser can page through all 2000+ drives instead
// of being silently capped at the first window."
const PAGE = 500;
// A bound, because this walks pages: a corpus that grows past it should show a truncated list rather than
// hold the editor open making requests.
const MAX_SESSIONS = 5000;

async function fetchAll(): Promise<SessionRow[]> {
  const out: SessionRow[] = [];
  let total = Infinity;
  while (out.length < total && out.length < MAX_SESSIONS) {
    const r = await api.sessionsPage({ limit: PAGE, offset: out.length });
    total = r.total;
    if (!r.sessions.length) break;
    out.push(...r.sessions);
  }
  return out;
}

let stateCache: Promise<Map<string, SessionState>> | null = null;
const driveStates = () => (stateCache ??= api.sessionStates()
  .then((rs) => new Map(rs.map((r) => [r.session_id, r])))
  .catch((e) => { stateCache = null; throw e; }));

let cache: Promise<SessionRow[]> | null = null;
const sessions = () => (cache ??= fetchAll().catch((e) => { cache = null; throw e; }));

export default function SessionPicker({ sessionId, onPick }: {
  sessionId: string;
  /** Given the first frame of the chosen session. The caller routes, so unsaved work is flushed first. */
  onPick: (frameId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [rows, setRows] = useState<SessionRow[] | null>(null);
  const [states, setStates] = useState<Map<string, SessionState>>(new Map());
  const [recent, setRecent] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const { anchorRef, style, place } = useAnchoredDropdown(open);
  // Live job rows, so a drive being labelled shows it here instead of in the jobs page. One shared
  // connection: this hook rides the stream every other watcher already uses.
  const { data: jobs } = useJobStream();
  const [starting, setStarting] = useState<string | null>(null);
  // Read inside the click handler, which would otherwise close over the empty map from first render.
  const statesRef = useRef<Map<string, SessionState>>(new Map());
  statesRef.current = states;

  // Fetched on mount rather than on open, because the button's whole job is to NAME the current session
  // and it cannot do that without the list. One request per page load, shared across frame navigation.
  useEffect(() => { sessions().then(setRows).catch((e) => setErr(humanizeError(e))); }, []);

  // Which drives are started, in progress, or cannot be opened at all. A separate request from the list
  // because it is an aggregate over every frame and object and should not hold up naming the current
  // drive, which is the button's first job.
  useEffect(() => {
    driveStates().then((m) => setStates(m)).catch(() => { /* the picker still lists drives without it */ });
  }, []);

  // Recorded here rather than in the picker's click handler, so a drive reached any other way (a deep
  // link, the back button, triage) also counts as somewhere you have been.
  useEffect(() => {
    recordVisit(sessionId);
    setRecent(recentSessions());
  }, [sessionId]);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    inputRef.current?.focus();
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (boxRef.current?.contains(t) || anchorRef.current?.contains(t)) return;
      setOpen(false);
    };
    // Capture, and stopped, for the reason components/editor/properties/Popover.tsx documents at length:
    // the editor binds Escape on window and its handler clears the annotator's selection.
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopPropagation();
      setOpen(false);
      anchorRef.current?.focus();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey, true);
    };
  }, [open, anchorRef]);

  const current = rows?.find((s) => s.session_id === sessionId) ?? null;

  // One bounded batch, never the whole drive. There is one GPU in this box and the server refuses a
  // second local job while one is running, so the worst a double-click can do is a 409 the user is told
  // about rather than two passes competing for the card.
  const startRun = useCallback(async (s: SessionRow) => {
    setStarting(s.session_id);
    try {
      await api.startAutolabel(s.session_id, AUTOLABEL_BATCH, "local", true);
      toast(`labelling the next ${AUTOLABEL_BATCH} frames of ${sessionLabel(s)}`, "success");
    } catch (e) {
      // The refusals are the interesting ones and each says something true: the GPU is held by training,
      // another drive is already running, or this session failed its health checks.
      toast(humanizeError(e), "warn");
    } finally {
      setStarting(null);
    }
  }, []);

  const stopRun = useCallback(async (jobId: string) => {
    try {
      const r = await api.cancelJob(`/api/jobs/autolabel/${jobId}/cancel`);
      toast(r.stopped ? "auto-label stopped" : (r.detail || "cancel requested"), "info");
    } catch (e) {
      toast(`could not stop it: ${humanizeError(e)}`, "error");
    }
  }, []);

  const choose = useCallback(async (s: SessionRow) => {
    if (s.session_id === sessionId) { setOpen(false); return; }
    if (!canOpen(statesRef.current.get(s.session_id))) {
      // 126 sessions are LiDAR and 3D captures with no camera frames. Letting the click through would
      // return a 404 from first-frame and read as a broken picker rather than an empty drive.
      setErr("that drive has no camera frames, so the editor cannot open it");
      return;
    }
    setBusy(s.session_id);
    try {
      const { frame_id } = await api.firstFrame(s.session_id);
      setOpen(false);
      onPick(frame_id);
    } catch (e) {
      // Named rather than silently doing nothing: a session with no frames is a real state in this corpus,
      // where 42 sessions have never been through detection.
      setErr(humanizeError(e));
    } finally {
      setBusy(null);
    }
  }, [sessionId, onPick]);

  const listed = orderByVisit((rows ?? []).filter((s) => matchesSession(s, query)), sessionId, recent);
  const prevId = previousSession(sessionId);
  const prev = prevId ? rows?.find((s) => s.session_id === prevId) ?? null : null;

  return (
    <div className="relative shrink-0">
      <button ref={anchorRef} onClick={() => { if (!open) place(); setOpen((o) => !o); }}
        aria-haspopup="dialog" aria-expanded={open}
        title="which drive this frame belongs to; pick another to jump to its first frame"
        className="flex items-center gap-1.5 max-w-[190px] px-1.5 py-1 rounded hover:bg-line/40 group">
        <span className="flex flex-col leading-tight min-w-0 text-left">
          <span className="font-mono text-[11px] text-ink truncate">
            {current ? sessionLabel(current) : `SESSION ${sessionId.slice(0, 8)}`}
          </span>
          <span className="font-mono text-[9.5px] text-ink-3 truncate">
            {current ? sessionDetail(current) : (err ? "session list unavailable" : "loading drives...")}
          </span>
        </span>
        <span className="text-ink-3 group-hover:text-ink shrink-0"><Icon name="chevD" size={12} /></span>
      </button>

      {open && (
        <div ref={boxRef} role="dialog" aria-label="Choose session" style={style}
          className="z-50 w-[320px] panel border border-line rounded shadow-xl">
          <div className="p-1.5 border-b hairline">
            <input ref={inputRef} value={query} onChange={(e) => setQuery(e.target.value)}
              placeholder="search drives by vehicle, city, date or id" aria-label="search sessions"
              className="w-full bg-bg-2 border border-line rounded px-2 py-1 font-mono text-[11px] text-ink placeholder:text-ink-3/70 focus:border-accent outline-none" />
          </div>
          <div className="max-h-[60vh] overflow-auto p-1 space-y-0.5">
            {err && <div className="px-2 py-2 font-mono text-[11px] text-block">{err}</div>}
            {!rows && !err && <div className="px-2 py-2 font-mono text-[11px] text-ink-3">loading drives...</div>}
            {rows && !listed.length && (
              <div className="px-2 py-2 font-mono text-[11px] text-ink-3">no drive matches that.</div>
            )}
            {listed.map((s) => {
              const here = s.session_id === sessionId;
              return (
                <button key={s.session_id} onClick={() => void choose(s)}
                  disabled={busy != null || !canOpen(states.get(s.session_id))}
                  className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left hover:bg-line/40 disabled:opacity-45 disabled:cursor-not-allowed ${here ? "bg-line/25" : ""}`}>
                  <span className="flex flex-col leading-tight min-w-0 flex-1">
                    <span className={`font-mono text-[11.5px] truncate ${here ? "text-ink" : "text-ink-2"}`}>
                      {sessionLabel(s)}
                    </span>
                    <span className="font-mono text-[9.5px] text-ink-3 truncate">{sessionDetail(s)}</span>
                  </span>
                  {here && <span className="font-mono text-[9px] uppercase tracking-wide text-accent shrink-0">here</span>}
                  {!here && s.session_id === prevId && (
                    <span className="font-mono text-[9px] uppercase tracking-wide text-info shrink-0"
                          title="the drive you were in before this one">back</span>
                  )}
                  {!here && s.session_id !== prevId && recent.includes(s.session_id) && (
                    <span className="font-mono text-[9px] uppercase tracking-wide text-ink-3 shrink-0"
                          title="you have opened this drive before">seen</span>
                  )}
                  {DRIVE_STATUS[driveStatus(states.get(s.session_id))].label && (
                    <span className={`font-mono text-[9px] uppercase tracking-wide shrink-0 ${DRIVE_STATUS[driveStatus(states.get(s.session_id))].tone}`}
                          title={DRIVE_STATUS[driveStatus(states.get(s.session_id))].tip}>
                      {DRIVE_STATUS[driveStatus(states.get(s.session_id))].label}
                    </span>
                  )}
                  {busy === s.session_id && <span className="font-mono text-[9px] text-ink-3 shrink-0">opening...</span>}
                  {(() => {
                    const job = jobForSession(jobs?.autolabel ?? [], s.session_id);
                    if (job) {
                      const pct = Math.round((job.progress ?? 0) * 100);
                      return (
                        <span className="flex items-center gap-1.5 shrink-0"
                              onClick={(e) => e.stopPropagation()}>
                          {/* The animation is the point: a bar that only moves when a number changes reads
                              as frozen on a slow frame, so the fill also breathes while it is running. */}
                          <span className="relative w-12 h-1 rounded-sm bg-line overflow-hidden">
                            <span className="absolute inset-y-0 left-0 bg-accent progress-live rounded-sm"
                                  style={{ width: `${Math.max(4, pct)}%` }} />
                          </span>
                          <span className="font-mono text-[9px] text-accent tabular-nums w-7 text-right">{pct}%</span>
                          <button onClick={() => void stopRun(job.job_id)}
                            title="stop this auto-label run"
                            className="font-mono text-[9px] uppercase tracking-wide text-block hover:text-block/80">stop</button>
                        </span>
                      );
                    }
                    if (!canAutolabel(states.get(s.session_id))) return null;
                    return (
                      <button
                        onClick={(e) => { e.stopPropagation(); void startRun(s); }}
                        disabled={starting != null}
                        title={`auto-label the next ${AUTOLABEL_BATCH} unlabelled frames of this drive`}
                        className="font-mono text-[9px] uppercase tracking-wide text-ink-3 hover:text-accent disabled:opacity-40 shrink-0">
                        {starting === s.session_id ? "starting..." : "label"}
                      </button>
                    );
                  })()}
                </button>
              );
            })}
          </div>
          {prev && (
            <button onClick={() => void choose(prev)}
              className="w-full flex items-center gap-2 px-2 py-1.5 border-t hairline text-left hover:bg-line/40">
              <span className="font-mono text-[9px] uppercase tracking-wide text-info shrink-0">back to</span>
              <span className="font-mono text-[11px] text-ink-2 truncate">{sessionLabel(prev)}</span>
            </button>
          )}
          {rows && (
            <div className="px-2 py-1.5 border-t hairline font-mono text-[10px] text-ink-3">
              {listed.length} of {rows.length} drives · opens each at its first frame
            </div>
          )}
        </div>
      )}
    </div>
  );
}
