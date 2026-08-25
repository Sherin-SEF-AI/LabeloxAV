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
import { matchesSession, orderSessions, sessionDetail, sessionLabel } from "@/lib/sessionPicker";
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

let cache: Promise<SessionRow[]> | null = null;
const sessions = () => (cache ??= fetchAll().catch((e) => { cache = null; throw e; }));

export default function SessionPicker({ sessionId, onPick }: {
  sessionId: string;
  /** Given the first frame of the chosen session. The caller routes, so unsaved work is flushed first. */
  onPick: (frameId: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [rows, setRows] = useState<SessionRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const { anchorRef, style, place } = useAnchoredDropdown(open);

  // Fetched on mount rather than on open, because the button's whole job is to NAME the current session
  // and it cannot do that without the list. One request per page load, shared across frame navigation.
  useEffect(() => { sessions().then(setRows).catch((e) => setErr(humanizeError(e))); }, []);

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

  const choose = useCallback(async (s: SessionRow) => {
    if (s.session_id === sessionId) { setOpen(false); return; }
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

  const listed = orderSessions((rows ?? []).filter((s) => matchesSession(s, query)), sessionId);

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
                <button key={s.session_id} onClick={() => void choose(s)} disabled={busy != null}
                  className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left hover:bg-line/40 disabled:opacity-50 ${here ? "bg-line/25" : ""}`}>
                  <span className="flex flex-col leading-tight min-w-0 flex-1">
                    <span className={`font-mono text-[11.5px] truncate ${here ? "text-ink" : "text-ink-2"}`}>
                      {sessionLabel(s)}
                    </span>
                    <span className="font-mono text-[9.5px] text-ink-3 truncate">{sessionDetail(s)}</span>
                  </span>
                  {here && <span className="font-mono text-[9px] uppercase tracking-wide text-accent shrink-0">here</span>}
                  {busy === s.session_id && <span className="font-mono text-[9px] text-ink-3 shrink-0">opening...</span>}
                </button>
              );
            })}
          </div>
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
