"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import { matchesSession, orderSessions, sessionDetail, sessionLabel } from "@/lib/sessionPicker";
import type { SessionRow } from "@/lib/types";

// Choosing a session by what it is (vehicle, city, date) instead of pasting its id. Forms used to ask for a
// raw 36-character id, which meant opening another page, finding the drive, copying the id and coming back.
//
// A combobox: type to filter by vehicle, city, route, date or id (a pasted id still works), arrows to move,
// Enter to choose. The value is the session id; the text shown is the drive's name.

const MAX_SHOWN = 50;

export default function SessionSelect({ value, onChange, placeholder = "Choose a session", className = "" }: {
  value: string; onChange: (sessionId: string) => void; placeholder?: string; className?: string;
}) {
  const [sessions, setSessions] = useState<SessionRow[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [cur, setCur] = useState(0);
  const listId = useId();
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.sessions().then(setSessions).catch(() => setFailed(true));
  }, []);
  useEffect(() => {
    const close = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const chosen = sessions?.find((s) => s.session_id === value) ?? null;
  const matches = useMemo(
    () => orderSessions((sessions ?? []).filter((s) => matchesSession(s, query)), null).slice(0, MAX_SHOWN),
    [sessions, query]);

  const pick = (s: SessionRow) => { onChange(s.session_id); setQuery(""); setOpen(false); };

  return (
    <div ref={box} className={`relative ${className}`}>
      <input
        role="combobox" aria-expanded={open} aria-controls={listId} aria-autocomplete="list"
        className="field w-full"
        value={open ? query : chosen ? `${sessionLabel(chosen)}  (${sessionDetail(chosen)})` : value}
        placeholder={failed ? "sessions could not be loaded; paste a session id" : placeholder}
        onFocus={() => { setOpen(true); setCur(0); }}
        onChange={(e) => {
          const v = e.target.value;
          setQuery(v);
          setOpen(true);
          setCur(0);
          // A pasted full id is taken as-is, which is also the fallback when the list failed to load.
          if (/^[0-9a-f-]{36}$/i.test(v.trim())) onChange(v.trim());
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setOpen(true); setCur((c) => Math.min(c + 1, matches.length - 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setCur((c) => Math.max(c - 1, 0)); }
          else if (e.key === "Enter" && open && matches[cur]) { e.preventDefault(); pick(matches[cur]); }
          else if (e.key === "Escape") setOpen(false);
        }}
      />
      {open && sessions && (
        <ul id={listId} role="listbox"
          className="absolute z-50 mt-1 w-full max-h-72 overflow-y-auto rounded-md border hairline bg-panel shadow-xl py-1">
          {matches.map((s, i) => (
            <li key={s.session_id} role="option" aria-selected={s.session_id === value}
              onMouseDown={(e) => { e.preventDefault(); pick(s); }} onMouseEnter={() => setCur(i)}
              className={`px-3 py-1.5 cursor-pointer flex items-baseline justify-between gap-3 ${i === cur ? "bg-head" : ""}`}>
              <span className="text-[12.5px] text-ink truncate">{sessionLabel(s)}</span>
              <span className="text-[11.5px] text-ink-3 shrink-0">{sessionDetail(s)}</span>
            </li>
          ))}
          {!matches.length && <li className="px-3 py-2 text-[12.5px] text-ink-3">No session matches &ldquo;{query}&rdquo;</li>}
          {matches.length === MAX_SHOWN && (
            <li className="px-3 py-1.5 text-[11.5px] text-ink-3">Showing the newest {MAX_SHOWN}; type to narrow.</li>
          )}
        </ul>
      )}
    </div>
  );
}
