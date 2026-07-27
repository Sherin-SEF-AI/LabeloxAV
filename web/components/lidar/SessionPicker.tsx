"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { SessionRow } from "@/lib/types";

// The 3D workspaces asked the user to type a raw session UUID into a text box, which meant they were
// unusable without querying the database by hand. This offers the real session list and keeps the free-text
// field for the case where someone genuinely has an id from elsewhere.
export default function SessionPicker({
  value,
  onChange,
  onLoad,
}: {
  value: string;
  onChange: (id: string) => void;
  onLoad: (id: string) => void;
}) {
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [err, setErr] = useState(false);

  useEffect(() => {
    api
      .sessionsPage({ limit: 200 })
      .then((p) => setSessions(p.sessions))
      .catch(() => setErr(true));
  }, []);

  return (
    <div className="flex flex-col gap-2">
      <label className="sr-only" htmlFor="lidar-session-select">
        capture session
      </label>
      <select
        id="lidar-session-select"
        value={sessions.some((s) => s.session_id === value) ? value : ""}
        onChange={(e) => {
          onChange(e.target.value);
          if (e.target.value) onLoad(e.target.value);
        }}
        className="min-w-0 rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs"
      >
        <option value="">
          {err ? "session list unavailable" : sessions.length ? "select a session" : "loading sessions..."}
        </option>
        {sessions.map((s) => (
          <option key={s.session_id} value={s.session_id}>
            {s.vehicle_id}
            {s.city ? ` / ${s.city}` : ""} / {s.session_id.slice(0, 8)}
          </option>
        ))}
      </select>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          onLoad(value);
        }}
        className="flex gap-2"
      >
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="or paste a session id"
          aria-label="session id"
          className="min-w-0 flex-1 rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs"
        />
        <button className="rounded bg-cyan-700 px-3 py-1 text-xs hover:bg-cyan-600">Load</button>
      </form>
    </div>
  );
}
