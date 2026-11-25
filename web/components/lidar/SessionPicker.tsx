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
      <label className="text-[12px] font-medium text-ink-2" htmlFor="lidar-session-select">
        Session
      </label>
      <select
        id="lidar-session-select"
        value={sessions.some((s) => s.session_id === value) ? value : ""}
        onChange={(e) => {
          onChange(e.target.value);
          if (e.target.value) onLoad(e.target.value);
        }}
        className="field min-w-0"
      >
        <option value="">
          {err ? "Session list unavailable" : sessions.length ? "Choose a session" : "Loading sessions..."}
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
          className="field min-w-0 flex-1"
        />
        <button type="submit" className="btn btn-primary">Load</button>
      </form>
    </div>
  );
}
