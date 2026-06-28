"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { UserRow } from "@/lib/types";
import { getUser, setUser, type CurrentUser } from "@/lib/user";

// The current-user chip in the nav: shows who is acting (name + role), switch users, or add one.
// Lightweight, no password; the choice rides on every request as attribution.

const ROLE_COLOR: Record<string, string> = { admin: "text-accent", reviewer: "text-info", annotator: "text-ink-2" };

export default function UserPicker() {
  const [cur, setCur] = useState<CurrentUser | null>(null);
  const [users, setUsers] = useState<UserRow[]>([]);
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [role, setRole] = useState("annotator");

  useEffect(() => {
    setCur(getUser());
    api.users().then((us) => {
      setUsers(us);
      const cur = getUser();
      // also re-pick when the cached user is stale (deleted / from a reset DB) so mutations do not 401
      if ((!cur || !us.some((u) => u.user_id === cur.user_id)) && us.length) {
        pick(us.find((u) => u.role === "admin") ?? us[0]);
      }
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function pick(u: UserRow) {
    const c = { user_id: u.user_id, name: u.name, role: u.role };
    setUser(c);
    setCur(c);
    setOpen(false);
  }

  async function add() {
    if (!name.trim()) return;
    try {
      const u = await api.createUser(name.trim(), role);
      setUsers((us) => [...us, u]);
      setName("");
      pick(u);
    } catch (e) {
      alert(String(e));
    }
  }

  return (
    <div className="relative">
      <button onClick={() => setOpen((o) => !o)} className="font-mono text-xs border border-line px-2 py-0.5 hover:border-accent">
        <span className="text-ink-2">{cur?.name ?? "no user"}</span>{" "}
        <span className={ROLE_COLOR[cur?.role ?? ""] ?? "text-ink-3"}>{cur?.role ?? ""}</span>
      </button>
      {open && (
        <div className="absolute right-0 mt-1 w-60 panel z-50 p-2 space-y-1">
          <div className="font-mono text-[10px] uppercase text-ink-3">switch user</div>
          {users.map((u) => (
            <button key={u.user_id} onClick={() => pick(u)}
              className={`w-full flex justify-between px-1 py-0.5 font-mono text-[11px] ${cur?.user_id === u.user_id ? "text-ink" : "text-ink-3 hover:text-ink-2"}`}>
              <span>{u.name}</span>
              <span className="text-ink-3">{u.role} · {u.reviews} rev</span>
            </button>
          ))}
          <div className="border-t hairline pt-1 flex gap-1">
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="new user"
              className="flex-1 bg-bg border border-line px-1 py-0.5 font-mono text-[11px] text-ink min-w-0" />
            <select value={role} onChange={(e) => setRole(e.target.value)} className="bg-bg border border-line px-1 font-mono text-[11px] text-ink">
              <option value="annotator">ann</option>
              <option value="reviewer">rev</option>
              <option value="admin">adm</option>
            </select>
            <button onClick={add} className="border border-line px-1.5 text-ink-2 hover:border-accent">+</button>
          </div>
        </div>
      )}
    </div>
  );
}
