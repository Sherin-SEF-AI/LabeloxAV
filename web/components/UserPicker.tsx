"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import Icon from "@/components/shell/Icon";
import { api, apiPost, humanizeError } from "@/lib/api";
import { useFeatures } from "@/lib/features";
import { toast } from "@/lib/toast";
import type { UserRow } from "@/lib/types";
import { getUser, setUser, type CurrentUser } from "@/lib/user";

// The account menu in the top bar: who you are, your account, the team and keys for administrators, and
// sign out.
//
// It used to be a "switch user" list with an add-user box and no password. Under real authentication
// picking another person dropped your own token, so every request after it was refused; on a team install
// the chip read as a way to act as somebody else. Switching now exists only where it means something: a
// dev server running with auth off, where identity is a header and the list is a convenience.

const ROLE_LABEL: Record<string, string> = { admin: "Administrator", reviewer: "Reviewer", annotator: "Annotator" };

function initials(name: string): string {
  const parts = name.trim().split(/[\s._-]+/).filter(Boolean);
  return ((parts[0]?.[0] ?? "?") + (parts[1]?.[0] ?? "")).toUpperCase();
}

export default function UserPicker() {
  const router = useRouter();
  const features = useFeatures();
  const devSwitch = features.auth_enabled === false;
  const [cur, setCur] = useState<CurrentUser | null>(null);
  const [users, setUsers] = useState<UserRow[]>([]);
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => { setCur(getUser()); }, []);
  useEffect(() => {
    const close = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", close); document.removeEventListener("keydown", esc); };
  }, []);
  // The switch list is fetched only where switching exists.
  useEffect(() => {
    if (!devSwitch) return;
    api.users().then(setUsers).catch((e) => toast(humanizeError(e), "error"));
  }, [devSwitch]);

  const switchTo = (u: UserRow) => {
    const c: CurrentUser = { user_id: u.user_id, name: u.name, role: u.role };
    setUser(c);
    setCur(c);
    setOpen(false);
    window.location.reload();
  };

  const signOut = async () => {
    setOpen(false);
    try {
      await apiPost("/api/auth/logout", {});
    } catch (e) {
      // Signed out here regardless: the token lives in this browser and is dropped below. Say what the server
      // could not do rather than pretending it happened.
      toast(`Signed out in this browser; the server did not clear its cookies: ${humanizeError(e)}`, "error");
    }
    setUser(null);
    router.push("/login");
  };

  const isAdmin = cur?.role === "admin";
  const name = cur?.name ?? "Not signed in";

  return (
    <div ref={box} className="relative">
      <button onClick={() => setOpen((o) => !o)} aria-haspopup="menu" aria-expanded={open}
        className="flex items-center gap-2 h-8 pl-1 pr-2 rounded-md hover:bg-head transition-colors">
        <span className="w-6 h-6 rounded-full bg-accent/20 text-accent-2 text-[10.5px] font-semibold flex items-center justify-center">
          {cur ? initials(cur.name) : "?"}
        </span>
        <span className="hidden lg:block text-[12.5px] text-ink-2 max-w-[120px] truncate">{name}</span>
        <Icon name="chevD" size={12} />
      </button>
      {open && (
        <div role="menu" className="absolute right-0 mt-1.5 w-64 panel z-50 py-1.5 shadow-2xl">
          <div className="px-3 pt-1 pb-2 border-b hairline">
            <div className="text-[13px] font-medium text-ink truncate">{name}</div>
            {cur && <div className="text-[12px] text-ink-3">{ROLE_LABEL[cur.role] ?? cur.role}</div>}
          </div>
          <div className="py-1">
            {[
              { href: "/profile", label: "Your account", icon: "user", show: true },
              { href: "/settings/team", label: "Team", icon: "person", show: isAdmin },
              { href: "/settings/keys", label: "API keys", icon: "terminal", show: isAdmin },
              { href: "/settings", label: "Settings", icon: "gear", show: true },
            ].filter((i) => i.show).map((i) => (
              <Link key={i.href} href={i.href} role="menuitem" onClick={() => setOpen(false)}
                className="flex items-center gap-2.5 px-3 py-1.5 text-[12.5px] text-ink-2 hover:bg-head hover:text-ink">
                <span className="text-ink-3"><Icon name={i.icon} size={14} /></span>{i.label}
              </Link>
            ))}
          </div>
          {devSwitch && users.length > 0 && (
            <div className="py-1 border-t hairline">
              <div className="px-3 pt-1 pb-1 text-[10.5px] font-medium uppercase tracking-[0.06em] text-ink-3">Act as (dev server, auth off)</div>
              {users.map((u) => (
                <button key={u.user_id} role="menuitem" onClick={() => switchTo(u)}
                  className={`w-full flex justify-between px-3 py-1 text-[12.5px] hover:bg-head ${cur?.user_id === u.user_id ? "text-ink" : "text-ink-3"}`}>
                  <span className="truncate">{u.name}</span>
                  <span className="text-ink-3">{u.role}</span>
                </button>
              ))}
            </div>
          )}
          <div className="pt-1 border-t hairline">
            <button role="menuitem" onClick={() => void signOut()}
              className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[12.5px] text-ink-2 hover:bg-head hover:text-ink">
              <span className="text-ink-3"><Icon name="chevR" size={14} /></span>Sign out
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
