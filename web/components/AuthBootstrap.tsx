"use client";

import { useEffect } from "react";
import { getUser, setUser } from "@/lib/user";

// Local-dev sign-in. With deny-by-default auth on, a fresh browser has no token in localStorage, so every
// gated call the shell makes on mount 401s and there is no way in (the UserPicker itself reads the gated
// /api/users). This asks the backend's dev-login for a real signed admin token and stores it.
//
// It is not enough to check that *some* user is stored: an earlier session can leave a user with no token,
// or one signed by a since-changed key, and either wedges the app as "logged in" while every request 401s.
// So a stored token is verified against /api/users/me first, and only a token that actually authenticates is
// trusted. Anything else re-bootstraps.
//
// Inert unless the backend hands out a token: dev-login returns 404 when env is not "local" or the flag is
// off, leaving the real login flow untouched. The reload (so the calls that already 401'd re-fire with the
// credential) happens at most once per tab via a sessionStorage sentinel, so a persistently bad state cannot
// loop.

async function tokenIsGood(token: string): Promise<boolean> {
  try {
    const r = await fetch("/api/users/me", { headers: { Authorization: `Bearer ${token}` } });
    return r.ok;
  } catch {
    return false; // backend unreachable: do not tear down whatever is stored
  }
}

export default function AuthBootstrap() {
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const stored = getUser();
      if (stored?.token && (await tokenIsGood(stored.token))) return;
      if (cancelled) return;

      try {
        const r = await fetch("/api/auth/dev-login", { method: "POST" });
        if (!r.ok || cancelled) return; // 404 in prod / flag off: leave the real login flow alone
        const u = await r.json();
        setUser({ user_id: u.user_id, name: u.name, role: u.role, token: u.token });
        if (!sessionStorage.getItem("lbx_bootstrapped")) {
          sessionStorage.setItem("lbx_bootstrapped", "1");
          window.location.reload();
        }
      } catch {
        // backend unreachable: nothing to bootstrap, the app will surface its own connection error
      }
    })();
    return () => { cancelled = true; };
  }, []);
  return null;
}
