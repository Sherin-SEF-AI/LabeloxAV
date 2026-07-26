"use client";

export const dynamic = "force-dynamic";

import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { setUser } from "@/lib/user";
import { humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";

// The sign-in surface. Deny-by-default auth means the app needs a real credential before it can load any
// gated data, and a 401 anywhere in the app routes here (?next carries where to return). Two ways in: the
// one-click local dev sign-in (the backend's dev-login, only available on a dev box), or pasting a signed
// token issued for a user. On success the user is stored and we return to wherever they were headed.

function LoginBody() {
  const next = useSearchParams().get("next") || "/";
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);

  const finish = (u: { user_id: string; name: string; role: string; token: string }) => {
    setUser(u);
    toast(`Signed in as ${u.name}`, "success");
    window.location.href = next;
  };

  const devLogin = async () => {
    setBusy(true);
    try {
      const r = await fetch("/api/auth/dev-login", { method: "POST" });
      if (!r.ok) throw new Error(r.status === 404 ? "dev sign-in is disabled on this server" : `sign-in failed (${r.status})`);
      finish(await r.json());
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setBusy(false);
    }
  };

  const tokenLogin = async () => {
    if (!token.trim()) return;
    setBusy(true);
    try {
      // Verify the pasted token against /users/me, then adopt the identity it resolves to.
      const r = await fetch("/api/users/me", { headers: { Authorization: `Bearer ${token.trim()}` } });
      if (!r.ok) throw new Error("that token is not valid");
      const me = await r.json();
      finish({ user_id: me.user_id, name: me.name, role: me.role, token: token.trim() });
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-bg">
      <div className="w-[min(24rem,92vw)] panel p-5 space-y-4">
        <div>
          <div className="font-display font-bold text-lg text-ink">Labelox<span className="text-accent">AV</span></div>
          <div className="font-mono text-[11px] text-ink-3">Sign in to continue</div>
        </div>

        <button onClick={devLogin} disabled={busy}
          className="w-full border border-accent text-accent px-3 py-2 font-mono text-xs hover:bg-accent/10 disabled:opacity-50">
          {busy ? "signing in..." : "Sign in (local dev)"}
        </button>

        <div className="flex items-center gap-2 font-mono text-[10px] text-ink-4">
          <div className="h-px bg-line flex-1" /> or paste a token <div className="h-px bg-line flex-1" />
        </div>

        <div className="space-y-2">
          <input value={token} onChange={(e) => setToken(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && tokenLogin()}
            placeholder="lbx1...." aria-label="access token"
            className="w-full bg-bg border border-line px-2 py-1.5 font-mono text-[11px] text-ink" />
          <button onClick={tokenLogin} disabled={busy || !token.trim()}
            className="w-full border border-line text-ink-2 px-3 py-1.5 font-mono text-xs hover:border-accent disabled:opacity-40">
            Sign in with token
          </button>
        </div>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return <Suspense fallback={null}><LoginBody /></Suspense>;
}
