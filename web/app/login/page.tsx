"use client";

export const dynamic = "force-dynamic";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { setUser } from "@/lib/user";
import { humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";

// The sign-in surface.
//
// It used to offer exactly two things: a local dev shortcut, and a box to paste a token an administrator had
// minted on the server. That was the whole identity story, and it is why this could not be deployed anywhere
// real: there was no way for a person to obtain access, recover an account, or add a second factor.
//
// What is drawn is decided by the server, not guessed here. /api/auth/methods reports which methods the
// deployment actually offers, so a directory-only install shows a single sign-on button instead of a
// password form that would refuse every submission, and a fresh install offers "create the first
// administrator" instead of a login nobody can yet pass.

type Methods = {
  password: boolean; self_signup: boolean; signup_domains: string[];
  oidc: boolean; oidc_issuer: string | null; dev_login: boolean; bootstrap: boolean;
};
type Issued = { user_id: string; name: string; role: string; token: string };
type Stage = "signin" | "signup" | "mfa" | "enrol" | "forgot" | "reset";

function Field({ label, ...rest }: { label: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <label className="block space-y-1">
      <span className="font-mono text-[10px] uppercase text-ink-3">{label}</span>
      <input {...rest}
        className="w-full bg-bg border border-line px-2 py-1.5 font-mono text-[12px] text-ink
                   focus:border-accent outline-none" />
    </label>
  );
}

function Divider({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 font-mono text-[10px] text-ink-3">
      <h1 className="sr-only">Sign in to LabeloxAV</h1>
      <div className="h-px bg-line flex-1" />{children}<div className="h-px bg-line flex-1" />
    </div>
  );
}

function Shell({ title, subtitle, children }: {
  title: string; subtitle?: string; children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-bg p-4">
      <div className="w-[min(26rem,94vw)] panel p-5 space-y-4">
        <div>
          <div className="font-display font-bold text-lg text-ink">
            Labelox<span className="text-accent">AV</span>
          </div>
          <div className="font-mono text-[11px] text-ink-3">{subtitle ?? title}</div>
        </div>
        {children}
      </div>
    </div>
  );
}

function LoginBody() {
  const next = useSearchParams().get("next") || "/";
  const [methods, setMethods] = useState<Methods | null>(null);
  const [stage, setStage] = useState<Stage>("signin");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [resetToken, setResetToken] = useState("");
  const [mfaHandle, setMfaHandle] = useState("");
  const [enrolUri, setEnrolUri] = useState<string | null>(null);
  const [enrolSecret, setEnrolSecret] = useState<string | null>(null);
  const [recovery, setRecovery] = useState<string[] | null>(null);
  const [pending, setPending] = useState<Issued | null>(null);

  const finish = useCallback((u: Issued) => {
    setUser(u);
    toast(`Signed in as ${u.name}`, "success");
    window.location.href = next;
  }, [next]);

  useEffect(() => {
    fetch("/api/auth/methods")
      .then((r) => (r.ok ? r.json() : null))
      .then((m: Methods | null) => {
        setMethods(m);
        if (m?.bootstrap) setStage("signup");
      })
      .catch(() => setMethods(null));
  }, []);

  // A single-sign-on return leaves the issued credential in a one-shot cookie, because a token in the URL
  // fragment survives in browser history and in the query string it reaches every proxy log on the way.
  useEffect(() => {
    const raw = document.cookie.split("; ").find((c) => c.startsWith("lbx_sso_handoff="));
    if (!raw) return;
    document.cookie = "lbx_sso_handoff=; Path=/; Max-Age=0";
    try { finish(JSON.parse(decodeURIComponent(raw.split("=").slice(1).join("=")))); } catch { /* ignore */ }
  }, [finish]);

  const call = async (path: string, body: unknown): Promise<Record<string, unknown>> => {
    const r = await fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(String(data?.detail ?? `request failed (${r.status})`));
    return data;
  };

  const run = (fn: () => Promise<void>) => async () => {
    setBusy(true); setErr(null);
    try { await fn(); } catch (e) { setErr(humanizeError(e)); } finally { setBusy(false); }
  };

  const submitSignin = run(async () => {
    const d = await call("/api/auth/login", { name: name.trim(), password });
    if (d.mfa_required) {
      setMfaHandle(String(d.mfa_handle));
      // A role that requires a second factor must be unusable until one exists, so enrolment happens here
      // at the door rather than after a token has already been handed out.
      setStage(d.mfa_enrol ? "enrol" : "mfa");
      if (d.mfa_enrol && d.detail) setErr(String(d.detail));
      return;
    }
    finish(d as unknown as Issued);
  });

  const submitSignup = run(async () => {
    const d = await call("/api/auth/signup",
      { name: name.trim(), password, email: email.trim() || null });
    if (d.mfa_required) { setMfaHandle(String(d.mfa_handle)); setStage("enrol"); return; }
    finish(d as unknown as Issued);
  });

  const submitMfa = run(async () => {
    const d = await call("/api/auth/login/mfa", { mfa_handle: mfaHandle, code: code.trim() });
    finish(d as unknown as Issued);
  });

  const startEnrol = run(async () => {
    // Mid-sign-in there is no token yet, which is the whole point, so the secret is minted through the
    // challenge that stopped the login rather than through the authenticated setup route.
    const d = await call("/api/auth/mfa/setup-pending", { mfa_handle: mfaHandle });
    setEnrolSecret(String(d.secret));
    setEnrolUri(String(d.otpauth_uri));
    setMfaHandle(String(d.mfa_handle));
  });

  const confirmEnrol = run(async () => {
    const d = await call("/api/auth/mfa/confirm", { mfa_handle: mfaHandle, code: code.trim() });
    if (Array.isArray(d.recovery_codes)) {
      // Held on screen rather than navigating away: these are shown exactly once, and redirecting straight
      // into the app would lose them.
      setPending(d as unknown as Issued);
      setRecovery(d.recovery_codes as string[]);
      return;
    }
    finish(d as unknown as Issued);
  });

  const submitForgot = run(async () => {
    const d = await call("/api/auth/password/reset-request", { name: name.trim() });
    if (d.reset_token) { setResetToken(String(d.reset_token)); setStage("reset"); }
    else setErr(String(d.detail ?? "if that account exists, a reset link has been created for it"));
  });

  const submitReset = run(async () => {
    const d = await call("/api/auth/password/reset", { token: resetToken.trim(), password });
    finish(d as unknown as Issued);
  });

  const devLogin = run(async () => {
    const r = await fetch("/api/auth/dev-login", { method: "POST" });
    if (!r.ok) throw new Error(r.status === 404 ? "dev sign-in is disabled on this server" : "sign-in failed");
    finish(await r.json());
  });

  const onEnter = (fn: () => void) => (e: React.KeyboardEvent) => { if (e.key === "Enter") fn(); };

  if (recovery && pending) {
    return (
      <Shell title="Save your recovery codes" subtitle="these are not shown again">
        <div className="grid grid-cols-2 gap-1 font-mono text-[12px] text-ink select-all">
          {recovery.map((c) => <div key={c} className="border border-line px-2 py-1">{c}</div>)}
        </div>
        <div className="font-mono text-[10px] text-ink-3">
          Each works once, and only if you lose your authenticator. Store them somewhere other than the
          device that generates your codes.
        </div>
        <button onClick={() => finish(pending)}
          className="w-full border border-accent text-accent px-3 py-2 font-mono text-xs hover:bg-accent/10">
          I have saved them, continue
        </button>
      </Shell>
    );
  }

  if (!methods) return <Shell title="Sign in" subtitle="contacting the server..."><div /></Shell>;

  const title = stage === "signup" ? (methods.bootstrap ? "Create the first administrator" : "Create an account")
    : stage === "mfa" ? "Two-factor code"
    : stage === "enrol" ? "Set up two-factor authentication"
    : stage === "forgot" ? "Reset your password"
    : stage === "reset" ? "Choose a new password"
    : "Sign in";

  return (
    <Shell title={title} subtitle={stage === "signup" && methods.bootstrap
      ? "this deployment has no accounts yet, so the first one is an administrator"
      : title}>
      {err && <div className="font-mono text-[11px] text-block border border-block px-2 py-1">{err}</div>}

      {stage === "signin" && (
        <>
          {methods.password && (
            <div className="space-y-2">
              <Field label="user name" value={name} autoComplete="username"
                onChange={(e) => setName(e.target.value)} onKeyDown={onEnter(submitSignin)} />
              <Field label="password" type="password" value={password} autoComplete="current-password"
                onChange={(e) => setPassword(e.target.value)} onKeyDown={onEnter(submitSignin)} />
              <button onClick={submitSignin} disabled={busy || !name.trim() || !password}
                className="w-full border border-accent text-accent px-3 py-2 font-mono text-xs
                           hover:bg-accent/10 disabled:opacity-40">
                {busy ? "signing in..." : "Sign in"}
              </button>
              <div className="flex justify-between font-mono text-[10px] text-ink-3">
                <button onClick={() => { setErr(null); setStage("forgot"); }} className="hover:text-accent">
                  forgot password
                </button>
                {methods.self_signup && (
                  <button onClick={() => { setErr(null); setStage("signup"); }} className="hover:text-accent">
                    create an account
                  </button>
                )}
              </div>
            </div>
          )}

          {methods.oidc && (
            <>
              {methods.password && <Divider>or</Divider>}
              <a href={`/api/auth/oidc/start?next=${encodeURIComponent(next)}`}
                className="block w-full border border-line text-ink-2 px-3 py-2 font-mono text-xs text-center
                           hover:border-accent">
                Sign in with single sign-on
              </a>
            </>
          )}

          {methods.dev_login && (
            <>
              <Divider>local development</Divider>
              <button onClick={devLogin} disabled={busy}
                className="w-full border border-line text-ink-3 px-3 py-1.5 font-mono text-[11px]
                           hover:border-accent disabled:opacity-40">
                Sign in as the dev administrator
              </button>
            </>
          )}

          {!methods.password && !methods.oidc && !methods.dev_login && (
            <div className="font-mono text-[11px] text-ink-3">
              No sign-in method is enabled on this deployment. An administrator issues credentials directly.
            </div>
          )}
        </>
      )}

      {stage === "signup" && (
        <div className="space-y-2">
          <Field label="user name" value={name} autoComplete="username"
            onChange={(e) => setName(e.target.value)} />
          <Field label={methods.signup_domains.length
            ? `email (${methods.signup_domains.join(", ")})` : "email (optional)"}
            type="email" value={email} autoComplete="email" onChange={(e) => setEmail(e.target.value)} />
          <Field label="password" type="password" value={password} autoComplete="new-password"
            onChange={(e) => setPassword(e.target.value)} onKeyDown={onEnter(submitSignup)} />
          <div className="font-mono text-[10px] text-ink-3">
            at least 12 characters; length matters more than symbols
          </div>
          <button onClick={submitSignup} disabled={busy || !name.trim() || !password}
            className="w-full border border-accent text-accent px-3 py-2 font-mono text-xs
                       hover:bg-accent/10 disabled:opacity-40">
            {busy ? "creating..." : "Create account"}
          </button>
          {!methods.bootstrap && (
            <button onClick={() => { setErr(null); setStage("signin"); }}
              className="w-full font-mono text-[10px] text-ink-3 hover:text-accent">back to sign in</button>
          )}
        </div>
      )}

      {stage === "mfa" && (
        <div className="space-y-2">
          <div className="font-mono text-[11px] text-ink-3">
            Enter the six-digit code from your authenticator, or one of your recovery codes.
          </div>
          <Field label="code" value={code} autoComplete="one-time-code"
            onChange={(e) => setCode(e.target.value)} onKeyDown={onEnter(submitMfa)} />
          <button onClick={submitMfa} disabled={busy || !code.trim()}
            className="w-full border border-accent text-accent px-3 py-2 font-mono text-xs
                       hover:bg-accent/10 disabled:opacity-40">
            Verify
          </button>
        </div>
      )}

      {stage === "enrol" && (
        <div className="space-y-2">
          {!enrolUri ? (
            <button onClick={startEnrol} disabled={busy}
              className="w-full border border-accent text-accent px-3 py-2 font-mono text-xs
                         hover:bg-accent/10 disabled:opacity-40">
              Generate my secret
            </button>
          ) : (
            <>
              <div className="font-mono text-[10px] text-ink-3">
                Add this to your authenticator, then enter the code it shows.
              </div>
              <div className="font-mono text-[11px] text-ink border border-line px-2 py-1 break-all select-all">
                {enrolSecret}
              </div>
              <div className="font-mono text-[9px] text-ink-3 break-all">{enrolUri}</div>
              <Field label="code" value={code} inputMode="numeric" autoComplete="one-time-code"
                onChange={(e) => setCode(e.target.value)} onKeyDown={onEnter(confirmEnrol)} />
              <button onClick={confirmEnrol} disabled={busy || code.trim().length < 6}
                className="w-full border border-accent text-accent px-3 py-2 font-mono text-xs
                           hover:bg-accent/10 disabled:opacity-40">
                Activate
              </button>
            </>
          )}
        </div>
      )}

      {stage === "forgot" && (
        <div className="space-y-2">
          <Field label="user name" value={name} onChange={(e) => setName(e.target.value)}
            onKeyDown={onEnter(submitForgot)} />
          <button onClick={submitForgot} disabled={busy || !name.trim()}
            className="w-full border border-accent text-accent px-3 py-2 font-mono text-xs
                       hover:bg-accent/10 disabled:opacity-40">
            Request a reset
          </button>
          <button onClick={() => { setErr(null); setStage("signin"); }}
            className="w-full font-mono text-[10px] text-ink-3 hover:text-accent">back to sign in</button>
        </div>
      )}

      {stage === "reset" && (
        <div className="space-y-2">
          <Field label="reset token" value={resetToken} onChange={(e) => setResetToken(e.target.value)} />
          <Field label="new password" type="password" value={password} autoComplete="new-password"
            onChange={(e) => setPassword(e.target.value)} onKeyDown={onEnter(submitReset)} />
          <button onClick={submitReset} disabled={busy || !resetToken.trim() || !password}
            className="w-full border border-accent text-accent px-3 py-2 font-mono text-xs
                       hover:bg-accent/10 disabled:opacity-40">
            Set password and sign in
          </button>
        </div>
      )}
    </Shell>
  );
}

export default function LoginPage() {
  return <Suspense fallback={null}><LoginBody /></Suspense>;
}
