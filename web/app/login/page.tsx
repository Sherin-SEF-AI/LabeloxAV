"use client";

export const dynamic = "force-dynamic";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { setUser } from "@/lib/user";
import { humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import Icon from "@/components/shell/Icon";

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

// The full-width call to action on every stage, and the quieter full-width alternative beneath it.
const PRIMARY = "btn btn-primary w-full justify-center h-10 text-[13.5px]";
const SECONDARY = "btn w-full justify-center h-10 text-[13px]";
const TEXT_LINK = "text-[12.5px] text-ink-3 hover:text-accent-2 hover:underline underline-offset-2";

function Field({ label, hint, ...rest }: { label: string; hint?: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <label className="block space-y-1.5">
      <span className="block text-[12.5px] font-medium text-ink-2">{label}</span>
      <input {...rest} className="field w-full h-10 text-[13.5px]" />
      {hint && <span className="block text-[12px] text-ink-3">{hint}</span>}
    </label>
  );
}

function Divider({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 text-[12px] text-ink-3">
      <div className="h-px bg-line flex-1" />{children}<div className="h-px bg-line flex-1" />
    </div>
  );
}

function ErrorNote({ children }: { children: React.ReactNode }) {
  return (
    <div role="alert" className="flex items-start gap-2 rounded-md border border-block/40 bg-block/5 px-3 py-2.5 text-[12.5px] text-ink">
      <span className="text-block shrink-0 mt-px" aria-hidden><Icon name="info" size={14} /></span>
      {/* Server messages arrive in lower case ("that user name and password do not match"). */}
      <span className="min-w-0 break-words">
        {typeof children === "string" && children ? children[0].toUpperCase() + children.slice(1) : children}
      </span>
    </div>
  );
}

function Shell({ title, subtitle, children }: {
  title: string; subtitle?: string; children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-bg px-4 py-10">
      <div className="w-[min(25rem,100%)]">
        <div className="mb-6 text-center font-display font-bold text-[22px] tracking-tight text-ink">
          Labelox<span className="text-accent">AV</span>
        </div>
        <div className="panel p-7 space-y-5 shadow-lg">
          <div>
            <h1 className="text-[20px] font-semibold tracking-tight text-ink">{title}</h1>
            {subtitle && subtitle !== title && <p className="mt-1 text-[13px] text-ink-3 leading-relaxed">{subtitle}</p>}
          </div>
          {children}
        </div>
        <p className="mt-6 text-center text-[12px] text-ink-3">Labelling for autonomous-driving data</p>
      </div>
    </div>
  );
}

function LoginBody() {
  const params = useSearchParams();
  const next = params.get("next") || "/";
  // A sign-in link from an administrator (Settings > Team) carries its token here, so the person lands on
  // "choose a password" with nothing to paste.
  const invite = params.get("invite");
  const [methods, setMethods] = useState<Methods | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const [stage, setStage] = useState<Stage>(invite ? "reset" : "signin");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [resetToken, setResetToken] = useState(invite ?? "");
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
        setUnreachable(!m);
        if (m?.bootstrap) setStage("signup");
      })
      .catch(() => { setMethods(null); setUnreachable(true); });
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
    else setErr(String(d.detail ?? "If that account exists, a reset link has been created for it. Ask an administrator for it."));
  });

  const submitReset = run(async () => {
    const d = await call("/api/auth/password/reset", { token: resetToken.trim(), password });
    finish(d as unknown as Issued);
  });

  const devLogin = run(async () => {
    const r = await fetch("/api/auth/dev-login", { method: "POST" });
    if (!r.ok) throw new Error(r.status === 404 ? "Development sign-in is turned off on this server" : "Sign-in failed");
    finish(await r.json());
  });

  const onEnter = (fn: () => void) => (e: React.KeyboardEvent) => { if (e.key === "Enter") fn(); };

  if (recovery && pending) {
    return (
      <Shell title="Save your recovery codes" subtitle="They are shown once and never again.">
        <div className="grid grid-cols-2 gap-1.5 font-mono text-[13px] text-ink select-all">
          {recovery.map((c) => <div key={c} className="rounded border border-line bg-bg-2 px-2.5 py-1.5">{c}</div>)}
        </div>
        <p className="text-[12.5px] text-ink-3 leading-relaxed">
          Each works once, and only if you lose your authenticator. Store them somewhere other than the
          device that generates your codes.
        </p>
        <button type="button" onClick={() => finish(pending)} className={PRIMARY}>
          I have saved them, continue
        </button>
      </Shell>
    );
  }

  if (!methods) {
    return unreachable ? (
      <Shell title="Cannot reach the server" subtitle="The sign-in options could not be loaded.">
        <ErrorNote>Check that the LabeloxAV server is running and that this address is correct, then try again.</ErrorNote>
        <button type="button" onClick={() => window.location.reload()} className={PRIMARY}>Try again</button>
      </Shell>
    ) : (
      <Shell title="Sign in" subtitle="Contacting the server…">
        <div className="space-y-3" aria-hidden>
          <span className="skeleton block h-10 w-full" />
          <span className="skeleton block h-10 w-full" />
          <span className="skeleton block h-10 w-full" />
        </div>
      </Shell>
    );
  }

  const title = stage === "signup" ? (methods.bootstrap ? "Create the first administrator" : "Create an account")
    : stage === "mfa" ? "Two-factor code"
    : stage === "enrol" ? "Set up two-factor authentication"
    : stage === "forgot" ? "Reset your password"
    : stage === "reset" ? (invite ? "Welcome. Choose a password" : "Choose a new password")
    : "Sign in";

  return (
    <Shell title={title} subtitle={stage === "signup" && methods.bootstrap
      ? "This deployment has no accounts yet, so the first one is an administrator."
      : stage === "reset" && invite ? "You will use it to sign in from now on."
      : stage === "signin" ? "Welcome back. Sign in to continue labelling."
      : stage === "forgot" ? "Enter your user name and a reset link is created for you."
      : title}>
      {err && <ErrorNote>{err}</ErrorNote>}

      {stage === "signin" && (
        <>
          {methods.password && (
            <div className="space-y-3.5">
              <Field label="User name" value={name} autoComplete="username"
                onChange={(e) => setName(e.target.value)} onKeyDown={onEnter(submitSignin)} />
              <Field label="Password" type="password" value={password} autoComplete="current-password"
                onChange={(e) => setPassword(e.target.value)} onKeyDown={onEnter(submitSignin)} />
              <button type="button" onClick={submitSignin} disabled={busy || !name.trim() || !password}
                className={PRIMARY}>
                {busy ? "Signing in…" : "Sign in"}
              </button>
              <div className="flex justify-between">
                <button type="button" onClick={() => { setErr(null); setStage("forgot"); }} className={TEXT_LINK}>
                  Forgot password?
                </button>
                {methods.self_signup && (
                  <button type="button" onClick={() => { setErr(null); setStage("signup"); }} className={TEXT_LINK}>
                    Create an account
                  </button>
                )}
              </div>
            </div>
          )}

          {methods.oidc && (
            <>
              {methods.password && <Divider>or</Divider>}
              <a href={`/api/auth/oidc/start?next=${encodeURIComponent(next)}`} className={SECONDARY}>
                Sign in with single sign-on
              </a>
            </>
          )}

          {methods.dev_login && (
            <>
              <Divider>Local development</Divider>
              <button type="button" onClick={devLogin} disabled={busy} className={SECONDARY}>
                Sign in as the dev administrator
              </button>
            </>
          )}

          {!methods.password && !methods.oidc && !methods.dev_login && (
            <ErrorNote>
              No sign-in method is turned on for this deployment. Ask an administrator for a sign-in link.
            </ErrorNote>
          )}
        </>
      )}

      {stage === "signup" && (
        <div className="space-y-3.5">
          <Field label="User name" value={name} autoComplete="username"
            onChange={(e) => setName(e.target.value)} />
          <Field label={methods.signup_domains.length
            ? `Email (${methods.signup_domains.join(", ")})` : "Email (optional)"}
            type="email" value={email} autoComplete="email" onChange={(e) => setEmail(e.target.value)} />
          <Field label="Password" type="password" value={password} autoComplete="new-password"
            hint="At least 12 characters. Length matters more than symbols."
            onChange={(e) => setPassword(e.target.value)} onKeyDown={onEnter(submitSignup)} />
          <button type="button" onClick={submitSignup} disabled={busy || !name.trim() || !password}
            className={PRIMARY}>
            {busy ? "Creating…" : "Create account"}
          </button>
          {!methods.bootstrap && (
            <button type="button" onClick={() => { setErr(null); setStage("signin"); }}
              className={`w-full ${TEXT_LINK}`}>Back to sign in</button>
          )}
        </div>
      )}

      {stage === "mfa" && (
        <div className="space-y-3.5">
          <p className="text-[13px] text-ink-2">
            Enter the six-digit code from your authenticator, or one of your recovery codes.
          </p>
          <Field label="Code" value={code} autoComplete="one-time-code"
            onChange={(e) => setCode(e.target.value)} onKeyDown={onEnter(submitMfa)} />
          <button type="button" onClick={submitMfa} disabled={busy || !code.trim()}
            className={PRIMARY}>
            Verify
          </button>
        </div>
      )}

      {stage === "enrol" && (
        <div className="space-y-3.5">
          {!enrolUri ? (
            <button type="button" onClick={startEnrol} disabled={busy}
              className={PRIMARY}>
              Generate my secret
            </button>
          ) : (
            <>
              <p className="text-[13px] text-ink-2">
                Add this key to your authenticator app, then enter the six-digit code it shows.
              </p>
              <div className="font-mono text-[13px] text-ink rounded border border-line bg-bg-2 px-2.5 py-2 break-all select-all">
                {enrolSecret}
              </div>
              <details className="text-[12px] text-ink-3">
                <summary className="cursor-pointer hover:text-ink-2">Setup link, for apps that accept one</summary>
                <div className="mt-1.5 font-mono text-[11.5px] break-all select-all">{enrolUri}</div>
              </details>
              <Field label="Code" value={code} inputMode="numeric" autoComplete="one-time-code"
                onChange={(e) => setCode(e.target.value)} onKeyDown={onEnter(confirmEnrol)} />
              <button type="button" onClick={confirmEnrol} disabled={busy || code.trim().length < 6}
                className={PRIMARY}>
                Activate
              </button>
            </>
          )}
        </div>
      )}

      {stage === "forgot" && (
        <div className="space-y-3.5">
          <Field label="User name" value={name} onChange={(e) => setName(e.target.value)}
            onKeyDown={onEnter(submitForgot)} />
          <button type="button" onClick={submitForgot} disabled={busy || !name.trim()}
            className={PRIMARY}>
            Request a reset
          </button>
          <button type="button" onClick={() => { setErr(null); setStage("signin"); }}
            className={`w-full ${TEXT_LINK}`}>Back to sign in</button>
        </div>
      )}

      {stage === "reset" && (
        <div className="space-y-3.5">
          {!invite && <Field label="Reset token" value={resetToken} onChange={(e) => setResetToken(e.target.value)} />}
          <Field label="New password" type="password" value={password} autoComplete="new-password"
            hint="At least 12 characters. Length matters more than symbols."
            onChange={(e) => setPassword(e.target.value)} onKeyDown={onEnter(submitReset)} />
          <button type="button" onClick={submitReset} disabled={busy || !resetToken.trim() || !password}
            className={PRIMARY}>
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
