"use client";

import { useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import MyIssues from "@/components/labelops/MyIssues";
import LocalePicker from "@/components/shell/LocalePicker";
import { useConfirm } from "@/components/ConfirmProvider";
import { toast } from "@/lib/toast";
import { useQuery } from "@/lib/useQuery";
import { getUser, setUser } from "@/lib/user";
import type { Profile } from "@/lib/types";

// Your own account: password, second factor, and sessions.
//
// None of this was reachable before, because none of it existed: the only credential was a token an
// administrator minted on the server, so there was nothing for a person to manage and no page to manage it
// on. Everything here acts on the caller's own account only, which is why it needs no role gate.

function Section({ title, children, hint }: {
  title: string; children: React.ReactNode; hint?: string;
}) {
  return (
    <section className="panel">
      <div className="panel-head"><h2 className="section-title">{title}</h2></div>
      <div className="p-4 space-y-3">
        {hint && <p className="section-desc">{hint}</p>}
        {children}
      </div>
    </section>
  );
}

function Field({ label, ...rest }: { label: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <label className="block space-y-1.5">
      <span className="block text-[12.5px] text-ink-2">{label}</span>
      <input {...rest} className="field w-full max-w-sm" />
    </label>
  );
}

function Pill({ tone, children }: { tone: "pass" | "warn" | "muted"; children: React.ReactNode }) {
  const cls = tone === "pass" ? "border-pass/40 text-pass" : tone === "warn" ? "border-warn/40 text-warn" : "border-line text-ink-3";
  return <span className={`inline-flex items-center px-2 py-0.5 rounded-full border text-[12px] ${cls}`}>{children}</span>;
}

export default function ProfilePage() {
  const profileQ = useQuery(["profile"], () => api.profile());
  const profile: Profile | null = profileQ.data ?? null;
  const profileErr = profileQ.error;
  useEffect(() => { if (profileErr) toast(humanizeError(profileErr), "error"); }, [profileErr]);
  const load = profileQ.reload;
  const [busy, setBusy] = useState(false);
  const confirm = useConfirm();

  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [mfaSecret, setMfaSecret] = useState<string | null>(null);
  const [mfaUri, setMfaUri] = useState<string | null>(null);
  const [mfaCode, setMfaCode] = useState("");
  const [recovery, setRecovery] = useState<string[] | null>(null);
  const [disablePw, setDisablePw] = useState("");

  // Several of these actions deliberately invalidate every existing token, which includes the one this tab
  // is holding. The server hands back a fresh one so the page keeps working rather than throwing the user
  // out of the thing they just secured.
  const adopt = (token: string) => {
    const u = getUser();
    if (u) setUser({ ...u, token });
  };

  const run = (fn: () => Promise<void>) => async () => {
    setBusy(true);
    try { await fn(); } catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };

  const changePassword = run(async () => {
    const r = await api.changePassword(profile?.has_password ? current : null, next);
    adopt(r.token);
    setCurrent(""); setNext("");
    toast("Password changed. Every other session was signed out.", "success");
    await load();
  });

  const startMfa = run(async () => {
    const r = await api.mfaSetup();
    setMfaSecret(r.secret);
    setMfaUri(r.otpauth_uri);
  });

  const confirmMfa = run(async () => {
    const r = await api.mfaConfirm(mfaCode.trim());
    adopt(r.token);
    setRecovery(r.recovery_codes);
    setMfaSecret(null); setMfaUri(null); setMfaCode("");
    await load();
  });

  const disableMfa = run(async () => {
    if (!(await confirm({
      title: "Remove two-factor authentication?",
      body: "Your account will be protected by its password alone.",
      danger: true, confirmLabel: "Remove",
    }))) return;
    await api.mfaDisable(disablePw);
    setDisablePw("");
    toast("Two-factor authentication removed", "success");
    await load();
  });

  const revoke = run(async () => {
    if (!(await confirm({
      title: "Sign out everywhere?",
      body: "Every token issued to you stops working, on every device. This tab stays signed in.",
      confirmLabel: "Sign out everywhere",
    }))) return;
    adopt((await api.revokeSessions()).token);
    toast("All other sessions signed out", "success");
  });

  return (
    <PageShell active="PROFILE" title="Your account"
      subtitle="Feedback on your work, password, two-factor authentication and sessions">
      <div className="page-wrap-narrow space-y-4">
        {profile && (
          <section className="panel px-4 py-3.5 flex items-center gap-3.5 flex-wrap">
            <span className="w-10 h-10 rounded-full bg-accent/15 text-accent-2 flex items-center justify-center text-[16px] font-semibold uppercase" aria-hidden>
              {profile.name.slice(0, 1)}
            </span>
            <div className="min-w-0">
              <div className="text-[15px] font-semibold text-ink">{profile.name}</div>
              <div className="text-[12.5px] text-ink-3">
                <span className="capitalize">{profile.role}</span>{profile.email ? ` · ${profile.email}` : ""}
              </div>
            </div>
            <div className="ml-auto flex gap-2 flex-wrap">
              <Pill tone={profile.has_password ? "pass" : "muted"}>
                {profile.has_password ? "Password set" : "No password"}
              </Pill>
              <Pill tone={profile.mfa_enabled ? "pass" : "warn"}>
                {profile.mfa_enabled ? `Two-factor on · ${profile.recovery_codes_left} recovery codes left` : "Two-factor off"}
              </Pill>
              {profile.sso && <Pill tone="muted">SSO via {profile.sso_issuer}</Pill>}
            </div>
          </section>
        )}

        {recovery && (
          <Section title="Your recovery codes" hint="Shown once and never again. Each code signs you in once if you lose your authenticator.">
            <div className="grid grid-cols-2 gap-1.5 font-mono text-[13px] text-ink select-all max-w-md">
              {recovery.map((c) => <div key={c} className="rounded border border-line bg-bg-2 px-2.5 py-1.5">{c}</div>)}
            </div>
            <button type="button" onClick={() => setRecovery(null)} className="btn btn-primary">
              I have saved them
            </button>
          </Section>
        )}

        {/* Feedback on your own labels. Issues were creatable in the editor and readable nowhere, and the
            notification they raise is addressed to the reviewer rota rather than to the person who drew
            the label - so this is where an annotator can actually go and look. */}
        <MyIssues />

        <Section title="Password"
          hint="Changing it signs out every other session, which is the point of changing it.">
          {profile?.has_password && (
            <Field label="Current password" type="password" value={current} autoComplete="current-password"
              onChange={(e) => setCurrent(e.target.value)} />
          )}
          <Field label="New password" type="password" value={next} autoComplete="new-password"
            onChange={(e) => setNext(e.target.value)} />
          <p className="text-[12px] text-ink-3">
            At least 12 characters. Length matters more than symbols.
          </p>
          <button type="button" onClick={changePassword} disabled={busy || !next} className="btn btn-primary">
            {profile?.has_password ? "Change password" : "Set a password"}
          </button>
        </Section>

        <Section title="Two-factor authentication"
          hint="A code from an authenticator app, asked for in addition to your password.">
          {profile?.mfa_enabled ? (
            <>
              <div className="flex items-center gap-2 text-[12.5px] text-pass">
                <span className="w-1.5 h-1.5 rounded-full bg-pass" aria-hidden />Turned on
              </div>
              <Field label="Password, to turn it off" type="password" value={disablePw}
                autoComplete="current-password" onChange={(e) => setDisablePw(e.target.value)} />
              <button type="button" onClick={disableMfa} disabled={busy || !disablePw} className="btn btn-danger">
                Remove two-factor authentication
              </button>
            </>
          ) : mfaSecret ? (
            <>
              <p className="text-[12.5px] text-ink-2">
                Add this key to your authenticator app, then enter the six-digit code it shows.
              </p>
              <div className="font-mono text-[13px] text-ink rounded border border-line bg-bg-2 px-2.5 py-1.5 max-w-sm break-all select-all">
                {mfaSecret}
              </div>
              {mfaUri && (
                <details className="text-[12px] text-ink-3 max-w-lg">
                  <summary className="cursor-pointer hover:text-ink-2">Setup link, for apps that accept one</summary>
                  <div className="mt-1.5 font-mono text-[11.5px] break-all select-all">{mfaUri}</div>
                </details>
              )}
              <Field label="Code" value={mfaCode} inputMode="numeric" autoComplete="one-time-code"
                onChange={(e) => setMfaCode(e.target.value)} />
              <button type="button" onClick={confirmMfa} disabled={busy || mfaCode.trim().length < 6} className="btn btn-primary">
                Turn on
              </button>
            </>
          ) : (
            <button type="button" onClick={startMfa} disabled={busy} className="btn btn-primary">
              Set up two-factor authentication
            </button>
          )}
        </Section>

        <Section title="Sessions"
          hint="Signs out every other browser and device you are signed in on. Use it if a device is lost or a sign-in may have leaked.">
          <button type="button" onClick={revoke} disabled={busy} className="btn">
            Sign out everywhere else
          </button>
        </Section>

        {/* Four locales were implemented and reachable only from inside the onboarding tour, which a user
            sees once. A setting you can only get to from a thing you have already dismissed is not a
            setting. */}
        <Section title="Language">
          <LocalePicker />
        </Section>
      </div>
    </PageShell>
  );
}
