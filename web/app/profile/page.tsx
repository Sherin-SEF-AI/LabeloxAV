"use client";

import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import MyIssues from "@/components/labelops/MyIssues";
import LocalePicker from "@/components/shell/LocalePicker";
import { useConfirm } from "@/components/ConfirmProvider";
import { toast } from "@/lib/toast";
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
      <div className="border-b hairline px-3 py-2">
        <div className="font-mono text-[11px] uppercase text-ink-3">{title}</div>
        {hint && <div className="font-mono text-[10px] text-ink-3">{hint}</div>}
      </div>
      <div className="p-3 space-y-2">{children}</div>
    </section>
  );
}

function Field({ label, ...rest }: { label: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <label className="block space-y-1">
      <span className="font-mono text-[10px] uppercase text-ink-3">{label}</span>
      <input {...rest}
        className="w-full max-w-sm bg-bg border border-line px-2 py-1 font-mono text-[12px] text-ink
                   focus:border-accent outline-none" />
    </label>
  );
}

export default function ProfilePage() {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [busy, setBusy] = useState(false);
  const confirm = useConfirm();

  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [mfaSecret, setMfaSecret] = useState<string | null>(null);
  const [mfaUri, setMfaUri] = useState<string | null>(null);
  const [mfaCode, setMfaCode] = useState("");
  const [recovery, setRecovery] = useState<string[] | null>(null);
  const [disablePw, setDisablePw] = useState("");

  const load = useCallback(async () => {
    try { setProfile(await api.profile()); }
    catch (e) { toast(humanizeError(e), "error"); }
  }, []);

  useEffect(() => { load(); }, [load]);

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
      subtitle="feedback on your work, password, second factor, and sessions">
      <div className="p-4 space-y-4 max-w-3xl">
        {/* Feedback on your own labels. Issues were creatable in the editor and readable nowhere, and the
            notification they raise is addressed to the reviewer rota rather than to the person who drew
            the label - so this is where an annotator can actually go and look. */}
        <MyIssues />

        {/* Four locales were implemented and reachable only from inside the onboarding tour, which a user
            sees once. A setting you can only get to from a thing you have already dismissed is not a
            setting. */}
        <section className="panel p-3">
          <LocalePicker />
        </section>
        {profile && (
          <section className="panel p-3 flex items-center gap-4 flex-wrap">
            <div>
              <div className="font-mono text-[13px] text-ink">{profile.name}</div>
              <div className="font-mono text-[10px] text-ink-3">
                {profile.role}{profile.email ? ` · ${profile.email}` : ""}
              </div>
            </div>
            <div className="ml-auto flex gap-2 font-mono text-[10px]">
              <span className={profile.has_password ? "text-pass" : "text-ink-3"}>
                {profile.has_password ? "password set" : "no password"}
              </span>
              <span className={profile.mfa_enabled ? "text-pass" : "text-warn"}>
                {profile.mfa_enabled ? `2FA on · ${profile.recovery_codes_left} recovery codes left` : "2FA off"}
              </span>
              {profile.sso && <span className="text-ink-3">SSO via {profile.sso_issuer}</span>}
            </div>
          </section>
        )}

        {recovery && (
          <Section title="your recovery codes" hint="shown once, never again">
            <div className="grid grid-cols-2 gap-1 font-mono text-[12px] text-ink select-all max-w-md">
              {recovery.map((c) => <div key={c} className="border border-line px-2 py-1">{c}</div>)}
            </div>
            <button onClick={() => setRecovery(null)}
              className="border border-line px-2 py-0.5 font-mono text-[11px] text-ink-2 hover:border-accent">
              I have saved them
            </button>
          </Section>
        )}

        <Section title="password"
          hint="changing it signs out every other session, which is the point of changing it">
          {profile?.has_password && (
            <Field label="current password" type="password" value={current} autoComplete="current-password"
              onChange={(e) => setCurrent(e.target.value)} />
          )}
          <Field label="new password" type="password" value={next} autoComplete="new-password"
            onChange={(e) => setNext(e.target.value)} />
          <div className="font-mono text-[10px] text-ink-3">
            at least 12 characters; length matters more than symbols
          </div>
          <button onClick={changePassword} disabled={busy || !next}
            className="border border-line px-2 py-0.5 font-mono text-[11px] text-ink-2
                       hover:border-accent disabled:opacity-40">
            {profile?.has_password ? "change password" : "set a password"}
          </button>
        </Section>

        <Section title="two-factor authentication"
          hint="an authenticator app code in addition to your password">
          {profile?.mfa_enabled ? (
            <>
              <div className="font-mono text-[11px] text-pass">enrolled</div>
              <Field label="password (to remove it)" type="password" value={disablePw}
                autoComplete="current-password" onChange={(e) => setDisablePw(e.target.value)} />
              <button onClick={disableMfa} disabled={busy || !disablePw}
                className="border border-line px-2 py-0.5 font-mono text-[11px] text-ink-3
                           hover:text-block hover:border-block disabled:opacity-40">
                remove two-factor authentication
              </button>
            </>
          ) : mfaSecret ? (
            <>
              <div className="font-mono text-[10px] text-ink-3">
                Add this to your authenticator, then enter the code it shows.
              </div>
              <div className="font-mono text-[12px] text-ink border border-line px-2 py-1 max-w-sm break-all select-all">
                {mfaSecret}
              </div>
              <div className="font-mono text-[9px] text-ink-3 break-all max-w-lg">{mfaUri}</div>
              <Field label="code" value={mfaCode} inputMode="numeric" autoComplete="one-time-code"
                onChange={(e) => setMfaCode(e.target.value)} />
              <button onClick={confirmMfa} disabled={busy || mfaCode.trim().length < 6}
                className="border border-accent px-2 py-0.5 font-mono text-[11px] text-accent
                           hover:bg-accent/10 disabled:opacity-40">
                activate
              </button>
            </>
          ) : (
            <button onClick={startMfa} disabled={busy}
              className="border border-line px-2 py-0.5 font-mono text-[11px] text-ink-2
                         hover:border-accent disabled:opacity-40">
              set up two-factor authentication
            </button>
          )}
        </Section>

        <Section title="sessions"
          hint="a token is stateless, so this is how a lost or stolen one is actually taken away">
          <button onClick={revoke} disabled={busy}
            className="border border-line px-2 py-0.5 font-mono text-[11px] text-ink-2
                       hover:border-accent disabled:opacity-40">
            sign out everywhere else
          </button>
        </Section>
      </div>
    </PageShell>
  );
}
