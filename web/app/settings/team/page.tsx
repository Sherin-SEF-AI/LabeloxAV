"use client";

import { useCallback, useEffect, useState } from "react";

import PageShell from "@/components/shell/PageShell";
import Icon from "@/components/shell/Icon";
import LoadState from "@/components/shell/LoadState";
import { useConfirm } from "@/components/ConfirmProvider";
import { apiGet, apiPost, humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import { useCurrentUser } from "@/lib/user";

// The team: who has an account, with what role, and how a new person gets in.
//
// Inviting someone gives the administrator a link to send them. It opens "choose a password" at /login,
// works once and expires (settings.auth.invite_ttl_hours), and leaves the person with an ordinary password
// sign-in. The same link, issued again, is how a lost password is recovered. Nothing here emails: the link
// is handed over however the team talks, which is also what keeps this working on an offline lab network.

type Member = { user_id: string; name: string; role: string; reviews: number };
type Link = { user_id: string; name: string; role: string; invite_token: string; expires_in_hours: number };

const ROLES: [string, string][] = [
  ["annotator", "Annotator: labels and reviews their own work"],
  ["reviewer", "Reviewer: also accepts and rejects others' labels"],
  ["admin", "Administrator: everything, including the team and keys"],
];
const ROLE_TONE: Record<string, string> = { admin: "text-accent-2", reviewer: "text-info", annotator: "text-ink-2" };

function SignInLink({ link, onDone }: { link: Link; onDone: () => void }) {
  const url = `${typeof window !== "undefined" ? window.location.origin : ""}/login?invite=${encodeURIComponent(link.invite_token)}`;
  const copy = async () => {
    try { await navigator.clipboard.writeText(url); toast("Link copied", "success"); }
    catch { toast("Copy is blocked in this browser; select the link and copy it by hand", "error"); }
  };
  return (
    <section className="panel p-5 border-pass/40" role="alert">
      <h2 className="text-[14px] font-semibold text-ink">Sign-in link for {link.name}</h2>
      <p className="mt-1 text-[12.5px] text-ink-2">
        Send this to {link.name}. Opening it lets them choose a password and sign in. It works once and expires in {link.expires_in_hours} hours;
        it is not shown again, so issue a new one if it is lost.
      </p>
      <div className="mt-3 flex items-stretch gap-2">
        <code className="flex-1 min-w-0 font-mono text-[12px] text-ink bg-bg-2 border hairline rounded px-3 py-2 break-all select-all">{url}</code>
        <button className="btn btn-primary shrink-0" onClick={() => void copy()}>Copy link</button>
      </div>
      <div className="mt-3"><button className="btn" onClick={onDone}>Done</button></div>
    </section>
  );
}

export default function TeamPage() {
  const me = useCurrentUser();
  const confirm = useConfirm();
  const [members, setMembers] = useState<Member[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [link, setLink] = useState<Link | null>(null);
  const [name, setName] = useState("");
  const [role, setRole] = useState("annotator");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try { setMembers((await apiGet<{ users: Member[] }>("/api/users?limit=500")).users); }
    catch (e) { setError(e); }
  }, []);
  useEffect(() => { if (me?.role === "admin") void load(); }, [load, me?.role]);

  const invite = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      setLink(await apiPost<Link>("/api/users/invite", { name: name.trim(), role }));
      setName("");
      await load();
    } catch (err) { toast(humanizeError(err), "error"); }
    finally { setBusy(false); }
  };
  const newLink = async (m: Member) => {
    try { setLink(await apiPost<Link>(`/api/users/${m.user_id}/sign-in-link`, {})); }
    catch (err) { toast(humanizeError(err), "error"); }
  };
  const signOutEverywhere = async (m: Member) => {
    const self = m.user_id === me?.user_id;
    if (!(await confirm({
      title: `Sign ${self ? "yourself" : m.name} out everywhere?`,
      body: self ? "Every session you have open, including this one, stops working. You will need to sign in again."
        : `Every session ${m.name} has open stops working now. Their password still works; to stop that too, issue them a new sign-in link and do not send it.`,
      confirmLabel: "Sign out everywhere", danger: true,
    }))) return;
    try {
      await apiPost(`/api/users/${m.user_id}/revoke-tokens`, {});
      toast(`${m.name} is signed out everywhere`, "success");
    } catch (err) { toast(humanizeError(err), "error"); }
  };

  if (me && me.role !== "admin") {
    return (
      <PageShell active="Team" title="Team">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-8">
          <section className="panel p-5 text-[13px] text-ink-2">The team is managed by administrators. Ask one to invite a teammate.</section>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell active="Team" title="Team" subtitle="who has an account, and how new people get in">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 flex flex-col gap-5">
        {link && <SignInLink link={link} onDone={() => setLink(null)} />}

        <form onSubmit={invite} className="panel p-5 grid gap-4 sm:grid-cols-[1fr_1fr_auto] items-end">
          <div className="sm:col-span-3">
            <h2 className="text-[14px] font-semibold text-ink">Invite a teammate</h2>
            <p className="mt-0.5 text-[12.5px] text-ink-3">You get a link to send them. They choose their own password.</p>
          </div>
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] text-ink-2">Name they sign in with</span>
            <input className="field" value={name} onChange={(e) => setName(e.target.value)} required maxLength={64} placeholder="e.g. priya" />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="text-[12px] text-ink-2">Role</span>
            <select className="field" value={role} onChange={(e) => setRole(e.target.value)}>
              {ROLES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </label>
          <button className="btn btn-primary" type="submit" disabled={busy || !name.trim()}>
            <Icon name="plus" size={13} />{busy ? "Inviting" : "Invite"}
          </button>
        </form>

        <section className="panel">
          <header className="flex items-center justify-between px-4 pt-3.5 pb-2.5">
            <h2 className="text-[13.5px] font-semibold text-ink">Members{members ? ` (${members.length})` : ""}</h2>
          </header>
          <div className="px-4 pb-4">
            <LoadState loading={!members && !error} error={error} onRetry={() => void load()} />
            {members && (
              <div className="overflow-x-auto -mx-1">
                <table className="w-full text-[12.5px]">
                  <thead>
                    <tr className="text-left text-ink-3 border-b hairline">
                      <th className="font-medium px-1 py-2">Name</th>
                      <th className="font-medium px-1 py-2">Role</th>
                      <th className="font-medium px-1 py-2 text-right">Reviews</th>
                      <th className="px-1 py-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {members.map((m) => (
                      <tr key={m.user_id} className="border-b hairline last:border-0">
                        <td className="px-1 py-2.5 text-ink">{m.name}{m.user_id === me?.user_id && <span className="text-ink-3"> (you)</span>}</td>
                        <td className={`px-1 py-2.5 capitalize ${ROLE_TONE[m.role] ?? "text-ink-2"}`}>{m.role}</td>
                        <td className="px-1 py-2.5 text-right text-ink-2">{m.reviews.toLocaleString()}</td>
                        <td className="px-1 py-2.5 text-right whitespace-nowrap">
                          <button className="btn" onClick={() => void newLink(m)}>New sign-in link</button>{" "}
                          <button className="btn" onClick={() => void signOutEverywhere(m)}>Sign out everywhere</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>
      </div>
    </PageShell>
  );
}
