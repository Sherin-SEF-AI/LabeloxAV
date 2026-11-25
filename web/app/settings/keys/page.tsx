"use client";

import { useCallback, useState } from "react";

import PageShell from "@/components/shell/PageShell";
import Icon from "@/components/shell/Icon";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { useConfirm } from "@/components/ConfirmProvider";
import { apiGet, apiPost, humanizeError } from "@/lib/api";
import { invalidate, useQuery } from "@/lib/useQuery";
import { toast } from "@/lib/toast";
import { useCurrentUser } from "@/lib/user";

// API keys: long-lived credentials for scripts, notebooks and CI, so nothing automated has to borrow a
// person's sign-in token (which expires after hours and leaves with that person).
//
// The server stores only a hash. The key is in the response to create or rotate and nowhere else, ever, so
// this page shows it once, with a copy button and the lines that use it, and then forgets it.

type Account = {
  service_account_id: string; name: string; key_prefix: string; description: string | null; role: string | null;
  created_at: string | null; last_used_at: string | null; expires_at: string | null; revoked_at: string | null;
  active: boolean;
};
type Minted = Account & { api_key: string };

const EXPIRY: [string, number | null][] = [["Never", null], ["30 days", 30], ["90 days", 90], ["1 year", 365]];
const ROLES: [string, string][] = [
  ["annotator", "Annotator: read data, create and edit labels"],
  ["reviewer", "Reviewer: also accept and reject labels"],
  ["admin", "Administrator: everything, including users and keys"],
];

function when(iso: string | null): string {
  return iso ? new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "";
}

function OneTimeKey({ minted, onDone }: { minted: Minted; onDone: () => void }) {
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  const copy = async (text: string, what: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast(`${what} copied`, "success");
    } catch {
      toast("Copy is blocked in this browser; select the text and copy it by hand", "error");
    }
  };
  const usage = `pip install ./sdk          # from a LabeloxAV checkout (package: labelox-client)
export LBX_URL=${origin}
export LBX_API_KEY=${minted.api_key}
labelox health`;
  return (
    <section className="panel p-5 border-pass/40" role="alert">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 text-pass"><Icon name="check" size={18} /></span>
        <div className="min-w-0 flex-1">
          <h2 className="section-title text-[14px]">Key for &ldquo;{minted.name}&rdquo; created</h2>
          <p className="mt-1 text-[12.5px] text-ink-2">Copy it now. Only its first characters are stored in readable form, so it cannot be shown again; if it is lost, rotate it.</p>
          <div className="mt-3 flex items-stretch gap-2">
            <code className="flex-1 min-w-0 font-mono text-[12.5px] text-ink bg-bg-2 border hairline rounded px-3 py-2 break-all select-all">{minted.api_key}</code>
            <button className="btn btn-primary shrink-0" onClick={() => void copy(minted.api_key, "Key")}>Copy key</button>
          </div>
          <div className="mt-4 text-[12.5px] text-ink-3">Use it from a script or CI job:</div>
          <pre className="mt-1 font-mono text-[12px] text-ink-2 bg-bg-2 border hairline rounded px-3 py-2 overflow-x-auto">{usage}</pre>
          <div className="mt-3 flex gap-2">
            <button className="btn" onClick={() => void copy(usage, "Commands")}>Copy commands</button>
            <button className="btn" onClick={onDone}>I have saved it</button>
          </div>
        </div>
      </div>
    </section>
  );
}

function CreateForm({ onMinted }: { onMinted: (m: Minted) => void }) {
  const [name, setName] = useState("");
  const [role, setRole] = useState("annotator");
  const [expires, setExpires] = useState<number | null>(90);
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      const m = await apiPost<Minted>("/api/service-accounts", {
        name: name.trim(), role, expires_in_days: expires, description: description.trim() || null,
      });
      setName(""); setDescription("");
      onMinted(m);
    } catch (err) {
      toast(humanizeError(err), "error");
    } finally {
      setBusy(false);
    }
  };
  return (
    <form onSubmit={submit} className="panel p-5 grid gap-4 sm:grid-cols-2">
      <div className="sm:col-span-2">
        <h2 className="section-title text-[14px]">Create a key</h2>
        <p className="mt-0.5 text-[12.5px] text-ink-3">One key per script or pipeline, named after it, so revoking one never breaks another.</p>
      </div>
      <label className="flex flex-col gap-1.5">
        <span className="text-[12.5px] text-ink-2">Name</span>
        <input className="field" value={name} onChange={(e) => setName(e.target.value)} required maxLength={64}
          placeholder="e.g. nightly-export" />
      </label>
      <label className="flex flex-col gap-1.5">
        <span className="text-[12.5px] text-ink-2">Expires</span>
        <select className="field" value={expires ?? ""} onChange={(e) => setExpires(e.target.value ? Number(e.target.value) : null)}>
          {EXPIRY.map(([label, days]) => <option key={label} value={days ?? ""}>{label}</option>)}
        </select>
      </label>
      <label className="flex flex-col gap-1.5">
        <span className="text-[12.5px] text-ink-2">Permissions</span>
        <select className="field" value={role} onChange={(e) => setRole(e.target.value)}>
          {ROLES.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
        </select>
      </label>
      <label className="flex flex-col gap-1.5">
        <span className="text-[12.5px] text-ink-2">What it is for <span className="text-ink-3">(optional)</span></span>
        <input className="field" value={description} onChange={(e) => setDescription(e.target.value)} maxLength={200}
          placeholder="e.g. exports the reviewed set every night" />
      </label>
      <div className="sm:col-span-2">
        <button className="btn btn-primary" type="submit" disabled={busy || !name.trim()}>
          <Icon name="plus" size={13} />{busy ? "Creating…" : "Create key"}
        </button>
      </div>
    </form>
  );
}

export default function ApiKeysPage() {
  const user = useCurrentUser();
  const confirm = useConfirm();
  const [showRevoked, setShowRevoked] = useState(false);
  const [minted, setMinted] = useState<Minted | null>(null);

  // Administrators only, so it waits for the role. keepPrevious: toggling "show revoked" leaves the
  // current rows up until the other list arrives.
  const accountsQ = useQuery(["/api/service-accounts", showRevoked],
    () => apiGet<{ accounts: Account[] }>(`/api/service-accounts?include_revoked=${showRevoked}`),
    { enabled: user?.role === "admin", keepPrevious: true });
  const rows: Account[] | null = accountsQ.data?.accounts ?? null;
  const error = accountsQ.validating ? null : accountsQ.error ?? null;
  // Both lists: a mint, a revoke or a rotate changes the one with revoked keys as well as the one without.
  const load = useCallback(() => invalidate("/api/service-accounts"), []);

  const revoke = async (a: Account) => {
    if (!(await confirm({ title: `Revoke "${a.name}"?`, body: "Anything using this key stops working immediately. This cannot be undone; create a new key instead.", confirmLabel: "Revoke", danger: true }))) return;
    try {
      await apiPost(`/api/service-accounts/${a.service_account_id}/revoke`, {});
      toast(`Revoked ${a.name}`, "success");
      await load();
    } catch (e) { toast(humanizeError(e), "error"); }
  };
  const rotate = async (a: Account) => {
    if (!(await confirm({ title: `Rotate "${a.name}"?`, body: "A new key replaces this one now. The old key stops working, so update whatever uses it.", confirmLabel: "Rotate" }))) return;
    try {
      setMinted(await apiPost<Minted>(`/api/service-accounts/${a.service_account_id}/rotate`, {}));
      await load();
    } catch (e) { toast(humanizeError(e), "error"); }
  };

  if (user && user.role !== "admin") {
    return (
      <PageShell active="API keys" title="API keys">
        <div className="page-wrap-narrow">
          <section className="panel">
            <EmptyState icon="shield" title="Administrators manage API keys">
              Ask an administrator for a key for your script, notebook or CI job.
            </EmptyState>
          </section>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell active="API keys" title="API keys" subtitle="Credentials for scripts, notebooks and CI">
      <div className="page-wrap-narrow flex flex-col gap-4">
        {minted && <OneTimeKey minted={minted} onDone={() => setMinted(null)} />}
        <CreateForm onMinted={(m) => { setMinted(m); void load(); }} />

        <section className="panel">
          <header className="panel-head">
            <h2 className="section-title">Keys</h2>
            <label className="ml-auto flex items-center gap-2 text-[12.5px] text-ink-3 cursor-pointer">
              <input type="checkbox" checked={showRevoked} onChange={(e) => setShowRevoked(e.target.checked)} className="accent-primary" />
              Show revoked
            </label>
          </header>
          <div className={rows && rows.length > 0 ? "" : "p-3"}>
            <LoadState loading={!rows && !error} error={error} onRetry={() => void load()}
              empty={rows?.length === 0} emptyIcon="terminal"
              emptyTitle={showRevoked ? "No keys" : "No active keys"}
              emptyText="Create one above for each script or pipeline, so revoking one never breaks another." />
            {rows && rows.length > 0 && (
              <div className="overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Key</th>
                      <th>Permissions</th>
                      <th>Created</th>
                      <th>Last used</th>
                      <th>Expires</th>
                      <th><span className="sr-only">Actions</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((a) => (
                      <tr key={a.service_account_id} className="align-top">
                        <td>
                          <div className="text-ink">{a.name}</div>
                          {a.description && <div className="text-[12px] text-ink-3">{a.description}</div>}
                        </td>
                        <td className="font-mono text-[12px]">lbxk_{a.key_prefix.slice(0, 8)}…</td>
                        <td className="capitalize">{a.role ?? ""}</td>
                        <td>{when(a.created_at)}</td>
                        <td>{a.last_used_at ? when(a.last_used_at) : <span className="text-ink-3">Never</span>}</td>
                        <td>
                          {a.revoked_at ? <span className="text-block">Revoked {when(a.revoked_at)}</span>
                            : !a.active ? <span className="text-warn">Expired</span>
                            : a.expires_at ? when(a.expires_at) : <span className="text-ink-3">Never</span>}
                        </td>
                        <td className="text-right whitespace-nowrap">
                          {!a.revoked_at && (
                            <>
                              <button className="btn" onClick={() => void rotate(a)}>Rotate</button>{" "}
                              <button className="btn btn-danger" onClick={() => void revoke(a)}>Revoke</button>
                            </>
                          )}
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
