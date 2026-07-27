"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api , humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { useConfirm } from "@/components/ConfirmProvider";
import { toast } from "@/lib/toast";

// Outbound integrations: webhook subscriptions and registered storage buckets.
//
// The signing secret is shown exactly once, at creation, because that is the only moment the server will ever
// return it: the receiver needs it to verify deliveries, and storing it anywhere retrievable would defeat the
// point of signing. The UI says so rather than letting someone close the dialog and lose it silently.

type Webhook = {
  webhook_id: string; url: string; events: string[]; active: boolean;
  last_status: number | null; last_error: string | null; failure_count: number;
  last_delivery_at: string | null;
};
type Source = {
  source_id: string; name: string; provider: string; bucket: string; prefix: string | null;
  uri: string; credential_profile: string | null; last_object_count: number | null;
};

function Section({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="flex items-center gap-2 font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
        <span>{title}</span>
        {right && <span className="ml-auto normal-case">{right}</span>}
      </div>
      <div className="p-3">{children}</div>
    </section>
  );
}

export default function IntegrationsPage() {
  // useSearchParams forces a client-side-render bailout, which Next requires be under a Suspense boundary or
  // the static prerender of this route fails the build. Match the pattern used by login and search.
  return <Suspense fallback={null}><IntegrationsBody /></Suspense>;
}

function IntegrationsBody() {
  const params = useSearchParams();
  const [hooks, setHooks] = useState<Webhook[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [events, setEvents] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [freshSecret, setFreshSecret] = useState<{ url: string; secret: string } | null>(null);
  const confirm = useConfirm();

  // webhook form
  const [url, setUrl] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  // source form
  const [sName, setSName] = useState("");
  const [sProvider, setSProvider] = useState("s3");
  const [sBucket, setSBucket] = useState("");
  const [sPrefix, setSPrefix] = useState("");
  const [sProfile, setSProfile] = useState("");

  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(null), 5000); };

  const refresh = useCallback(async () => {
    try {
      const [e, s] = await Promise.all([api.integrationEvents(), api.storageSources()]);
      setEvents(e.events); setSources(s.sources);
    } catch (err) { flash(humanizeError(err)); }
    // Webhooks are admin-gated, so a non-admin simply sees an empty list rather than an error wall.
    try { setHooks((await api.webhooks()).webhooks); } catch { setHooks([]); }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const addHook = async () => {
    if (!url.trim()) { flash("give the webhook a url"); return; }
    setBusy(true);
    try {
      const wh = await api.createWebhook(url.trim(), picked);
      setFreshSecret({ url: wh.url, secret: wh.secret });
      setUrl(""); setPicked([]);
      await refresh();
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const addSource = async () => {
    if (!sName.trim() || !sBucket.trim()) { flash("name and bucket are required"); return; }
    setBusy(true);
    try {
      await api.registerSource({
        name: sName.trim(), provider: sProvider, bucket: sBucket.trim(),
        prefix: sPrefix.trim() || null, credential_profile: sProfile.trim() || null,
      });
      setSName(""); setSBucket(""); setSPrefix(""); setSProfile("");
      await refresh();
      flash("source registered");
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const preview = async (id: string) => {
    setBusy(true);
    try {
      const r = await api.previewSource(id);
      flash(r.detail ? r.detail : `${r.count ?? 0} objects under ${r.uri}`);
      await refresh();
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const highlight = params.get("tab") === "webhooks";

  return (
    <PageShell
      active="INTEGRATIONS"
      title="Integrations"
      subtitle="outbound webhooks and registered storage buckets"
      right={msg ? <span className="font-mono text-[11px] text-accent">{msg}</span> : null}
    >
      <div className="p-4 space-y-4 max-w-5xl">
        {/* the one-time secret */}
        {freshSecret && (
          <div className="panel border border-accent p-3 space-y-1">
            <div className="font-mono text-[11px] text-accent">
              copy this signing secret now, it is not shown again
            </div>
            <div className="font-mono text-[11px] text-ink-3">{freshSecret.url}</div>
            <div className="font-mono text-[12px] text-ink break-all select-all">{freshSecret.secret}</div>
            <div className="font-mono text-[10px] text-ink-3">
              verify each delivery: X-Labelox-Signature = sha256=HMAC(secret, raw body)
            </div>
            <button onClick={() => setFreshSecret(null)}
              className="mt-1 border border-line px-2 py-0.5 font-mono text-[11px] text-ink-2 hover:border-accent">
              dismiss
            </button>
          </div>
        )}

        <Section title={`webhooks (${hooks.length})`}
          right={highlight ? <span className="font-mono text-[11px] text-accent">from the menu</span> : null}>
          <div className="space-y-2">
            <div className="flex items-center gap-2 font-mono text-[11px] flex-wrap">
              <input value={url} onChange={(e) => setUrl(e.target.value)}
                placeholder="https://your-pipeline/hook"
                className="flex-1 min-w-[260px] bg-bg border border-line px-1.5 py-0.5 text-ink" />
              <button onClick={addHook} disabled={busy}
                className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
                add webhook
              </button>
            </div>
            <div className="flex items-center gap-1 flex-wrap">
              <span className="font-mono text-[10px] uppercase text-ink-3 mr-1">events</span>
              {events.map((e) => (
                <button key={e}
                  onClick={() => setPicked((p) => p.includes(e) ? p.filter((x) => x !== e) : [...p, e])}
                  className={`px-1.5 py-0.5 font-mono text-[10.5px] border ${
                    picked.includes(e) ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
                  {e}
                </button>
              ))}
              <span className="font-mono text-[10px] text-ink-3 ml-1">
                {picked.length ? "" : "none picked = every event"}
              </span>
            </div>

            {hooks.length > 0 && (
              <table className="w-full font-mono text-[11px] mt-2">
                <thead>
                  <tr className="text-ink-3 text-left border-b hairline">
                    <th className="py-1">url</th><th>events</th><th>state</th><th>last</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {hooks.map((h) => (
                    <tr key={h.webhook_id} className="border-b hairline">
                      <td className="py-1 text-ink-2 truncate max-w-[280px]">{h.url}</td>
                      <td className="text-ink-3">{h.events.length ? h.events.join(", ") : "all"}</td>
                      <td className={h.active ? "text-pass" : "text-block"}>
                        {h.active ? "active" : `off (${h.failure_count} fails)`}
                      </td>
                      <td className="text-ink-3">{h.last_error ?? (h.last_status ?? "-")}</td>
                      <td className="text-right">
                        <button onClick={async () => { if (!(await confirm({ title: "Remove this webhook?", body: h.url, danger: true, confirmLabel: "Remove" }))) return; try { await api.deleteWebhook(h.webhook_id); toast("Webhook removed", "success"); } catch (e) { toast(humanizeError(e), "error"); } refresh(); }}
                          className="text-ink-3 hover:text-block">remove</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Section>

        <Section title={`storage sources (${sources.length})`}>
          <div className="space-y-2">
            <div className="flex items-center gap-2 font-mono text-[11px] flex-wrap">
              <input value={sName} onChange={(e) => setSName(e.target.value)} placeholder="name"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-32" />
              <select value={sProvider} onChange={(e) => setSProvider(e.target.value)}
                className="bg-bg border border-line px-1.5 py-0.5 text-ink">
                {["s3", "minio", "gcs", "azure"].map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
              <input value={sBucket} onChange={(e) => setSBucket(e.target.value)} placeholder="bucket"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-40" />
              <input value={sPrefix} onChange={(e) => setSPrefix(e.target.value)} placeholder="prefix (optional)"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-40" />
              <input value={sProfile} onChange={(e) => setSProfile(e.target.value)}
                placeholder="credential profile"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-40" />
              <button onClick={addSource} disabled={busy}
                className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
                register
              </button>
            </div>
            <div className="font-mono text-[10px] text-ink-3">
              credentials are never stored here: the profile names a key held in the server environment
            </div>

            {sources.length > 0 && (
              <table className="w-full font-mono text-[11px] mt-2">
                <thead>
                  <tr className="text-ink-3 text-left border-b hairline">
                    <th className="py-1">name</th><th>uri</th><th>profile</th><th>objects</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {sources.map((s) => (
                    <tr key={s.source_id} className="border-b hairline">
                      <td className="py-1 text-ink-2">{s.name}</td>
                      <td className="text-ink-3 truncate max-w-[260px]">{s.uri}</td>
                      <td className="text-ink-3">{s.credential_profile ?? "-"}</td>
                      <td className="text-ink-3">{s.last_object_count ?? "-"}</td>
                      <td className="text-right space-x-2">
                        <button onClick={() => preview(s.source_id)} disabled={busy}
                          className="text-ink-3 hover:text-accent disabled:opacity-40">preview</button>
                        <button onClick={async () => { if (!(await confirm({ title: "Remove this storage source?", body: s.name, danger: true, confirmLabel: "Remove" }))) return; try { await api.deleteSource(s.source_id); toast("Source removed", "success"); } catch (e) { toast(humanizeError(e), "error"); } refresh(); }}
                          className="text-ink-3 hover:text-block">remove</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Section>
      </div>
    </PageShell>
  );
}
