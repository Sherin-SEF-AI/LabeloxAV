"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { T } from "@/lib/i18n";
import { useSearchParams } from "next/navigation";
import { api , humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { useConfirm } from "@/components/ConfirmProvider";
import { toast } from "@/lib/toast";
import { invalidate, useQuery } from "@/lib/useQuery";
import { humanize } from "@/lib/format";
import EmptyState from "@/components/ui/EmptyState";

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

function Section({ title, count, desc, children, right }: {
  title: string; count?: number; desc?: string; children: React.ReactNode; right?: React.ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="section-title">
          {title}{count !== undefined && <span className="ml-1 font-normal text-ink-3 tabular-nums">({count})</span>}
        </h2>
        {right && <span className="ml-auto">{right}</span>}
      </div>
      <div className="p-4 space-y-3">
        {desc && <p className="section-desc">{desc}</p>}
        {children}
      </div>
    </section>
  );
}

// `annotation.created` -> "Annotation created", `anpr.watchlist_hit` -> "ANPR watchlist hit". The event name
// itself is what a receiver matches on, so it stays in the tooltip and is what the API is sent.
const eventLabel = (e: string) => humanize(e.replace(/\./g, " ")).replace(/^Anpr\b/, "ANPR");


export default function IntegrationsPage() {
  // useSearchParams forces a client-side-render bailout, which Next requires be under a Suspense boundary or
  // the static prerender of this route fails the build. Match the pattern used by login and search.
  return <Suspense fallback={null}><IntegrationsBody /></Suspense>;
}

const NO_HOOKS: Webhook[] = [];
const NO_SOURCES: Source[] = [];
const NO_EVENTS: string[] = [];

function IntegrationsBody() {
  const params = useSearchParams();
  const eventsQ = useQuery(["integrationEvents"], () => api.integrationEvents());
  const sourcesQ = useQuery(["storageSources"], () => api.storageSources());
  // Webhooks are admin-gated, so a non-admin simply sees an empty list rather than an error wall.
  const hooksQ = useQuery(["webhooks"], () => api.webhooks());
  const hooks: Webhook[] = hooksQ.error ? NO_HOOKS : hooksQ.data?.webhooks ?? NO_HOOKS;
  const sources: Source[] = sourcesQ.data?.sources ?? NO_SOURCES;
  const events: string[] = eventsQ.data?.events ?? NO_EVENTS;
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

  // A failed read of the events or the sources is flashed, as it was.
  const failure = eventsQ.error ?? sourcesQ.error;
  useEffect(() => {
    if (failure) flash(humanizeError(failure));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [failure]);
  const refresh = useCallback(() => invalidate("integrationEvents", "storageSources", "webhooks"), []);

  const addHook = async () => {
    if (!url.trim()) { flash("Give the webhook an address"); return; }
    setBusy(true);
    try {
      const wh = await api.createWebhook(url.trim(), picked);
      setFreshSecret({ url: wh.url, secret: wh.secret });
      setUrl(""); setPicked([]);
      await refresh();
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const addSource = async () => {
    if (!sName.trim() || !sBucket.trim()) { flash("A name and a bucket are required"); return; }
    setBusy(true);
    try {
      await api.registerSource({
        name: sName.trim(), provider: sProvider, bucket: sBucket.trim(),
        prefix: sPrefix.trim() || null, credential_profile: sProfile.trim() || null,
      });
      setSName(""); setSBucket(""); setSPrefix(""); setSProfile("");
      await refresh();
      flash("Source registered");
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
      subtitle="Webhooks that tell other systems what happened, and storage buckets to import from"
      right={msg ? <span className="text-[12px] text-accent-2" role="status">{msg}</span> : null}
    >
      <div className="page-wrap-narrow space-y-4">
        {/* the one-time secret */}
        {freshSecret && (
          <section className="panel border-accent/60 p-4 space-y-2" role="alert">
            <h2 className="section-title">Copy this signing secret now. It is not shown again.</h2>
            <div className="text-[12.5px] text-ink-3">For {freshSecret.url}</div>
            <div className="font-mono text-[13px] text-ink rounded border border-line bg-bg-2 px-3 py-2 break-all select-all">{freshSecret.secret}</div>
            <p className="text-[12px] text-ink-3">
              Your receiver verifies each delivery by checking that the <code className="font-mono text-ink-2">X-Labelox-Signature</code> header
              equals <code className="font-mono text-ink-2">sha256=HMAC(secret, raw body)</code>.
            </p>
            <button type="button" onClick={() => setFreshSecret(null)} className="btn">
              <T k="action.dismiss">Dismiss</T>
            </button>
          </section>
        )}

        <Section title="Webhooks" count={hooks.length}
          desc="Each event is sent as a signed POST to the address you give, so a pipeline can react without polling."
          right={highlight ? <span className="text-[12px] text-accent-2">Opened from the menu</span> : null}>
          <label className="block space-y-1.5">
            <span className="block text-[12.5px] text-ink-2">Address</span>
            <div className="flex items-center gap-2 flex-wrap">
              <input value={url} onChange={(e) => setUrl(e.target.value)}
                placeholder="https://your-pipeline/hook" className="field flex-1 min-w-[260px]" />
              <button type="button" onClick={addHook} disabled={busy} className="btn btn-primary">
                Add webhook
              </button>
            </div>
          </label>
          <div className="space-y-1.5">
            <div className="text-[12.5px] text-ink-2">
              Events <span className="text-ink-3">{picked.length ? `(${picked.length} picked)` : "(none picked sends every event)"}</span>
            </div>
            <div className="flex items-center gap-1.5 flex-wrap">
              {events.map((e) => (
                <button key={e} type="button" title={e} aria-pressed={picked.includes(e)}
                  onClick={() => setPicked((p) => p.includes(e) ? p.filter((x) => x !== e) : [...p, e])}
                  className={`chip ${picked.includes(e) ? "chip-on" : ""}`}>
                  {eventLabel(e)}
                </button>
              ))}
            </div>
          </div>

          {hooks.length > 0 ? (
            <div className="overflow-x-auto rounded-md border hairline">
              <table className="data-table">
                <thead>
                  <tr><th>Address</th><th>Events</th><th>State</th><th>Last delivery</th><th><span className="sr-only">Actions</span></th></tr>
                </thead>
                <tbody>
                  {hooks.map((h) => (
                    <tr key={h.webhook_id}>
                      <td className="text-ink truncate max-w-[280px]" title={h.url}>{h.url}</td>
                      <td>{h.events.length ? h.events.map(eventLabel).join(", ") : "All events"}</td>
                      <td className={h.active ? "text-pass" : "text-block"}>
                        {h.active ? "Active" : `Off after ${h.failure_count} failures`}
                      </td>
                      <td className="text-ink-3">{h.last_error ?? (h.last_status != null ? `HTTP ${h.last_status}` : "None yet")}</td>
                      <td className="text-right">
                        <button type="button" onClick={async () => { if (!(await confirm({ title: "Remove this webhook?", body: h.url, danger: true, confirmLabel: "Remove" }))) return; try { await api.deleteWebhook(h.webhook_id); toast("Webhook removed", "success"); } catch (e) { toast(humanizeError(e), "error"); } refresh(); }}
                          className="btn btn-danger"><T k="action.remove">Remove</T></button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded-md border hairline">
              <EmptyState compact icon="link" title="No webhooks yet">
                Add an address above to be told when labels are created, exports finish or a model is promoted.
              </EmptyState>
            </div>
          )}
        </Section>

        <Section title="Storage sources" count={sources.length}
          desc="Buckets the import page can read from. Credentials are never stored here: the profile names a key held in the server's environment.">
          <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-3 items-end">
            <label className="flex flex-col gap-1.5">
              <span className="text-[12.5px] text-ink-2">Name</span>
              <input value={sName} onChange={(e) => setSName(e.target.value)} placeholder="e.g. Fleet uploads" className="field" />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-[12.5px] text-ink-2">Provider</span>
              <select value={sProvider} onChange={(e) => setSProvider(e.target.value)} className="field">
                {[["s3", "S3"], ["minio", "MinIO"], ["gcs", "GCS"], ["azure", "Azure"]].map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-[12.5px] text-ink-2">Bucket</span>
              <input value={sBucket} onChange={(e) => setSBucket(e.target.value)} placeholder="bucket-name" className="field" />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-[12.5px] text-ink-2">Prefix <span className="text-ink-3">(optional)</span></span>
              <input value={sPrefix} onChange={(e) => setSPrefix(e.target.value)} placeholder="drives/2026/" className="field" />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-[12.5px] text-ink-2">Credential profile</span>
              <input value={sProfile} onChange={(e) => setSProfile(e.target.value)} placeholder="e.g. fleet-readonly" className="field" />
            </label>
            <button type="button" onClick={addSource} disabled={busy} className="btn btn-primary justify-self-start">
              Register
            </button>
          </div>

          {sources.length > 0 ? (
            <div className="overflow-x-auto rounded-md border hairline">
              <table className="data-table">
                <thead>
                  <tr><th>Name</th><th>Location</th><th>Profile</th><th className="num">Objects</th><th><span className="sr-only">Actions</span></th></tr>
                </thead>
                <tbody>
                  {sources.map((s) => (
                    <tr key={s.source_id}>
                      <td className="text-ink">{s.name}</td>
                      <td className="font-mono text-[12px] text-ink-3 truncate max-w-[260px]" title={s.uri}>{s.uri}</td>
                      <td className="text-ink-3">{s.credential_profile ?? "-"}</td>
                      <td className="num">{s.last_object_count != null ? s.last_object_count.toLocaleString() : "-"}</td>
                      <td className="text-right whitespace-nowrap space-x-2">
                        <button type="button" onClick={() => preview(s.source_id)} disabled={busy} className="btn">Preview</button>
                        <button type="button" onClick={async () => { if (!(await confirm({ title: "Remove this storage source?", body: s.name, danger: true, confirmLabel: "Remove" }))) return; try { await api.deleteSource(s.source_id); toast("Source removed", "success"); } catch (e) { toast(humanizeError(e), "error"); } refresh(); }}
                          className="btn btn-danger"><T k="action.remove">Remove</T></button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded-md border hairline">
              <EmptyState compact icon="database" title="No storage sources yet">
                Register a bucket above and it becomes a place to import recordings from.
              </EmptyState>
            </div>
          )}
        </Section>
      </div>
    </PageShell>
  );
}
