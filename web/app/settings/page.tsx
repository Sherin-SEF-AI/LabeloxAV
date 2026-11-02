"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import PageShell from "@/components/shell/PageShell";
import Icon from "@/components/shell/Icon";
import LoadState from "@/components/shell/LoadState";
import { apiGet } from "@/lib/api";
import { LABS_EVENT, LABS_PREF, readPref, setLabs } from "@/lib/prefs";
import { NAV } from "@/lib/sidebar";
import { useCurrentUser } from "@/lib/user";

// Settings: is this installation healthy, how does this browser show the app, and where the account and
// integration pages are.
//
// System status is the same set of checks `scripts/doctor.sh` runs inside the API container
// (services/doctor.py), so an administrator sees "the GPU worker is not running, start it with ..." here
// instead of inferring it from a job bar that never moves. The API restricts it to administrators, because
// the results name hosts, ports and model files.

type Check = { name: string; status: "ok" | "warn" | "fail" | "info"; detail: string; fix: string; group: string };
type Checks = { status: "ok" | "warn" | "fail"; checks: Check[] };

const GROUPS: [string, string][] = [["services", "Services"], ["models", "Models and GPU"], ["jobs", "Jobs"]];

const STATUS: Record<Check["status"], { label: string; cls: string; dot: string }> = {
  ok: { label: "OK", cls: "text-pass", dot: "bg-pass" },
  warn: { label: "Attention", cls: "text-warn", dot: "bg-warn" },
  fail: { label: "Failing", cls: "text-block", dot: "bg-block" },
  info: { label: "Info", cls: "text-ink-3", dot: "bg-ink-3" },
};

function Card({ title, description, children, action }: {
  title: string; description?: string; children: React.ReactNode; action?: React.ReactNode;
}) {
  return (
    <section className="panel">
      <header className="flex items-start justify-between gap-4 px-4 pt-4 pb-3">
        <div className="min-w-0">
          <h2 className="text-[14px] font-semibold text-ink">{title}</h2>
          {description && <p className="mt-0.5 text-[12.5px] text-ink-3 max-w-[65ch]">{description}</p>}
        </div>
        {action}
      </header>
      <div className="px-4 pb-4">{children}</div>
    </section>
  );
}

function SystemStatus() {
  const user = useCurrentUser();
  const [data, setData] = useState<Checks | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await apiGet<Checks>("/api/system/checks"));
      setCheckedAt(new Date());
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  if (user && user.role !== "admin") {
    return (
      <Card title="System status" description="Installation health is visible to administrators.">
        <p className="text-[12.5px] text-ink-3">Ask an administrator, or have them run <code className="font-mono text-ink-2">labeloxav doctor</code> on the server.</p>
      </Card>
    );
  }

  const overall = data ? STATUS[data.status] : null;
  return (
    <Card
      title="System status"
      description="The same checks as labeloxav doctor, run inside the application. Anything not OK says how to fix it."
      action={
        <button className="btn shrink-0" onClick={() => void load()} disabled={loading}>
          <Icon name="undo" size={13} />{loading ? "Checking" : "Check again"}
        </button>
      }>
      <LoadState loading={loading && !data} error={error} onRetry={() => void load()} />
      {data && overall && (
        <>
          <div className="flex items-center gap-2 mb-3 text-[12.5px]">
            <span className={`w-2 h-2 rounded-full ${overall.dot}`} aria-hidden />
            <span className={`font-medium ${overall.cls}`}>
              {data.status === "ok" ? "Everything is working" : data.status === "warn" ? "Working, with warnings" : "Something needs fixing"}
            </span>
            {checkedAt && <span className="text-ink-3">checked {checkedAt.toLocaleTimeString()}</span>}
          </div>
          <div className="flex flex-col gap-4">
            {GROUPS.map(([group, label]) => {
              const rows = data.checks.filter((c) => c.group === group);
              if (!rows.length) return null;
              return (
                <div key={group}>
                  <h3 className="side-section px-0 mb-1.5">{label}</h3>
                  <ul className="rounded-md border hairline divide-y divide-line overflow-hidden">
                    {rows.map((c) => {
                      const st = STATUS[c.status];
                      return (
                        <li key={c.name} className="grid grid-cols-[1fr_auto] sm:grid-cols-[minmax(150px,220px)_1fr_auto] gap-x-4 gap-y-1 px-3 py-2.5 bg-bg/40">
                          <span className="text-[12.5px] text-ink">{c.name}</span>
                          <span className="order-3 col-span-2 sm:order-none sm:col-span-1 text-[12.5px] text-ink-2 min-w-0 break-words">{c.detail}</span>
                          <span className={`order-2 sm:order-none text-[11.5px] font-medium ${st.cls} flex items-center gap-1.5 justify-self-end`}>
                            <span className={`w-1.5 h-1.5 rounded-full ${st.dot}`} aria-hidden />{st.label}
                          </span>
                          {c.fix && (c.status === "warn" || c.status === "fail") && (
                            <code className="order-4 col-span-2 sm:order-none sm:col-start-2 sm:col-span-2 block font-mono text-[11.5px] text-ink-2 bg-bg-2 border hairline rounded px-2 py-1.5 whitespace-pre-wrap break-words">
                              {c.fix}
                            </code>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              );
            })}
          </div>
        </>
      )}
    </Card>
  );
}

function Preferences() {
  const [labs, setLabsState] = useState(false);
  useEffect(() => {
    setLabsState(readPref(LABS_PREF) === "1");
    const on = () => setLabsState(readPref(LABS_PREF) === "1");
    window.addEventListener(LABS_EVENT, on);
    return () => window.removeEventListener(LABS_EVENT, on);
  }, []);
  const labsCount = NAV.filter((s) => s.labs).reduce((n, s) => n + s.items.length, 0);
  return (
    <Card title="Preferences" description="How this browser shows the app. Saved in this browser only.">
      <label className="flex items-start gap-3 cursor-pointer">
        <input type="checkbox" checked={labs} onChange={(e) => { setLabs(e.target.checked); setLabsState(e.target.checked); }}
          className="mt-0.5 h-4 w-4 accent-[#2f6fd6]" />
        <span>
          <span className="block text-[13px] text-ink">Show Labs in the sidebar</span>
          <span className="block text-[12.5px] text-ink-3 max-w-[65ch]">
            {labsCount} research tools outside the everyday labelling workflow: agent console, autonomy, calibration
            monitoring, pseudo ground truth, slice evaluation, edge deployment and more. They stay reachable from
            the command palette either way.
          </span>
        </span>
      </label>
    </Card>
  );
}

const LINKS: { href: string; label: string; hint: string; icon: string }[] = [
  { href: "/profile", label: "Your account", hint: "password, two-factor authentication, sessions", icon: "user" },
  { href: "/settings/team", label: "Team", hint: "invite teammates, issue sign-in links (administrators)", icon: "person" },
  { href: "/settings/projects", label: "Projects", hint: "who sees which sessions, members, moving sessions (administrators)", icon: "folder" },
  { href: "/settings/keys", label: "API keys", hint: "credentials for scripts, notebooks and CI (administrators)", icon: "terminal" },
  { href: "/integrations", label: "Integrations", hint: "storage buckets and webhooks", icon: "link" },
  { href: "/activity", label: "Activity", hint: "what you and the team have done", icon: "activity" },
  { href: "/govern/pii-access", label: "Personal data access log", hint: "who viewed unblurred data (administrators)", icon: "shield" },
  { href: "/settings/about", label: "About", hint: "version, where the source is, and the licences that apply", icon: "info" },
  { href: "/console", label: "Resource console", hint: "GPU, memory, disk and everything running now", icon: "server" },
];

export default function SettingsPage() {
  return (
    <PageShell active="Settings" title="Settings" subtitle="System status, preferences and account">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 flex flex-col gap-5">
        <SystemStatus />
        <Preferences />
        <Card title="More settings">
          <ul className="grid sm:grid-cols-2 gap-2">
            {LINKS.map((l) => (
              <li key={l.href}>
                <Link href={l.href} className="flex items-start gap-3 rounded-md border hairline px-3 py-2.5 hover:bg-head transition-colors">
                  <span className="text-ink-3 mt-0.5"><Icon name={l.icon} size={16} /></span>
                  <span className="min-w-0">
                    <span className="block text-[13px] text-ink">{l.label}</span>
                    <span className="block text-[12px] text-ink-3">{l.hint}</span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </PageShell>
  );
}
