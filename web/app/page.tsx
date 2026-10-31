"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import PageShell from "@/components/shell/PageShell";
import Icon from "@/components/shell/Icon";
import LoadState from "@/components/shell/LoadState";
import { api, apiGet, humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import { useCurrentUser } from "@/lib/user";

// Home: what is waiting, what is running, where work was last happening, and, for a new team, the five steps
// from an empty install to a trained model. Every number comes from GET /api/home, which counts the tables
// at request time. The ranked object queue that used to live here is "My queue" (/queue).

type Home = {
  corpus: { sessions: number; frames: number; objects: number; waiting: number; settled: number;
            by_state: Record<string, number> };
  jobs: { active: Record<string, number>; active_total: number; running: number; failed_window_days: number;
          failed: { kind: string; job_id: string; error: string; at: string | null }[] };
  recent_sessions: { session_id: string; vehicle_id: string; city: string | null; start_ts_ns: number;
                     frames: number; objects: number; waiting: number }[];
  steps: { data: boolean; autolabel: boolean; review: boolean; export: boolean; model: boolean };
};
type Check = { name: string; status: "ok" | "warn" | "fail" | "info"; detail: string; fix: string };

const STEPS: { key: keyof Home["steps"]; title: string; body: string; actions: { label: string; href: string }[] }[] = [
  { key: "data", title: "Bring in data", body: "Upload dashcam video or images, or import a dataset already labelled in COCO, YOLO, KITTI, nuScenes, CVAT and more. Faces and plates are blurred as frames are stored.",
    actions: [{ label: "Upload footage", href: "/annotate/new" }, { label: "Import a dataset", href: "/import" }] },
  { key: "autolabel", title: "Auto-label a session", body: "Open a session and run auto-label. Three detectors propose every object; confident ones are accepted, the rest wait for a person.",
    actions: [{ label: "Open sessions", href: "/annotations" }] },
  { key: "review", title: "Review the labels", body: "Accept, fix or reject what the models proposed, one keystroke per object. The most useful objects come first.",
    actions: [{ label: "Start reviewing", href: "/review/queue" }] },
  { key: "export", title: "Export a dataset", body: "Freeze a versioned dataset in the format your training code reads.",
    actions: [{ label: "Datasets", href: "/datasets" }] },
  { key: "model", title: "Train a model", body: "Fine-tune a detector on your reviewed labels and compare it with the current one, class by class.",
    actions: [{ label: "Training", href: "/training" }] },
];

const n = (v: number) => v.toLocaleString();
const KIND: Record<string, string> = { autolabel: "Auto-label", import: "Import", training: "Training" };

function ago(iso: string | null): string {
  if (!iso) return "";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return "just now";
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 129600) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} days ago`;
}

function Stat({ label, value, sub, href, tone }: {
  label: string; value: string; sub?: string; href?: string; tone?: "warn" | "accent";
}) {
  const body = (
    <>
      <div className="text-[12px] text-ink-3">{label}</div>
      <div className={`mt-1 text-[24px] leading-none font-semibold tabular-nums tracking-tight ${tone === "warn" ? "text-warn" : tone === "accent" ? "text-accent-2" : "text-ink"}`}>{value}</div>
      {sub && <div className="mt-1.5 text-[12px] text-ink-3 truncate">{sub}</div>}
    </>
  );
  return href ? (
    <Link href={href} className="panel px-4 py-3.5 hover:border-[#383b42] hover:bg-head transition-colors">{body}</Link>
  ) : (
    <div className="panel px-4 py-3.5">{body}</div>
  );
}

function Card({ title, action, children }: { title: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="panel flex flex-col min-w-0">
      <header className="flex items-center justify-between gap-3 px-4 pt-3.5 pb-2.5">
        <h2 className="text-[13.5px] font-semibold text-ink">{title}</h2>
        {action}
      </header>
      <div className="px-4 pb-4 min-w-0">{children}</div>
    </section>
  );
}

function Setup({ steps }: { steps: Home["steps"] }) {
  const done = STEPS.filter((s) => steps[s.key]).length;
  const next = STEPS.find((s) => !steps[s.key]);
  return (
    <section className="panel p-5">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-[16px] font-semibold text-ink">Get started</h2>
          <p className="mt-1 text-[13px] text-ink-3">From raw footage to a trained detector. Each step ticks itself off when the data says it is done.</p>
        </div>
        <span className="text-[12.5px] text-ink-2 tabular-nums">{done} of {STEPS.length} done</span>
      </div>
      <div className="mt-3 h-1 rounded-full bg-line overflow-hidden" aria-hidden>
        <div className="h-full bg-accent" style={{ width: `${(done / STEPS.length) * 100}%` }} />
      </div>
      <ol className="mt-4 grid gap-2">
        {STEPS.map((s, i) => {
          const isDone = steps[s.key];
          const isNext = next?.key === s.key;
          return (
            <li key={s.key} className={`flex items-start gap-3 rounded-md px-3 py-2.5 ${isNext ? "bg-head border hairline" : ""}`}>
              <span className={`mt-0.5 shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-[11px] font-semibold ${isDone ? "bg-pass/15 text-pass" : isNext ? "bg-accent/20 text-accent-2" : "bg-line text-ink-3"}`}>
                {isDone ? <Icon name="check" size={12} /> : i + 1}
              </span>
              <div className="flex-1 min-w-0">
                <div className={`text-[13.5px] ${isDone ? "text-ink-3 line-through decoration-ink-3/40" : "text-ink"}`}>{s.title}</div>
                {isNext && <p className="mt-0.5 text-[12.5px] text-ink-2 max-w-[70ch]">{s.body}</p>}
              </div>
              {isNext && (
                <div className="flex gap-2 shrink-0 flex-wrap justify-end">
                  {s.actions.map((a, j) => (
                    <Link key={a.href} href={a.href} className={`btn ${j === 0 ? "btn-primary" : ""}`}>{a.label}</Link>
                  ))}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function SystemAttention() {
  const user = useCurrentUser();
  const [bad, setBad] = useState<Check[] | null>(null);
  const [failed, setFailed] = useState<unknown>(null);
  useEffect(() => {
    if (user?.role !== "admin") return;
    apiGet<{ checks: Check[] }>("/api/system/checks")
      .then((r) => setBad(r.checks.filter((c) => c.status === "warn" || c.status === "fail")))
      .catch((e) => setFailed(e));
  }, [user?.role]);
  // Not knowing is not the same as everything being fine, so a failed check says so.
  if (failed) {
    return <p className="text-[12.5px] text-warn">Could not load system status: {humanizeError(failed)}</p>;
  }
  if (!bad || !bad.length) return null;
  return (
    <ul className="flex flex-col gap-2">
      {bad.map((c) => (
        <li key={c.name} className="flex items-start gap-2.5">
          <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${c.status === "fail" ? "bg-block" : "bg-warn"}`} aria-hidden />
          <div className="min-w-0">
            <div className="text-[12.5px] text-ink">{c.name}</div>
            <div className="text-[12px] text-ink-3 break-words">{c.detail}</div>
          </div>
        </li>
      ))}
      <li><Link href="/settings" className="text-[12.5px] text-accent hover:text-accent-2">How to fix these</Link></li>
    </ul>
  );
}

export default function HomePage() {
  const router = useRouter();
  const [data, setData] = useState<Home | null>(null);
  const [error, setError] = useState<unknown>(null);

  const load = useCallback(async () => {
    setError(null);
    try { setData(await apiGet<Home>("/api/home")); } catch (e) { setError(e); }
  }, []);
  useEffect(() => { void load(); }, [load]);

  const open = async (sessionId: string) => {
    try {
      const { frame_id } = await api.firstFrame(sessionId);
      router.push(`/frame/${frame_id}`);
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  };

  const setupDone = data ? Object.values(data.steps).every(Boolean) : true;
  const c = data?.corpus;

  return (
    <PageShell active="Home" title="Home">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6 flex flex-col gap-5">
        <LoadState loading={!data && !error} error={error} onRetry={() => void load()} />

        {data && !setupDone && <Setup steps={data.steps} />}

        {data && c && (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
              <Stat label="Sessions" value={n(c.sessions)} href="/annotations" sub="recordings and uploads" />
              <Stat label="Frames" value={n(c.frames)} sub="stored, faces and plates blurred" />
              <Stat label="Objects" value={n(c.objects)} href="/explore" sub={`${n(c.settled)} settled`} />
              <Stat label="Waiting for review" value={n(c.waiting)} href="/review/queue" tone={c.waiting ? "accent" : undefined}
                sub={c.objects ? `${Math.round((c.waiting / c.objects) * 100)}% of all objects` : "nothing yet"} />
              <Stat label="Jobs queued or running" value={n(data.jobs.active_total)} href="/jobs"
                sub={data.jobs.active_total
                  ? `${n(data.jobs.running)} running: ${Object.entries(data.jobs.active).filter(([, v]) => v).map(([k, v]) => `${v} ${k}`).join(", ")}`
                  : "none"} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[1.7fr_1fr] gap-5">
              <Card title="Recent sessions" action={<Link href="/annotations" className="text-[12.5px] text-accent hover:text-accent-2">All sessions</Link>}>
                {data.recent_sessions.length ? (
                  <div className="overflow-x-auto -mx-1">
                    <table className="w-full text-[12.5px] tabular-nums">
                      <thead>
                        <tr className="text-left text-ink-3 border-b hairline">
                          <th className="font-medium px-1 py-2">Session</th>
                          <th className="font-medium px-1 py-2">Recorded</th>
                          <th className="font-medium px-1 py-2 text-right">Frames</th>
                          <th className="font-medium px-1 py-2 text-right">Objects</th>
                          <th className="font-medium px-1 py-2 text-right">To review</th>
                          <th className="px-1 py-2" />
                        </tr>
                      </thead>
                      <tbody>
                        {data.recent_sessions.map((s) => (
                          <tr key={s.session_id} className="border-b hairline last:border-0">
                            <td className="px-1 py-2 text-ink">{s.vehicle_id}{s.city && <span className="text-ink-3"> · {s.city}</span>}</td>
                            <td className="px-1 py-2 text-ink-2">{new Date(s.start_ts_ns / 1e6).toLocaleDateString()}</td>
                            <td className="px-1 py-2 text-right text-ink-2">{n(s.frames)}</td>
                            <td className="px-1 py-2 text-right text-ink-2">{n(s.objects)}</td>
                            <td className={`px-1 py-2 text-right ${s.waiting ? "text-ink" : "text-ink-3"}`}>{n(s.waiting)}</td>
                            <td className="px-1 py-2 text-right">
                              <button className="btn" onClick={() => void open(s.session_id)} disabled={!s.frames}>Open</button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="py-6 text-center">
                    <p className="text-[13px] text-ink-3">No sessions yet.</p>
                    <div className="mt-3 flex justify-center gap-2">
                      <Link href="/annotate/new" className="btn btn-primary"><Icon name="upload" size={13} />Upload footage</Link>
                      <Link href="/import" className="btn">Import a dataset</Link>
                    </div>
                  </div>
                )}
              </Card>

              <div className="flex flex-col gap-5 min-w-0">
                <Card title="Quick actions">
                  <div className="grid grid-cols-2 gap-2">
                    {[["Upload footage", "/annotate/new", "upload"], ["Import a dataset", "/import", "database"],
                      ["Review labels", "/review/queue", "check"], ["Train a model", "/training", "chip"]].map(([label, href, icon]) => (
                      <Link key={href} href={href} className="flex items-center gap-2 rounded-md border hairline px-3 py-2 text-[12.5px] text-ink-2 hover:bg-head hover:text-ink transition-colors">
                        <span className="text-ink-3"><Icon name={icon} size={14} /></span>{label}
                      </Link>
                    ))}
                  </div>
                </Card>

                <Card title="Needs attention" action={<Link href="/jobs" className="text-[12.5px] text-accent hover:text-accent-2">Jobs</Link>}>
                  <div className="flex flex-col gap-3">
                    <SystemAttention />
                    {data.jobs.failed.length ? (
                      <ul className="flex flex-col gap-2.5">
                        {data.jobs.failed.map((f) => (
                          <li key={f.job_id} className="flex items-start gap-2.5">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 bg-block" aria-hidden />
                            <div className="min-w-0">
                              <div className="text-[12.5px] text-ink">{KIND[f.kind] ?? f.kind} failed <span className="text-ink-3">{ago(f.at)}</span></div>
                              <div className="text-[12px] text-ink-3 break-words line-clamp-2">{f.error || "no reason recorded"}</div>
                            </div>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-[12.5px] text-ink-3">No failed jobs in the last {data.jobs.failed_window_days} days.</p>
                    )}
                  </div>
                </Card>
              </div>
            </div>
          </>
        )}
      </div>
    </PageShell>
  );
}
