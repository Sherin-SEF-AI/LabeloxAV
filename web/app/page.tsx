"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import PageShell from "@/components/shell/PageShell";
import Icon from "@/components/shell/Icon";
import LoadState from "@/components/shell/LoadState";
import { api, apiGet, humanizeError } from "@/lib/api";
import { useT, type Translate } from "@/lib/i18n";
import { toast } from "@/lib/toast";
import { invalidate, useQuery } from "@/lib/useQuery";
import { useCurrentUser } from "@/lib/user";
import SampleDrives from "@/components/SampleDrives";
import { humanize, pct } from "@/lib/format";

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
  waiting_for_people: (
    | { kind: "blind_audit"; audit_id: string; job_id: string; frame_id: string;
        n_frames: number; n_labeled: number; created_at: string | null }
    | { kind: "control_samples"; count: number; href: string }
    | { kind: "settlement_lot"; lot_id: string; class_name: string; judged: number; of: number;
        population: number; href: string }
  )[];
};
type Check = { name: string; status: "ok" | "warn" | "fail" | "info"; detail: string; fix: string };

// The English here is what renders in English and what a missing translation falls back to; `k` is the
// dictionary key (lib/locales). Kept side by side so the page still reads as what it shows.
type Text = { k: string; en: string };
const STEPS: { key: keyof Home["steps"]; title: Text; body: Text; actions: { label: Text; href: string }[] }[] = [
  { key: "data", title: { k: "home.step.data.title", en: "Bring in data" },
    body: { k: "home.step.data.body", en: "Upload dashcam video or images, or import a dataset already labelled in COCO, YOLO, KITTI, nuScenes, CVAT and more. Faces and plates are blurred as frames are stored." },
    actions: [{ label: { k: "home.upload_footage", en: "Upload footage" }, href: "/annotate/new" },
              { label: { k: "shell.import_a_dataset", en: "Import a dataset" }, href: "/import" }] },
  { key: "autolabel", title: { k: "home.step.autolabel.title", en: "Auto-label a session" },
    body: { k: "home.step.autolabel.body", en: "Open a session and run auto-label. Three detectors propose every object; confident ones are accepted, the rest wait for a person." },
    actions: [{ label: { k: "home.open_sessions", en: "Open sessions" }, href: "/annotations" }] },
  { key: "review", title: { k: "home.step.review.title", en: "Review the labels" },
    body: { k: "home.step.review.body", en: "Accept, fix or reject what the models proposed, one keystroke per object. The most useful objects come first." },
    actions: [{ label: { k: "home.start_reviewing", en: "Start reviewing" }, href: "/review/queue" }] },
  { key: "export", title: { k: "home.step.export.title", en: "Export a dataset" },
    body: { k: "home.step.export.body", en: "Freeze a versioned dataset in the format your training code reads." },
    actions: [{ label: { k: "shell.datasets", en: "Datasets" }, href: "/datasets" }] },
  { key: "model", title: { k: "home.step.model.title", en: "Train a model" },
    body: { k: "home.step.model.body", en: "Fine-tune a detector on your reviewed labels and compare it with the current one, class by class." },
    actions: [{ label: { k: "shell.training", en: "Training" }, href: "/training" }] },
];

const n = (v: number) => v.toLocaleString();
const KIND: Record<string, Text> = {
  autolabel: { k: "home.kind.autolabel", en: "Auto-label" },
  import: { k: "shell.import", en: "Import" },
  training: { k: "shell.training", en: "Training" },
};

function ago(iso: string | null, t: Translate): string {
  if (!iso) return "";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return t("home.ago.now", "just now");
  if (s < 5400) return t("home.ago.min", "{n} min ago", { n: Math.round(s / 60) });
  if (s < 129600) return t("home.ago.hours", "{n} h ago", { n: Math.round(s / 3600) });
  return t("home.ago.days", "{n} days ago", { n: Math.round(s / 86400) });
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
    <Link href={href} className="panel px-4 py-3.5 hover:border-line-3 hover:bg-head transition-colors">{body}</Link>
  ) : (
    <div className="panel px-4 py-3.5">{body}</div>
  );
}

function Card({ title, action, children }: { title: string; action?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="panel flex flex-col min-w-0">
      <header className="flex items-center justify-between gap-3 px-4 pt-3.5 pb-2.5">
        <h2 className="section-title">{title}</h2>
        {action}
      </header>
      <div className="px-4 pb-4 min-w-0">{children}</div>
    </section>
  );
}

function AutolabelNow({ session, onChange }: { session: Home["recent_sessions"][number]; onChange: () => void }) {
  const [busy, setBusy] = useState(false);
  const [queued, setQueued] = useState<string | null>(null);
  const t = useT();
  const start = async () => {
    setBusy(true);
    try {
      const r = await api.startAutolabel(session.session_id);
      setQueued(r.job_id);
      toast(t("home.autolabel.started", "Auto-label started on {vehicle}", { vehicle: session.vehicle_id }), "success");
      onChange();
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setBusy(false);
    }
  };
  if (queued) {
    return <Link href="/jobs" className="btn">{t("home.autolabel.follow", "Follow it in Jobs")}</Link>;
  }
  return (
    <button className="btn btn-primary" onClick={() => void start()} disabled={busy}>
      {busy ? t("home.autolabel.starting", "Starting")
        : t("home.autolabel.run", "Auto-label {vehicle}", { vehicle: session.vehicle_id })}
    </button>
  );
}

function Setup({ steps, recent, onChange }: { steps: Home["steps"]; recent: Home["recent_sessions"]; onChange: () => void }) {
  // The session a newcomer most likely wants labelled: the newest one with frames and no objects yet.
  const unlabelled = recent.find((r) => r.frames > 0 && r.objects === 0);
  const done = STEPS.filter((s) => steps[s.key]).length;
  const next = STEPS.find((s) => !steps[s.key]);
  const t = useT();
  return (
    <section className="panel p-5">
      <div className="flex items-baseline justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-[16px] font-semibold text-ink">{t("home.setup.title", "Get started")}</h2>
          <p className="mt-1 text-[13px] text-ink-3">{t("home.setup.body", "From raw footage to a trained detector. Each step ticks itself off when the data says it is done.")}</p>
        </div>
        <span className="text-[12.5px] text-ink-2 tabular-nums">{t("home.setup.progress", "{done} of {total} done", { done, total: STEPS.length })}</span>
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
                <div className={`text-[13.5px] ${isDone ? "text-ink-3 line-through decoration-ink-3/40" : "text-ink"}`}>{t(s.title.k, s.title.en)}</div>
                {isNext && <p className="mt-0.5 text-[12.5px] text-ink-2 max-w-[70ch]">{t(s.body.k, s.body.en)}</p>}
                {isNext && s.key === "data" && (
                  <div className="mt-3">
                    <div className="mb-1.5 text-[12px] text-ink-3">{t("home.setup.sample", "No footage yet? Load a sample drive to try the whole loop:")}</div>
                    <SampleDrives onLoaded={onChange} />
                  </div>
                )}
              </div>
              {isNext && (
                <div className="flex gap-2 shrink-0 flex-wrap justify-end">
                  {s.key === "autolabel" && unlabelled && <AutolabelNow session={unlabelled} onChange={onChange} />}
                  {s.actions.map((a, j) => (
                    <Link key={a.href} href={a.href}
                      className={`btn ${j === 0 && !(s.key === "autolabel" && unlabelled) ? "btn-primary" : ""}`}>{t(a.label.k, a.label.en)}</Link>
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
  const t = useT();
  // Same key as the settings page, which reads the same checks: opening one after the other is one request.
  const checks = useQuery(["/api/system/checks"], () => apiGet<{ checks: Check[] }>("/api/system/checks"),
    { enabled: user?.role === "admin" });
  const bad = checks.data ? checks.data.checks.filter((c) => c.status === "warn" || c.status === "fail") : null;
  const failed = checks.data ? null : checks.error;
  // Not knowing is not the same as everything being fine, so a failed check says so.
  if (failed) {
    return <p className="text-[12.5px] text-warn">{t("home.system.failed", "Could not load system status: {error}", { error: humanizeError(failed) })}</p>;
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
      <li><Link href="/settings" className="text-[12.5px] text-accent hover:text-accent-2">{t("home.system.fix", "How to fix these")}</Link></li>
    </ul>
  );
}

export default function HomePage() {
  const router = useRouter();
  const t = useT();
  // Cached under the path, so coming back to Home shows the last numbers at once and refreshes them
  // behind, instead of a spinner over a page that was there a moment ago.
  const home = useQuery(["/api/home"], () => apiGet<Home>("/api/home"));
  const data = home.data ?? null;
  // A failed refresh keeps the last numbers on screen and says so above them, as the old load() did.
  const error = home.error;
  // Starting an auto-label or loading a sample drive changes the counts here and the job lists elsewhere.
  const load = async () => { await Promise.all([home.reload(), invalidate("jobs", "sessions", "sessionsPage", "sessionStats")]); };

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
      <div className="page-wrap flex flex-col gap-5">
        <LoadState loading={home.loading} error={home.loadState.error} onRetry={() => void load()} />

        {data && !setupDone && <Setup steps={data.steps} recent={data.recent_sessions} onChange={() => void load()} />}

        {data && (data.waiting_for_people ?? []).length > 0 && (
          <section className="panel" aria-label={t("home.waiting.title", "Waiting for you")}>
            <div className="panel-head">
              <h2 className="section-title">{t("home.waiting.title", "Waiting for you")}</h2>
              <span className="ml-auto text-[12px] text-ink-3 tabular-nums">{data.waiting_for_people.length}</span>
            </div>
            <ul className="grid gap-3 p-3.5 grid-cols-[repeat(auto-fill,minmax(300px,1fr))]">
            {data.waiting_for_people.map((w) => {
              // Each task is a card: what to do, one plain sentence on why, its progress when it has any, and
              // the button directly beneath, so the action is never a wide empty gap away from what it does.
              const row = (key: string, icon: string, title: string, why: string, href: string, label: string,
                progress?: { done: number; of: number }) => (
                <li key={key} className="flex flex-col gap-2 rounded-md border hairline bg-head/40 px-3.5 py-3">
                  <div className="flex items-start gap-2.5">
                    <span className="shrink-0 mt-0.5 text-ink-3" aria-hidden><Icon name={icon} size={15} /></span>
                    <div className="min-w-0">
                      <div className="text-[13.5px] text-ink font-medium leading-snug">{title}</div>
                      <div className="mt-1 text-[12.5px] text-ink-3 leading-relaxed">{why}</div>
                    </div>
                  </div>
                  <div className="mt-auto pt-1 flex items-center gap-3">
                    <Link href={href} className="btn btn-primary shrink-0">{label}</Link>
                    {progress && progress.of > 0 && (
                      <span className="flex-1 flex items-center gap-2 min-w-0" aria-hidden>
                        <span className="flex-1 h-1.5 rounded-full bg-line overflow-hidden">
                          <span className="block h-full bg-accent" style={{ width: `${Math.min(100, (progress.done / progress.of) * 100)}%` }} />
                        </span>
                        <span className="text-[11.5px] text-ink-3 tabular-nums">{pct(progress.done / progress.of)}</span>
                      </span>
                    )}
                  </div>
                </li>
              );
              if (w.kind === "blind_audit") {
                // Two whole sentences rather than one with a clause bolted on: where "done so far" goes is
                // the translator's call, and a suffix concatenated in English order would take it away.
                const why = "The model's boxes are hidden while you label. Comparing your labels with the model's " +
                  "afterwards shows how much it misses. You can split the work across people and sittings";
                return row(w.audit_id, "flag",
                  t("home.audit.title", "Label {remaining} of {total} frames from scratch (blind check)",
                    { remaining: w.n_frames - w.n_labeled, total: w.n_frames }),
                  w.n_labeled ? t("home.audit.why_progress", `${why}; {n} done so far.`, { n: w.n_labeled })
                    : t("home.audit.why", `${why}.`),
                  `/frame/${w.frame_id}?job=${w.job_id}`,
                  w.n_labeled ? t("home.audit.continue", "Continue labelling") : t("home.audit.start", "Start labelling"),
                  { done: w.n_labeled, of: w.n_frames });
              }
              if (w.kind === "control_samples") {
                return row("controls", "check", t("home.controls.title", "Check {n} labels that were accepted without a person", { n: n(w.count) }),
                  t("home.controls.why", "A random sample of labels the system accepted on its own. Your answers show how often it gets them right. One key per label."),
                  w.href, t("home.controls.go", "Check them"));
              }
              // The class name is stored as `traffic_sign`; the sentence reads "traffic sign".
              return row(w.lot_id, "layers",
                t("home.lot.title", "Check {of} sample {klass} labels from a batch of {population} ({judged} done)",
                  { population: n(w.population), klass: humanize(w.class_name).toLowerCase(), judged: w.judged, of: w.of }),
                t("home.lot.why", "Your answers decide whether the whole batch is accepted or sent back. One key per label."),
                w.href, w.judged ? t("home.lot.continue", "Continue") : t("home.lot.start", "Start"),
                { done: w.judged, of: w.of });
            })}
            </ul>
          </section>
        )}

        {data && c && (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
              <Stat label={t("shell.sessions", "Sessions")} value={n(c.sessions)} href="/annotations" sub={t("home.stat.sessions_sub", "recordings and uploads")} />
              <Stat label={t("home.stat.frames", "Frames")} value={n(c.frames)} sub={t("home.stat.frames_sub", "faces and plates blurred")} />
              <Stat label={t("home.stat.objects", "Objects")} value={n(c.objects)} href="/explore" sub={t("home.stat.objects_sub", "{n} settled", { n: n(c.settled) })} />
              <Stat label={t("home.stat.waiting", "Waiting for review")} value={n(c.waiting)} href="/review/queue" tone={c.waiting ? "accent" : undefined}
                sub={c.objects ? t("home.stat.waiting_sub", "{pct}% of all objects", { pct: Math.round((c.waiting / c.objects) * 100) })
                  : t("home.stat.nothing", "nothing yet")} />
              <Stat label={t("home.stat.jobs", "Jobs queued or running")} value={n(data.jobs.active_total)} href="/jobs"
                sub={data.jobs.active_total
                  ? t("home.stat.jobs_sub", "{list} ({n} running)", { n: n(data.jobs.running),
                      list: Object.entries(data.jobs.active).filter(([, v]) => v).map(([k, v]) => `${v} ${k}`).join(", ") })
                  : t("home.stat.none", "none")} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-[1.7fr_1fr] gap-5">
              <Card title={t("home.recent.title", "Recent sessions")} action={<Link href="/annotations" className="text-[12.5px] text-accent hover:text-accent-2">{t("home.recent.all", "All sessions")}</Link>}>
                {data.recent_sessions.length ? (
                  <div className="overflow-x-auto -mx-1">
                    <table className="w-full text-[12.5px] tabular-nums">
                      <thead>
                        <tr className="text-left text-ink-3 border-b hairline">
                          <th scope="col" className="font-medium px-1 py-2">{t("home.recent.session", "Session")}</th>
                          <th scope="col" className="font-medium px-1 py-2">{t("home.recent.recorded", "Recorded")}</th>
                          <th scope="col" className="font-medium px-1 py-2 text-right">{t("home.stat.frames", "Frames")}</th>
                          <th scope="col" className="font-medium px-1 py-2 text-right">{t("home.stat.objects", "Objects")}</th>
                          <th scope="col" className="font-medium px-1 py-2 text-right">{t("home.recent.to_review", "To review")}</th>
                          <td className="px-1 py-2" />
                        </tr>
                      </thead>
                      <tbody>
                        {data.recent_sessions.map((s) => (
                          <tr key={s.session_id} className="border-b hairline last:border-0">
                            <td className="px-1 py-2 text-ink">{s.vehicle_id}{s.city && <span className="text-ink-3"> · {s.city}</span>}</td>
                            <td className="px-1 py-2 text-ink-2">{new Date(s.start_ts_ns / 1e6).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}</td>
                            <td className="px-1 py-2 text-right text-ink-2">{n(s.frames)}</td>
                            <td className="px-1 py-2 text-right text-ink-2">{n(s.objects)}</td>
                            <td className={`px-1 py-2 text-right ${s.waiting ? "text-ink" : "text-ink-3"}`}>{n(s.waiting)}</td>
                            <td className="px-1 py-2 text-right">
                              <button className="btn" onClick={() => void open(s.session_id)} disabled={!s.frames}
                                aria-label={`${t("common.open", "Open")} ${s.vehicle_id}`}>{t("common.open", "Open")}</button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="py-6 text-center">
                    <p className="text-[13px] text-ink-3">{t("home.recent.empty", "No sessions yet.")}</p>
                    <div className="mt-3 flex justify-center gap-2">
                      <Link href="/annotate/new" className="btn btn-primary"><Icon name="upload" size={13} />{t("home.upload_footage", "Upload footage")}</Link>
                      <Link href="/import" className="btn">{t("shell.import_a_dataset", "Import a dataset")}</Link>
                    </div>
                  </div>
                )}
              </Card>

              <div className="flex flex-col gap-5 min-w-0">
                <Card title={t("home.quick.title", "Quick actions")}>
                  <div className="grid grid-cols-2 gap-2">
                    {[["home.upload_footage", "Upload footage", "/annotate/new", "upload"], ["shell.import_a_dataset", "Import a dataset", "/import", "database"],
                      ["home.quick.review", "Review labels", "/review/queue", "check"], ["home.step.model.title", "Train a model", "/training", "chip"]].map(([k, label, href, icon]) => (
                      <Link key={href} href={href} className="flex items-center gap-2 rounded-md border hairline px-3 py-2 text-[12.5px] text-ink-2 hover:bg-head hover:text-ink transition-colors">
                        <span className="text-ink-3"><Icon name={icon} size={14} /></span>{t(k, label)}
                      </Link>
                    ))}
                  </div>
                </Card>

                <Card title={t("home.attention.title", "Needs attention")} action={<Link href="/jobs" className="text-[12.5px] text-accent hover:text-accent-2">{t("shell.jobs", "Jobs")}</Link>}>
                  <div className="flex flex-col gap-3">
                    <SystemAttention />
                    {data.jobs.failed.length ? (
                      <ul className="flex flex-col gap-2.5">
                        {data.jobs.failed.map((f) => (
                          <li key={f.job_id} className="flex items-start gap-2.5">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 bg-block" aria-hidden />
                            <div className="min-w-0">
                              <div className="text-[12.5px] text-ink">{t("home.attention.failed", "{kind} failed", { kind: KIND[f.kind] ? t(KIND[f.kind].k, KIND[f.kind].en) : f.kind })} <span className="text-ink-3">{ago(f.at, t)}</span></div>
                              <div className="text-[12px] text-ink-3 break-words line-clamp-2">{f.error || t("home.attention.no_reason", "no reason recorded")}</div>
                            </div>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-[12.5px] text-ink-3">{t("home.attention.none", "No failed jobs in the last {days} days.", { days: data.jobs.failed_window_days })}</p>
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
