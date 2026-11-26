"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { T } from "@/lib/i18n";
import { useSearchParams } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import Icon from "@/components/shell/Icon";
import SessionSelect from "@/components/shell/SessionSelect";
import { apiGet, apiPost, humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import { useCurrentUser } from "@/lib/user";
import EmptyState from "@/components/ui/EmptyState";
import { Segmented } from "@/components/ui/kit";
import { ago, humanize, num } from "@/lib/format";
import {
  approvalState, approveEdit, editTable, getEditProposal, leftAloneText, listEditProposals, previewEdit,
  proposeEdit, rejectEdit, revertEdit, type EditPreview, type EditProposal,
} from "@/lib/tools/agentsApi";

// Ask in words; the assistant answers from the data with read-only tools, and anything that would change
// labels or start work comes back as a proposal a reviewer approves (services/assistant). Nothing on this
// page changes data except the Approve button (and Undo, which reverses an applied edit).
//
// "Edit data" mode (#17): one sentence changes labels across many sessions ("relabel every pickup in Delhi
// sessions as LCV"). It previews per session and class with sample crops and what it leaves alone, then becomes
// a proposal like any other; a reviewer approves it and it lands as one reversible agent run.

type Status = { provider: string; model: string; ready: boolean; local: boolean; detail: string | null };
type TraceStep = { tool: string; arguments: Record<string, unknown>; kind: string; error: string | null };
type Proposal = {
  proposal_id: string; tool: string; summary: string; status: string; request: string | null;
  preview: { count?: number | null; unit?: string; human_labels_left_alone?: number; sessions?: number;
             sample?: { class_name: string; frame_id: string }[] };
  result: Record<string, unknown> | null; created_at: string | null; decided_at: string | null;
};
type Turn = { role: "user" | "assistant"; content: string; trace?: TraceStep[]; proposals?: string[] };

const EXAMPLES = [
  "How many autorickshaws are waiting for review?",
  "Which drives do we have from Delhi?",
  "Send the cyclists in this session back for review",
  "Look for label mistakes in this session",
];

const TOOL_LABEL: Record<string, string> = {
  list_classes: "looked up classes", count_objects: "counted labels", find_sessions: "searched sessions",
  session_overview: "read the session", queue_overview: "read the review queue", error_overview: "read error candidates",
  propose_reclassify: "proposed a class change", propose_flag_for_review: "proposed a review",
  propose_autolabel: "proposed auto-labelling", propose_error_detection: "proposed error detection",
  list_datasets: "read datasets", model_status: "read models and training", list_projects: "read projects",
  job_overview: "read jobs", propose_export: "proposed an export", propose_training: "proposed training",
  propose_bulk_edit: "proposed an edit across sessions",
};

const EDIT_EXAMPLES = [
  "relabel every pickup in Delhi sessions as LCV",
  "send all cyclists at night in Bengaluru drives back for review",
  "send unreviewed autorickshaws from the last 7 days back for review",
];

const STATUS_TONE: Record<string, string> = {
  pending: "text-warn", approved: "text-pass", rejected: "text-ink-3", failed: "text-block",
};

function ProposalCard({ p, canDecide, onDecided }: { p: Proposal; canDecide: boolean; onDecided: () => void }) {
  const [busy, setBusy] = useState(false);
  const decide = async (verb: "approve" | "reject") => {
    setBusy(true);
    try {
      const r = await apiPost<Proposal>(`/api/assistant/proposals/${p.proposal_id}/${verb}`, {});
      toast(r.status === "approved" ? "Approved and applied" : r.status === "failed"
        ? `Could not apply: ${String((r.result as { error?: string } | null)?.error ?? "unknown reason")}` : "Rejected",
      r.status === "failed" ? "error" : "success");
      onDecided();
    } catch (e) { toast(humanizeError(e), "error"); }
    finally { setBusy(false); }
  };
  const classes = [...new Set((p.preview.sample ?? []).map((s) => s.class_name))];
  return (
    <li className="border hairline rounded p-3 flex flex-col gap-2">
      <div className="flex items-start gap-2">
        <span className="text-[13px] text-ink flex-1">{p.summary}</span>
        <span className={`text-[12px] font-medium shrink-0 ${STATUS_TONE[p.status] ?? "text-ink-3"}`}>{humanize(p.status)}</span>
      </div>
      {p.request && <p className="text-[12px] text-ink-3">Asked: &ldquo;{p.request}&rdquo;{p.created_at ? ` · ${ago(p.created_at)}` : ""}</p>}
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-ink-2 tabular-nums">
        {p.preview.count != null && <span>{p.preview.count.toLocaleString()} {p.preview.unit ?? "labels"}</span>}
        {(p.preview.sessions ?? 0) > 1 && <span>{p.preview.sessions} sessions</span>}
        {classes.length > 0 && <span>{classes.map(humanize).join(", ")}</span>}
        {(p.preview.human_labels_left_alone ?? 0) > 0 && <span>{p.preview.human_labels_left_alone} human labels left alone</span>}
      </div>
      {p.status === "failed" && p.result && <p className="text-[12px] text-block">{String((p.result as { error?: string }).error ?? "")}</p>}
      {p.status === "pending" && (canDecide ? (
        <div className="flex gap-2">
          <button className="btn btn-primary" disabled={busy} onClick={() => void decide("approve")}>Approve</button>
          <button className="btn" disabled={busy} onClick={() => void decide("reject")}><T k="common.reject">Reject</T></button>
        </div>
      ) : <p className="text-[12px] text-ink-3">A reviewer approves or rejects this.</p>)}
    </li>
  );
}

function EditCounts({ p }: { p: EditPreview }) {
  const t = editTable(p);
  return (
    <div className="overflow-x-auto">
      <table className="data-table w-full">
        <thead>
          <tr><th>Session</th><th>Recorded</th>{t.classes.map((c) => <th key={c} className="text-right">{humanize(c)}</th>)}
            {t.rows.some((r) => r.other > 0) && <th className="text-right">Other</th>}<th className="text-right">Total</th></tr>
        </thead>
        <tbody>
          {t.rows.slice(0, 40).map(({ session: s, cells, other }) => (
            <tr key={s.session_id}>
              <td><a className="link" href={`/?session=${s.session_id}`}>{s.vehicle ?? s.session_id.slice(0, 8)}</a>
                {s.city && <span className="text-ink-3 ml-1.5">{s.city}</span>}</td>
              <td className="text-ink-3">{s.date ?? "-"}</td>
              {cells.map((n, i) => <td key={i} className="text-right tabular-nums">{n ? num(n) : ""}</td>)}
              {t.rows.some((r) => r.other > 0) && <td className="text-right tabular-nums">{other ? num(other) : ""}</td>}
              <td className="text-right tabular-nums font-medium">{num(s.total)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {t.rows.length > 40 && <p className="text-[11.5px] text-ink-3 mt-1">and {t.rows.length - 40} more sessions</p>}
    </div>
  );
}

function EditDataPanel({ sessionId, canDecide, onProposed }: { sessionId: string; canDecide: boolean; onProposed: () => void }) {
  const [cmd, setCmd] = useState("");
  const [preview, setPreview] = useState<EditPreview | null>(null);
  const [previewedFor, setPreviewedFor] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [proposal, setProposal] = useState<EditProposal | null>(null);
  const [recent, setRecent] = useState<EditProposal[]>([]);

  const loadRecent = useCallback(async () => {
    try { setRecent((await listEditProposals(8)).proposals); } catch { /* secondary */ }
  }, []);
  useEffect(() => { void loadRecent(); }, [loadRecent]);
  // Follow the proposal until a reviewer decides it, so the person who asked sees it land.
  useEffect(() => {
    if (!proposal || proposal.status !== "pending") return;
    const t = setTimeout(() => { void getEditProposal(proposal.proposal_id).then(setProposal).catch(() => undefined); }, 4000);
    return () => clearTimeout(t);
  }, [proposal]);

  const run = async (raw: string) => {
    const c = raw.trim();
    if (c.length < 3 || busy) return;
    setCmd(c); setBusy("preview"); setErr(null); setProposal(null);
    try { setPreview(await previewEdit(c, sessionId || undefined)); setPreviewedFor(c + "|" + sessionId); }
    catch (e) { setErr(humanizeError(e)); setPreview(null); }
    finally { setBusy(null); }
  };
  const propose = async () => {
    setBusy("propose"); setErr(null);
    try { setProposal(await proposeEdit(cmd, sessionId || undefined)); onProposed(); void loadRecent(); toast("Sent for approval", "success"); }
    catch (e) { setErr(humanizeError(e)); }
    finally { setBusy(null); }
  };
  const decide = async (fn: () => Promise<unknown>, ok: string) => {
    if (!proposal) return;
    setBusy("decide");
    try { await fn(); setProposal(await getEditProposal(proposal.proposal_id)); toast(ok, "success"); onProposed(); void loadRecent(); }
    catch (e) { toast(humanizeError(e), "error"); }
    finally { setBusy(null); }
  };
  const stale = preview && previewedFor !== cmd.trim() + "|" + sessionId;
  const left = leftAloneText(preview?.left_alone);
  const st = proposal ? approvalState(proposal) : null;

  return (
    <div className="flex-1 flex flex-col">
      <form className="px-4 py-3 border-b hairline flex gap-2" onSubmit={(e) => { e.preventDefault(); void run(cmd); }}>
        <input className="field flex-1 min-w-0" value={cmd} onChange={(e) => setCmd(e.target.value)} maxLength={500}
               placeholder="e.g. relabel every pickup in Delhi sessions as LCV" aria-label="Edit command" disabled={!!busy} />
        <button className="btn btn-primary shrink-0" type="submit" disabled={!!busy || cmd.trim().length < 3}>
          {busy === "preview" ? "Reading..." : "Preview"}
        </button>
      </form>
      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4 max-h-[66vh] [&>*]:shrink-0">
        {err && <div role="alert" className="text-[12.5px] text-block border border-block/30 bg-block/5 rounded px-3 py-2">{err}</div>}
        {!preview && !err && (
          <EmptyState icon="wand" title="Change labels across many sessions in one sentence"
            action={
              <div className="flex flex-col items-stretch gap-2 w-full max-w-md">
                <span className="label-caps text-left">For example</span>
                {EDIT_EXAMPLES.map((x) => <button key={x} className="btn justify-start text-left" onClick={() => void run(x)}>{x}</button>)}
              </div>
            }>
            Name the classes, the change (relabel as a class, or send back for review) and where: a city, a vehicle,
            dates, night or rain, or a dataset query after &ldquo;where&rdquo;. You see every count before anything
            changes, and a reviewer approves it. Labels a person confirmed are never changed.
          </EmptyState>
        )}
        {preview && (
          <>
            <div className="flex flex-col gap-1.5">
              <div className="label-caps">Read as</div>
              <div className="flex flex-wrap gap-1.5">{preview.interpretation.map((x) => <span key={x} className="chip chip-on">{x}</span>)}</div>
            </div>
            {preview.warnings.length > 0 && (
              <ul className="text-[12.5px] text-warn border border-warn/40 bg-warn/5 rounded px-3 py-2 list-disc pl-6">
                {preview.warnings.map((w) => <li key={w}>{w}</li>)}
              </ul>
            )}
            <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
              <span className="text-[22px] font-semibold tabular-nums text-ink">{num(preview.count)}</span>
              <span className="text-[13px] text-ink-2">label{preview.count === 1 ? "" : "s"} would change, in {preview.sessions} session{preview.sessions === 1 ? "" : "s"}</span>
            </div>
            {left && <p className="text-[12.5px] text-ink-2">{left}</p>}
            {preview.by_session.length > 0 && <EditCounts p={preview} />}
            {!proposal && (
              <div className="flex items-center gap-3">
                <button className="btn btn-primary" disabled={!preview.can_propose || !!busy || !!stale} onClick={() => void propose()}>
                  {busy === "propose" ? "Sending..." : "Send for approval"}
                </button>
                <span className="text-[12px] text-ink-3">
                  {stale ? "The command changed; preview it again first." : preview.can_propose
                    ? "Nothing changes until a reviewer approves." : "Nothing to send: see the notes above."}
                </span>
              </div>
            )}
            {proposal && st && (
              <div className="border hairline rounded p-3 flex flex-col gap-2" role="status">
                <div className="flex items-center gap-2">
                  <span className={`text-[12.5px] font-medium ${st.tone === "pass" ? "text-pass" : st.tone === "warn" ? "text-warn" : st.tone === "block" ? "text-block" : "text-ink-3"}`}>{st.label}</span>
                  <span className="text-[12px] text-ink-3 ml-auto">{proposal.summary}</span>
                </div>
                {proposal.status === "failed" && <p className="text-[12px] text-block">{proposal.result?.error}</p>}
                {proposal.status === "approved" && proposal.result?.edited != null && (
                  <p className="text-[12px] text-ink-2">{num(proposal.result.edited)} label(s) changed and sent to review
                    {proposal.result.skipped && Object.keys(proposal.result.skipped).length > 0
                      ? `; ${Object.entries(proposal.result.skipped).map(([k, v]) => `${v} ${humanize(k)}`).join(", ")} left alone since the preview` : ""}.</p>
                )}
                <div className="flex gap-2">
                  {proposal.status === "pending" && (canDecide ? (
                    <>
                      <button className="btn btn-primary" disabled={!!busy} onClick={() => void decide(() => approveEdit(proposal.proposal_id), "Approved and applied")}>Approve</button>
                      <button className="btn" disabled={!!busy} onClick={() => void decide(() => rejectEdit(proposal.proposal_id), "Rejected")}>Reject</button>
                    </>
                  ) : <span className="text-[12px] text-ink-3">A reviewer approves or rejects this; it is in the Proposals list.</span>)}
                  {proposal.status === "approved" && proposal.run?.status === "committed" && canDecide && (
                    <button className="btn" disabled={!!busy} onClick={() => void decide(() => revertEdit(proposal.proposal_id), "Undone")}>Undo</button>
                  )}
                </div>
              </div>
            )}
            {preview.sample.length > 0 && (
              <div>
                <div className="label-caps mb-1.5">Sample</div>
                <div className="grid grid-cols-4 sm:grid-cols-6 xl:grid-cols-8 gap-2">
                  {preview.sample.map((o) => (
                    <a key={o.object_id} href={`/frame/${o.frame_id}`} className="block border hairline rounded overflow-hidden bg-bg-2" title={`${o.class_name} · ${humanize(o.state)} · conf ${o.conf}`}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={o.crop_url} alt={o.class_name} loading="lazy" className="w-full aspect-square object-contain" />
                      <div className="px-1.5 py-0.5 text-[11px] text-ink-2 truncate">{humanize(o.class_name)}</div>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
        {recent.length > 0 && (
          <div className="border-t hairline pt-3">
            <div className="label-caps mb-1.5">Recent edits across sessions</div>
            <ul className="flex flex-col gap-1.5 text-[12.5px]">
              {recent.map((r) => {
                const s = approvalState(r);
                return (
                  <li key={r.proposal_id} className="flex items-center gap-2 min-w-0">
                    <button className="link truncate text-left" onClick={() => { setProposal(r); setCmd(r.request ?? ""); }}>{r.request ?? r.summary}</button>
                    <span className="text-ink-3 shrink-0">{r.created_at ? ago(r.created_at) : ""}</span>
                    <span className={`ml-auto shrink-0 ${s.tone === "pass" ? "text-pass" : s.tone === "warn" ? "text-warn" : s.tone === "block" ? "text-block" : "text-ink-3"}`}>{s.label}</span>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

function AssistantPage() {
  const me = useCurrentUser();
  const params = useSearchParams();
  const [status, setStatus] = useState<Status | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState(params.get("session") ?? "");
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [showDecided, setShowDecided] = useState(false);
  const [mode, setMode] = useState<"ask" | "edit">(params.get("mode") === "edit" ? "edit" : "ask");
  const end = useRef<HTMLDivElement | null>(null);
  const canDecide = me?.role === "reviewer" || me?.role === "admin";

  const loadProposals = useCallback(async () => {
    try {
      const r = await apiGet<{ proposals: Proposal[] }>(`/api/assistant/proposals?status=${showDecided ? "all" : "pending"}&limit=50`);
      setProposals(r.proposals);
    } catch { /* the list is secondary; the chat still works */ }
  }, [showDecided]);
  useEffect(() => {
    apiGet<Status>("/api/assistant/status").then(setStatus).catch((e) => setStatus({
      provider: "", model: "assistant", ready: false, local: false, detail: humanizeError(e) }));
  }, []);
  useEffect(() => { void loadProposals(); }, [loadProposals]);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [turns, busy]);

  const send = async (raw: string) => {
    const content = raw.trim();
    if (!content || busy) return;
    const next: Turn[] = [...turns, { role: "user", content }];
    setTurns(next);
    setText("");
    setBusy(true);
    try {
      const r = await apiPost<{ reply: string; trace: TraceStep[]; proposals: Proposal[] }>("/api/assistant/chat", {
        messages: next.map(({ role, content }) => ({ role, content })),
        session_id: sessionId || undefined,
      });
      setTurns([...next, { role: "assistant", content: r.reply || "(no answer)", trace: r.trace,
                           proposals: r.proposals.map((p) => p.proposal_id) }]);
      if (r.proposals.length) void loadProposals();
    } catch (e) {
      setTurns([...next, { role: "assistant", content: humanizeError(e) }]);
    } finally { setBusy(false); }
  };

  const pending = proposals.filter((p) => p.status === "pending").length;

  return (
    <PageShell active="Assistant" title="Assistant" subtitle="Ask about the data in plain words; any change it suggests waits for a reviewer"
      right={status && (
        <span className={`text-[12px] ${status.ready ? "text-ink-3" : "text-warn"}`} title={status.detail ?? undefined}>
          {status.model}{status.local ? " · on this machine" : ` · ${status.provider}`}{status.ready ? "" : " · not ready"}
        </span>
      )}>
      <div className="page-wrap grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,380px)] items-start">
        <section className="panel flex flex-col min-h-[60vh]">
          {status && !status.ready && (
            <div role="alert" className="px-4 py-2.5 border-b hairline text-[12.5px] text-warn bg-warn/5">The assistant&apos;s model is not ready{status.detail ? `: ${status.detail}` : "."}</div>
          )}
          <div className="px-4 py-3 border-b hairline flex flex-wrap items-center gap-2 text-[12.5px]">
            <span className="text-ink-3">Working in</span>
            <div className="w-full sm:w-80"><SessionSelect value={sessionId} onChange={setSessionId} placeholder="Any session" /></div>
            {sessionId && <button className="link" onClick={() => setSessionId("")}><T k="action.clear">Clear</T></button>}
            <div className="ml-auto">
              <Segmented value={mode} onChange={setMode} options={[
                { value: "ask", label: "Ask", tip: "Ask about the data" },
                { value: "edit", label: "Edit data", tip: "Change labels across many sessions in one sentence" },
              ]} />
            </div>
          </div>
          {mode === "edit" ? <EditDataPanel sessionId={sessionId} canDecide={canDecide} onProposed={() => void loadProposals()} /> : (<>
          <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4 max-h-[62vh]" aria-live="polite">
            {turns.length === 0 && (
              <EmptyState icon="comment" title="Ask about labels, sessions, the review queue or likely errors"
                action={
                  <div className="flex flex-col items-stretch gap-2 w-full max-w-md">
                    <span className="label-caps text-left">Try asking</span>
                    {EXAMPLES.map((x) => (
                      <button key={x} className="btn justify-start text-left" onClick={() => void send(x)}>{x}</button>
                    ))}
                  </div>
                }>
                Answers come from the data. Anything that would change labels or start work becomes a proposal
                for a reviewer to approve.
              </EmptyState>
            )}
            {turns.map((t, i) => (
              <div key={i} className={`flex flex-col gap-1.5 ${t.role === "user" ? "items-end" : "items-start"}`}>
                <div className={`max-w-[85%] rounded px-3 py-2 text-[13px] whitespace-pre-wrap ${t.role === "user" ? "bg-accent/10 text-ink" : "bg-bg-2 text-ink"}`}>
                  {t.content}
                </div>
                {t.trace && t.trace.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 text-[11px] text-ink-3">
                    {t.trace.map((s, j) => (
                      <span key={j} className={`border rounded px-1.5 py-0.5 ${s.error ? "border-warn/40 text-warn" : "hairline"}`}
                            title={`${s.tool}(${JSON.stringify(s.arguments)})${s.error ? `: ${s.error}` : ""}`}>
                        {TOOL_LABEL[s.tool] ?? s.tool}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {busy && <div role="status" className="self-start rounded px-3 py-2 bg-bg-2 text-[13px] text-ink-3">Thinking...</div>}
            <div ref={end} />
          </div>
          <form className="border-t hairline p-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); void send(text); }}>
            <input className="field flex-1 min-w-0" value={text} onChange={(e) => setText(e.target.value)} maxLength={2000}
                   placeholder="Ask a question, e.g. how many cattle are in this session?" aria-label="Message" disabled={busy} />
            <button className="btn btn-primary shrink-0" type="submit" disabled={busy || !text.trim()}>
              <Icon name="next" size={13} />Send
            </button>
          </form>
          </>)}
        </section>

        <section className="panel flex flex-col">
          <header className="panel-head">
            <h2 className="section-title">Proposals{pending ? ` (${pending} waiting)` : ""}</h2>
            <label className="ml-auto flex items-center gap-1.5 text-[12px] text-ink-3 cursor-pointer">
              <input type="checkbox" checked={showDecided} onChange={(e) => setShowDecided(e.target.checked)} />
              Show decided
            </label>
          </header>
          <div className="p-3.5 flex flex-col gap-3">
            <p className="section-desc">
              Approving makes the change the way a person would: class changes go back to review and can be undone;
              auto-labelling starts a job.
            </p>
            {proposals.length === 0
              ? <EmptyState compact icon="check" title={showDecided ? "No proposals yet" : "Nothing waiting"}>
                  When the assistant suggests a change, it appears here for a reviewer.
                </EmptyState>
              : <ul className="flex flex-col gap-2.5">{proposals.map((p) => (
                  <ProposalCard key={p.proposal_id} p={p} canDecide={canDecide} onDecided={() => void loadProposals()} />
                ))}</ul>}
          </div>
        </section>
      </div>
    </PageShell>
  );
}

export default function Page() {
  return <Suspense fallback={null}><AssistantPage /></Suspense>;
}
