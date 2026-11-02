"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import Icon from "@/components/shell/Icon";
import SessionSelect from "@/components/shell/SessionSelect";
import { apiGet, apiPost, humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import { useCurrentUser } from "@/lib/user";

// Ask in words; the assistant answers from the data with read-only tools, and anything that would change
// labels or start work comes back as a proposal a reviewer approves (services/assistant). Nothing on this
// page changes data except the Approve button.

type Status = { provider: string; model: string; ready: boolean; local: boolean; detail: string | null };
type TraceStep = { tool: string; arguments: Record<string, unknown>; kind: string; error: string | null };
type Proposal = {
  proposal_id: string; tool: string; summary: string; status: string; request: string | null;
  preview: { count?: number | null; unit?: string; human_labels_left_alone?: number; sample?: { class_name: string; frame_id: string }[] };
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
};

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
        <span className={`text-[11px] uppercase tracking-[0.06em] shrink-0 ${STATUS_TONE[p.status] ?? "text-ink-3"}`}>{p.status}</span>
      </div>
      {p.request && <p className="text-[12px] text-ink-3">Asked: &ldquo;{p.request}&rdquo;</p>}
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[12px] text-ink-2 tabular-nums">
        {p.preview.count != null && <span>{p.preview.count.toLocaleString()} {p.preview.unit ?? "labels"}</span>}
        {classes.length > 0 && <span>{classes.join(", ")}</span>}
        {(p.preview.human_labels_left_alone ?? 0) > 0 && <span>{p.preview.human_labels_left_alone} human labels left alone</span>}
      </div>
      {p.status === "failed" && p.result && <p className="text-[12px] text-block">{String((p.result as { error?: string }).error ?? "")}</p>}
      {p.status === "pending" && (canDecide ? (
        <div className="flex gap-2">
          <button className="btn btn-primary" disabled={busy} onClick={() => void decide("approve")}>Approve</button>
          <button className="btn" disabled={busy} onClick={() => void decide("reject")}>Reject</button>
        </div>
      ) : <p className="text-[12px] text-ink-3">A reviewer approves or rejects this.</p>)}
    </li>
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
    <PageShell active="Assistant" title="Assistant" subtitle="ask about the data; changes wait for a reviewer"
      right={status && (
        <span className={`text-[12px] ${status.ready ? "text-ink-3" : "text-warn"}`} title={status.detail ?? undefined}>
          {status.model}{status.local ? " · on this machine" : ` · ${status.provider}`}{status.ready ? "" : " · not ready"}
        </span>
      )}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,380px)] items-start">
        <section className="panel flex flex-col min-h-[60vh]">
          {status && !status.ready && (
            <div className="px-4 py-2.5 border-b hairline text-[12.5px] text-warn">The model is not ready: {status.detail}</div>
          )}
          <div className="px-4 py-3 border-b hairline flex flex-wrap items-center gap-2 text-[12.5px]">
            <span className="text-ink-3">Working in</span>
            <div className="w-full sm:w-80"><SessionSelect value={sessionId} onChange={setSessionId} placeholder="no particular session" /></div>
            {sessionId && <button className="text-accent hover:text-accent-2" onClick={() => setSessionId("")}>clear</button>}
          </div>
          <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4 max-h-[62vh]" aria-live="polite">
            {turns.length === 0 && (
              <div className="flex flex-col gap-3 text-[13px] text-ink-2">
                <p>Ask about labels, sessions, the review queue or likely errors. Questions are answered from the data; anything that would change labels or start work becomes a proposal for a reviewer to approve.</p>
                <div className="flex flex-wrap gap-2">
                  {EXAMPLES.map((x) => <button key={x} className="btn text-left" onClick={() => void send(x)}>{x}</button>)}
                </div>
              </div>
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
            {busy && <div className="text-[12.5px] text-ink-3">Thinking...</div>}
            <div ref={end} />
          </div>
          <form className="border-t hairline p-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); void send(text); }}>
            <input className="field flex-1 min-w-0" value={text} onChange={(e) => setText(e.target.value)} maxLength={2000}
                   placeholder="e.g. how many cattle are there in this session?" aria-label="Message" disabled={busy} />
            <button className="btn btn-primary shrink-0" type="submit" disabled={busy || !text.trim()}>
              <Icon name="next" size={13} />Send
            </button>
          </form>
        </section>

        <section className="panel p-4 flex flex-col gap-3">
          <header className="flex items-center gap-2">
            <h2 className="text-[13.5px] font-semibold text-ink">Proposals{pending ? ` (${pending} waiting)` : ""}</h2>
            <label className="ml-auto flex items-center gap-1.5 text-[12px] text-ink-3">
              <input type="checkbox" checked={showDecided} onChange={(e) => setShowDecided(e.target.checked)} />
              show decided
            </label>
          </header>
          <p className="text-[12px] text-ink-3">
            Approving runs the change the way a person would: class changes go back to review and can be reverted; auto-labelling queues a job.
          </p>
          {proposals.length === 0
            ? <p className="text-[12.5px] text-ink-3">Nothing waiting.</p>
            : <ul className="flex flex-col gap-2.5">{proposals.map((p) => (
                <ProposalCard key={p.proposal_id} p={p} canDecide={canDecide} onDecided={() => void loadProposals()} />
              ))}</ul>}
        </section>
      </div>
    </PageShell>
  );
}

export default function Page() {
  return <Suspense fallback={null}><AssistantPage /></Suspense>;
}
