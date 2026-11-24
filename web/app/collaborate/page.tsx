"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { invalidate, useQuery } from "@/lib/useQuery";
import { useCurrentUser } from "@/lib/user";
import type { AssignmentRow, MergeRequestRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { StateBadge } from "@/components/StateBadge";

// M4.3 collaboration console: lakeFS branches, annotator assignments, and merge requests. Annotators work
// on isolated branches; reviewers approve, merge to main, and can revert a bad merge. Color is earned: a
// merged MR is pass-green, reverted is block-red, open is info-blue.

const NO_BRANCHES: string[] = [];
const NO_ASGS: AssignmentRow[] = [];
const NO_MRS: MergeRequestRow[] = [];

export default function CollaboratePage() {
  const branchesQ = useQuery(["collabBranches"], () => api.collabBranches());
  const asgsQ = useQuery(["collabAssignments"], () => api.collabAssignments());
  const mrsQ = useQuery(["collabMRs"], () => api.collabMRs());
  const branches: string[] = branchesQ.data?.branches ?? NO_BRANCHES;
  const asgs: AssignmentRow[] = asgsQ.data ?? NO_ASGS;
  const mrs: MergeRequestRow[] = mrsQ.data ?? NO_MRS;
  const [msg, setMsg] = useState<string | null>(null);
  // Without this the page kept rendering an empty shell after a failed request, which is
  // indistinguishable from having no data.
  const loadErr = branchesQ.loadState.error ?? asgsQ.loadState.error ?? mrsQ.loadState.error ?? null;
  // Read in an effect, not in the body: getUser() touches localStorage and rendering it directly made
  // the server say one thing and the client another, which React resolves by discarding the server HTML.
  const me = useCurrentUser();
  const canReview = me?.role === "reviewer" || me?.role === "admin";

  // An approve, a merge or a revert moves a merge request and can move branches and assignments with it.
  const load = useCallback(() => invalidate("collabBranches", "collabAssignments", "collabMRs"), []);

  const act = async (fn: () => Promise<unknown>, label: string) => {
    if (!me) { setMsg("pick a user first"); return; }
    const r = (await fn()) as { error?: string; status?: string };
    setMsg(r.error ? r.error : `${label}: ${r.status ?? "ok"}`);
    await load();
  };

  return (
    <PageShell
      active="COLLABORATE"
      title="Collaboration Console"
      right={msg ? <span className="text-[12px] text-warn">{msg}</span> : undefined}
    >
      <div className="p-4 space-y-4 text-[12px]">
      {loadErr != null && <LoadState error={loadErr} onRetry={() => void load()} />}
        <div className="flex items-center gap-2 text-ink-3">
          <span>actor: <span className="text-ink-2">{me ? `${me.name} (${me.role})` : "none"}</span></span>
          {!canReview && <span className="text-warn">reviewer or admin role needed to approve/merge</span>}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <section className="panel">
            <div className="uppercase text-[10px] text-ink-3 border-b hairline px-3 py-2">branches ({branches.length})</div>
            <div className="p-2 space-y-0.5">
              {branches.map((b) => (
                <div key={b} className="flex items-center gap-2 px-1">
                  <span className={`w-1.5 h-1.5 rounded-full ${b === "main" ? "bg-accent" : "bg-ink-3"}`} />
                  <span className={b === "main" ? "text-ink" : "text-ink-2"}>{b}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="uppercase text-[10px] text-ink-3 border-b hairline px-3 py-2">assignments ({asgs.length})</div>
            <div className="p-2 space-y-1">
              {asgs.length ? asgs.map((a) => (
                <div key={a.assignment_id} className="flex items-center gap-2">
                  <span className="text-ink-2 truncate flex-1">{a.user} · {a.item_id}</span>
                  <span className="text-ink-3">{a.status}</span>
                </div>
              )) : <div className="text-ink-3 text-center py-4">none</div>}
            </div>
          </section>

          <section className="panel lg:col-span-1">
            <div className="uppercase text-[10px] text-ink-3 border-b hairline px-3 py-2">merge requests ({mrs.length})</div>
            <div className="p-2 space-y-2">
              {mrs.length ? mrs.map((m) => (
                <div key={m.mr_id} className="border-b hairline pb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-ink-2 truncate flex-1">{m.title}</span>
                    <StateBadge state={m.status} />
                  </div>
                  <div className="text-ink-3 truncate">{m.source_branch} → {m.target_branch}</div>
                  {canReview && (
                    <div className="flex gap-2 mt-1">
                      {m.status === "open" && <button onClick={() => act(() => api.collabApprove(m.mr_id, me!.user_id), "approve")} className="text-info hover:text-accent">approve</button>}
                      {(m.status === "approved" || m.status === "open") && <button onClick={() => act(() => api.collabMerge(m.mr_id, me!.user_id), "merge")} className="text-pass hover:text-accent">merge</button>}
                      {m.status === "merged" && <button onClick={() => act(() => api.collabRevert(m.mr_id, me!.user_id), "revert")} className="text-block hover:text-accent">revert</button>}
                    </div>
                  )}
                </div>
              )) : <div className="text-ink-3 text-center py-4">none</div>}
            </div>
          </section>
        </div>
      </div>
    </PageShell>
  );
}
