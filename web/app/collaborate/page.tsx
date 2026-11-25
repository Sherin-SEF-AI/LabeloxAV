"use client";

import { useCallback, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { invalidate, useQuery } from "@/lib/useQuery";
import { useCurrentUser } from "@/lib/user";
import type { AssignmentRow, MergeRequestRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { StateBadge } from "@/components/StateBadge";
import EmptyState from "@/components/ui/EmptyState";
import { ago, humanize, num } from "@/lib/format";

// Branches (collaboration console): lakeFS branches, annotator assignments, and merge requests. Annotators work
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
  // The branch list alone needs lakeFS, an optional service; the server says so with a 503. That is a fact
  // about the installation, shown in the branches panel, not a page failure: assignments and merge
  // requests come from the database and still work.
  const branchErr = branchesQ.loadState.error;
  const lakefsDown = branchErr instanceof ApiError && branchErr.status === 503;
  const loadErr = (lakefsDown ? null : branchErr) ?? asgsQ.loadState.error ?? mrsQ.loadState.error ?? null;
  // Read in an effect, not in the body: getUser() touches localStorage and rendering it directly made
  // the server say one thing and the client another, which React resolves by discarding the server HTML.
  const me = useCurrentUser();
  const canReview = me?.role === "reviewer" || me?.role === "admin";

  // An approve, a merge or a revert moves a merge request and can move branches and assignments with it.
  const load = useCallback(() => invalidate("collabBranches", "collabAssignments", "collabMRs"), []);

  const act = async (fn: () => Promise<unknown>, label: string) => {
    if (!me) { setMsg("Sign in first"); return; }
    const r = (await fn()) as { error?: string; status?: string };
    setMsg(r.error ? r.error : `${label}: ${humanize(r.status ?? "done")}`);
    await load();
  };

  return (
    <PageShell
      active="Branches"
      title="Branches"
      subtitle="Each annotator works on their own copy of the data; a reviewer merges it back."
      right={msg ? <span className="text-[12px] text-warn">{msg}</span> : undefined}
    >
      <div className="page-wrap space-y-4">
        {loadErr != null && <LoadState error={loadErr} onRetry={() => void load()} />}
        {me && !canReview && (
          <p className="section-desc">You can see merge requests; approving, merging and reverting need the reviewer or admin role.</p>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <section className="panel">
            <div className="panel-head">
              <h2 className="section-title">Branches</h2>
              {!lakefsDown && <span className="text-[12px] text-ink-3">{num(branches.length)}</span>}
            </div>
            {lakefsDown ? (
              <EmptyState compact icon="link" title="Branch service is not running">
                Listing branches needs the data versioning service (lakeFS), which is not running on this
                server. Assignments and merge requests do not need it and work as usual.
              </EmptyState>
            ) : branchesQ.loading ? (
              <div className="p-3.5 space-y-2">{[60, 45, 52].map((w) => <div key={w} className="skeleton h-3" style={{ width: `${w}%` }} />)}</div>
            ) : branches.length ? (
              <ul className="p-2">
                {branches.map((b) => (
                  <li key={b} className="flex items-center gap-2 px-2 py-1.5 text-[12.5px] rounded hover:bg-head/60">
                    <span className={`w-1.5 h-1.5 rounded-full ${b === "main" ? "bg-accent" : "bg-ink-3"}`} aria-hidden />
                    <span className={`font-mono text-[12px] truncate ${b === "main" ? "text-ink" : "text-ink-2"}`}>{b}</span>
                    {b === "main" && <span className="ml-auto text-[11px] text-ink-3">shared</span>}
                  </li>
                ))}
              </ul>
            ) : (
              <EmptyState compact icon="link" title="No branches yet">
                A branch is created when work is assigned to an annotator.
              </EmptyState>
            )}
          </section>

          <section className="panel lg:col-span-2">
            <div className="panel-head">
              <h2 className="section-title">Assignments</h2>
              <span className="text-[12px] text-ink-3">{num(asgs.length)}</span>
            </div>
            {asgs.length ? (
              <div className="max-h-[320px] overflow-auto">
                <table className="data-table">
                  <thead><tr><th>Annotator</th><th>Item</th><th>Branch</th><th>Status</th></tr></thead>
                  <tbody>
                    {asgs.map((a) => (
                      <tr key={a.assignment_id}>
                        <td className="text-ink">{a.user}</td>
                        <td className="font-mono text-[12px]">{a.item_id}</td>
                        <td className="font-mono text-[12px] text-ink-3">{a.branch || "-"}</td>
                        <td><StateBadge state={a.status} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : asgsQ.loading ? (
              <div className="p-3.5 space-y-2">{[60, 45, 52].map((w) => <div key={w} className="skeleton h-3" style={{ width: `${w}%` }} />)}</div>
            ) : (
              <EmptyState compact icon="person" title="Nothing assigned">
                Work handed to an annotator appears here, with the branch they are working on.
              </EmptyState>
            )}
          </section>
        </div>

        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Merge requests</h2>
            <span className="text-[12px] text-ink-3">{num(mrs.length)}</span>
          </div>
          {mrs.length ? (
            <table className="data-table">
              <thead><tr><th>Title</th><th>From</th><th>Into</th><th>Opened</th><th>Status</th>{canReview && <th className="num">Actions</th>}</tr></thead>
              <tbody>
                {mrs.map((m) => (
                  <tr key={m.mr_id}>
                    <td className="text-ink">{m.title}</td>
                    <td className="font-mono text-[12px]">{m.source_branch}</td>
                    <td className="font-mono text-[12px]">{m.target_branch}</td>
                    <td className="text-ink-3">{m.created_at ? ago(m.created_at) : "-"}</td>
                    <td><StateBadge state={m.status} /></td>
                    {canReview && (
                      <td className="text-right whitespace-nowrap">
                        <span className="inline-flex gap-2">
                          {m.status === "open" && <button type="button" onClick={() => act(() => api.collabApprove(m.mr_id, me!.user_id), "Approve")} className="btn">Approve</button>}
                          {(m.status === "approved" || m.status === "open") && <button type="button" onClick={() => act(() => api.collabMerge(m.mr_id, me!.user_id), "Merge")} className="btn btn-primary">Merge</button>}
                          {m.status === "merged" && <button type="button" onClick={() => act(() => api.collabRevert(m.mr_id, me!.user_id), "Revert")} className="btn btn-danger">Revert</button>}
                        </span>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : mrsQ.loading ? (
            <div className="p-3.5 space-y-2">{[60, 45, 52].map((w) => <div key={w} className="skeleton h-3" style={{ width: `${w}%` }} />)}</div>
          ) : (
            <EmptyState compact icon="check" title="No merge requests">
              When an annotator finishes work on their branch, it comes here for a reviewer to approve and merge.
            </EmptyState>
          )}
        </section>
      </div>
    </PageShell>
  );
}
