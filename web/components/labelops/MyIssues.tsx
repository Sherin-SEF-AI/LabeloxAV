"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { api, humanizeError } from "@/lib/api";
import type { IssueRow } from "@/lib/types";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { humanize, shortId } from "@/lib/format";
import { useQuery } from "@/lib/useQuery";

// Feedback on your own labels, in a place you can go and look.
//
// Issues were creatable in the frame editor and readable nowhere. The notification they emit is addressed
// to the reviewer ROLE - correctly, since who picks an issue up is a duty rota - so the annotator who drew
// the label was the one participant never told, and found out when the job came back, if at all. The
// notification half is fixed at the emitter; this is the other half: somewhere to see the list, including
// the ones already read and the ones already resolved.

const KIND_TONE: Record<string, string> = {
  wrong_class: "text-block border-block/45 bg-block/10",
  bad_box: "text-warn border-warn/40 bg-warn/10",
  missing: "text-warn border-warn/40 bg-warn/10",
  comment: "text-ink-3 border-line",
};

export default function MyIssues({ limit = 25 }: { limit?: number }) {
  const router = useRouter();
  const [openOnly, setOpenOnly] = useState(true);

  // Keyed by the filter, so the inbox paints from the cache on return and refreshes behind it.
  const params = useMemo(() => ({ mine: true, ...(openOnly ? { status: "open" } : {}) }), [openOnly]);
  const q = useQuery(["lopIssues", params], () => api.lopIssues(params));
  const rows: IssueRow[] = (q.data?.issues ?? []).slice(0, limit);
  // An empty inbox is good news and a failed fetch is not; they must not look the same.
  const { loading, error: err } = q.loadState;
  const load = q.reload;

  const open = (i: IssueRow) => {
    // The frame is where the object is, and the object is what the issue is about.
    if (i.frame_id) router.push(`/frame/${i.frame_id}${i.object_id ? `?focus=${i.object_id}` : ""}`);
    else if (i.object_id) router.push(`/object/${i.object_id}`);
  };

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="section-title">Feedback on my labels</h2>
        <label className="ml-auto flex items-center gap-1.5 text-[12px] text-ink-2 cursor-pointer">
          <input type="checkbox" checked={openOnly} onChange={(e) => setOpenOnly(e.target.checked)} className="accent-primary" />
          Open only
        </label>
        <button type="button" onClick={() => void load()} className="btn">
          Refresh
        </button>
      </div>
      <div className="p-3">
        {err != null ? (
          <LoadState error={err} onRetry={() => void load()} />
        ) : loading ? (
          <LoadState loading rows={2} />
        ) : rows.length === 0 ? (
          <EmptyState compact icon="comment" title={openOnly ? "Nothing outstanding on your labels" : "No feedback on your labels yet"}>
            When a reviewer flags one of your labels, it shows here with a link to the frame.
          </EmptyState>
        ) : (
          <div className="space-y-1.5">
            {rows.map((i) => (
              <button key={i.issue_id} type="button" onClick={() => open(i)}
                className="w-full text-left flex items-center gap-2.5 rounded border border-line hover:border-accent/60 hover:bg-head/50 px-2.5 py-2">
                <span className={`text-[11px] leading-none px-1.5 py-1 rounded border ${KIND_TONE[i.kind] ?? KIND_TONE.comment}`}>
                  {humanize(i.kind)}
                </span>
                <span className="text-[12.5px] text-ink-2 truncate">
                  {i.object_id ? "On an object" : "On a frame"}
                  <span className="ml-1.5 font-mono text-[11.5px] text-ink-3">{shortId(i.object_id ?? i.frame_id)}</span>
                </span>
                {i.n_comments ? <span className="text-[12px] text-ink-3">{i.n_comments} comment{i.n_comments === 1 ? "" : "s"}</span> : null}
                <span className={`ml-auto text-[12px] ${i.status === "open" ? "text-warn" : "text-pass"}`}>
                  {humanize(i.status)}
                </span>
                <span className="text-[12px] text-ink-3">
                  {i.created_at ? new Date(i.created_at).toLocaleDateString() : ""}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
