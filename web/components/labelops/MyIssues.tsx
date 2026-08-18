"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { api, humanizeError } from "@/lib/api";
import type { IssueRow } from "@/lib/types";
import LoadState from "@/components/shell/LoadState";

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
  const [rows, setRows] = useState<IssueRow[]>([]);
  const [err, setErr] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [openOnly, setOpenOnly] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await api.lopIssues({ mine: true, ...(openOnly ? { status: "open" } : {}) });
      setRows(r.issues.slice(0, limit));
    } catch (e) {
      // An empty inbox is good news and a failed fetch is not; they must not look the same.
      setErr(e);
    } finally {
      setLoading(false);
    }
  }, [openOnly, limit]);

  useEffect(() => { void load(); }, [load]);

  const open = (i: IssueRow) => {
    // The frame is where the object is, and the object is what the issue is about.
    if (i.frame_id) router.push(`/frame/${i.frame_id}${i.object_id ? `?focus=${i.object_id}` : ""}`);
    else if (i.object_id) router.push(`/object/${i.object_id}`);
  };

  return (
    <section className="panel">
      <div className="flex items-center gap-2 font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
        <span>feedback on my labels</span>
        <label className="ml-auto normal-case flex items-center gap-1 text-ink-3">
          <input type="checkbox" checked={openOnly} onChange={(e) => setOpenOnly(e.target.checked)} />
          open only
        </label>
        <button onClick={() => void load()} className="normal-case border border-line px-2 py-0.5 hover:border-accent">
          refresh
        </button>
      </div>
      <div className="p-3">
        {err != null ? (
          <LoadState error={err} onRetry={() => void load()} />
        ) : loading ? (
          <div className="font-mono text-xs text-ink-3/50 animate-pulse py-4 text-center">loading...</div>
        ) : rows.length === 0 ? (
          <div className="font-mono text-xs text-ink-3 py-4 text-center">
            {openOnly ? "nothing outstanding on your labels" : "no feedback on your labels yet"}
          </div>
        ) : (
          <div className="space-y-1">
            {rows.map((i) => (
              <button key={i.issue_id} onClick={() => open(i)}
                className="w-full text-left flex items-center gap-2 border border-line hover:border-accent px-2 py-1.5">
                <span className={`text-[10px] leading-none px-1.5 py-1 rounded border ${KIND_TONE[i.kind] ?? KIND_TONE.comment}`}>
                  {i.kind.replace(/_/g, " ")}
                </span>
                <span className="font-mono text-[11px] text-ink-2 truncate">
                  {i.object_id ? `object ${i.object_id.slice(0, 8)}` : `frame ${(i.frame_id ?? "").slice(0, 8)}`}
                </span>
                {i.n_comments ? <span className="font-mono text-[10px] text-ink-3">{i.n_comments} comment{i.n_comments === 1 ? "" : "s"}</span> : null}
                <span className={`ml-auto font-mono text-[10px] ${i.status === "open" ? "text-warn" : "text-pass"}`}>
                  {i.status}
                </span>
                <span className="font-mono text-[10px] text-ink-3">
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
