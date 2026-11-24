"use client";

import { useEffect, useMemo, useState } from "react";
import { api , humanizeError } from "@/lib/api";
import type { IssueRow } from "@/lib/types";
import { invalidate, useQuery } from "@/lib/useQuery";

// Issue threads for the frame currently open in the editor. A reviewer rejecting a job teaches nobody
// anything; an issue anchored to the exact object turns review into specific feedback that outlives the
// conversation. Anchor to the selected object when there is one, otherwise to the frame (which is how you
// report something MISSING, since there is no object to point at).

const KINDS = ["comment", "wrong_class", "bad_geometry", "missing", "duplicate", "unclear"] as const;

const KIND_TONE: Record<string, string> = {
  wrong_class: "text-block",
  bad_geometry: "text-block",
  missing: "text-info",
  duplicate: "text-ink-2",
  unclear: "text-ink-2",
  comment: "text-ink-3",
};

export default function IssuePanel({ frameId, objectId }: { frameId: string; objectId?: string | null }) {
  const [open, setOpen] = useState<IssueRow | null>(null);
  const [kind, setKind] = useState<string>("comment");
  const [body, setBody] = useState("");
  const [reply, setReply] = useState("");
  const [showResolved, setShowResolved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Keyed by the frame and the filter. The list on screen stays up while the next frame's loads, as the
  // state it replaces did, and a failure goes to the same error line the actions below use.
  const params = useMemo(
    () => ({ frame_id: frameId, ...(showResolved ? {} : { status: "open" }) }), [frameId, showResolved]);
  const q = useQuery(frameId ? ["lopIssues", params] : null, () => api.lopIssues(params), { keepPrevious: true });
  const issues: IssueRow[] = q.data?.issues ?? [];
  useEffect(() => { if (q.error) setErr(humanizeError(q.error)); }, [q.error]);
  // Every cached issue list, not only this one: the same issue is in its author's inbox on the profile page.
  const refresh = () => invalidate("lopIssues");

  const create = async () => {
    if (!body.trim()) { setErr("describe the problem first"); return; }
    setBusy(true);
    setErr(null);
    try {
      await api.lopCreateIssue({
        kind, body: body.trim(), frame_id: frameId,
        object_id: objectId ?? null,
      });
      setBody("");
      await refresh();
    } catch (e) { setErr(humanizeError(e)); } finally { setBusy(false); }
  };

  const openThread = async (i: IssueRow) => {
    try { setOpen(await api.lopIssue(i.issue_id)); } catch (e) { setErr(humanizeError(e)); }
  };

  const addComment = async () => {
    if (!open || !reply.trim()) return;
    setBusy(true);
    try {
      setOpen(await api.lopComment(open.issue_id, reply.trim()));
      setReply("");
    } catch (e) { setErr(humanizeError(e)); } finally { setBusy(false); }
  };

  const toggleResolve = async (i: IssueRow) => {
    setBusy(true);
    try {
      const updated = await api.lopResolveIssue(i.issue_id, i.status === "resolved");
      if (open?.issue_id === i.issue_id) setOpen(updated);
      await refresh();
    } catch (e) { setErr(humanizeError(e)); } finally { setBusy(false); }
  };

  return (
    <div className="p-2 space-y-2 text-[12px]">
      <div className="flex items-center gap-2">
        <span className="uppercase text-[10px] text-ink-3">issues</span>
        <label className="ml-auto flex items-center gap-1 text-ink-3 cursor-pointer">
          <input type="checkbox" checked={showResolved} onChange={(e) => setShowResolved(e.target.checked)} />
          show resolved
        </label>
      </div>

      {/* raise */}
      <div className="space-y-1.5 border-b hairline pb-2">
        <div className="flex items-center gap-1 flex-wrap">
          {KINDS.map((k) => (
            <button key={k} onClick={() => setKind(k)}
              className={`px-1.5 py-0.5 border ${kind === k ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
              {k.replace("_", " ")}
            </button>
          ))}
        </div>
        <textarea value={body} onChange={(e) => setBody(e.target.value)} rows={2}
          placeholder={objectId ? "problem with the selected object" : "problem with this frame"}
          className="w-full bg-bg border border-line px-1.5 py-1 text-ink resize-none" />
        <div className="flex items-center gap-2">
          <span className="text-ink-3">
            {objectId ? `on object ${objectId.slice(0, 8)}` : "on this frame"}
          </span>
          <button onClick={create} disabled={busy}
            className="ml-auto px-2 py-0.5 border border-line text-ink-2 hover:border-accent disabled:opacity-40">
            raise
          </button>
        </div>
      </div>

      {err && <div className="text-block">{err}</div>}

      {/* list */}
      {issues.length === 0 ? (
        <div className="text-ink-3">no {showResolved ? "" : "open "}issues on this frame</div>
      ) : (
        <div className="space-y-1">
          {issues.map((i) => (
            <div key={i.issue_id} className="border border-line">
              <button onClick={() => openThread(i)}
                className="w-full text-left px-1.5 py-1 hover:bg-bg-2/50 flex items-center gap-2">
                <span className={KIND_TONE[i.kind] ?? "text-ink-3"}>{i.kind.replace("_", " ")}</span>
                {i.object_id && <span className="text-ink-3">obj {i.object_id.slice(0, 6)}</span>}
                <span className="ml-auto text-ink-3">
                  {i.n_comments ?? 0} {i.status === "resolved" ? "- resolved" : ""}
                </span>
              </button>

              {open?.issue_id === i.issue_id && (
                <div className="border-t hairline px-1.5 py-1 space-y-1">
                  {(open.comments ?? []).map((c) => (
                    <div key={c.comment_id}>
                      <span className="text-ink-3">{c.author ?? "anon"}: </span>
                      <span className="text-ink-2">{c.body}</span>
                    </div>
                  ))}
                  <div className="flex gap-1 pt-1">
                    <input value={reply} onChange={(e) => setReply(e.target.value)} placeholder="reply"
                      onKeyDown={(e) => { if (e.key === "Enter") addComment(); }}
                      className="flex-1 min-w-0 bg-bg border border-line px-1.5 py-0.5 text-ink" />
                    <button onClick={addComment} disabled={busy}
                      className="px-1.5 border border-line text-ink-2 hover:border-accent disabled:opacity-40">send</button>
                    <button onClick={() => toggleResolve(i)} disabled={busy}
                      className="px-1.5 border border-line text-ink-3 hover:border-accent disabled:opacity-40">
                      {i.status === "resolved" ? "reopen" : "resolve"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
