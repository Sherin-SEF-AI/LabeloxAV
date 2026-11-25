"use client";

import { useEffect, useState } from "react";
import { T } from "@/lib/i18n";
import { useRouter } from "next/navigation";
import { api , humanizeError } from "@/lib/api";
import { invalidate, useQuery } from "@/lib/useQuery";
import type { DiscoveryCandidate, SessionRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import { StateBadge } from "@/components/StateBadge";
import ScoreBar from "@/components/shell/ScoreBar";
import EmptyState from "@/components/ui/EmptyState";
import LoadState from "@/components/shell/LoadState";
import Thumb from "@/components/engine/Thumb";
import { humanize, num, plural, shortId } from "@/lib/format";

// M1.5 rare-scenario discovery queue: unusual frames surfaced by embedding novelty (outlier / sparse
// cluster) or rare classes, ranked by score, for a human to confirm/dismiss/tag. Feeds active learning.

const NO_ITEMS: DiscoveryCandidate[] = [];
const NO_SESSIONS: SessionRow[] = [];

export default function DiscoveryPage() {
  const router = useRouter();
  // A failed read was never shown here and still is not: the queue stays as it was.
  const itemsQ = useQuery(["discoveryQueue", "pending"], () => api.discoveryQueue("pending"));
  const items: DiscoveryCandidate[] = itemsQ.data ?? NO_ITEMS;
  const sessionsQ = useQuery(["sessions"], () => api.sessions());
  const sessions: SessionRow[] = sessionsQ.data ?? NO_SESSIONS;
  const [session, setSession] = useState("");
  // Start on the newest session, without overriding one already picked when the list refreshes.
  useEffect(() => {
    if (sessions[0]) setSession((cur) => cur || sessions[0].session_id);
  }, [sessions]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // A run adds candidates in every state, not only the pending ones listed here.
  const load = () => invalidate("discoveryQueue");

  const run = async () => {
    if (!session) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.discoveryRun(session);
      const kinds = Object.entries(r.by_kind).map(([k, v]) => `${num(v)} ${humanize(k).toLowerCase()}`).join(", ");
      setMsg(`Found ${plural(r.candidates, "unusual frame")}${kinds ? ` (${kinds})` : ""}.`);
      await load();
    } catch (e) {
      setMsg(humanizeError(e));
    } finally {
      setBusy(false);
    }
  };

  const act = async (c: DiscoveryCandidate, state: string, tag?: string) => {
    await api.discoverySetState(c.candidate_id, state, tag);
    itemsQ.setData((xs) => (xs ?? []).filter((x) => x.candidate_id !== c.candidate_id));
  };

  const primaryAction = (
    <div className="flex items-center gap-2 text-[12.5px]">
      <label htmlFor="discovery-session" className="text-ink-3">Search session</label>
      <select id="discovery-session" value={session} onChange={(e) => setSession(e.target.value)}
        className="field max-w-xs">
        {sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.vehicle_id} · {shortId(s.session_id)}</option>)}
      </select>
      <button type="button" onClick={run} disabled={busy || !session}
        className="btn btn-primary">{busy ? "Searching..." : "Find unusual frames"}</button>
    </div>
  );

  return (
    <PageShell active="DISCOVERY" title="Discovery" primaryAction={primaryAction}
      subtitle="Unusual frames found in a session. Confirm the ones worth labelling, dismiss the rest."
      right={<span className="text-[12.5px] text-ink-3">{num(items.length)} waiting</span>}>
      <div className="page-wrap space-y-4">
        {msg && <div role="status" className="panel px-3.5 py-2.5 text-[12.5px] text-ink-2">{msg}</div>}

        {itemsQ.loading && !items.length ? (
          <LoadState loading rows={4} />
        ) : items.length ? (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {items.map((c) => (
              <div key={c.candidate_id} className="panel overflow-hidden flex flex-col">
                <button type="button" className="block w-full" aria-label="open this frame in the editor"
                  onClick={() => router.push(`/frame/${c.frame_id}`)}>
                  <Thumb src={c.image_url} className="w-full h-28" />
                </button>
                <div className="p-2.5 space-y-1.5 text-[12px] flex-1 flex flex-col">
                  <div className="flex items-center justify-between gap-1.5">
                    <StateBadge state={humanize(c.kind)} />
                    <span data-tip="How unusual this frame is"><ScoreBar value={c.score} tone="info" /></span>
                  </div>
                  {c.rare_classes.length > 0 && (
                    <div className="text-ink-2 truncate" data-tip={c.rare_classes.map(humanize).join(", ")}>
                      {c.rare_classes.map(humanize).join(", ")}
                    </div>
                  )}
                  <div className="text-ink-3 text-[11px]">{c.vehicle_id}</div>
                  <div className="flex gap-1.5 pt-1 mt-auto">
                    <button type="button" onClick={() => act(c, "confirmed", c.rare_classes[0])}
                      className="btn flex-1 text-pass"><T k="action.confirm_short">Confirm</T></button>
                    <button type="button" onClick={() => act(c, "dismissed")}
                      className="btn flex-1"><T k="action.dismiss">Dismiss</T></button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="panel">
            <EmptyState icon="search" title="No unusual frames waiting"
              action={<button type="button" onClick={run} disabled={busy || !session} className="btn btn-primary">{busy ? "Searching..." : "Find unusual frames"}</button>}>
              Pick a session above and search it. Frames that look unlike the rest of the corpus, or that
              contain rare classes, land here for you to confirm or dismiss.
            </EmptyState>
          </div>
        )}
      </div>
    </PageShell>
  );
}
