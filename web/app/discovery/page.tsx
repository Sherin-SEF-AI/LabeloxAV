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
      setMsg(`found ${r.candidates} candidates (${Object.entries(r.by_kind).map(([k, v]) => `${k}:${v}`).join(", ")})`);
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
    <div className="flex items-center gap-2 text-[12px]">
      <span className="text-ink-3">run discovery on</span>
      <select value={session} onChange={(e) => setSession(e.target.value)}
        className="bg-bg border border-line px-2 py-1 text-ink max-w-xs">
        {sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.vehicle_id} / {s.session_id.slice(0, 8)}</option>)}
      </select>
      <button onClick={run} disabled={busy || !session}
        className="border border-accent text-accent px-2 py-1 hover:bg-accent/10 disabled:opacity-50">{busy ? "..." : "run"}</button>
    </div>
  );

  return (
    <PageShell active="DISCOVERY" title="Discovery" primaryAction={primaryAction}
      right={<span className="text-[12px] text-ink-3">{items.length} pending</span>}>
      <div className="p-4 space-y-4">
        {msg && <div className="panel p-2 text-[12px] text-warn">{msg}</div>}

        {items.length ? (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {items.map((c) => (
              <div key={c.candidate_id} className="panel">
                <button type="button" className="block w-full" aria-label="open this frame in the editor"
                  onClick={() => router.push(`/frame/${c.frame_id}`)}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={c.image_url} alt="" className="w-full h-28 object-cover bg-bg-2 cursor-pointer" />
                </button>
                <div className="p-2 space-y-1 text-[11px]">
                  <div className="flex items-center justify-between gap-1">
                    <StateBadge state={c.kind.replace("_", " ")} />
                    <ScoreBar value={c.score} tone="info" />
                  </div>
                  {c.rare_classes.length > 0 && <div className="text-accent truncate">{c.rare_classes.join(", ")}</div>}
                  <div className="text-ink-3">{c.vehicle_id}</div>
                  <div className="flex gap-1 pt-1">
                    <button onClick={() => act(c, "confirmed", c.rare_classes[0])}
                      className="flex-1 border border-pass text-pass px-1 py-0.5 hover:bg-pass/10"><T k="action.confirm_short">confirm</T></button>
                    <button onClick={() => act(c, "dismissed")}
                      className="flex-1 border border-line text-ink-3 px-1 py-0.5 hover:border-block hover:text-block"><T k="action.dismiss">dismiss</T></button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="panel px-3 py-10 text-center text-[12.5px] text-ink-3">
            no pending candidates. Pick a session and click run to surface unusual frames.
          </div>
        )}
      </div>
    </PageShell>
  );
}
