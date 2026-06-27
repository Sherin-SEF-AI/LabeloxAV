"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { DiscoveryCandidate, SessionRow } from "@/lib/types";
import TopNav from "@/components/TopNav";

// M1.5 rare-scenario discovery queue: unusual frames surfaced by embedding novelty (outlier / sparse
// cluster) or rare classes, ranked by score, for a human to confirm/dismiss/tag. Feeds active learning.

const KIND_COLOR: Record<string, string> = {
  embedding_outlier: "text-warn",
  sparse_cluster: "text-info",
  rare_class: "text-accent",
};

export default function DiscoveryPage() {
  const router = useRouter();
  const [items, setItems] = useState<DiscoveryCandidate[]>([]);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [session, setSession] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    setItems(await api.discoveryQueue("pending"));
  }, []);

  useEffect(() => {
    load();
    api.sessions().then((s) => { setSessions(s); if (s[0]) setSession(s[0].session_id); });
  }, [load]);

  const run = async () => {
    if (!session) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.discoveryRun(session);
      setMsg(`found ${r.candidates} candidates (${Object.entries(r.by_kind).map(([k, v]) => `${k}:${v}`).join(", ")})`);
      await load();
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  };

  const act = async (c: DiscoveryCandidate, state: string, tag?: string) => {
    await api.discoverySetState(c.candidate_id, state, tag);
    setItems((xs) => xs.filter((x) => x.candidate_id !== c.candidate_id));
  };

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="DISCOVERY" />
      <main className="flex-1 overflow-auto p-4 space-y-4">
        <div className="panel p-3 flex items-center gap-2 flex-wrap font-mono text-[11px]">
          <span className="text-ink-3">run discovery on</span>
          <select value={session} onChange={(e) => setSession(e.target.value)}
            className="bg-bg border border-line px-2 py-1 text-ink max-w-xs">
            {sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.vehicle_id} / {s.session_id.slice(0, 8)}</option>)}
          </select>
          <button onClick={run} disabled={busy || !session}
            className="border border-accent text-accent px-2 py-1 hover:bg-accent/10 disabled:opacity-50">{busy ? "..." : "run"}</button>
          <span className="ml-auto text-ink-3">{items.length} pending</span>
          {msg && <span className="text-warn w-full">{msg}</span>}
        </div>

        {items.length ? (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {items.map((c) => (
              <div key={c.candidate_id} className="panel">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={c.image_url} alt="" onClick={() => router.push(`/frame/${c.frame_id}`)}
                  className="w-full h-28 object-cover bg-bg-2 cursor-pointer" />
                <div className="p-2 space-y-1 font-mono text-[10px]">
                  <div className="flex items-center justify-between">
                    <span className={KIND_COLOR[c.kind] || "text-ink-2"}>{c.kind.replace("_", " ")}</span>
                    <span className="text-ink-3">{c.score.toFixed(2)}</span>
                  </div>
                  {c.rare_classes.length > 0 && <div className="text-accent truncate">{c.rare_classes.join(", ")}</div>}
                  <div className="text-ink-3">{c.vehicle_id}</div>
                  <div className="flex gap-1 pt-1">
                    <button onClick={() => act(c, "confirmed", c.rare_classes[0])}
                      className="flex-1 border border-pass text-pass px-1 py-0.5 hover:bg-pass/10">confirm</button>
                    <button onClick={() => act(c, "dismissed")}
                      className="flex-1 border border-line text-ink-3 px-1 py-0.5 hover:border-block hover:text-block">dismiss</button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="panel px-3 py-10 text-center font-mono text-xs text-ink-3">
            no pending candidates. Pick a session and click run to surface unusual frames.
          </div>
        )}
      </main>
    </div>
  );
}
