"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import type { LineageGraph, LineageNode } from "@/lib/types";

// The lineage graph: sessions to labels to gold to trainset to model to promotion.
//
// Every edge here was already recorded and the graph was not, so the chain from a shipped model back to
// the footage it is made of could only be walked by hand, one query at a time. That chain is the answer
// to the questions an audit actually asks: which footage is this model made of, was any of it from a
// subject who has since withdrawn consent, which promotion introduced this regression.
//
// Laid out in columns by the server-supplied rank rather than by a force simulation. A force layout looks
// impressive and puts the same graph in a different place every time it renders, which makes it useless
// for the one thing this is for: pointing at a node and saying "that one".

const KIND_TONE: Record<string, string> = {
  session: "border-ink-3 text-ink-2",
  dataset: "border-accent text-accent",
  gold: "border-pass text-pass",
  training_job: "border-ink-2 text-ink",
  model: "border-warn text-warn",
  promotion: "border-pass text-pass",
  deployment: "border-accent-2 text-accent-2",
};

function Column({ kind, nodes, onPick, selected }: {
  kind: string; nodes: LineageNode[]; onPick: (n: LineageNode) => void; selected: string | null;
}) {
  if (!nodes.length) return null;
  return (
    <div className="min-w-[13rem] space-y-1">
      <div className="font-mono text-[10px] uppercase text-ink-3">
        {kind.replace(/_/g, " ")} ({nodes.length})
      </div>
      {nodes.map((n) => (
        <button key={n.id} onClick={() => onPick(n)}
          className={`block w-full text-left border px-2 py-1 font-mono text-[10.5px]
                      ${KIND_TONE[n.kind] ?? "border-line text-ink-2"}
                      ${selected === n.id ? "bg-panel-2" : ""}
                      ${n.incomplete ? "border-dashed opacity-70" : ""}`}>
          <div className="truncate">{n.label}</div>
          {n.incomplete && (
            // Rendered as a break rather than dropped: a missing gold set is a fact about the lineage.
            <div className="text-[9px] text-block">record is gone</div>
          )}
        </button>
      ))}
    </div>
  );
}

function LineageBody() {
  const params = useSearchParams();
  const [mode, setMode] = useState<"model" | "session" | "dataset">(
    (params.get("mode") as "model" | "session" | "dataset") || "model");
  const [target, setTarget] = useState(params.get("id") || "");
  const [graph, setGraph] = useState<LineageGraph | null>(null);
  const [picked, setPicked] = useState<LineageNode | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async (m: string, id: string) => {
    if (!id.trim()) return;
    setBusy(true);
    try {
      const g = m === "model" ? await api.modelLineage(id.trim())
        : m === "session" ? await api.sessionLineage(id.trim())
        : await api.datasetLineage(id.trim());
      setGraph(g);
      setPicked(null);
    } catch (e) { toast(humanizeError(e), "error"); setGraph(null); }
    finally { setBusy(false); }
  }, []);

  useEffect(() => { if (target) load(mode, target); }, [load, mode, target]);

  const edgesFor = (id: string) =>
    (graph?.edges ?? []).filter((e) => e.source === id || e.target === id);

  return (
    <PageShell active="LINEAGE" title="Lineage"
      subtitle="what a model is made of, and what a session ended up in"
      filters={
        <>
          {(["model", "session", "dataset"] as const).map((m) => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-2 py-0.5 border ${
                mode === m ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
              {m}
            </button>
          ))}
          <input value={target} onChange={(e) => setTarget(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") load(mode, target); }}
            placeholder={mode === "model" ? "model version"
              : mode === "session" ? "session id" : "dataset commit id"}
            className="bg-bg border border-line px-1.5 py-0.5 text-ink w-72" />
          <button onClick={() => load(mode, target)} disabled={busy || !target.trim()}
            className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
            {busy ? "loading..." : "trace"}
          </button>
        </>
      }>
      <div className="p-4 space-y-4">
        {!graph ? (
          <div className="panel p-4 font-mono text-[11px] text-ink-3 max-w-3xl space-y-2">
            <div>Trace a model backwards to the footage it is made of, or a session forwards to
              everything it ended up in.</div>
            <div className="text-ink-3">
              The forward direction is what an erasure request asks: if a subject withdraws consent for a
              session, this is what has to be re-examined. Answering it by hand meant reading every dataset
              commit&apos;s slice spec.
            </div>
          </div>
        ) : (
          <>
            <div className="panel p-3 overflow-x-auto">
              <div className="flex gap-6 items-start min-w-max">
                {graph.kinds.map((kind) => (
                  <Column key={kind} kind={kind} selected={picked?.id ?? null}
                    nodes={graph.nodes.filter((n) => n.kind === kind)}
                    onPick={setPicked} />
                ))}
              </div>
            </div>

            <div className="flex gap-4 flex-wrap">
              <div className="panel px-3 py-2">
                <div className="font-mono text-[10px] uppercase text-ink-3">nodes</div>
                <div className="font-mono text-[18px] text-ink tabular-nums">{graph.nodes.length}</div>
              </div>
              <div className="panel px-3 py-2">
                <div className="font-mono text-[10px] uppercase text-ink-3">edges</div>
                <div className="font-mono text-[18px] text-ink tabular-nums">{graph.edges.length}</div>
              </div>
              {graph.sessions_truncated && (
                <div className="panel px-3 py-2 border border-warn">
                  <div className="font-mono text-[10px] uppercase text-warn">truncated</div>
                  <div className="font-mono text-[10.5px] text-ink-3 max-w-sm">
                    showing {graph.sessions_shown} sessions; the model draws on more
                  </div>
                </div>
              )}
            </div>

            {graph.detail && (
              <div className="panel p-3 font-mono text-[10.5px] text-ink-3 max-w-4xl">
                {graph.detail}
              </div>
            )}

            {picked && (
              <section className="panel">
                <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
                  {picked.kind.replace(/_/g, " ")} · {picked.label}
                </div>
                <div className="p-3 space-y-2">
                  {Object.keys(picked.meta).length > 0 && (
                    <div className="flex gap-1 flex-wrap">
                      {Object.entries(picked.meta).map(([k, v]) => (
                        <span key={k}
                          className="border border-line px-1.5 py-0.5 font-mono text-[10px] text-ink-2">
                          {k} <span className="text-ink-3">{String(v).slice(0, 40)}</span>
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="font-mono text-[10px] uppercase text-ink-3">connections</div>
                  <ul className="space-y-0.5">
                    {edgesFor(picked.id).map((e, i) => (
                      <li key={i} className="font-mono text-[10.5px] text-ink-2">
                        {e.source === picked.id ? "→" : "←"} {e.kind.replace(/_/g, " ")}{" "}
                        <span className="text-ink-3">
                          {(graph.nodes.find(
                            (n) => n.id === (e.source === picked.id ? e.target : e.source))?.label)
                            ?? "?"}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </PageShell>
  );
}

export default function LineagePage() {
  return <Suspense fallback={null}><LineageBody /></Suspense>;
}
