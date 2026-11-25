"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
import { useQuery } from "@/lib/useQuery";
import type { LineageGraph, LineageNode } from "@/lib/types";
import EmptyState from "@/components/ui/EmptyState";
import { Stat } from "@/components/ui/kit";
import { humanize, num, plural } from "@/lib/format";

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
    <div className="min-w-[13rem] max-w-[16rem] space-y-1.5">
      <div className="label-caps">
        {humanize(kind)} ({nodes.length})
      </div>
      {nodes.map((n) => (
        <button key={n.id} onClick={() => onPick(n)}
          aria-pressed={selected === n.id} title={n.label}
          className={`block w-full text-left border rounded px-2.5 py-1.5 text-[12.5px] hover:bg-head
                      ${KIND_TONE[n.kind] ?? "border-line text-ink-2"}
                      ${selected === n.id ? "bg-head ring-1 ring-accent" : ""}
                      ${n.incomplete ? "border-dashed opacity-70" : ""}`}>
          <div className="truncate">{n.label}</div>
          {n.incomplete && (
            // Rendered as a break rather than dropped: a missing gold set is a fact about the lineage.
            <div className="text-[11px] text-block">Record is gone</div>
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
  const [picked, setPicked] = useState<LineageNode | null>(null);

  // One cached graph per thing traced, keyed by the read that produces it. keepPrevious: the last graph
  // stays up while the next one loads.
  const id = target.trim();
  const method = mode === "model" ? "modelLineage" : mode === "session" ? "sessionLineage" : "datasetLineage";
  const graphQ = useQuery<LineageGraph>(id ? [method, id] : null,
    () => (mode === "model" ? api.modelLineage(id) : mode === "session" ? api.sessionLineage(id) : api.datasetLineage(id)),
    { keepPrevious: true });
  // A failed trace clears the graph rather than leaving the previous subject's up under a new id.
  const graph: LineageGraph | null = id && !graphQ.error ? graphQ.data ?? null : null;
  const busy = graphQ.validating;
  const loaded = graphQ.data;
  const failure = graphQ.error;
  useEffect(() => { setPicked(null); }, [loaded]);
  useEffect(() => { if (failure) toast(humanizeError(failure), "error"); }, [failure]);
  const load = (_m: string, _id: string) => { void graphQ.reload(); };

  const edgesFor = (id: string) =>
    (graph?.edges ?? []).filter((e) => e.source === id || e.target === id);

  const placeholder = mode === "model" ? "Model version" : mode === "session" ? "Session id" : "Dataset commit id";
  return (
    <PageShell active="Lineage" title="Lineage"
      subtitle="What a model is made of, and everything a session ended up in."
      filters={
        <>
          <span className="text-ink-3">Trace a</span>
          {(["model", "session", "dataset"] as const).map((m) => (
            <button key={m} type="button" onClick={() => setMode(m)} aria-pressed={mode === m}
              className={`chip ${mode === m ? "chip-on" : ""}`}>
              {humanize(m)}
            </button>
          ))}
          <input value={target} onChange={(e) => setTarget(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") load(mode, target); }}
            placeholder={placeholder} aria-label={placeholder}
            className="field w-80 font-mono" />
          <button type="button" onClick={() => load(mode, target)} disabled={busy || !target.trim()}
            className="btn btn-primary">
            {busy ? "Tracing..." : "Trace"}
          </button>
        </>
      }>
      <div className="page-wrap space-y-4">
        {!graph ? (
          <div className="panel">
            <EmptyState icon="route" title={busy ? "Tracing..." : "Trace a model or a session"}>
              Enter a model version to see the footage it was trained on, or a session id to see every dataset
              and model it ended up in. When someone withdraws consent for a session, the forward trace is the
              list of what has to be re-examined.
            </EmptyState>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <Stat label="Items" value={num(graph.nodes.length)} />
              <Stat label="Links" value={num(graph.edges.length)} />
              {graph.sessions_truncated && (
                <Stat label="Truncated" tone="warn" value={num(graph.sessions_shown)}
                  sub="sessions shown; the model draws on more" className="lg:col-span-2" />
              )}
            </div>

            {graph.detail && <p className="section-desc max-w-4xl">{graph.detail}</p>}

            <div className="panel p-4 overflow-x-auto">
              <div className="flex gap-6 items-start min-w-max">
                {graph.kinds.map((kind) => (
                  <Column key={kind} kind={kind} selected={picked?.id ?? null}
                    nodes={graph.nodes.filter((n) => n.kind === kind)}
                    onPick={setPicked} />
                ))}
              </div>
            </div>

            {picked ? (
              <section className="panel">
                <div className="panel-head">
                  <span className="label-caps">{humanize(picked.kind)}</span>
                  <h2 className="section-title truncate">{picked.label}</h2>
                </div>
                <div className="p-3.5 space-y-3">
                  {Object.keys(picked.meta).length > 0 && (
                    <div className="flex gap-1.5 flex-wrap">
                      {Object.entries(picked.meta).map(([k, v]) => (
                        <span key={k} className="chip">
                          {humanize(k)} <span className="text-ink-3">{String(v).slice(0, 40)}</span>
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="label-caps">{plural(edgesFor(picked.id).length, "connection")}</div>
                  <ul className="space-y-1">
                    {edgesFor(picked.id).map((e, i) => (
                      <li key={i} className="text-[12.5px] text-ink-2">
                        <span className="text-ink-3" aria-hidden>{e.source === picked.id ? "→" : "←"}</span> {humanize(e.kind)}{" "}
                        <span className="text-ink">
                          {(graph.nodes.find(
                            (n) => n.id === (e.source === picked.id ? e.target : e.source))?.label)
                            ?? "?"}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </section>
            ) : (
              <p className="section-desc">Select an item above to see its details and what it connects to.</p>
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
