"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { toast } from "@/lib/toast";

// Build a dataset by describing it.
//
// The engine behind this shipped as an SDK call, which is right for a training config and wrong for finding
// out what the corpus can give you. Somebody deciding whether "night AND vru" is worth training on wants to
// see the count and the class mix before they commit, and they want the terms they can use in front of them
// rather than in a docstring.
//
// Two things this deliberately shows that a search box normally hides. The compiled predicate, because a
// dataset whose contents cannot be explained is one nobody can defend in a review. And the refusal, because
// an unknown term silently widening the selection is how a training set ends up missing the class it was
// assembled for.

type Preview = {
  query: string;
  predicate: Record<string, unknown>;
  terms: { term: string; kind: string; expands_to?: string[] }[];
  frames: number;
  objects: number;
  classes: Record<string, number>;
  sealed: boolean;
};

type Vocab = { scene: string[]; state: string[]; group: string[]; class: string[] };

const KIND_TONE: Record<string, string> = {
  scene: "text-info", state: "text-warn", group: "text-accent", class: "text-ink",
};

export default function DatasetQueryPage() {
  const [q, setQ] = useState("night AND vru");
  const [vocab, setVocab] = useState<Vocab | null>(null);
  const [pv, setPv] = useState<Preview | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [building, setBuilding] = useState(false);
  const [built, setBuilt] = useState<{ shards: string[]; samples: number; index_uri: string | null } | null>(null);

  useEffect(() => {
    api.datasetVocabulary().then(setVocab).catch(() => setVocab(null));
  }, []);

  const run = useCallback(async (query: string) => {
    if (!query.trim()) return;
    setBusy(true);
    setErr(null);
    setBuilt(null);
    try {
      setPv(await api.datasetPreview(query));
    } catch (e) {
      setPv(null);
      setErr(e);
    } finally {
      setBusy(false);
    }
  }, []);

  // Debounced, because every keystroke is a corpus-wide count and the point is to explore, not to hammer.
  useEffect(() => {
    const t = setTimeout(() => void run(q), 450);
    return () => clearTimeout(t);
  }, [q, run]);

  const build = useCallback(async () => {
    setBuilding(true);
    try {
      const r = await api.datasetBuildShards(q);
      setBuilt(r);
      toast(`${r.samples} samples in ${r.shards.length} shards`, "success");
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setBuilding(false);
    }
  }, [q]);

  const addTerm = useCallback((t: string) => {
    setQ((cur) => (cur.trim() ? `${cur.trim()} AND ${t}` : t));
  }, []);

  const snippet = useMemo(
    () => `import labelox\nlabelox.configure(BASE_URL, token=TOKEN)\nds = labelox.load(${JSON.stringify(q)})`,
    [q],
  );

  return (
    <PageShell active="DATASETS" title="Dataset query"
      subtitle="describe the data, see what the corpus can give you">
      <div className="p-4 space-y-3 max-w-6xl">

        <div className="panel px-3 py-3 space-y-2">
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            spellCheck={false}
            placeholder="night AND vru AND reviewed"
            className="input w-full font-mono text-[13px]"
          />
          {/* Terms are clickable rather than only documented: the compiler refuses what it does not know,
              so the list of what works has to be in reach of the box that rejects you. */}
          {vocab && (
            <div className="space-y-1">
              {([["scene", vocab.scene], ["group", vocab.group], ["state", vocab.state]] as const).map(
                ([kind, items]) => (
                  <div key={kind} className="flex flex-wrap items-center gap-1">
                    <span className="font-mono text-[10px] uppercase text-ink-3 w-14">{kind}</span>
                    {items.map((t) => (
                      <button key={t} onClick={() => addTerm(t)}
                        className={`font-mono text-[10px] px-1.5 py-0.5 border border-line hover:bg-line/40 ${KIND_TONE[kind]}`}>
                        {t}
                      </button>
                    ))}
                  </div>
                ),
              )}
              <div className="font-mono text-[10px] text-ink-3">
                plus any of the {vocab.class.length} ontology class names
              </div>
            </div>
          )}
        </div>

        {err != null && (
          <div className="panel px-3 py-2">
            {/* The refusal is the feature. An unknown term that widened the selection instead would give a
                plausible dataset missing the thing it was built for. */}
            <LoadState error={err} onRetry={() => void run(q)} />
          </div>
        )}

        {busy && !pv && <div className="font-mono text-[11px] text-ink-3">counting...</div>}

        {pv && !err && (
          <>
            <div className="flex flex-wrap gap-2 font-mono text-[11px]">
              <Stat label="frames" value={pv.frames.toLocaleString()} />
              <Stat label="objects" value={pv.objects.toLocaleString()} />
              <Stat label="classes" value={String(Object.keys(pv.classes).length)} />
              {pv.sealed && <Stat label="pinned" value="sealed" tone="text-accent" />}
            </div>

            {pv.frames === 0 && (
              <div className="panel px-3 py-4 font-mono text-[11px] text-ink-3">
                Nothing matches. That is an answer: this corpus holds no frames meeting all of those terms.
              </div>
            )}

            <div className="grid gap-3 md:grid-cols-2">
              <section className="panel">
                <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
                  what this compiles to
                </div>
                <div className="p-3 space-y-2">
                  {/* Shown, not hidden: this is the difference between a dataset somebody can defend and one
                      they can only describe. */}
                  <pre className="font-mono text-[10px] text-ink-2 whitespace-pre-wrap break-all">
                    {JSON.stringify(pv.predicate, null, 1)}
                  </pre>
                  <div className="space-y-1">
                    {pv.terms.map((t) => (
                      <div key={t.term} className="font-mono text-[10px]">
                        <span className={KIND_TONE[t.kind] ?? "text-ink"}>{t.term}</span>
                        <span className="text-ink-3"> is a {t.kind}</span>
                        {t.expands_to && (
                          <span className="text-ink-3"> and covers {t.expands_to.length} classes</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </section>

              <section className="panel">
                <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
                  class mix
                </div>
                <div className="p-3 space-y-1">
                  {Object.entries(pv.classes).slice(0, 14).map(([name, n]) => {
                    const max = Math.max(...Object.values(pv.classes), 1);
                    return (
                      <div key={name} className="flex items-center gap-2 font-mono text-[10px]">
                        <span className="w-32 truncate text-ink-2">{name}</span>
                        <span className="flex-1 bg-line/40 h-2">
                          <span className="block bg-accent h-2" style={{ width: `${(n / max) * 100}%` }} />
                        </span>
                        <span className="w-14 text-right tabular-nums text-ink-3">{n}</span>
                      </div>
                    );
                  })}
                  {!Object.keys(pv.classes).length && (
                    <div className="font-mono text-[10px] text-ink-3">no labelled objects in this selection</div>
                  )}
                </div>
              </section>
            </div>

            <section className="panel">
              <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2 flex items-center justify-between">
                <span>take it away</span>
                <button onClick={() => void build()} disabled={building || pv.frames === 0}
                  className="border border-accent text-accent px-2 py-0.5 hover:bg-accent/10 disabled:opacity-30">
                  {building ? "building..." : "build shards"}
                </button>
              </div>
              <div className="p-3 space-y-2">
                {/* The query is the artifact. A path names where somebody put a zip; this names what the
                    data is, which is what whoever inherits the config actually needs. */}
                <pre className="font-mono text-[10px] text-ink-2 whitespace-pre-wrap">{snippet}</pre>
                {built && (
                  <div className="font-mono text-[10px] text-ink-3 space-y-0.5">
                    <div className="text-ink">{built.samples} samples in {built.shards.length} shards</div>
                    {built.shards.slice(0, 3).map((s) => <div key={s}>{s}</div>)}
                    {built.index_uri && <div>index: {built.index_uri}</div>}
                  </div>
                )}
              </div>
            </section>
          </>
        )}
      </div>
    </PageShell>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="panel px-3 py-2 min-w-[110px]">
      <div className="text-[10px] uppercase text-ink-3">{label}</div>
      <div className={`text-[18px] tabular-nums ${tone ?? "text-ink"}`}>{value}</div>
    </div>
  );
}
