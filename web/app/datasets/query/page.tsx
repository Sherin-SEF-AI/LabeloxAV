"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { toast } from "@/lib/toast";
import { useQuery } from "@/lib/useQuery";
import { Stat } from "@/components/ui/kit";
import EmptyState from "@/components/ui/EmptyState";
import { humanize, num, plural } from "@/lib/format";

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
  scene: "text-info", state: "text-warn", group: "text-accent-2", class: "text-ink",
};
const KIND_LABEL: Record<string, string> = { scene: "Scene", group: "Group", state: "State", class: "Class" };

// One clause of the compiled predicate as a line a reviewer can read. The JSON stays one click away.
function clauseText(v: unknown): string {
  if (Array.isArray(v)) return v.length > 6 ? `${v.slice(0, 6).join(", ")} and ${v.length - 6} more` : v.join(", ");
  if (v && typeof v === "object") return Object.entries(v as Record<string, unknown>).map(([k, x]) => `${humanize(k)}: ${clauseText(x)}`).join("; ");
  return String(v);
}

export default function DatasetQueryPage() {
  const [q, setQ] = useState("night AND vru");
  // The vocabulary is the same for every visit; a failure reads as no vocabulary, as it always has.
  const vocabQuery = useQuery(["datasetVocabulary"], () => api.datasetVocabulary());
  const vocab: Vocab | null = vocabQuery.error ? null : vocabQuery.data ?? null;
  const [pv, setPv] = useState<Preview | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const [building, setBuilding] = useState(false);
  const [built, setBuilt] = useState<{ shards: string[]; samples: number; index_uri: string | null } | null>(null);

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
      toast(`${num(r.samples)} samples in ${plural(r.shards.length, "shard")}`, "success");
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
      subtitle="Describe the data you want and see what the corpus can give you before you build it.">
      <div className="page-wrap space-y-4">

        <section className="panel">
          <div className="panel-head"><h2 className="section-title">Query</h2>
            {busy && <span className="ml-auto inline-flex items-center gap-1.5 text-[12px] text-ink-3"><span className="running-dot" />Counting...</span>}
          </div>
          <div className="p-4 space-y-3">
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              spellCheck={false}
              aria-label="Dataset query"
              placeholder="night AND vru AND reviewed"
              className="field w-full font-mono text-[13.5px] py-2"
            />
            {/* Terms are clickable rather than only documented: the compiler refuses what it does not know,
                so the list of what works has to be in reach of the box that rejects you. */}
            {vocab && (
              <div className="space-y-2">
                <div className="text-[12px] text-ink-3">Click a term to add it with AND:</div>
                {([["scene", vocab.scene], ["group", vocab.group], ["state", vocab.state]] as const).map(
                  ([kind, items]) => (
                    <div key={kind} className="flex flex-wrap items-center gap-1.5">
                      <span className="text-[12px] font-medium text-ink-2 w-16 shrink-0">{KIND_LABEL[kind]}</span>
                      {items.map((t) => (
                        <button key={t} type="button" onClick={() => addTerm(t)}
                          className={`chip font-mono text-[12px] ${KIND_TONE[kind]}`}>
                          {t}
                        </button>
                      ))}
                    </div>
                  ),
                )}
                <div className="text-[12px] text-ink-3">
                  Any of the {num(vocab.class.length)} ontology class names also work as terms.
                </div>
              </div>
            )}
          </div>
        </section>

        {err != null && (
          // The refusal is the feature. An unknown term that widened the selection instead would give a
          // plausible dataset missing the thing it was built for.
          <LoadState error={err} onRetry={() => void run(q)} />
        )}

        {busy && !pv && !err && <LoadState loading rows={3} />}

        {!pv && !busy && err == null && !q.trim() && (
          <div className="panel">
            <EmptyState icon="layers" title="Describe the data you need">
              Combine scene, group, state and class terms with AND, for example <span className="font-mono text-ink-2">night AND vru</span>.
              The counts update as you type.
            </EmptyState>
          </div>
        )}

        {pv && !err && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Frames" value={num(pv.frames)} />
              <Stat label="Objects" value={num(pv.objects)} />
              <Stat label="Classes" value={num(Object.keys(pv.classes).length)} />
              {pv.sealed && <Stat label="Pinned" value="Sealed" tone="accent" />}
            </div>

            {pv.frames === 0 && (
              <div className="panel">
                <EmptyState compact icon="search" title="Nothing matches">
                  That is an answer: this corpus holds no frames meeting all of those terms. Try removing one.
                </EmptyState>
              </div>
            )}

            <div className="grid gap-4 md:grid-cols-2">
              <section className="panel">
                <div className="panel-head">
                  <h2 className="section-title">What this selects</h2>
                </div>
                <div className="p-4 space-y-3">
                  {/* Shown, not hidden: this is the difference between a dataset somebody can defend and one
                      they can only describe. Read as words first, with the exact JSON a click away. */}
                  <div className="space-y-1.5">
                    {pv.terms.map((t) => (
                      <div key={t.term} className="text-[12.5px]">
                        <span className={`font-mono text-[12px] ${KIND_TONE[t.kind] ?? "text-ink"}`}>{t.term}</span>
                        <span className="text-ink-3"> is a {t.kind}</span>
                        {t.expands_to && (
                          <span className="text-ink-3" title={t.expands_to.join(", ")}> covering {plural(t.expands_to.length, "class", "classes")}</span>
                        )}
                      </div>
                    ))}
                  </div>
                  {Object.keys(pv.predicate).length > 0 && (
                    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-[12.5px] border-t hairline pt-3">
                      {Object.entries(pv.predicate).map(([k, v]) => (
                        <div key={k} className="contents">
                          <dt className="text-ink-3">{humanize(k)}</dt>
                          <dd className="text-ink-2 break-words">{clauseText(v)}</dd>
                        </div>
                      ))}
                    </dl>
                  )}
                  <details className="text-[12px]">
                    <summary className="cursor-pointer text-ink-3 hover:text-ink-2">Show the compiled filter (JSON)</summary>
                    <pre className="mt-2 p-2.5 rounded bg-bg-2 border hairline text-[12px] text-ink-2 font-mono whitespace-pre-wrap break-all">
                      {JSON.stringify(pv.predicate, null, 2)}
                    </pre>
                  </details>
                </div>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <h2 className="section-title">Class mix</h2>
                </div>
                <div className="p-4 space-y-1.5">
                  {Object.entries(pv.classes).slice(0, 14).map(([name, n]) => {
                    const max = Math.max(...Object.values(pv.classes), 1);
                    return (
                      <div key={name} className="flex items-center gap-3 text-[12.5px]">
                        <span className="w-36 truncate text-ink-2" title={name}>{humanize(name).replace(/\b(suv|vru|lcv|hcv|muv)\b/gi, (m) => m.toUpperCase())}</span>
                        <span className="flex-1 bg-bg-2 h-2 rounded-full overflow-hidden">
                          <span className="block bg-accent h-2 rounded-full" style={{ width: `${(n / max) * 100}%` }} />
                        </span>
                        <span className="w-16 text-right tabular-nums text-ink-3">{num(n)}</span>
                      </div>
                    );
                  })}
                  {!Object.keys(pv.classes).length && (
                    <div className="text-[12.5px] text-ink-3">No labelled objects in this selection.</div>
                  )}
                  {Object.keys(pv.classes).length > 14 && (
                    <div className="text-[12px] text-ink-3 pt-1">And {plural(Object.keys(pv.classes).length - 14, "more class", "more classes")}.</div>
                  )}
                </div>
              </section>
            </div>

            <section className="panel">
              <div className="panel-head">
                <h2 className="section-title">Use this dataset</h2>
                <button onClick={() => void build()} disabled={building || pv.frames === 0}
                  data-tip="Write the selection as training shards in storage"
                  className="btn btn-primary ml-auto">
                  {building && <span className="running-dot mr-1" />}
                  {building ? "Building shards..." : "Build shards"}
                </button>
              </div>
              <div className="p-4 space-y-3">
                <p className="section-desc">
                  The query is the dataset: load it by name in a training script, or build shards to hand off.
                </p>
                {/* The query is the artifact. A path names where somebody put a zip; this names what the
                    data is, which is what whoever inherits the config actually needs. */}
                <pre className="p-3 rounded bg-bg-2 border hairline text-[12px] text-ink-2 font-mono whitespace-pre-wrap">{snippet}</pre>
                {built && (
                  <div className="text-[12.5px] text-ink-3 space-y-1">
                    <div className="text-pass font-medium">{num(built.samples)} samples in {plural(built.shards.length, "shard")}</div>
                    {built.shards.slice(0, 3).map((sh) => <div key={sh} className="font-mono text-[12px] break-all">{sh}</div>)}
                    {built.index_uri && <div>Index: <span className="font-mono text-[12px] break-all">{built.index_uri}</span></div>}
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
