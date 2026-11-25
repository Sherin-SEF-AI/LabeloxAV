"use client";

import { useState } from "react";
import { T } from "@/lib/i18n";
import { api, humanizeError } from "@/lib/api";
import { useRouter } from "next/navigation";
import { toast } from "@/lib/toast";
import { useQuery } from "@/lib/useQuery";
import type { Scenario } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import ScoreBar from "@/components/shell/ScoreBar";
import EmptyState from "@/components/ui/EmptyState";
import { duration, fromNs, humanize, pct, plural, shortId } from "@/lib/format";

// Scenario mining surface (the second moat): NL search over behaviourally-defined scenarios,
// ranked by criticality (TTC/PET-derived).
const TYPE_COLOR: Record<string, string> = {
  near_miss: "text-block border-block",
  cut_in: "text-warn border-warn",
  wrong_side: "text-accent border-accent",
  hard_brake: "text-info border-info",
  animal_on_road: "text-warn border-warn",
  illegal_park: "text-ink-2 border-line",
  congestion: "text-ink-2 border-line",
};

const NO_ROWS: Scenario[] = [];

export default function ScenariosPage() {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [semantic, setSemantic] = useState(false);
  // What was last asked for, as opposed to what is in the box: the list only changes on submit, so typing
  // does not fire a request per keystroke. Starts as the unfiltered list.
  const [asked, setAsked] = useState<{ q: string; semantic: boolean }>({ q: "", semantic: false });
  // Each answer is cached under the read that produced it. keepPrevious: the last rows stay up while the
  // next search runs, and when it fails.
  const scenariosQ = useQuery<Scenario[] | { results: Scenario[] }>(
    asked.q ? ["scenarioSearch", asked.q, asked.semantic] : ["scenarios", { limit: "200" }],
    () => (asked.q ? api.scenarioSearch(asked.q, asked.semantic) : api.scenarios({ limit: "200" })),
    { keepPrevious: true });
  const found = scenariosQ.data;
  const rows: Scenario[] = !found ? NO_ROWS : Array.isArray(found) ? found : found.results;
  const loading = scenariosQ.validating;
  // A search that fails must not look like a search that found nothing.
  const loadErr = scenariosQ.validating ? null : scenariosQ.error ?? null;

  const load = () => {
    const next = { q: q.trim(), semantic };
    // The same question again is a refresh; a new one changes the key and fetches by itself.
    if (next.q === asked.q && (next.semantic === asked.semantic || !next.q)) void scenariosQ.reload();
    else setAsked(next);
  };

  // A scenario is a window on a session's clock; the frame nearest its start is what a person wants to see.
  const openAt = async (s: Scenario) => {
    try {
      const f = await api.inspectorFrameAt(s.session_id, s.t_in_ns);
      if (f.frame_id) router.push(`/frame/${f.frame_id}`);
      else toast("No frame was captured at this moment", "error");
    } catch (e) { toast(humanizeError(e), "error"); }
  };

  // Every cut-in the miner finds today scores the same maximum, so a bar per row repeats one fact 200 times.
  // Say it once in the header and show the bar only when the rows actually differ.
  const crits = new Set(rows.map((s) => s.criticality.toFixed(2)));
  const uniformCrit = rows.length > 1 && crits.size === 1 ? rows[0].criticality : null;

  return (
    <PageShell
      active="SCENARIOS"
      title="Scenarios"
      subtitle="Behaviours mined from drives (cut-ins, near misses, hard braking), ranked by how critical they were."
      right={
        <span className="inline-flex items-center gap-2 text-[12px] text-ink-2">
          <span className={`w-2 h-2 rounded-full ${loading ? "bg-warn" : "bg-pass"}`} aria-hidden />
          {loading ? "Searching..." : plural(rows.length, "scenario")}
        </span>
      }
    >
      <div className="page-wrap space-y-4">
        <section className="panel p-4 space-y-2">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              load();
            }}
            className="flex gap-2"
          >
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              aria-label="Describe the scenario to find"
              placeholder='Describe a scenario, e.g. "wrong-side autorickshaw cutting in at night on wet road"'
              className="field flex-1 text-[13px] py-2"
            />
            <button className="btn btn-primary px-4">
              <T k="action.search">Search</T>
            </button>
          </form>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-[12px] text-ink-3">
              Your words are turned into filters on type, actor, light and road surface.
            </div>
            <label className="text-[12px] text-ink-2 flex items-center gap-1.5 cursor-pointer"
              data-tip="Also match by image similarity (CLIP), not only by the parsed filters">
              <input type="checkbox" checked={semantic} onChange={(e) => setSemantic(e.target.checked)} />
              Match by image similarity
            </label>
          </div>
        </section>

        {loadErr != null && <LoadState error={loadErr} onRetry={() => void load()} />}

        {!found && loading ? (
          <LoadState loading rows={8} />
        ) : !rows.length ? (
          <div className="panel">
            <EmptyState icon="route" title={asked.q ? "No scenarios match that search" : "No scenarios mined yet"}>
              {asked.q
                ? "Try fewer words, or turn on image similarity to match by what the frames look like."
                : "Scenarios appear here once a session has been mined for cut-ins, near misses and other behaviours."}
            </EmptyState>
            {!asked.q && (
              <details className="px-4 pb-4 text-[12px] text-ink-3">
                <summary className="cursor-pointer select-none">For administrators</summary>
                <div className="mt-1.5">Mine a session from a terminal: <span className="font-mono">make mine ARGS=&quot;--session &lt;uuid&gt;&quot;</span></div>
              </details>
            )}
          </div>
        ) : (
        <section className="panel overflow-x-auto">
          {uniformCrit !== null && (
            <div className="panel-head text-[12px] font-normal">
              All {plural(rows.length, "scenario")} shown have the same criticality ({pct(uniformCrit)}), so it is not repeated on each row.
            </div>
          )}
          <table className="data-table">
            <thead>
              <tr>
                <th className="w-36">Type</th>
                {uniformCrit === null && <th className="w-36">Criticality</th>}
                <th>Actors</th>
                <th className="w-48">Session</th>
                <th className="w-48">Starts</th>
                <th className="w-24 num">Lasts</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((s) => {
                const actors = (s.meta?.actor_classes as string[] | undefined) ??
                  (s.meta?.class ? [s.meta.class as string] : []);
                // Tags repeat the type and the actor class in storage form; show only what they add.
                const extra = s.tags.filter((t) => t !== s.type && !actors.includes(t));
                const lasts = (s.t_out_ns - s.t_in_ns) / 1e9;
                return (
                <tr key={s.scenario_id}>
                  <td>
                    <span className={`inline-flex text-[12px] font-medium border rounded px-1.5 py-0.5 whitespace-nowrap ${TYPE_COLOR[s.type] ?? "text-ink-2 border-line"}`}>
                      {humanize(s.type)}
                    </span>
                  </td>
                  {uniformCrit === null && (
                    <td>
                      <ScoreBar value={s.criticality} tone="warn" />
                    </td>
                  )}
                  <td className="text-ink">
                    {actors.length ? actors.map(humanize).join(", ") : "Ego vehicle"}
                    {extra.length > 0 && <span className="text-ink-3"> · {extra.map(humanize).join(", ")}</span>}
                  </td>
                  <td>
                    <button type="button" onClick={() => void openAt(s)} className="link"
                      data-tip="Open the frame where this scenario starts">
                      {s.vehicle_id ?? shortId(s.session_id)}
                    </button>
                    {s.city && <span className="text-ink-3"> · {s.city}</span>}
                  </td>
                  <td className="text-ink-2 whitespace-nowrap">{fromNs(s.t_in_ns)}</td>
                  <td className="num text-ink-3">{Number.isFinite(lasts) && lasts > 0 ? duration(lasts) : "-"}</td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </section>
        )}
      </div>
    </PageShell>
  );
}
