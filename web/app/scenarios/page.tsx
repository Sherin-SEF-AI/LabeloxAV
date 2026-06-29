"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { Scenario } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import ScoreBar from "@/components/shell/ScoreBar";

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

export default function ScenariosPage() {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [semantic, setSemantic] = useState(false);
  const [rows, setRows] = useState<Scenario[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      if (q.trim()) {
        const res = await api.scenarioSearch(q.trim(), semantic);
        setRows(res.results);
      } else {
        setRows(await api.scenarios({ limit: "200" }));
      }
    } finally {
      setLoading(false);
    }
  }, [q, semantic]);

  useEffect(() => {
    load();
  }, []); // initial

  return (
    <PageShell
      active="SCENARIOS"
      right={
        <>
          <span className="border border-line px-2 py-0.5">{rows.length} found</span>
          <span className={`w-2 h-2 rounded-full ${loading ? "bg-warn" : "bg-pass"}`} />
        </>
      }
      filters={
        <div className="w-full">
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
              placeholder='e.g. "wrong-side autorickshaw cutting in at night on wet road"'
              className="flex-1 bg-panel border hairline text-ink text-sm px-3 py-2 font-mono"
            />
            <button className="border border-accent text-accent px-4 text-sm font-mono hover:bg-accent/10">
              search
            </button>
          </form>
          <div className="flex items-center justify-between mt-1">
            <div className="font-mono text-[11px] text-ink-3">
              parsed into structured filters over the scenario index (type, actor class, light, surface)
            </div>
            <label className="font-mono text-[11px] text-ink-2 flex items-center gap-1.5 cursor-pointer">
              <input type="checkbox" checked={semantic} onChange={(e) => setSemantic(e.target.checked)} />
              semantic (CLIP)
            </label>
          </div>
        </div>
      }
    >
      <table className="w-full text-sm">
          <thead className="text-ink-3 font-mono text-[11px] uppercase border-b hairline sticky top-0 bg-bg">
            <tr>
              <th className="text-left font-normal px-3 py-2 w-40">type</th>
              <th className="text-left font-normal px-3 py-2 w-40">criticality</th>
              <th className="text-left font-normal px-3 py-2">actors / tags</th>
              <th className="text-left font-normal px-3 py-2 w-28">city</th>
              <th className="text-left font-normal px-3 py-2 w-44">window (ts_ns)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => (
              <tr key={s.scenario_id} className="border-b hairline hover:bg-bg-2">
                <td className="px-3 py-2">
                  <span className={`font-mono text-[11px] uppercase border px-1.5 py-0.5 ${TYPE_COLOR[s.type] ?? "text-ink-2 border-line"}`}>
                    {s.type}
                  </span>
                </td>
                <td className="px-3 py-2">
                  <ScoreBar value={s.criticality} tone="warn" />
                </td>
                <td className="px-3 py-2 font-mono text-xs text-ink-2">
                  {(s.meta?.actor_classes as string[] | undefined)?.join(", ") ||
                    (s.meta?.class as string) ||
                    "ego"}
                  <span className="text-ink-3"> · {s.tags.join(" ")}</span>
                </td>
                <td className="px-3 py-2 font-mono text-xs">{s.city ?? "?"}</td>
                <td className="px-3 py-2 font-mono text-[11px] text-ink-3">
                  {String(s.t_in_ns).slice(0, 13)}
                </td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={5} className="px-3 py-8 text-center text-ink-3">
                  no scenarios. mine a session: <span className="font-mono">make mine ARGS=&quot;--session &lt;uuid&gt;&quot;</span>
                </td>
              </tr>
            )}
          </tbody>
        </table>
    </PageShell>
  );
}
