"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";

// What a past relabel run did to the corpus, grouped by the decision rather than by the object.
//
// A run rewrote 50,812 objects across 611 class pairs. Most were refinements. Some were category errors, and
// an operator who spots one bus labelled as a bus shelter has no way to learn that 1,046 more share it, or
// that 708 traffic signs and 522 hoardings reached the same class by different routes. One object at a time
// is not a way through fifty thousand of them.
//
// `refused` marks the lineages the ontology guard would reject today: a move that turns a countable thing
// into uncountable stuff, or crosses the l0 boundary. Those need no judgement call to identify, only the
// time to clean up, which is why they sort first and are the default view.

type Lineage = {
  from_name: string; to_name: string;
  /** How many the run moved. History, and it does not change when they are put right. */
  count: number;
  /** How many still carry the class the move produced and nobody has ruled on since. This is the work. */
  outstanding: number;
  refused_now: boolean; reason: string | null;
  examples: { object_id: string; frame_id: string }[];
};

export default function ContaminationPanel() {
  const router = useRouter();
  const [rows, setRows] = useState<Lineage[]>([]);
  const [summary, setSummary] = useState<{ lineages: number; objects: number;
    refused_lineages: number; refused_objects: number;
    outstanding: number; refused_outstanding: number } | null>(null);
  const [refusedOnly, setRefusedOnly] = useState(true);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    api.agentContamination(25, refusedOnly)
      .then((r) => { setRows(r.lineages); setSummary(r.summary); setErr(null); })
      .catch((e) => setErr(humanizeError(e)))
      .finally(() => setLoading(false));
  };

  // Put a refused move back. Offered only where the ontology forbids the result, because a refinement is a
  // judgement call and undoing it in bulk would replace one bulk opinion with another.
  const revert = async (l: Lineage) => {
    const key = `${l.from_name}->${l.to_name}`;
    setBusy(key);
    try {
      const r = await api.agentContaminationRevert(l.from_name, l.to_name);
      toast(`${r.reverted} objects restored to ${l.from_name}, routed to review`, "success", 12000,
        r.run_id ? {
          label: "undo",
          run: async () => {
            const u = await api.agentRevert(r.run_id as string);
            toast(`undone: ${u.reverted} back to ${l.to_name}`, "info");
            load();
          },
        } : undefined);
      load();
    } catch (e) {
      toast(`could not revert: ${humanizeError(e)}`, "error");
    } finally {
      setBusy(null);
    }
  };

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(load, [refusedOnly]);

  // Into the frame editor with the object focused, which is where the correction dialog lives: one fix
  // there now offers to clear the rest of the lineage. A lineage with no example is a number nobody can act
  // on, so the row only offers to open when it can.
  const open = (l: Lineage) => {
    const e = l.examples[0];
    if (e) router.push(`/frame/${e.frame_id}?focus=${e.object_id}`);
  };

  return (
    <div className="panel">
      <div className="flex items-center gap-3 px-4 py-3 border-b hairline">
        <div className="text-ink font-medium">Relabel lineages</div>
        <div className="text-ink-3 text-xs">what a past run changed, grouped by the decision</div>
        <label className="ml-auto flex items-center gap-1.5 font-mono text-[11px] text-ink-3 cursor-pointer">
          <input type="checkbox" checked={refusedOnly} className="accent-accent"
            onChange={(e) => setRefusedOnly(e.target.checked)} />
          only ones refused today
        </label>
      </div>

      {summary && (
        <div className="px-4 py-2 border-b hairline font-mono text-[11px] text-ink-3">
          {/* The work first, the history second. Leading with what the run once did meant the board still
              read 3,057 outstanding immediately after 3,010 of them had been put right. */}
          {summary.refused_outstanding > 0 ? (
            <span className="text-block">
              {summary.refused_outstanding.toLocaleString()} objects still carry a class the ontology refuses
            </span>
          ) : (
            <span className="text-pass">nothing outstanding</span>
          )}
          <span className="text-ink-3">
            {" "}· {summary.objects.toLocaleString()} moved in all, across {summary.lineages} lineages
          </span>
        </div>
      )}

      {err ? <div className="px-4 py-6 text-center text-block text-xs font-mono">{err}</div>
        : loading ? <div className="px-4 py-6 text-center text-ink-3 text-xs font-mono">loading...</div>
        : rows.length ? (
          <table className="w-full text-sm">
            <thead className="text-ink-3 font-mono text-[11px] uppercase border-b hairline">
              <tr>
                <th className="text-left font-normal px-3 py-2 w-28">still wrong</th>
                <th className="text-left font-normal px-3 py-2">move</th>
                <th className="text-left font-normal px-3 py-2">why it would be refused</th>
                <th className="px-3 py-2 w-40"></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((l) => (
                <tr key={`${l.from_name}->${l.to_name}`} className="border-b hairline hover:bg-bg-2">
                  <td className="px-3 py-2 font-mono">
                    <span className={l.outstanding ? "text-ink-2" : "text-pass"}>
                      {l.outstanding.toLocaleString()}
                    </span>
                    <span className="text-ink-3/60"> / {l.count.toLocaleString()}</span>
                  </td>
                  <td className="px-3 py-2 font-mono text-[11px]">
                    <span className="text-ink-3">{l.from_name}</span>
                    <span className="text-ink-3"> → </span>
                    <span className={l.refused_now ? "text-block" : "text-ink-2"}>{l.to_name}</span>
                  </td>
                  <td className="px-3 py-2 text-ink-3 text-xs">{l.reason ?? "-"}</td>
                  <td className="px-3 py-2 text-right font-mono text-[10px] whitespace-nowrap">
                    {l.examples[0] && (
                      <button onClick={() => open(l)}
                        title="open an example, where one correction can be applied to the rest"
                        className="border border-line px-1.5 py-0.5 rounded text-ink-3 hover:border-accent">
                        open
                      </button>
                    )}
                    {l.refused_now && l.outstanding > 0 && (
                      <button onClick={() => revert(l)} disabled={busy !== null}
                        title={`put the remaining ${l.outstanding} back to ${l.from_name} and route them to review`}
                        className="ml-1 border border-line px-1.5 py-0.5 rounded text-ink-3 hover:border-block hover:text-block disabled:opacity-40">
                        {busy === `${l.from_name}->${l.to_name}` ? "..." : "revert"}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="px-4 py-10 text-center text-ink-3 text-sm">
            {refusedOnly
              ? "No relabel lineage would be refused by the ontology guard."
              : "No relabel run has rewritten this corpus."}
          </div>
        )}
    </div>
  );
}
