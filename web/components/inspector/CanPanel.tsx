"use client";

import { useEffect, useRef, useState } from "react";
import { flattenNumeric } from "@/lib/inspector/mcap";
import { useClock } from "@/lib/inspector/clock";
import { useMcap } from "@/lib/inspector/mcapContext";

// CAN table: the decoded CAN signals at the current clock time. The platform's recorder writes decoded
// signals (speed 0x247, torque 0x111, regen 0x249) on the CAN topics; this shows their current values, one
// row per signal, refreshed as the clock moves.

type Row = { topic: string; field: string; value: number };

export default function CanPanel() {
  const clock = useClock();
  const { mcap } = useMcap();
  const [rows, setRows] = useState<Row[]>([]);
  const [note, setNote] = useState("locating CAN...");
  const topics = useRef<string[]>([]);
  const busy = useRef(false);

  useEffect(() => {
    topics.current = mcap.topics().filter((t) => /can/i.test(t.topic)).map((t) => t.topic);
    if (topics.current.length === 0) { setNote("no CAN topic"); return; }
    setNote("");
    const run = async (ns: bigint) => {
      if (busy.current) return;
      busy.current = true;
      try {
        const out: Row[] = [];
        for (const tp of topics.current) {
          const m = await mcap.latestAt(tp, ns, 2_000_000_000n);
          if (m) for (const [k, v] of Object.entries(flattenNumeric(m.value))) out.push({ topic: tp, field: k, value: v });
        }
        setRows(out);
      } finally { busy.current = false; }
    };
    return clock.subscribe(() => run(clock.nowNs()));
  }, [clock, mcap]);

  return (
    <div className="h-full overflow-auto no-scrollbar p-2">
      {note ? <div className="font-mono text-[10px] text-ink-3">{note}</div> : (
        <table className="w-full font-mono text-[10.5px]">
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b hairline/40">
                <td className="text-ink-3 pr-2 truncate">{r.topic}</td>
                <td className="text-ink-2 pr-2">{r.field}</td>
                <td className="text-accent text-right tabular-nums">{Number.isInteger(r.value) ? r.value : r.value.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
