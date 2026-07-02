"use client";

import { useEffect, useRef, useState } from "react";
import { useClock } from "@/lib/inspector/clock";
import { useMcap } from "@/lib/inspector/mcapContext";

// The raw message panel: the decoded message on a topic at the current clock time, monospace. Subscribes to
// the shared clock and refetches the latest message at each seek (throttled), so scrubbing moves it in
// lockstep with every other panel on one ts_ns clock.

export default function RawPanel({ topic }: { topic: string }) {
  const clock = useClock();
  const { mcap } = useMcap();
  const [text, setText] = useState("(no message yet)");
  const busy = useRef(false);
  const pending = useRef<bigint | null>(null);

  useEffect(() => {
    const run = async (ns: bigint) => {
      if (busy.current) { pending.current = ns; return; }
      busy.current = true;
      try {
        const m = await mcap.latestAt(topic, ns);
        setText(m ? JSON.stringify(m.value, null, 2) : "(no message at or before this time)");
      } catch (e) {
        setText("read error: " + String(e));
      } finally {
        busy.current = false;
        if (pending.current !== null) { const n = pending.current; pending.current = null; run(n); }
      }
    };
    return clock.subscribe(() => run(clock.nowNs()));
  }, [clock, mcap, topic]);

  return (
    <div className="h-full overflow-auto no-scrollbar p-2">
      <pre className="font-mono text-[10.5px] text-ink-2 whitespace-pre-wrap">{text}</pre>
    </div>
  );
}
