"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { Spinner } from "@/components/Spinner";

// The Session Inspector home: pick an MCAP session to inspect. The health verdict chip answers "did my
// recording work" before a single panel is opened.

type Row = { session_id: string; vehicle_id: string; city: string | null; start_ts_ns: number; end_ts_ns: number; verdict: string | null };

const V: Record<string, string> = { pass: "text-pass border-pass", warn: "text-warn border-warn", fail: "text-block border-block" };

export default function InspectorHome() {
  const router = useRouter();
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.inspectorSessions().then(setRows).catch(() => setRows([])).finally(() => setLoading(false));
  }, []);

  return (
    <PageShell active="INSPECT" title="Session Inspector" subtitle="MCAP" right={loading ? <Spinner label="loading" /> : <span className="font-mono text-xs text-ink-3">{rows.length} sessions</span>}>
      <div className="h-full overflow-auto">
        <div className="max-w-4xl mx-auto p-6">
          <p className="text-ink-3 text-sm mb-4 max-w-2xl">Foxglove-class MCAP inspection inside LabeloxAV: one ts_ns clock across camera, IMU, CAN, GPS, and raw messages, with annotation overlays and session-health verdicts. Open any session below, or use the Lichtblick escape hatch for the long tail.</p>
          {rows.length === 0 && !loading ? (
            <div className="panel p-6 text-center text-ink-3 text-sm">No MCAP sessions yet. Ingest an .mcap on the Import page.</div>
          ) : (
            <div className="panel divide-y divide-line/50">
              {rows.map((r) => (
                <button key={r.session_id} onClick={() => router.push(`/inspect/${r.session_id}`)}
                  className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-bg-2 text-left">
                  <span className="font-mono text-[12px] text-ink-2">{r.vehicle_id}</span>
                  <span className="font-mono text-[10px] text-ink-3">{r.city || ""}</span>
                  <span className="font-mono text-[9.5px] text-ink-3/70 truncate">{r.session_id}</span>
                  {r.verdict && <span className={`ml-auto border px-1.5 rounded uppercase font-mono text-[10px] ${V[r.verdict] || "border-line text-ink-3"}`}>{r.verdict}</span>}
                  <span className={r.verdict ? "font-mono text-[10px] text-accent" : "ml-auto font-mono text-[10px] text-accent"}>inspect →</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}
