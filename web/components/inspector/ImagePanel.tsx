"use client";

import { useEffect, useRef, useState } from "react";
import { api, type InspectorPanel } from "@/lib/api";
import { useClock } from "@/lib/inspector/clock";
import { useMcap } from "@/lib/inspector/mcapContext";

// Image panel: the camera frame at the current clock time. JPEG CompressedImage decodes directly from the
// MCAP (base64 data URL). When the payload is not a decodable JSON image (e.g. a protobuf or H.264 session),
// it falls back to the already-extracted frame keyed by ts_ns, the fast path for ingested sessions. Annotation
// overlays are drawn on top in M-I.5.

export default function ImagePanel({ panel, onFrame }: { panel: InspectorPanel; onFrame?: (frameId: string | null, tsNs: string) => void }) {
  const clock = useClock();
  const { mcap, sessionId } = useMcap();
  const [src, setSrc] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const busy = useRef(false);
  const pending = useRef<bigint | null>(null);

  useEffect(() => {
    if (!panel.topic) return;
    const run = async (ns: bigint) => {
      if (busy.current) { pending.current = ns; return; }
      busy.current = true;
      try {
        const m = await mcap.latestAt(panel.topic!, ns, 3_000_000_000n);
        const v = m?.value as { data?: string; format?: string } | null;
        if (v && typeof v.data === "string") {
          setSrc(`data:image/${v.format || "jpeg"};base64,${v.data}`);
          setNote("");
          onFrame?.(null, ns.toString());
        } else {
          // fast path: the extracted frame nearest this ts_ns
          const f = await api.inspectorFrameAt(sessionId, Number(ns)).catch(() => null);
          if (f?.image_url) { setSrc(f.image_url); setNote(""); onFrame?.(f.frame_id, ns.toString()); }
          else { setNote("no frame at this time"); }
        }
      } catch (e) {
        setNote("image read error: " + String(e));
      } finally {
        busy.current = false;
        if (pending.current !== null) { const n = pending.current; pending.current = null; run(n); }
      }
    };
    return clock.subscribe(() => run(clock.nowNs()));
  }, [clock, mcap, panel.topic, sessionId, onFrame]);

  return (
    <div className="relative h-full w-full bg-black flex items-center justify-center overflow-hidden">
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt="" className="max-h-full max-w-full object-contain" />
      ) : (
        <div className="font-mono text-[10px] text-ink-3">{note || "loading frame..."}</div>
      )}
    </div>
  );
}
