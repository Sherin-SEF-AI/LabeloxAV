"use client";

import { useEffect, useRef, useState } from "react";
import { api, type InspectorPanel } from "@/lib/api";
import { useClock } from "@/lib/inspector/clock";
import { useMcap } from "@/lib/inspector/mcapContext";

// Image panel: the camera frame at the current clock time. JPEG CompressedImage decodes directly from the
// MCAP (base64 data URL). When the payload is not a decodable JSON image (e.g. a protobuf or H.264 session),
// it falls back to the already-extracted frame keyed by ts_ns, the fast path for ingested sessions. Annotation
// overlays are drawn on top in M-I.5.

type Ann = { class_name: string; conf: number; bbox: number[]; source: string };

export default function ImagePanel({ panel, onFrame }: { panel: InspectorPanel; onFrame?: (frameId: string | null, tsNs: string) => void }) {
  const clock = useClock();
  const { mcap, sessionId } = useMcap();
  const [src, setSrc] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [overlays, setOverlays] = useState(true);
  const [ann, setAnn] = useState<{ width: number; height: number; objects: Ann[] } | null>(null);
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
        } else {
          // fast path: the extracted frame nearest this ts_ns
          const f = await api.inspectorFrameAt(sessionId, Number(ns)).catch(() => null);
          if (f?.image_url) { setSrc(f.image_url); setNote(""); }
          else { setNote("no frame at this time"); }
        }
        // annotation overlay: the nearest extracted frame's boxes, synchronized to playback
        if (overlays) {
          const a = await api.inspectorAnnotationsAt(sessionId, Number(ns)).catch(() => null);
          if (a && a.width && a.height) { setAnn({ width: a.width, height: a.height, objects: a.objects }); onFrame?.(a.frame_id, ns.toString()); }
          else { setAnn(null); onFrame?.(null, ns.toString()); }
        } else {
          setAnn(null);
        }
      } catch (e) {
        setNote("image read error: " + String(e));
      } finally {
        busy.current = false;
        if (pending.current !== null) { const n = pending.current; pending.current = null; run(n); }
      }
    };
    return clock.subscribe(() => run(clock.nowNs()));
  }, [clock, mcap, panel.topic, sessionId, onFrame, overlays]);

  const STROKE: Record<string, string> = { human: "#56D364", auto_accept: "#FF7A2F", fused: "#58A6FF" };

  return (
    <div className="relative h-full w-full bg-black flex items-center justify-center overflow-hidden">
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt="" className="max-h-full max-w-full object-contain" />
      ) : (
        <div className="font-mono text-[10px] text-ink-3">{note || "loading frame..."}</div>
      )}
      {overlays && ann && ann.objects.length > 0 && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${ann.width} ${ann.height}`} preserveAspectRatio="xMidYMid meet">
          {ann.objects.map((o, i) => (
            <g key={i}>
              <rect x={o.bbox[0]} y={o.bbox[1]} width={Math.max(0, o.bbox[2] - o.bbox[0])} height={Math.max(0, o.bbox[3] - o.bbox[1])}
                fill="none" stroke={STROKE[o.source] || "#FF7A2F"} strokeWidth={1.5} />
              <text x={o.bbox[0]} y={Math.max(8, o.bbox[1] - 2)} fill={STROKE[o.source] || "#FF7A2F"} fontSize={9} fontFamily="monospace">{o.class_name} {o.conf.toFixed(2)}</text>
            </g>
          ))}
        </svg>
      )}
      <button onClick={() => setOverlays((v) => !v)} title="toggle annotation overlays"
        className={`absolute top-1 right-1 font-mono text-[9px] border px-1 rounded ${overlays ? "border-accent/50 text-accent bg-accent/10" : "border-line text-ink-3"}`}>
        overlays {overlays ? "on" : "off"}
      </button>
    </div>
  );
}
