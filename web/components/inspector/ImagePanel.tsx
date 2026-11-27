"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError, type InspectorPanel, humanizeError } from "@/lib/api";
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
  // Distinct from `note`, because "we have not looked yet" and "we looked and there is nothing" are
  // different facts and only one of them is worth waiting for. Rendering the first message for both is the
  // same defect the analytics page had: an unfinished request shown as a settled answer.
  const [loaded, setLoaded] = useState(false);
  const [overlays, setOverlays] = useState(true);
  const [ann, setAnn] = useState<{ width: number; height: number; objects: Ann[] } | null>(null);
  const busy = useRef(false);
  const pending = useRef<bigint | null>(null);

  useEffect(() => {
    // Playback ticks the clock every few milliseconds, and this panel used to fetch on every tick: one
    // request finished, the next started, so a playing session drove ~100 requests a second per panel at
    // the API, past the rate limit. Once 429s came back they came back instantly, which made the loop spin
    // faster, and each one blanked the image. Three things stop that:
    //  - the server says over which playhead window its answer stays the nearest frame, and the panel does
    //    not ask again while the playhead is inside it (once per extracted frame, not once per tick);
    //  - rounds are spaced at least MIN_GAP_MS apart, so scrubbing cannot outrun the server either;
    //  - a 429 pauses the panel for the Retry-After interval and keeps the last image on screen.
    const MIN_GAP_MS = 80;
    const BACKOFF_MS = 1000;
    type Win = { from: bigint | null; to: bigint | null };
    let frameWin: Win | null = null;
    let annWin: Win | null = null;
    let lastRun = 0;
    let backoffUntil = 0;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let alive = true;
    const inside = (w: Win | null, ns: bigint) =>
      w !== null && (w.from === null || ns >= w.from) && (w.to === null || ns < w.to);
    const toWin = (r: { valid_from_ns?: number | null; valid_to_ns?: number | null }): Win | null =>
      r.valid_from_ns === undefined && r.valid_to_ns === undefined ? null
        : { from: r.valid_from_ns == null ? null : BigInt(r.valid_from_ns),
            to: r.valid_to_ns == null ? null : BigInt(r.valid_to_ns) };
    const rateLimited = (e: unknown) => e instanceof ApiError && e.status === 429;

    const run = async (ns: bigint) => {
      busy.current = true;
      lastRun = Date.now();
      try {
        // A session with no recording has no topic to read, and the panel used to return here and sit on
        // "loading frame..." forever. The extracted-frame path needs only a session and a timestamp, which
        // that session has, so it is the only path when there is no topic rather than the fallback.
        const m = panel.topic ? await mcap.latestAt(panel.topic, ns, 3_000_000_000n) : null;
        const v = m?.value as { data?: string; format?: string } | null;
        if (v && typeof v.data === "string") {
          setSrc(`data:image/${v.format || "jpeg"};base64,${v.data}`);
          setNote("");
        } else if (!inside(frameWin, ns)) {
          // fast path: the extracted frame nearest this ts_ns
          const f = await api.inspectorFrameAt(sessionId, Number(ns));
          if (!alive) return;
          frameWin = toWin(f);
          if (f.image_url) { setSrc(f.image_url); setNote(""); }
          else { setSrc(null); setNote("no frame at this time"); }
        }
        // annotation overlay: the nearest extracted frame's boxes, synchronized to playback
        if (overlays) {
          if (!inside(annWin, ns)) {
            const a = await api.inspectorAnnotationsAt(sessionId, Number(ns));
            if (!alive) return;
            annWin = toWin(a);
            if (a.width && a.height) { setAnn({ width: a.width, height: a.height, objects: a.objects }); onFrame?.(a.frame_id, ns.toString()); }
            else { setAnn(null); onFrame?.(null, ns.toString()); }
          }
        } else {
          setAnn(null);
        }
      } catch (e) {
        if (rateLimited(e)) {
          backoffUntil = Date.now() + BACKOFF_MS;
          pending.current = pending.current ?? ns;   // retry this time once the pause is over
        } else if (alive) {
          setNote("image read error: " + humanizeError(e));
        }
      } finally {
        busy.current = false;
        if (alive) { setLoaded(true); drain(); }
      }
    };

    const drain = () => {
      if (busy.current || timer !== null || pending.current === null) return;
      const now = Date.now();
      const wait = Math.max(MIN_GAP_MS - (now - lastRun), backoffUntil - now, 0);
      const go = () => {
        timer = null;
        const n = pending.current;
        pending.current = null;
        if (n !== null && alive) void run(n);
      };
      if (wait > 0) timer = setTimeout(go, wait);
      else go();
    };
    const request = (ns: bigint) => { pending.current = ns; drain(); };

    // Once on mount as well as on every tick. Subscribing alone meant a panel added while playback was
    // paused never read anything at all, because the clock had no reason to fire.
    request(clock.nowNs());
    const unsubscribe = clock.subscribe(() => request(clock.nowNs()));
    return () => {
      alive = false;
      if (timer !== null) clearTimeout(timer);
      pending.current = null;
      unsubscribe();
    };
  }, [clock, mcap, panel.topic, sessionId, onFrame, overlays]);

  const STROKE: Record<string, string> = { human: "#56D364", auto_accept: "#FF7A2F", fused: "#58A6FF" };

  return (
    <div className="relative h-full w-full bg-black flex items-center justify-center overflow-hidden">
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt="" className="max-h-full max-w-full object-contain" />
      ) : (
        <div className="text-[11px] text-ink-3">{note || (loaded ? "no frame at this time" : "loading frame...")}</div>
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
      <button onClick={() => setOverlays((v) => !v)} data-tip="Show or hide the labels on the image"
        className={`absolute top-1 right-1 text-[11px] border px-1.5 py-0.5 rounded ${overlays ? "border-accent/60 text-accent-2 bg-panel/90" : "border-line text-ink-2 bg-panel/90"}`}>
        Overlays {overlays ? "on" : "off"}
      </button>
    </div>
  );
}
