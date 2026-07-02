"use client";

// M-MC.1 rig canvas: the multi-camera view state (NOT a new mode). The focused camera is the full annotation
// canvas (the EditorCanvas passed as children), so every tool keeps working exactly as in single-frame view;
// the other rig cameras render as lightweight context tiles around it, ordered by mounting yaw so the surround
// reads left, front, right, back. A dropped camera (in the group's missing_cams) shows as an empty tile rather
// than being silently omitted. Clicking a context tile focuses that camera (the page navigates to its frame,
// loading the full editor state there). Three layouts: grid (equal cells), surround-strip (one panoramic row),
// and focus+context (one large focused view over a context strip). Context tiles load progressively.

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

export type RigLayout = "grid" | "strip" | "focus";

// Nominal mounting yaw (deg) from vehicle forward, mirroring RigSettings.camera_yaw_deg, so the surround
// strip reads left -> front -> right -> back. Unknown cameras sort after, alphabetically.
const CAM_YAW: Record<string, number> = { cam_l: -90, cam_f: 0, cam_r: 90, cam_b: 180 };
const orderByYaw = (cams: string[]) =>
  [...cams].sort((a, b) => {
    const ya = a in CAM_YAW ? CAM_YAW[a] : 1e6;
    const yb = b in CAM_YAW ? CAM_YAW[b] : 1e6;
    return ya === yb ? a.localeCompare(b) : ya - yb;
  });

type TileObj = { bbox: number[]; class_name: string; state: string };

function ContextTile({ cam, frameId, focused, onFocus }: {
  cam: string; frameId: string | null; focused: boolean; onFocus: () => void;
}) {
  const [meta, setMeta] = useState<{ image_url: string; width: number; height: number } | null>(null);
  const [objs, setObjs] = useState<TileObj[]>([]);
  const [err, setErr] = useState(false);

  useEffect(() => {
    if (!frameId) return;
    let live = true;
    (async () => {
      try {
        const m = await api.frame(frameId);
        if (!live) return;
        setMeta({ image_url: m.image_url, width: m.width, height: m.height });
        // objects are context only (read-only overlay); best-effort, never block the tile
        api.frameObjects(frameId)
          .then((o) => live && setObjs(o.map((x) => ({ bbox: x.bbox, class_name: x.class_name, state: x.state }))))
          .catch(() => undefined);
      } catch { if (live) setErr(true); }
    })();
    return () => { live = false; };
  }, [frameId]);

  if (!frameId) {
    return (
      <div className="relative h-full w-full grid place-items-center bg-bg-2 border border-dashed border-block/40 rounded">
        <div className="text-center">
          <div className="font-mono text-[10px] uppercase text-block">{cam}</div>
          <div className="font-mono text-[9px] text-ink-3 mt-0.5">no frame · dropped</div>
        </div>
      </div>
    );
  }
  return (
    <button onClick={onFocus} title={`focus ${cam}`}
      className={`relative h-full w-full overflow-hidden rounded bg-bg-2 border ${focused ? "border-accent" : "border-line hover:border-ink-3"}`}>
      {meta && !err ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={meta.image_url} alt={cam} loading="lazy" className="absolute inset-0 h-full w-full object-contain" />
      ) : (
        <div className="absolute inset-0 grid place-items-center font-mono text-[10px] text-ink-3">{err ? "load failed" : "loading..."}</div>
      )}
      {meta && objs.length > 0 && (
        <svg viewBox={`0 0 ${meta.width} ${meta.height}`} preserveAspectRatio="xMidYMid meet"
          className="absolute inset-0 h-full w-full pointer-events-none">
          {objs.map((o, i) => (
            <rect key={i} x={o.bbox[0]} y={o.bbox[1]} width={o.bbox[2] - o.bbox[0]} height={o.bbox[3] - o.bbox[1]}
              fill="none" stroke={o.state === "rejected" ? "#F85149" : "#FF7A2F"} strokeWidth={Math.max(meta.width, meta.height) / 400} opacity={0.85} />
          ))}
        </svg>
      )}
      <span className="absolute left-1 top-1 font-mono text-[9px] uppercase text-ink bg-bg/70 px-1 rounded">{cam}</span>
      {focused && <span className="absolute right-1 top-1 font-mono text-[9px] uppercase text-accent bg-bg/70 px-1 rounded">focus</span>}
    </button>
  );
}

export default function RigView({ cameras, focusedCam, frameIds, layout, missingCams, onFocus, children }: {
  cameras: string[];
  focusedCam: string;
  frameIds: Record<string, string>;    // cam -> frame_id (present cameras only)
  missingCams: string[];
  layout: RigLayout;
  onFocus: (cam: string, frameId: string) => void;
  children: React.ReactNode;           // the focused EditorCanvas
}) {
  const ordered = orderByYaw(cameras);
  const others = ordered.filter((c) => c !== focusedCam);
  const wrapRef = useRef<HTMLDivElement>(null);

  const tile = (cam: string) => {
    if (cam === focusedCam) return <div key={cam} className="relative h-full w-full">{children}</div>;
    const fid = frameIds[cam] ?? null;
    return <ContextTile key={cam} cam={cam} frameId={missingCams.includes(cam) ? null : fid}
      focused={false} onFocus={() => fid && onFocus(cam, fid)} />;
  };

  if (layout === "focus") {
    return (
      <div ref={wrapRef} className="absolute inset-0 flex flex-col gap-1 p-1">
        <div className="relative flex-1 min-h-0">{children}</div>
        {others.length > 0 && (
          <div className="h-[110px] shrink-0 flex gap-1 overflow-x-auto">
            {others.map((cam) => (
              <div key={cam} className="relative h-full w-[150px] shrink-0">
                <ContextTile cam={cam} frameId={missingCams.includes(cam) ? null : (frameIds[cam] ?? null)}
                  focused={false} onFocus={() => frameIds[cam] && onFocus(cam, frameIds[cam])} />
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (layout === "strip") {
    return (
      <div ref={wrapRef} className="absolute inset-0 flex gap-1 p-1 overflow-x-auto">
        {ordered.map((cam) => (
          <div key={cam} className="relative h-full shrink-0" style={{ width: `max(240px, ${100 / Math.max(ordered.length, 1)}%)` }}>
            {tile(cam)}
          </div>
        ))}
      </div>
    );
  }

  // grid: equal cells, roughly square arrangement
  const cols = Math.ceil(Math.sqrt(ordered.length));
  return (
    <div ref={wrapRef} className="absolute inset-0 grid gap-1 p-1"
      style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gridAutoRows: "minmax(0, 1fr)" }}>
      {ordered.map((cam) => <div key={cam} className="relative min-h-0">{tile(cam)}</div>)}
    </div>
  );
}
