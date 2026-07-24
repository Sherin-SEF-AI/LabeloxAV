"use client";

import { useCallback, useRef, useState } from "react";
import type { AnnotationRow, LabelConfig } from "@/lib/types";

// Document / OCR: drag a box on the page image, then type what it says.
//
// Boxes are stored in IMAGE pixel coordinates, not screen coordinates, so an annotation stays correct when
// the page is displayed at a different zoom or on a different screen. The transcription is a separate field
// on the same annotation rather than a second row, because a region and its text are one fact.

type Props = {
  uri: string;
  annotations: AnnotationRow[];
  config: LabelConfig;
  activeLabel: string | null;
  onCreate: (bbox: [number, number, number, number]) => void;
  onSelect: (id: string | null) => void;
  selectedId: string | null;
};

function colorFor(config: LabelConfig, label: string | null): string {
  return (config.labels ?? []).find((x) => x.name === label)?.color || "#4c8dff";
}

export default function DocumentEditor({
  uri, annotations, config, activeLabel, onCreate, onSelect, selectedId,
}: Props) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [nat, setNat] = useState({ w: 0, h: 0 });
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);

  const boxes = annotations.filter((a) => a.kind === "bbox");

  // screen -> image pixels
  const toImage = useCallback((clientX: number, clientY: number) => {
    const el = imgRef.current;
    if (!el || !nat.w) return { x: 0, y: 0 };
    const r = el.getBoundingClientRect();
    const sx = nat.w / r.width, sy = nat.h / r.height;
    return {
      x: Math.max(0, Math.min(nat.w, (clientX - r.left) * sx)),
      y: Math.max(0, Math.min(nat.h, (clientY - r.top) * sy)),
    };
  }, [nat]);

  // image pixels -> percentage, so overlays scale with the rendered image automatically
  const pct = (b: number[]) => ({
    left: `${(b[0] / (nat.w || 1)) * 100}%`,
    top: `${(b[1] / (nat.h || 1)) * 100}%`,
    width: `${((b[2] - b[0]) / (nat.w || 1)) * 100}%`,
    height: `${((b[3] - b[1]) / (nat.h || 1)) * 100}%`,
  });

  return (
    <div className="p-4">
      <div ref={wrapRef} className="relative inline-block max-w-full">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img ref={imgRef} src={uri} alt="document page"
          onLoad={(e) => {
            const el = e.currentTarget;
            setNat({ w: el.naturalWidth, h: el.naturalHeight });
          }}
          className="block max-w-full select-none"
          draggable={false}
          onMouseDown={(e) => {
            const p = toImage(e.clientX, e.clientY);
            setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
          }}
          onMouseMove={(e) => {
            if (!drag) return;
            const p = toImage(e.clientX, e.clientY);
            setDrag((d) => (d ? { ...d, x1: p.x, y1: p.y } : null));
          }}
          onMouseUp={() => {
            const d = drag;
            setDrag(null);
            if (!d) return;
            const b: [number, number, number, number] = [
              Math.min(d.x0, d.x1), Math.min(d.y0, d.y1),
              Math.max(d.x0, d.x1), Math.max(d.y0, d.y1),
            ];
            if (b[2] - b[0] < 3 || b[3] - b[1] < 3) { onSelect(null); return; }
            onCreate(b);
          }}
          onMouseLeave={() => setDrag(null)} />

        {/* existing regions */}
        {boxes.map((a) => {
          const b = (a.payload.bbox as number[]) ?? [0, 0, 0, 0];
          const c = colorFor(config, a.label);
          const on = a.annotation_id === selectedId;
          return (
            <div key={a.annotation_id} onClick={() => onSelect(on ? null : a.annotation_id)}
              title={String(a.fields?.text ?? a.label ?? "region")}
              style={{ ...pct(b), borderColor: on ? "#fff" : c, backgroundColor: `${c}1f` }}
              className="absolute border-2 cursor-pointer" />
          );
        })}

        {/* in-progress */}
        {drag && nat.w > 0 && (
          <div style={pct([Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1),
                           Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)])}
            className="absolute border-2 border-accent bg-accent/10 pointer-events-none" />
        )}
      </div>
      <div className="font-mono text-[11px] text-ink-3 mt-2">
        {boxes.length} regions{nat.w ? ` - page ${nat.w}x${nat.h}` : ""}
        {!activeLabel && " - pick a label, then drag a region"}
      </div>
    </div>
  );
}
