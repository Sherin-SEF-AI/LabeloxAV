"use client";

// A frame (or a window of it) with labelled boxes drawn on top, in image coordinates. The SVG's viewBox is the
// window itself, so boxes stay registered with the pixels at any display size and nothing is recomputed on
// resize. Used by the error explanation, the disagreement queue and the rare-scene grid.

import { focusWindow, type Tone } from "@/lib/tools/quality";

export type DrawBox = { bbox: number[]; tone: Tone; label?: string; dashed?: boolean };

const STROKE: Record<Tone, string> = {
  pass: "rgb(var(--c-pass))", warn: "rgb(var(--c-warn))", block: "rgb(var(--c-block))",
  info: "rgb(var(--c-info))", muted: "rgb(var(--c-ink-3))",
};

export default function EvidenceImage({ src, width, height, boxes, focus = true, maxW = 480, maxH = 300,
  maxZoom = 3, alt = "", className = "" }: {
  src: string;
  width: number;
  height: number;
  boxes: DrawBox[];
  /** Zoom to the boxes (with padding) rather than show the whole frame. */
  focus?: boolean;
  maxW?: number;
  maxH?: number;
  maxZoom?: number;
  alt?: string;
  className?: string;
}) {
  const win = focus ? focusWindow(boxes.map((b) => b.bbox), width, height) : [0, 0, width, height];
  const ww = Math.max(1, win[2] - win[0]);
  const wh = Math.max(1, win[3] - win[1]);
  const s = Math.min(maxZoom, maxW / ww, maxH / wh);
  const dw = Math.round(ww * s);
  const dh = Math.round(wh * s);
  // Text and strokes are sized in screen pixels, converted to image units by the zoom.
  const px = 1 / s;
  return (
    <div className={`relative overflow-hidden rounded border hairline bg-bg-2 ${className}`} style={{ width: dw, height: dh }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} loading="lazy" decoding="async" draggable={false}
        style={{ position: "absolute", left: -win[0] * s, top: -win[1] * s, width: width * s, height: height * s, maxWidth: "none" }} />
      <svg className="absolute inset-0" width={dw} height={dh} viewBox={`${win[0]} ${win[1]} ${ww} ${wh}`} aria-hidden>
        {boxes.map((b, i) => {
          const [x1, y1, x2, y2] = b.bbox;
          return (
            <g key={i}>
              <rect x={x1} y={y1} width={Math.max(0, x2 - x1)} height={Math.max(0, y2 - y1)} fill="none"
                stroke={STROKE[b.tone]} strokeWidth={2 * px} strokeDasharray={b.dashed ? `${6 * px} ${4 * px}` : undefined} />
              {b.label && (
                <>
                  <rect x={x1} y={Math.max(win[1], y1 - 15 * px)} width={Math.min(ww, (b.label.length * 6.4 + 8) * px)}
                    height={14 * px} fill="rgb(var(--c-bg) / 0.85)" />
                  <text x={x1 + 4 * px} y={Math.max(win[1], y1 - 15 * px) + 10.5 * px} fontSize={11 * px}
                    fill={STROKE[b.tone]}>{b.label}</text>
                </>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
