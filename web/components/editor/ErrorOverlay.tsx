"use client";

// Likely label errors explained on the image (tool #12), drawn over the editor canvas.
//
// Mount it as a sibling of <EditorCanvas> inside the same relatively positioned box that the canvas fills, and pass
// the editor's viewport (the same object EditorCanvas takes). It fetches the frame's pending error candidates from
// GET /api/tools/quality/frames/{frameId}/why and draws, in screen space:
//   - a dashed red box and a "!" tag on each flagged object
//   - for the selected flag, the boxes the check compared it with (the box it duplicates, the vehicle a rider sits
//     inside) in amber, and a card with the plain-language reason, the numbers, the crops the check leaned on, the
//     vision model's opinion and a link to rule on it in the error queue
// Everything is pointer-events: none except the tags and the card, so drawing and selecting on the canvas work
// through it.
//
// Props:
//   frameId           the frame being edited
//   viewport          { scale, ox, oy }: screen = o + scale * image, exactly EditorCanvas's `viewport`
//   visible?          default true; false renders nothing (e.g. bound to a layer toggle)
//   selectedObjectId? the editor's selected object; its flag opens automatically
//   onSelectObject?   called with an object id when a tag or a neighbour on this frame is clicked
//   status?           "pending" (default) or "all" candidates
//   refreshKey?       change it (e.g. after a save) to refetch
//
// Suggested mount in web/app/frame/[id]/page.tsx (the coordinator's change), inside `editorCanvasEl`:
//   <CanvasErrorBoundary>
//     <div className="relative w-full h-full">
//       <EditorCanvas ... />
//       <ErrorOverlay frameId={id} viewport={st.viewport} selectedObjectId={st.selectedId}
//         onSelectObject={onCanvasSelect} visible={mode !== "lidar3d"} />
//     </div>
//   </CanvasErrorBoundary>
// (`onCanvasSelect` takes `string | null`, so it fits `onSelectObject` as is.)

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import ErrorWhyPanel from "@/components/quality/ErrorWhyPanel";
import { humanizeError } from "@/lib/api";
import { boxToScreen, type Viewport } from "@/lib/tools/quality";
import { getFrameWhy, type ErrorWhy } from "@/lib/tools/qualityApi";

export type ErrorOverlayProps = {
  frameId: string;
  viewport: Viewport;
  visible?: boolean;
  selectedObjectId?: string | null;
  onSelectObject?: (objectId: string) => void;
  status?: "pending" | "all";
  refreshKey?: unknown;
};

const CARD_W = 360;

export default function ErrorOverlay({ frameId, viewport, visible = true, selectedObjectId = null,
  onSelectObject, status = "pending", refreshKey }: ErrorOverlayProps) {
  const [items, setItems] = useState<ErrorWhy[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);   // object id whose card is open
  const [listOpen, setListOpen] = useState(false);

  useEffect(() => {
    if (!visible) return;
    let stop = false;
    setErr(null);
    getFrameWhy(frameId, status)
      .then((r) => { if (!stop) setItems(r.items); })
      .catch((e) => { if (!stop) { setItems([]); setErr(humanizeError(e)); } });
    return () => { stop = true; };
  }, [frameId, status, visible, refreshKey]);

  // Several checks can flag one object; one tag per object, its card lists them all.
  const byObject = useMemo(() => {
    const m = new Map<string, ErrorWhy[]>();
    for (const it of items) m.set(it.object_id, [...(m.get(it.object_id) ?? []), it]);
    return m;
  }, [items]);

  useEffect(() => {
    if (selectedObjectId && byObject.has(selectedObjectId)) setOpen(selectedObjectId);
  }, [selectedObjectId, byObject]);

  if (!visible) return null;
  const openItems = open ? byObject.get(open) ?? [] : [];
  const evidence = openItems.flatMap((w) => w.evidence.filter((e) => e.role !== "subject"));
  const anchor = openItems[0] ? boxToScreen(openItems[0].subject.bbox, viewport) : null;

  return (
    <div className="absolute inset-0 pointer-events-none z-[5]" aria-label="Likely label errors">
      <svg className="absolute inset-0 w-full h-full overflow-visible" aria-hidden>
        {[...byObject.entries()].map(([oid, ws]) => {
          const r = boxToScreen(ws[0].subject.bbox, viewport);
          const on = oid === open;
          return (
            <rect key={oid} x={r.x} y={r.y} width={r.w} height={r.h} fill={on ? "rgb(var(--c-block) / 0.08)" : "none"}
              stroke="rgb(var(--c-block))" strokeWidth={on ? 2.5 : 1.5} strokeDasharray="6 4" />
          );
        })}
        {evidence.map((e, i) => {
          const r = boxToScreen(e.bbox, viewport);
          return (
            <g key={`ev${i}`}>
              <rect x={r.x} y={r.y} width={r.w} height={r.h} fill="none" stroke="rgb(var(--c-warn))" strokeWidth={2}
                strokeDasharray="3 3" />
              <text x={r.x + 3} y={Math.max(11, r.y - 4)} fontSize={11} fill="rgb(var(--c-warn))">{e.label}</text>
            </g>
          );
        })}
      </svg>

      {[...byObject.entries()].map(([oid, ws]) => {
        const r = boxToScreen(ws[0].subject.bbox, viewport);
        return (
          <button key={oid} type="button"
            onClick={() => { setOpen(open === oid ? null : oid); onSelectObject?.(oid); }}
            className={`pointer-events-auto absolute flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded-full
              text-[11px] font-semibold leading-none ${oid === open ? "bg-block text-bg" : "bg-block/85 text-bg hover:bg-block"}`}
            style={{ left: r.x + r.w - 9, top: r.y - 9 }}
            aria-label={`Likely error on this ${ws[0].subject.class_name}: ${ws[0].detector}`}
            data-tip={ws.map((w) => w.detector).join(" · ")}>
            !{ws.length > 1 ? ws.length : ""}
          </button>
        );
      })}

      {/* The count, so a frame whose flags are off-screen at this zoom still says it has some. */}
      {(items.length > 0 || err) && (
        <div className="pointer-events-auto absolute right-2 top-2 max-w-[300px]">
          <button type="button" onClick={() => setListOpen((v) => !v)}
            className="chip chip-on text-[11.5px]" aria-expanded={listOpen}>
            {err ? "Error checks unavailable" : `${byObject.size} likely error${byObject.size === 1 ? "" : "s"}`}
          </button>
          {listOpen && !err && (
            <ul className="mt-1 panel p-1.5 space-y-0.5 max-h-[240px] overflow-auto">
              {[...byObject.entries()].map(([oid, ws]) => (
                <li key={oid}>
                  <button type="button" className="w-full text-left text-[12px] px-1.5 py-1 rounded hover:bg-head"
                    onClick={() => { setOpen(oid); onSelectObject?.(oid); }}>
                    <span className="text-ink">{ws[0].subject.class_name.replace(/_/g, " ")}</span>
                    <span className="text-ink-3"> · {ws.map((w) => w.detector).join(", ")}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {listOpen && err && <div className="mt-1 panel p-2 text-[12px] text-block">{err}</div>}
        </div>
      )}

      {open && anchor && openItems.length > 0 && (
        <div className="pointer-events-auto absolute panel shadow-lg p-3 space-y-3 max-h-[70vh] overflow-auto"
          style={{ width: CARD_W, left: Math.max(8, anchor.x + anchor.w + 12), top: Math.max(8, anchor.y) }}
          role="dialog" aria-label="Why this label was flagged">
          <div className="flex items-center gap-2">
            <h3 className="section-title">Why this was flagged</h3>
            <button type="button" className="ml-auto text-ink-3 hover:text-ink text-[13px] px-1" aria-label="Close"
              onClick={() => setOpen(null)}>×</button>
          </div>
          {openItems.map((w, i) => (
            <div key={w.candidate_id} className={i > 0 ? "pt-3 border-t hairline" : ""}>
              {/* The frame itself is right under the card, so the evidence is drawn on it instead. */}
              <ErrorWhyPanel why={w} showFrame={false} showVlm={i === openItems.length - 1} />
              <Link href={`/quality/errors?kind=${encodeURIComponent(w.kind)}`} className="link text-[12px]">
                Rule on it in the error queue
              </Link>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
