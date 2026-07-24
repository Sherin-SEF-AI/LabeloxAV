"use client";

import { useCallback, useRef } from "react";
import type { AnnotationRow, LabelConfig } from "@/lib/types";

// Text / NER: select a range in the body and it becomes a span annotation.
//
// Offsets come from the DOM selection mapped back onto the ORIGINAL string, not from the rendered markup.
// Once earlier spans are highlighted the text is split across many elements, so a naive
// selection.anchorOffset is an offset into whichever fragment was clicked and would silently label the wrong
// characters. Each fragment therefore carries its absolute start index and the offset is rebuilt from it.

type Props = {
  text: string;
  annotations: AnnotationRow[];
  config: LabelConfig;
  activeLabel: string | null;
  onCreate: (start: number, end: number) => void;
  onSelect: (id: string | null) => void;
  selectedId: string | null;
};

function colorFor(config: LabelConfig, label: string | null): string {
  const l = (config.labels ?? []).find((x) => x.name === label);
  return l?.color || "#4c8dff";
}

export default function TextSpanEditor({
  text, annotations, config, activeLabel, onCreate, onSelect, selectedId,
}: Props) {
  const ref = useRef<HTMLDivElement | null>(null);

  const spans = annotations
    .filter((a) => a.kind === "span")
    .map((a) => ({
      id: a.annotation_id,
      start: Number(a.payload.start ?? 0),
      end: Number(a.payload.end ?? 0),
      label: a.label,
    }))
    .sort((a, b) => a.start - b.start);

  // Build non-overlapping fragments over the original string, each tagged with its absolute offset.
  const fragments: { text: string; start: number; span?: (typeof spans)[number] }[] = [];
  let cursor = 0;
  for (const s of spans) {
    if (s.start > cursor) fragments.push({ text: text.slice(cursor, s.start), start: cursor });
    if (s.end > s.start) fragments.push({ text: text.slice(s.start, s.end), start: s.start, span: s });
    cursor = Math.max(cursor, s.end);
  }
  if (cursor < text.length) fragments.push({ text: text.slice(cursor), start: cursor });

  const handleMouseUp = useCallback(() => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !ref.current) return;

    // Map both ends back to absolute offsets using the data-start on the owning fragment.
    const abs = (node: Node | null, offset: number): number | null => {
      let el: HTMLElement | null = node instanceof HTMLElement ? node : node?.parentElement ?? null;
      while (el && el !== ref.current && el.dataset.start === undefined) el = el.parentElement;
      if (!el || el.dataset.start === undefined) return null;
      return Number(el.dataset.start) + offset;
    };

    const a = abs(sel.anchorNode, sel.anchorOffset);
    const b = abs(sel.focusNode, sel.focusOffset);
    if (a == null || b == null) return;
    const start = Math.min(a, b);
    const end = Math.max(a, b);
    if (end > start) onCreate(start, end);
    sel.removeAllRanges();
  }, [onCreate]);

  return (
    <div className="p-4">
      {!activeLabel && (
        <div className="font-mono text-[11px] text-ink-3 mb-2">
          pick a label first, then select text to tag it
        </div>
      )}
      <div ref={ref} onMouseUp={handleMouseUp}
        className="font-sans text-[15px] leading-7 text-ink whitespace-pre-wrap select-text cursor-text">
        {fragments.map((f, i) =>
          f.span ? (
            <mark key={i} data-start={f.start}
              onClick={() => onSelect(f.span!.id === selectedId ? null : f.span!.id)}
              title={`${f.span.label ?? "span"} [${f.span.start}, ${f.span.end})`}
              style={{
                backgroundColor: `${colorFor(config, f.span.label)}33`,
                borderBottom: `2px solid ${colorFor(config, f.span.label)}`,
                outline: f.span.id === selectedId ? `1px solid ${colorFor(config, f.span.label)}` : undefined,
              }}
              className="rounded-sm px-[1px] cursor-pointer text-ink">
              {f.text}
            </mark>
          ) : (
            <span key={i} data-start={f.start}>{f.text}</span>
          ),
        )}
      </div>
    </div>
  );
}
