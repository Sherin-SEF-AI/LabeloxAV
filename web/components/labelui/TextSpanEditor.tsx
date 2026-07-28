"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import type { AnnotationRow, LabelConfig } from "@/lib/types";

// Text / NER: select a range in the body and it becomes a span annotation; link two spans and it becomes a
// relation.
//
// Spans alone answer "what is this" and not "how do these connect", and the second is most of what NER is
// annotated for. A driver, a vehicle and a location can all be tagged in one sentence with no way to record
// that the driver was IN that vehicle AT that location, so the labels describe a bag of entities rather than
// an event. The `relation` kind was already defined and validated in the label config and had no way to be
// created.
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
  onCreateRelation?: (fromId: string, toId: string) => void;
};

function colorFor(config: LabelConfig, label: string | null): string {
  const l = (config.labels ?? []).find((x) => x.name === label);
  return l?.color || "#4c8dff";
}

export default function TextSpanEditor({
  text, annotations, config, activeLabel, onCreate, onSelect, selectedId, onCreateRelation,
}: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  // Relation drafting: click the source span, then the target. Two clicks rather than a drag because a drag
  // across text is already the gesture that creates a span, and one gesture cannot mean both.
  const [linkFrom, setLinkFrom] = useState<string | null>(null);

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

  const relations = useMemo(
    () => annotations.filter((a) => a.kind === "relation"), [annotations]);
  const spanById = useMemo(
    () => new Map(spans.map((sp) => [sp.id, sp])), [spans]);

  const clickSpan = (id: string) => {
    if (!linkFrom) { onSelect(id === selectedId ? null : id); return; }
    if (id === linkFrom) { setLinkFrom(null); return; }   // clicking the source again cancels
    onCreateRelation?.(linkFrom, id);
    setLinkFrom(null);
  };

  return (
    <div className="p-4">
      <div className="flex items-center gap-3 font-mono text-[11px] mb-2">
        {!activeLabel && <span className="text-ink-3">pick a label first, then select text to tag it</span>}
        {onCreateRelation && (
          <button
            onClick={() => setLinkFrom(linkFrom ? null : (selectedId ?? null))}
            disabled={!linkFrom && !selectedId}
            className={`ml-auto border px-1.5 py-0.5 ${
              linkFrom ? "border-accent text-accent" : "border-line text-ink-3 hover:text-ink-2"}
              disabled:opacity-40`}>
            {linkFrom ? "click the target span (esc cancels)" : "link this span"}
          </button>
        )}
      </div>
      <div ref={ref} onMouseUp={handleMouseUp}
        className="font-sans text-[15px] leading-7 text-ink whitespace-pre-wrap select-text cursor-text">
        {fragments.map((f, i) =>
          f.span ? (
            <mark key={i} data-start={f.start}
              onClick={() => clickSpan(f.span!.id)}
              title={`${f.span.label ?? "span"} [${f.span.start}, ${f.span.end})`}
              style={{
                backgroundColor: `${colorFor(config, f.span.label)}33`,
                borderBottom: `2px solid ${colorFor(config, f.span.label)}`,
                outline: f.span.id === selectedId || f.span.id === linkFrom
                  ? `1px solid ${colorFor(config, f.span.label)}` : undefined,
              }}
              className="rounded-sm px-[1px] cursor-pointer text-ink">
              {f.text}
            </mark>
          ) : (
            <span key={i} data-start={f.start}>{f.text}</span>
          ),
        )}
      </div>

      {relations.length > 0 && (
        // Listed rather than drawn as arcs over the text. Arcs are the conventional rendering and become
        // unreadable the moment two of them cross, which in any real sentence is immediately.
        <div className="mt-3 border-t hairline pt-2">
          <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">
            relations ({relations.length})
          </div>
          <ul className="space-y-0.5">
            {relations.map((r) => {
              const from = spanById.get(String(r.payload.from_annotation_id));
              const to = spanById.get(String(r.payload.to_annotation_id));
              return (
                <li key={r.annotation_id} className="font-mono text-[11px] text-ink-2">
                  <span className="text-ink">{from ? text.slice(from.start, from.end) : "?"}</span>
                  <span className="text-ink-3"> {r.label ?? "relates to"} </span>
                  <span className="text-ink">{to ? text.slice(to.start, to.end) : "?"}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
