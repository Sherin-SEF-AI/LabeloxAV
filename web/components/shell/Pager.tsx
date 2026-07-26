"use client";

// A minimal, honest pager for list views. The app hard-capped lists at a fixed limit with no way to see the
// rest: with 2,070 sessions and hundreds of thousands of objects, most lists physically could not show past
// the first page. This says "showing X-Y of N" and moves the window, so the data is actually reachable.
//
// Purely presentational: the parent owns offset/limit (usePager below) and re-fetches when they change.

export default function Pager({ offset, limit, total, onOffset }: {
  offset: number; limit: number; total: number; onOffset: (n: number) => void;
}) {
  if (total <= limit && offset === 0) return null;  // one page: no chrome
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + limit, total);
  const atStart = offset <= 0;
  const atEnd = offset + limit >= total;

  return (
    <div className="flex items-center gap-3 font-mono text-[11px] text-ink-3">
      <span>{from.toLocaleString()}-{to.toLocaleString()} of {total.toLocaleString()}</span>
      <div className="flex items-center gap-1">
        <button onClick={() => onOffset(0)} disabled={atStart} aria-label="first page"
          className="border border-line px-1.5 py-0.5 disabled:opacity-30 hover:border-accent">«</button>
        <button onClick={() => onOffset(Math.max(0, offset - limit))} disabled={atStart} aria-label="previous page"
          className="border border-line px-1.5 py-0.5 disabled:opacity-30 hover:border-accent">‹ prev</button>
        <button onClick={() => onOffset(offset + limit)} disabled={atEnd} aria-label="next page"
          className="border border-line px-1.5 py-0.5 disabled:opacity-30 hover:border-accent">next ›</button>
      </div>
    </div>
  );
}
