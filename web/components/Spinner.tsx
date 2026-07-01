// Shared loading primitives in the Operational Materialism language (matte, amber accent, mono).
// Used wherever a page fetches data so a slow endpoint shows motion instead of a blank panel.

export function Spinner({ label, className = "" }: { label?: string; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 font-mono text-[11px] text-ink-3 ${className}`}>
      <span
        className="inline-block w-3.5 h-3.5 rounded-full border-2 border-line border-t-accent animate-spin"
        aria-hidden
      />
      {label ? <span>{label}</span> : null}
      <span className="sr-only">loading</span>
    </span>
  );
}

// A full-panel centered loading state (for a page whose main content is still loading).
export function LoadingPanel({ label = "loading" }: { label?: string }) {
  return (
    <div className="panel flex items-center justify-center py-16" role="status" aria-live="polite">
      <Spinner label={label} />
    </div>
  );
}

// Pulsing skeleton rows that mirror a table grid, so the layout does not jump when data arrives.
// `cols` is the same grid-cols-[...] class the real rows use.
export function SkeletonRows({ rows = 6, cols }: { rows?: number; cols: string }) {
  return (
    <div aria-hidden>
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className={`grid ${cols} gap-2 px-3 py-2 border-b hairline items-center`}>
          {Array.from({ length: Math.max(1, cols.split("_").length) }).map((__, c) => (
            <span key={c} className="h-3 rounded bg-line/50 animate-pulse"
              style={{ animationDelay: `${(r * 3 + c) * 60}ms`, width: c === 1 ? "70%" : "90%" }} />
          ))}
        </div>
      ))}
    </div>
  );
}
