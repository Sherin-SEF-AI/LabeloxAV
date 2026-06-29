// The standalone title/action band. One hairline row: title (display) + subtitle + meta on the left, the
// primary action and other controls on the right. Used inside PageShell, and directly by editor-style
// routes that intentionally use BackButton instead of TopNav (object detail, timeline, multicam, the lidar
// editors) so they get the identical title bar without a TopNav they do not want.

export default function PageHeaderBar({ title, subtitle, meta, right, primaryAction, className = "" }: {
  title: string;
  subtitle?: string;
  meta?: React.ReactNode;
  right?: React.ReactNode;
  primaryAction?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`flex items-center gap-3 px-4 h-11 border-b hairline shrink-0 ${className}`}>
      <h1 className="font-display font-bold text-sm text-ink shrink-0">{title}</h1>
      {subtitle && <span className="font-mono text-[11px] text-ink-3 truncate">{subtitle}</span>}
      {meta && <div className="font-mono text-[11px] text-ink-3 flex items-center gap-2 min-w-0">{meta}</div>}
      <div className="ml-auto flex items-center gap-2 font-mono text-[11px] shrink-0">
        {right}
        {primaryAction}
      </div>
    </div>
  );
}
