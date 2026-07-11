// The standalone title/action band, Blender-style: a panel-toned strip with the title + subtitle + meta on
// the left and the primary action on the right. Used inside PageShell and directly by editor-style routes that
// use BackButton instead of TopNav, so they get the identical title bar.

export default function PageHeaderBar({ title, subtitle, meta, right, primaryAction, className = "" }: {
  title: string;
  subtitle?: string;
  meta?: React.ReactNode;
  right?: React.ReactNode;
  primaryAction?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`flex items-center gap-3 px-4 h-11 bg-panel border-b border-line shrink-0 ${className}`}>
      <h1 className="font-display font-semibold text-sm text-ink shrink-0">{title}</h1>
      {subtitle && <span className="text-[12px] text-ink-3 truncate">{subtitle}</span>}
      {meta && <div className="text-[12px] text-ink-3 flex items-center gap-2 min-w-0">{meta}</div>}
      <div className="ml-auto flex items-center gap-2 text-[12px] shrink-0">
        {right}
        {primaryAction}
      </div>
    </div>
  );
}
