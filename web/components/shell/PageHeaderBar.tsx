// The standalone title/action band: the title (and the sidebar entry it sits under, for a sub-page), subtitle
// and meta on the left, the primary action on the right. Used inside PageShell and directly by editor-style routes that
// use BackButton instead of TopNav, so they get the identical title bar.

export default function PageHeaderBar({ title, parent, subtitle, meta, right, primaryAction, className = "" }: {
  title: string;
  parent?: string;
  subtitle?: string;
  meta?: React.ReactNode;
  right?: React.ReactNode;
  primaryAction?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`flex items-center gap-3 px-4 sm:px-6 h-12 bg-bg border-b border-line shrink-0 ${className}`}>
      <h1 className="flex items-center gap-1.5 text-[15px] font-semibold tracking-tight text-ink shrink-0">
        {parent && <><span className="font-normal text-ink-3">{parent}</span><span className="font-normal text-ink-3" aria-hidden>/</span></>}
        {title}
      </h1>
      {subtitle && <span className="hidden sm:block text-[12.5px] text-ink-3 truncate">{subtitle}</span>}
      {meta && <div className="text-[12px] text-ink-3 flex items-center gap-2 min-w-0">{meta}</div>}
      <div className="ml-auto flex items-center gap-2 text-[12px] shrink-0">
        {right}
        {primaryAction}
      </div>
    </div>
  );
}
