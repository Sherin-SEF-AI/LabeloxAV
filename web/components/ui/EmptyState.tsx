"use client";

import Icon from "@/components/shell/Icon";

// What a page shows when there is genuinely nothing in it yet.
//
// Empty pages used to be one line of 11px grey text in a full-width panel ("No campaigns. One picks a class
// ..."), which reads as a page that failed to render, and several told the reader to run a `make` target in
// a terminal they may not have. An empty state is usually the first thing a new team sees on a page, so it
// says three things: what would be here, why it is empty, and the one thing to do about it.

export default function EmptyState({ icon = "info", title, children, action, compact = false, className = "" }: {
  icon?: string;
  title: string;
  children?: React.ReactNode;   // one or two sentences: why it is empty, what would fill it
  action?: React.ReactNode;     // the one thing to do next (a button or link)
  compact?: boolean;            // inside a panel rather than filling a page
  className?: string;
}) {
  return (
    <div role="status"
      className={`flex flex-col items-center justify-center text-center ${compact ? "py-6 px-4" : "py-14 px-6"} ${className}`}>
      <span className={`flex items-center justify-center rounded-full bg-head border border-line text-ink-3 ${compact ? "w-9 h-9 mb-2.5" : "w-12 h-12 mb-4"}`}
        aria-hidden>
        <Icon name={icon} size={compact ? 16 : 20} />
      </span>
      <div className={`font-medium text-ink ${compact ? "text-[13px]" : "text-[14.5px]"}`}>{title}</div>
      {children && (
        <div className={`mt-1.5 max-w-md text-ink-3 leading-relaxed ${compact ? "text-[12px]" : "text-[12.5px]"}`}>{children}</div>
      )}
      {action && <div className="mt-4 flex items-center gap-2">{action}</div>}
    </div>
  );
}
