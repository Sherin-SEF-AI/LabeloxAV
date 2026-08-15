"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  BackgroundPanel, CanvasPanel, GpuPanel, HostPanel, ProcessPanel, RunningNowPanel, useAgentRuns,
} from "@/components/console/Panels";
import Icon from "@/components/shell/Icon";
import { headline, uptime, workInFlight } from "@/lib/console";
import {
  type ConsoleSectionId, CONSOLE_SECTIONS, filterSections, groupSections, resolveSelection,
} from "@/lib/consoleSections";
import { useJobStream, useSystemStream } from "@/lib/useEventStream";

// The console as a dialog over the work, rather than a page instead of it.
//
// "Is anything happening, and is it my machine or my job" is a question people ask in the middle of
// something else. Answering it by navigating away means losing the frame they were annotating, the scroll
// position in a review queue, and the thing that raised the question in the first place. So it opens over
// the top and closes back onto exactly what was there.
//
// A section list with a search box is the shape that keeps working as this grows: every panel that would
// otherwise fight for room on one page is a row, and typing is the way in once there are more rows than fit
// the eye.
//
// The streams open only while it is open. A console that held two server-sent-event connections per tab for
// a dialog nobody has opened would cost more than the question it answers.

const EVENT = "lbx:console";

/** Open the console from anywhere: a chip, a menu item, a shortcut. */
export function openConsole(section?: ConsoleSectionId) {
  window.dispatchEvent(new CustomEvent(EVENT, { detail: section }));
}

export default function ConsoleModal() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [section, setSection] = useState<ConsoleSectionId>("overview");
  const searchRef = useRef<HTMLInputElement>(null);
  const closerRef = useRef<HTMLElement | null>(null);

  const { data: jobs, connected: jobsLive } = useJobStream();
  const { data: sys, connected: sysLive } = useSystemStream(open);
  const runs = useAgentRuns(open);

  const close = useCallback(() => {
    setOpen(false);
    setQuery("");
    // Focus goes back where it came from. A dialog that drops focus on the body leaves a keyboard user at
    // the top of the document, which on a frame editor means losing the canvas.
    closerRef.current?.focus?.();
  }, []);

  useEffect(() => {
    const onOpen = (e: Event) => {
      const want = (e as CustomEvent<ConsoleSectionId | undefined>).detail;
      closerRef.current = document.activeElement as HTMLElement | null;
      if (want) setSection(want);
      setOpen(true);
    };
    window.addEventListener(EVENT, onOpen);
    return () => window.removeEventListener(EVENT, onOpen);
  }, []);

  // Escape closes. Typed inside the search box it clears the query first, because a search that closed the
  // whole dialog on Escape is a search you cannot back out of.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (query) { setQuery(""); return; }
      close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, query, close]);

  useEffect(() => {
    if (open) searchRef.current?.focus();
  }, [open]);

  const matches = useMemo(() => filterSections(query), [query]);
  const shown = useMemo(() => resolveSelection(query, section), [query, section]);
  const groups = useMemo(() => groupSections(matches), [matches]);

  if (!open) return null;

  const { waitingTotal } = workInFlight(jobs);
  const live = jobsLive && sysLive;
  const current = CONSOLE_SECTIONS.find((s) => s.id === shown);

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center p-4">
      {/* The backdrop is a button so a click outside closes, which is the gesture people try first. */}
      <button aria-label="close console" onClick={close}
        className="absolute inset-0 bg-black/45 backdrop-blur-[2px] reveal cursor-default" />

      <div role="dialog" aria-modal="true" aria-label="Console"
        className="reveal-pop relative w-full max-w-[980px] h-[min(680px,88vh)] flex
                   bg-bg-2 border border-line rounded-lg shadow-2xl overflow-hidden">

        {/* SIDEBAR */}
        <div className="w-[232px] shrink-0 flex flex-col border-r hairline bg-bg/40">
          <div className="p-2.5">
            <div className="flex items-center gap-2 px-2 h-8 bg-bg border border-line rounded">
              <span className="text-ink-3 flex"><Icon name="search" size={13} /></span>
              <input ref={searchRef} value={query} onChange={(e) => setQuery(e.target.value)}
                placeholder="Search" aria-label="search console sections"
                className="flex-1 bg-transparent outline-none font-body text-[12px] text-ink placeholder:text-ink-3" />
            </div>
          </div>

          <div className="flex-1 overflow-auto px-2 pb-2 space-y-3">
            {groups.map((g) => (
              <div key={g.group}>
                <div className="px-2 py-1 font-display font-semibold text-[10px] uppercase tracking-wider text-ink-3">
                  {g.group}
                </div>
                {g.items.map((s) => (
                  <button key={s.id} onClick={() => setSection(s.id)}
                    className={`flex items-center gap-2.5 w-full px-2 py-1.5 rounded text-left
                                ${s.id === shown ? "bg-line/60 text-ink" : "text-ink-2 hover:bg-line/30"}`}>
                    <span className="flex text-ink-3"><Icon name={s.icon} size={14} /></span>
                    <span className="font-body text-[12.5px]">{s.label}</span>
                  </button>
                ))}
              </div>
            ))}
            {!matches.length && (
              <div className="px-2 py-6 text-center font-mono text-[11px] text-ink-3">
                nothing here matches “{query}”.
              </div>
            )}
          </div>

          {/* A console whose numbers freeze looks exactly like a quiet machine, so it says which it is. */}
          <div className="px-3 py-2 border-t hairline flex items-center gap-2 font-mono text-[10px] text-ink-3">
            <span className={`w-1.5 h-1.5 rounded-full ${live ? "bg-pass" : "bg-ink-3"}`} />
            <span>{live ? "live" : "not connected"}</span>
            {sys && <span className="ml-auto">up {uptime(sys.process.uptime_s)}</span>}
          </div>
        </div>

        {/* CONTENT */}
        <div className="flex-1 min-w-0 flex flex-col">
          <div className="flex items-start gap-3 px-6 pt-5 pb-3">
            <div className="min-w-0">
              <h2 className="font-display font-semibold text-[15px] text-ink">{current?.label ?? "Console"}</h2>
              <p className="font-mono text-[11px] text-ink-3 mt-0.5 truncate">{headline(jobs, sys)}</p>
            </div>
            <button onClick={close} aria-label="close"
              className="ml-auto shrink-0 w-7 h-7 flex items-center justify-center rounded
                         text-ink-3 hover:text-ink hover:bg-line/40">
              <Icon name="close" size={15} />
            </button>
          </div>

          <div className="flex-1 overflow-auto px-6 pb-6">
            {shown === "overview" && (
              <div className="space-y-5">
                <Section title="Running now"><RunningNowPanel jobs={jobs} /></Section>
                <Section title="GPU"><GpuPanel sys={sys} /></Section>
                <Section title="Host"><HostPanel sys={sys} /></Section>
              </div>
            )}
            {shown === "jobs" && <RunningNowPanel jobs={jobs} />}
            {shown === "background" && <BackgroundPanel runs={runs} />}
            {shown === "canvas" && <CanvasPanel />}
            {shown === "gpu" && <GpuPanel sys={sys} />}
            {shown === "machine" && <HostPanel sys={sys} />}
            {shown === "process" && <ProcessPanel sys={sys} waitingTotal={waitingTotal} />}
            {shown === null && (
              <div className="py-16 text-center font-mono text-[11px] text-ink-3">
                no section matches “{query}”.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="font-display font-semibold text-[11px] uppercase tracking-wider text-ink-3 mb-2">
        {title}
      </div>
      {children}
    </div>
  );
}
