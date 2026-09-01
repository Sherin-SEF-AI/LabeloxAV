"use client";

// What the guideline check said about this frame.
//
// The rules have existed as a corpus sweep with one caller and no way to reach a single frame, and no
// machine rule had ever created an Issue, so the panel below this one has been blind to every detector in
// the repo. This is the surface that changes that.
//
// Three things it must show and one it must not.
//
// It shows findings, ordered by severity, each clickable to select the object it is about. It shows rules
// that fired on most of the frame as a single counted line rather than as fifty rows: a rule objecting to
// every object is reporting one fact, and repeating it fifty times buries the four findings that matter.
// And it shows DORMANT rules, with the reason, because a rule that could not run because nobody is
// collecting the data it needs is a fact about the corpus, and hiding it would make the check look like it
// had inspected something it never looked at.
//
// What it must not do is open issue threads by itself. The editor autosaves every 700ms and the check runs
// after each save; a panel that opened a thread per finding would be unusable within a minute. Opening one
// is a decision, and it is a button.

import { useState } from "react";

import { usePanelFlag } from "./panelPrefs";

export type LintFinding = {
  object_id: string;
  rule: string;
  label: string;
  severity: string;
  score: number;
  reason: string;
};

export type LintResult = {
  findings: LintFinding[];
  systemic: Record<string, number>;
  dormant: { rule: string; label: string; reason: string }[];
  n_objects?: number;
  issues?: { opened: number };
};

const PREF_LINT_OPEN = "lbx.panel.lint";

const SEVERITY_CLASS: Record<string, string> = {
  high: "text-block",
  medium: "text-warn",
  low: "text-ink-3",
};

export default function LintCard({ lint, selectedIds, onSelect, onOpenIssues }: {
  lint: LintResult | null;
  selectedIds: string[];
  onSelect: (objectId: string) => void;
  /** Opens the findings as issue threads. Absent while one is in flight. */
  onOpenIssues?: () => void;
}) {
  const [open, setOpen] = usePanelFlag(PREF_LINT_OPEN, true);
  const [showDormant, setShowDormant] = useState(false);

  const findings = lint?.findings ?? [];
  const systemic = Object.entries(lint?.systemic ?? {});
  const dormant = lint?.dormant ?? [];
  const high = findings.filter((f) => f.severity === "high").length;

  return (
    <section className="border-b hairline">
      <button onClick={() => setOpen(!open)} aria-expanded={open}
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-line/30 group transition-colors">
        <span aria-hidden
          className={`text-ink-3 group-hover:text-ink w-3 text-center text-[9px] leading-none transition-transform duration-200 ${open ? "rotate-90" : ""}`}>▸</span>
        <span className="font-mono text-[10px] uppercase tracking-wide text-ink-2">checks</span>
        {!lint ? (
          <span className="font-mono text-[10px] text-ink-3">running...</span>
        ) : findings.length ? (
          <span className={`font-mono text-[10px] rounded px-1.5 ${high ? "bg-block/20 text-block" : "bg-warn/20 text-warn"}`}>
            {findings.length}
          </span>
        ) : systemic.length ? (
          // Not clean. A rule that objected to every object on the frame found something; it just found
          // one thing rather than a hundred, and the count below says what.
          <span className="font-mono text-[10px] text-warn">frame-wide</span>
        ) : (
          <span className="font-mono text-[10px] text-pass">clean</span>
        )}
        {systemic.length > 0 && (
          <span className="ml-auto font-mono text-[10px] text-ink-3"
            title="a rule that objected to most of the frame, counted once instead of queued per object">
            {systemic.length} frame-wide
          </span>
        )}
      </button>

      {open && (
        <div className="reveal">
          {lint && !findings.length && !systemic.length && (
            <div className="px-3 py-2 font-mono text-[11px] text-ink-3">
              Nothing breaks a guideline on this frame.
            </div>
          )}

          {findings.length > 0 && (
            <div className="max-h-[22vh] overflow-y-auto">
              {findings.map((f, i) => (
                <button key={`${f.object_id}-${f.rule}-${i}`} onClick={() => onSelect(f.object_id)}
                  title={`${f.reason} (${f.rule})`}
                  className={`w-full text-left flex items-start gap-1.5 pl-3 pr-1.5 py-1 font-mono text-[11px] border-l-2 ${
                    selectedIds.includes(f.object_id)
                      ? "bg-line text-ink border-accent"
                      : "text-ink-3 hover:text-ink-2 hover:bg-line/25 border-transparent"}`}>
                  <span className={`shrink-0 ${SEVERITY_CLASS[f.severity] ?? "text-ink-3"}`}>●</span>
                  <span className="flex-1 min-w-0">
                    <span className="block truncate text-ink-2">{f.label}</span>
                    <span className="block truncate text-ink-3 text-[10px]">{f.reason}</span>
                  </span>
                </button>
              ))}
            </div>
          )}

          {/* Counted, not listed. Same guard the reanalysis pass applies at 80%: a rule that fires on the
              whole frame is describing the pipeline, and one line saying so is worth more than fifty rows. */}
          {systemic.map(([rule, n]) => (
            <div key={rule} className="px-3 py-1.5 font-mono text-[10px] text-ink-3 border-t hairline">
              <span className="text-warn">{rule.replace(/_/g, " ")}</span> objected to {n} of{" "}
              {lint?.n_objects ?? n} objects on this frame, so it is describing the pipeline rather than
              any one box. Counted, not queued.
            </div>
          ))}

          {dormant.length > 0 && (
            <div className="border-t hairline">
              <button onClick={() => setShowDormant((v) => !v)}
                className="w-full text-left px-3 py-1 font-mono text-[10px] text-ink-3 hover:text-ink-2">
                {showDormant ? "−" : "+"} {dormant.length} check{dormant.length === 1 ? "" : "s"} could not run
              </button>
              {showDormant && dormant.map((d) => (
                <div key={d.rule} className="px-3 pb-1.5 font-mono text-[10px] text-ink-3">
                  <span className="text-ink-2">{d.label}</span>: {d.reason}
                </div>
              ))}
            </div>
          )}

          {findings.length > 0 && onOpenIssues && (
            <div className="px-3 py-1.5 border-t hairline">
              <button onClick={onOpenIssues}
                title="open these as issue threads in the panel below, so they survive leaving this frame"
                className="font-mono text-[10px] border border-line rounded px-2 py-1 text-ink-2 hover:border-accent hover:text-accent">
                open {findings.length} as issue{findings.length === 1 ? "" : "s"}
              </button>
              {lint?.issues && (
                <span className="ml-2 font-mono text-[10px] text-pass">
                  {lint.issues.opened} opened
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
