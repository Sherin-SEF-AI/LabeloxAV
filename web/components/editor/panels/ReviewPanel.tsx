"use client";

import { ConfBar, StateBadge } from "@/components/StateBadge";
import ScoreBar from "@/components/shell/ScoreBar";
import type { EdObject } from "@/components/editor/useEditor";
import { classColor } from "@/lib/colors";
import type { AlItem, ErrorCandidateRow } from "@/lib/types";

// The editor's right panel in review mode: accept or reject the selected object, the value queue, and the
// error candidates. Presentational: the frame editor owns the queues and passes the actions in.

export default function ReviewPanel({
  selected, selectedId, frameId, alItems, errItems, reviewLoaded,
  onReview, onPickItem, onConfirmError, onDismissError,
}: {
  selected: EdObject | null;
  selectedId: string | null;
  frameId: string;
  alItems: AlItem[];
  errItems: ErrorCandidateRow[];
  reviewLoaded: boolean;
  onReview: (verdict: "accept" | "reject") => void;
  onPickItem: (item: AlItem) => void;
  onConfirmError: (candidateId: string) => void;
  onDismissError: (candidateId: string) => void;
}) {
  return (
    <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2 text-[12px]">
      {selected ? (
        <div className="border-b hairline pb-2 space-y-1.5">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: classColor(selected.class_id) }} />
            <span className="truncate flex-1">{selected.class_name}</span>
            <StateBadge state={selected.state} />
          </div>
          <ConfBar conf={selected.conf} />
          <div className="flex gap-1">
            <button onClick={() => onReview("accept")} className="flex-1 border border-pass text-pass px-2 py-1 hover:bg-pass/10">accept (A)</button>
            <button onClick={() => onReview("reject")} className="flex-1 border border-block text-block px-2 py-1 hover:bg-block/10">reject (X)</button>
          </div>
        </div>
      ) : <div className="text-ink-3 border-b hairline pb-2">click an object on the canvas to accept or reject it.</div>}
      <div className="text-ink-3 uppercase text-[10px]">value queue ({alItems.length})</div>
      {alItems.slice(0, 60).map((it) => (
        <button key={it.object_id} onClick={() => onPickItem(it)}
          className={`block w-full text-left border-b hairline pb-1.5 ${it.object_id === selectedId ? "text-ink" : "text-ink-3 hover:text-ink-2"}`}>
          <div className="flex items-center gap-1.5">
            <span className="truncate flex-1">{it.class_name}</span>
            {it.frame_id !== frameId && <span className="text-info text-[10px]">other frame</span>}
            <span className="text-accent">{it.value.toFixed(3)}</span>
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-0.5">
            <ConfBar conf={it.conf} />
            <ScoreBar label="u" value={it.scores.uncertainty} showValue={false} />
            <ScoreBar label="d" value={it.scores.diversity} showValue={false} />
            <ScoreBar label="r" value={it.scores.rarity} showValue={false} />
            <ScoreBar label="e" value={it.scores.error_prone} showValue={false} tone="warn" />
          </div>
        </button>
      ))}
      {!alItems.length && <div className="text-ink-3 py-2 text-center">{reviewLoaded ? "value queue empty" : "loading value queue..."}</div>}
      {errItems.length > 0 && (
        <>
          <div className="text-ink-3 uppercase text-[10px] pt-1">error candidates ({errItems.length})</div>
          {errItems.slice(0, 40).map((ec) => (
            <div key={ec.candidate_id} className="border-b hairline pb-1.5 space-y-0.5">
              <div className="flex items-center gap-1.5">
                <span className="truncate flex-1">{ec.kind}</span>
                <span className="text-info">{ec.proposed_label?.class_name || "(review)"}</span>
              </div>
              <div className="flex items-center gap-2">
                <ScoreBar value={ec.score} tone="warn" />
                <button onClick={() => onConfirmError(ec.candidate_id)} className="text-block hover:text-accent">confirm</button>
                <button onClick={() => onDismissError(ec.candidate_id)} className="text-ink-3 hover:text-ink">dismiss</button>
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
