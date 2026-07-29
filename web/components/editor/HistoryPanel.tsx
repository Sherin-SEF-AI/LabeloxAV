"use client";

import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import type { Checkpoint, HistoryEntry } from "@/lib/types";

// Two kinds of going back, which people conflate until they lose an hour to the difference.
//
// The step list is this tab's undo stack made visible. It is capped, it is local, and it dies with a
// refresh, which is fine for "put that box back" and useless for anything longer. Making it visible turns
// undo from a key you press hopefully into a place you can look.
//
// Saves are the other kind: named, on the server, and permanent. They are what an annotator reaches for
// after an hour on a dense junction when they want to try a different reading of it and keep the first.
// Restoring one takes a save of what it replaced, so the destructive direction is also recoverable.

type Props = {
  frameId: string;
  past: HistoryEntry[];
  future: HistoryEntry[];
  onJump: (at: number) => void;
  onRestored: () => void;
  flash: (msg: string) => void;
};

function ago(iso: string | null): string {
  if (!iso) return "";
  const secs = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (secs < 60) return `${Math.round(secs)}s ago`;
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.round(secs / 3600)}h ago`;
  return `${Math.round(secs / 86400)}d ago`;
}

export default function HistoryPanel({ frameId, past, future, onJump, onRestored, flash }: Props) {
  const [tab, setTab] = useState<"steps" | "saves">("steps");
  const [saves, setSaves] = useState<Checkpoint[]>([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [showAuto, setShowAuto] = useState(false);

  const load = useCallback(async () => {
    try {
      const r = await api.listCheckpoints(frameId, true);
      setSaves(r.checkpoints);
    } catch {
      // A frame with no saves is the ordinary case, not an error worth interrupting anyone about.
      setSaves([]);
    }
  }, [frameId]);

  useEffect(() => {
    if (tab === "saves") void load();
  }, [tab, load]);

  async function saveAs() {
    const trimmed = name.trim();
    if (!trimmed) {
      flash("give the save a name, or you will not find it again");
      return;
    }
    setBusy(true);
    try {
      const c = await api.saveCheckpoint(frameId, trimmed);
      setName("");
      await load();
      flash(`saved "${c.name}" with ${c.object_count} objects`);
    } catch (e) {
      flash("save failed: " + humanizeError(e));
    } finally {
      setBusy(false);
    }
  }

  async function restore(c: Checkpoint) {
    setBusy(true);
    try {
      const r = await api.restoreCheckpoint(c.checkpoint_id);
      await load();
      onRestored();
      const stale = r.dropped_stale_tracks
        ? `, ${r.dropped_stale_tracks} stale track link dropped` : "";
      flash(`restored "${c.name}": ${r.updated} updated, ${r.created} back, ${r.deleted} removed${stale}`);
    } catch (e) {
      flash("restore failed: " + humanizeError(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove(c: Checkpoint) {
    setBusy(true);
    try {
      await api.deleteCheckpoint(c.checkpoint_id);
      await load();
    } catch (e) {
      flash("delete failed: " + humanizeError(e));
    } finally {
      setBusy(false);
    }
  }

  // Newest first, and the future (redo) above the past, so the list reads top to bottom as most recent
  // first in both directions rather than flipping halfway down.
  const steps = [
    ...future.map((h) => ({ ...h, ahead: true })),
    ...[...past].reverse().map((h) => ({ ...h, ahead: false })),
  ];
  const visibleSaves = showAuto ? saves : saves.filter((c) => !c.auto);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-1">
        {(["steps", "saves"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-2 py-0.5 font-mono text-[10px] uppercase border ${
              tab === t ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"
            }`}
          >
            {t === "steps" ? `steps (${past.length})` : `saves (${saves.length})`}
          </button>
        ))}
      </div>

      {tab === "steps" && (
        <>
          {steps.length === 0 && (
            <p className="font-mono text-[10px] text-ink-3">
              Nothing edited yet. Steps here are this tab&apos;s undo stack: they are capped and do not
              survive a refresh. For anything you want to keep, use a save.
            </p>
          )}
          <div className="max-h-[320px] overflow-auto">
            {steps.map((h) => (
              <button
                key={h.at}
                onClick={() => onJump(h.at)}
                title={h.ahead ? "redo forward to this point" : "go back to this point"}
                className={`w-full text-left px-2 py-0.5 font-mono text-[11px] flex items-center gap-2
                  hover:bg-line ${h.ahead ? "text-ink-3 italic" : "text-ink-2"}`}
              >
                <span className="text-ink-3 w-3">{h.ahead ? "↓" : "↑"}</span>
                <span className="flex-1 truncate">{h.label}</span>
                <span className="text-ink-3">{h.objects.length}</span>
              </button>
            ))}
          </div>
          {steps.length > 0 && (
            <p className="font-mono text-[10px] text-ink-3">
              Arrows show direction: up is undone, down is redone. The number is how many objects the frame
              had at that point.
            </p>
          )}
        </>
      )}

      {tab === "saves" && (
        <>
          <div className="flex gap-1">
            <input
              className="input font-mono text-[11px] flex-1"
              placeholder="name this state"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void saveAs();
                e.stopPropagation();
              }}
            />
            <button
              onClick={() => void saveAs()}
              disabled={busy}
              className="border border-line text-ink-2 px-2 py-1 font-mono text-[11px] hover:border-accent"
            >
              save as
            </button>
          </div>

          {saves.some((c) => c.auto) && (
            <label className="flex items-center gap-1.5 font-mono text-[10px] text-ink-3">
              <input type="checkbox" checked={showAuto} onChange={(e) => setShowAuto(e.target.checked)} />
              show the {saves.filter((c) => c.auto).length} automatic saves a restore left behind
            </label>
          )}

          {visibleSaves.length === 0 && (
            <p className="font-mono text-[10px] text-ink-3">
              No saves on this frame. A save is named, lives on the server, and survives a refresh, which
              undo does not.
            </p>
          )}

          <div className="max-h-[300px] overflow-auto">
            {visibleSaves.map((c) => (
              <div key={c.checkpoint_id} className="px-2 py-1 border-b border-line">
                <div className="flex items-center gap-2">
                  <span className={`flex-1 truncate font-mono text-[11px] ${c.auto ? "text-ink-3 italic" : "text-ink-2"}`}>
                    {c.name}
                  </span>
                  <button
                    onClick={() => void restore(c)}
                    disabled={busy}
                    className="border border-line text-ink-3 px-1.5 hover:border-accent font-mono text-[10px]"
                  >
                    restore
                  </button>
                  <button
                    onClick={() => void remove(c)}
                    disabled={busy}
                    className="text-ink-3 hover:text-block font-mono text-[10px] px-1"
                    title="delete this save"
                  >
                    &times;
                  </button>
                </div>
                <div className="font-mono text-[10px] text-ink-3">
                  {c.object_count} objects &middot; {ago(c.created_at)}
                  {c.created_by ? ` · ${c.created_by}` : ""}
                </div>
              </div>
            ))}
          </div>

          <p className="font-mono text-[10px] text-ink-3">
            Restoring takes a save of what it replaces first, so a restore you did not mean is itself one
            click from being undone.
          </p>
        </>
      )}
    </div>
  );
}
