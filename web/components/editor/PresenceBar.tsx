"use client";

// Who else is on this frame, and what that means for the person looking at it.
//
// Two pieces, both dumb: they render what lib/editor/presence.ts concluded and decide nothing themselves.
// The avatars sit in the editor's top bar. The banner is a full-width strip under it, because "you are
// not the one editing this" is not a detail to be found in a corner after the work has been done.

import type { PresenceBanner } from "@/lib/editor/presence";
import { initialsOf } from "@/lib/editor/presence";
import type { PresenceUser } from "@/lib/presenceApi";
import Icon from "@/components/shell/Icon";

/** How many avatars before the rest collapse into a count. The top bar is one row and does not wrap. */
const MAX_AVATARS = 4;

export function PresenceAvatars({ others, degraded }: { others: PresenceUser[]; degraded: boolean }) {
  if (degraded) {
    // Said rather than left blank: an empty avatar row and "presence is not working" look the same, and
    // only one of them means nobody else is here.
    return (
      <span
        title={"Live presence is unavailable, so this cannot show who else has the frame open or stop two "
               + "people editing it at once. Editing still works; a conflicting save is still refused."}
        className="text-[10.5px] font-medium uppercase tracking-[0.06em] px-1.5 py-0.5 border border-line text-ink-3 whitespace-nowrap">
        presence off
      </span>
    );
  }
  if (!others.length) return null;
  const shown = others.slice(0, MAX_AVATARS);
  const more = others.length - shown.length;
  const names = others.map((p) => p.name).join(", ");
  return (
    <ul className="flex items-center -space-x-1.5 shrink-0" aria-label={`Also on this frame: ${names}`}>
      {shown.map((p) => (
        <li key={p.user_id}
          title={`${p.name} (${p.role})${p.editing ? ", editing this frame" : ", viewing"}`}
          className={`flex items-center justify-center w-[24px] h-[24px] rounded-full border bg-panel
                      text-[10px] font-semibold select-none
                      ${p.editing ? "border-accent text-accent" : "border-line text-ink-2"}`}>
          <span aria-hidden="true">{initialsOf(p.name)}</span>
          <span className="sr-only">{p.name}{p.editing ? " (editing)" : ""}</span>
        </li>
      ))}
      {more > 0 && (
        <li title={others.slice(MAX_AVATARS).map((p) => p.name).join(", ")}
          className="flex items-center justify-center h-[24px] px-1.5 rounded-full border border-line bg-panel text-[10px] text-ink-3">
          +{more}
        </li>
      )}
    </ul>
  );
}

export function ReadOnlyBanner({ banner, onTakeOver, taking }: {
  banner: PresenceBanner | null;
  onTakeOver: () => void;
  /** A take-over is in flight. */
  taking?: boolean;
}) {
  if (!banner) return null;
  return (
    // role=status: announced when it appears, without stealing focus from whatever was being done.
    <div role="status"
      className={`shrink-0 flex items-center gap-2.5 px-3 py-1.5 border-b text-[12.5px] min-w-0
                  ${banner.kind === "lost" ? "border-block/60 bg-block/10 text-block" : "border-warn/60 bg-warn/10 text-warn"}`}>
      <span className="flex shrink-0"><Icon name="eye" size={15} /></span>
      <span className="font-medium whitespace-nowrap">{banner.text}</span>
      {banner.detail && <span className="text-ink-2 truncate" title={banner.detail}>{banner.detail}</span>}
      {banner.canTakeOver && (
        <button onClick={onTakeOver} disabled={taking}
          title="Take the edit lock. Whoever holds it is switched to view-only, and the take-over is recorded."
          className="ml-auto shrink-0 h-[24px] px-2.5 rounded-md border border-current font-medium hover:opacity-80 disabled:opacity-50">
          {taking ? "Taking over..." : "Take over"}
        </button>
      )}
    </div>
  );
}
