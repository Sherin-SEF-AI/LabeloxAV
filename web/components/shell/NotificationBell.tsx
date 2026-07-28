"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import Icon from "./Icon";
import { api, humanizeError } from "@/lib/api";
import { useEventStream } from "@/lib/useEventStream";
import type { NotificationRow } from "@/lib/types";

// The bell.
//
// Issue comments, job assignments, the kill switch, drift breaches and blocked promotions were all silent:
// the system knew, and nobody was told unless they happened to be on the right page at the right moment. A
// blocked promotion could sit unnoticed for a day while the flywheel idled.
//
// Fed by SSE rather than a poll, so an event that happens while you are annotating reaches you without a
// timer running behind every page. The stream carries the count and the newest few, which is exactly what
// this renders; opening the panel does not refetch.

const SEV_DOT: Record<string, string> = {
  info: "bg-ink-3",
  warn: "bg-warn",
  critical: "bg-block",
};

function ago(iso: string | null): string {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export default function NotificationBell() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [rows, setRows] = useState<NotificationRow[]>([]);
  const [unread, setUnread] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);

  const stream = useEventStream<{ unread: number; notifications: NotificationRow[] }>(
    "/api/events/notifications", "notifications");

  useEffect(() => {
    if (!stream.data) return;
    setUnread(stream.data.unread ?? 0);
    setRows(stream.data.notifications ?? []);
  }, [stream.data]);

  // The stream is the live path; this covers the first paint and any deployment where the stream cannot
  // connect, so the bell is never simply blank.
  const refresh = useCallback(async () => {
    try {
      const [c, l] = await Promise.all([api.notificationCount(), api.notifications(10)]);
      setUnread(c.unread);
      setRows(l.notifications);
    } catch { /* signed out or offline: the bell stays quiet rather than shouting an error */ }
  }, []);

  useEffect(() => { if (!stream.connected) refresh(); }, [stream.connected, refresh]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onEsc);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onEsc); };
  }, [open]);

  const openItem = async (n: NotificationRow) => {
    setOpen(false);
    try { await api.markNotificationRead(n.notification_id); } catch { /* navigating matters more */ }
    setUnread((u) => Math.max(0, u - (n.read ? 0 : 1)));
    if (n.href) router.push(n.href);
  };

  const markAll = async () => {
    try { await api.markAllNotificationsRead(); setUnread(0); await refresh(); }
    catch (e) { console.warn(humanizeError(e)); }
  };

  return (
    <div className="relative" ref={boxRef}>
      <button onClick={() => setOpen((o) => !o)}
        data-tip={unread ? `${unread} unread` : "Notifications"}
        aria-label={unread ? `Notifications, ${unread} unread` : "Notifications"}
        className="relative w-7 h-7 flex items-center justify-center rounded text-ink-3 hover:text-ink hover:bg-panel">
        <Icon name="bell" size={14} />
        {unread > 0 && (
          // A count, not a dot: "something happened" is not actionable, "three things happened" is.
          <span className="absolute -top-0.5 -right-0.5 min-w-[14px] h-[14px] px-[3px] rounded-full
                           bg-accent text-bg font-mono text-[9px] leading-[14px] text-center">
            {unread > 99 ? "99+" : unread}
          </span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 top-8 z-50 w-[min(26rem,92vw)] panel shadow-lg">
          <div className="flex items-center gap-2 px-3 py-2 border-b hairline font-mono text-[11px]">
            <span className="uppercase text-ink-3">notifications</span>
            {unread > 0 && (
              <button onClick={markAll} className="ml-auto text-ink-3 hover:text-accent">
                mark all read
              </button>
            )}
          </div>

          {rows.length === 0 ? (
            <div className="p-4 font-mono text-[11px] text-ink-3">
              Nothing yet. Assignments, raised issues, blocked promotions and drift breaches land here.
            </div>
          ) : (
            <ul className="max-h-[60vh] overflow-auto">
              {rows.map((n) => (
                <li key={n.notification_id}>
                  <button onClick={() => openItem(n)}
                    className={`w-full text-left px-3 py-2 border-b hairline hover:bg-panel-2
                                ${n.read ? "opacity-60" : ""}`}>
                    <div className="flex items-start gap-2">
                      <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${SEV_DOT[n.severity] ?? "bg-ink-3"}`} />
                      <div className="min-w-0 flex-1">
                        <div className="font-mono text-[11.5px] text-ink truncate">{n.title}</div>
                        {n.body && <div className="font-mono text-[10.5px] text-ink-3 line-clamp-2">{n.body}</div>}
                        <div className="font-mono text-[9.5px] text-ink-4">
                          {n.kind.replace(/_/g, " ")} · {ago(n.created_at)}
                        </div>
                      </div>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}

          <button onClick={() => { setOpen(false); router.push("/activity"); }}
            className="w-full px-3 py-2 font-mono text-[10.5px] text-ink-3 hover:text-accent text-left">
            open the activity feed
          </button>
        </div>
      )}
    </div>
  );
}
