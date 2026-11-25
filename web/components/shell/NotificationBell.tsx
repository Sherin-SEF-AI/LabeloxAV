"use client";

import { useEffect, useRef, useState } from "react";

import { useT } from "@/lib/i18n";
import { useQuery } from "@/lib/useQuery";
import { useRouter } from "next/navigation";

import Icon from "./Icon";
import EmptyState from "@/components/ui/EmptyState";
import { ago, humanize } from "@/lib/format";
import { api, humanizeError } from "@/lib/api";
import { useEventStream } from "@/lib/useEventStream";
import type { NotificationRow } from "@/lib/types";
import { useAnchoredDropdown } from "@/components/shell/useAnchoredDropdown";

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

// Relative time is computed from the clock, so the server renders one string and the client renders a
// different one a moment later, which React reports as a hydration mismatch. `mounted` below defers the
// relative form until after hydration; the absolute time is shown first, which is correct rather than a
// placeholder.
export default function NotificationBell() {
  const router = useRouter();
  const t = useT();
  const [open, setOpen] = useState(false);
  const [rows, setRows] = useState<NotificationRow[]>([]);
  const [unread, setUnread] = useState(0);
  const [mounted, setMounted] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);
  // Fixed rather than absolute: the top bar clips its children. See useAnchoredDropdown.
  const { anchorRef: btnRef, style, place } = useAnchoredDropdown(open);

  useEffect(() => { setMounted(true); }, []);

  const stream = useEventStream<{ unread: number; notifications: NotificationRow[] }>(
    "/api/events/notifications", "notifications");

  useEffect(() => {
    if (!stream.data) return;
    setUnread(stream.data.unread ?? 0);
    setRows(stream.data.notifications ?? []);
  }, [stream.data]);

  // The stream is the live path; this covers the first paint and any deployment where the stream cannot
  // connect, so the bell is never simply blank.
  // A failure is not rendered: signed out or offline, the bell stays quiet rather than shouting an error.
  const fallback = useQuery(["notificationCount+notifications", 10], async () => {
    const [c, l] = await Promise.all([api.notificationCount(), api.notifications(10)]);
    return { unread: c.unread, notifications: l.notifications };
  });
  useEffect(() => {
    if (!fallback.data) return;
    setUnread(fallback.data.unread);
    setRows(fallback.data.notifications);
  }, [fallback.data]);
  const refresh = fallback.reload;
  // Asked again whenever the stream drops, which is when it stops being the live path.
  const wasConnected = useRef(false);
  useEffect(() => {
    if (wasConnected.current && !stream.connected) void refresh();
    wasConnected.current = stream.connected;
  }, [stream.connected, refresh]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onEsc = (e: KeyboardEvent) => { if (e.key === "Escape") { setOpen(false); btnRef.current?.focus(); } };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onEsc);
    return () => { document.removeEventListener("mousedown", onDoc); document.removeEventListener("keydown", onEsc); };
  }, [open, btnRef]);

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
      <button ref={btnRef} onClick={() => { if (!open) place(); setOpen((o) => !o); }}
        data-tip={unread ? t("notify.unread", "{n} unread", { n: unread }) : t("notify.title", "Notifications")}
        aria-label={unread ? t("notify.title_unread", "Notifications, {n} unread", { n: unread }) : t("notify.title", "Notifications")}
        aria-haspopup="dialog" aria-expanded={open}
        className="relative w-7 h-7 flex items-center justify-center rounded text-ink-3 hover:text-ink hover:bg-panel">
        <Icon name="bell" size={14} />
        {unread > 0 && (
          // A count, not a dot: "something happened" is not actionable, "three things happened" is.
          <span className="absolute -top-0.5 -right-0.5 min-w-[14px] h-[14px] px-[3px] rounded-full
                           bg-accent text-bg text-[10.5px] leading-[14px] text-center">
            {unread > 99 ? "99+" : unread}
          </span>
        )}
      </button>

      {open && (
        <div style={style} role="group" aria-label={t("notify.title", "Notifications")} className="z-50 w-[min(26rem,92vw)] panel shadow-lg">
          <div className="flex items-center gap-2 px-3 py-2 border-b hairline text-[12px]">
            <span className="section-title">{t("notify.title", "Notifications")}</span>
            {unread > 0 && (
              <button onClick={markAll} className="ml-auto link text-[12px]">
                {t("notify.mark_all", "mark all read")}
              </button>
            )}
          </div>

          {rows.length === 0 ? (
            <EmptyState compact icon="bell" title={t("notify.empty", "You are all caught up")}>
              {t("notify.empty_long", "Assignments, raised issues, blocked promotions and drift alerts will appear here.")}
            </EmptyState>
          ) : (
            <ul className="max-h-[60vh] overflow-auto">
              {rows.map((n) => (
                <li key={n.notification_id}>
                  <button onClick={() => openItem(n)}
                    className={`w-full text-left px-3 py-2 border-b hairline hover:bg-head
                                ${n.read ? "opacity-60" : ""}`}>
                    <div className="flex items-start gap-2">
                      <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${SEV_DOT[n.severity] ?? "bg-ink-3"}`} aria-hidden />
                      <div className="min-w-0 flex-1">
                        <div className="text-[12.5px] font-medium text-ink truncate">{n.title}</div>
                        {n.body && <div className="text-[12px] text-ink-3 line-clamp-2 mt-0.5">{n.body}</div>}
                        <div className="text-[11px] text-ink-3 mt-1">
                          {humanize(n.kind)}{mounted ? ` · ${ago(n.created_at)}` : ""}
                        </div>
                      </div>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}

          <button onClick={() => { setOpen(false); router.push("/activity"); }}
            className="w-full px-3 py-2.5 text-[12px] link text-left">
            {t("notify.open_feed", "Open the activity feed")}
          </button>
        </div>
      )}
    </div>
  );
}
