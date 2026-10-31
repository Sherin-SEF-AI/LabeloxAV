"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import Icon from "./Icon";
import { NAV_FOOTER, activeKey, isLabsPath, sectionsFor, type NavItem } from "@/lib/sidebar";
import { packEnabled, useFeatures } from "@/lib/features";
import { readPref, writePref, LABS_PREF, SIDEBAR_COLLAPSED_PREF, LABS_EVENT } from "@/lib/prefs";

// The application sidebar: where a person on the team goes next. Sections come from lib/sidebar.ts.
//
// Collapsible to an icon rail, remembered per browser, because the review and editor pages want every pixel
// of width for imagery. Below the tablet breakpoint it is not in the layout at all and opens as a drawer from
// the top bar's menu button, since a fixed 220px column would leave a phone almost no room for content.

function Item({ item, active, collapsed, pathname, onNavigate }: {
  item: NavItem; active: boolean; collapsed: boolean; pathname: string; onNavigate?: () => void;
}) {
  // Sub-pages show under the entry while you are inside it, so the list stays short everywhere else.
  const children = active && !collapsed ? item.children ?? [] : [];
  const onChild = children.some((c) => c.href === pathname);
  return (
    <>
      <Link
        href={item.href}
        onClick={onNavigate}
        aria-current={active && !onChild ? "page" : undefined}
        aria-label={collapsed ? item.label : undefined}
        // A native title, not the app's data-tip: the list scrolls, and a pseudo-element tooltip inside a
        // scrolling container is clipped by it.
        title={collapsed ? `${item.label}: ${item.hint}` : item.hint}
        className={`side-item ${active ? "side-item-on" : ""} ${collapsed ? "justify-center px-0" : ""}`}
      >
        <span className={`shrink-0 ${active ? "text-accent-2" : ""}`}><Icon name={item.icon} size={16} /></span>
        {!collapsed && <span className="truncate">{item.label}</span>}
      </Link>
      {children.length > 0 && (
        <ul className="mt-px mb-1 ml-[17px] pl-3 border-l hairline flex flex-col gap-px">
          {children.map((c) => {
            const here = c.href === pathname;
            return (
              <li key={c.href}>
                <Link href={c.href} onClick={onNavigate} aria-current={here ? "page" : undefined}
                  className={`side-sub ${here ? "side-sub-on" : ""}`}>
                  <span className="truncate">{c.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </>
  );
}

export default function Sidebar() {
  const pathname = usePathname() ?? "/";
  const current = activeKey(pathname);
  const [collapsed, setCollapsed] = useState(false);
  const [labsOn, setLabsOn] = useState(false);
  const [labsOpen, setLabsOpen] = useState(true);
  const [drawer, setDrawer] = useState(false);
  const features = useFeatures();

  // Read after mount, never during render: the server has no browser storage, and reading it in render would
  // make the first client render disagree with the server's HTML.
  useEffect(() => {
    setCollapsed(readPref(SIDEBAR_COLLAPSED_PREF) === "1");
    setLabsOn(readPref(LABS_PREF) === "1");
    const onLabs = () => setLabsOn(readPref(LABS_PREF) === "1");
    const onDrawer = () => setDrawer((d) => !d);
    window.addEventListener(LABS_EVENT, onLabs);
    window.addEventListener("lbx:sidebar", onDrawer);
    return () => {
      window.removeEventListener(LABS_EVENT, onLabs);
      window.removeEventListener("lbx:sidebar", onDrawer);
    };
  }, []);

  // A navigation closes the drawer; leaving it open over the page just arrived at would hide that page.
  useEffect(() => { setDrawer(false); }, [pathname]);

  const toggleCollapsed = useCallback(() => {
    setCollapsed((c) => {
      writePref(SIDEBAR_COLLAPSED_PREF, c ? "0" : "1");
      return !c;
    });
  }, []);

  const onLabsPage = isLabsPath(pathname);
  const sections = sectionsFor((p) => packEnabled(features, p)).filter((s) => !s.labs || labsOn || onLabsPage);

  const body = (narrow: boolean) => (
    <>
      <div className={`flex items-center h-12 shrink-0 ${narrow ? "justify-center" : "justify-between pl-3.5 pr-2"}`}>
        {!narrow && (
          <Link href="/" className="font-display font-semibold text-[14px] tracking-tight text-ink">
            Labelox<span className="text-accent-2">AV</span>
          </Link>
        )}
        <button onClick={toggleCollapsed} className="side-icon-btn hidden md:flex"
          aria-label={narrow ? "Expand sidebar" : "Collapse sidebar"}
          data-tip={narrow ? "Expand sidebar" : "Collapse sidebar"}>
          <Icon name="sidebar" size={16} />
        </button>
      </div>

      <nav aria-label="Main" className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden px-2 pb-3 no-scrollbar">
        {sections.map((section) => {
          const isLabs = Boolean(section.labs);
          const open = !isLabs || labsOpen || onLabsPage;
          return (
            <div key={section.key} className="mt-3 first:mt-1">
              {narrow ? (
                <div className="mx-auto my-2 h-px w-6 bg-line" aria-hidden />
              ) : isLabs ? (
                <button onClick={() => setLabsOpen((o) => !o)} aria-expanded={open}
                  className="side-section w-full flex items-center justify-between hover:text-ink-2">
                  <span className="flex items-center gap-1.5"><Icon name="flask" size={12} />{section.label}</span>
                  <span className={`transition-transform ${open ? "" : "-rotate-90"}`}><Icon name="chevD" size={12} /></span>
                </button>
              ) : (
                <div className="side-section">{section.label}</div>
              )}
              {open && (
                <ul className="flex flex-col gap-px">
                  {section.items.map((item) => (
                    <li key={item.key}>
                      <Item item={item} active={current === item.key} collapsed={narrow} pathname={pathname}
                        onNavigate={() => setDrawer(false)} />
                    </li>
                  ))}
                </ul>
              )}
            </div>
          );
        })}
      </nav>

      <div className="shrink-0 border-t hairline px-2 py-2 flex flex-col gap-px">
        {NAV_FOOTER.map((item) => (
          <Item key={item.key} item={item} active={current === item.key} collapsed={narrow} pathname={pathname}
            onNavigate={() => setDrawer(false)} />
        ))}
        <button onClick={() => window.dispatchEvent(new Event("lbx:shortcuts"))}
          data-tip="Keyboard shortcuts (?)"
          className={`side-item ${narrow ? "justify-center px-0" : ""}`}>
          <span className="shrink-0"><Icon name="keyboard" size={16} /></span>
          {!narrow && <span className="truncate">Shortcuts</span>}
          {!narrow && <kbd className="ml-auto text-[10.5px] text-ink-3">?</kbd>}
        </button>
      </div>
    </>
  );

  return (
    <>
      <aside
        className={`hidden md:flex flex-col shrink-0 h-[100dvh] bg-rail border-r hairline transition-[width] duration-150 ${collapsed ? "w-[52px]" : "w-[220px]"}`}>
        {body(collapsed)}
      </aside>
      {drawer && (
        <div className="md:hidden fixed inset-0 z-[70] flex" role="dialog" aria-modal="true" aria-label="Navigation">
          <aside className="flex flex-col w-[260px] max-w-[85vw] h-full bg-rail border-r hairline shadow-2xl">
            {body(false)}
          </aside>
          <button className="flex-1 bg-black/50" aria-label="Close navigation" onClick={() => setDrawer(false)} />
        </div>
      )}
    </>
  );
}
