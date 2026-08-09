"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";

import { LOCALES, getLocale, setLocale, t, type Locale } from "@/lib/i18n";
import { getUser } from "@/lib/user";
import { isPreAuthRoute, shouldOpenOnboarding } from "@/lib/onboardingGate";

// The first five minutes.
//
// Every page explains itself well once you are on it, and nothing explains which page to be on. A new
// annotator signing in lands on a triage queue with no idea that the queue is ranked, that reviewing is
// where their time goes, or that there is a keyboard-driven mode that is several times faster. They find
// out from a colleague, or they do not.
//
// Deliberately not a tour that points at elements. Those break the moment the layout moves, and they demand
// attention at exactly the wrong time. This is four cards that name the four things worth knowing on day
// one, dismissible at any point, and shown exactly once per person.

const SEEN_KEY = "lbx_onboarded";

type Step = { title: string; body: string; href?: string; cta?: string };

const STEPS: Step[] = [
  {
    title: "Your queue is ranked, not a list",
    body: "Objects are ordered by how much labelling them would teach the model: uncertainty, rarity, and "
      + "disagreement between the detector and the tracker. Working top to bottom is the point.",
    href: "/review/queue", cta: "see the queue",
  },
  {
    title: "Rapid review is the fast path",
    body: "One crop fills the screen and one key decides it: A accepts, R rejects, C reclassifies, U undoes. "
      + "The next crop is already loaded. This is where most reviewing should happen.",
    href: "/review/rapid", cta: "try it",
  },
  {
    title: "The editor draws more than boxes",
    body: "Masks with SAM assistance, lanes, poses, and 3D cuboids, all on the same frame. Drag on empty "
      + "canvas to select several objects at once, and press ? at any time for the shortcuts.",
    href: "/annotations", cta: "open a session",
  },
  {
    title: "Nothing ships unreviewed",
    body: "Machine labels are proposals until a person accepts them. The promotion gate blocks a model that "
      + "regressed on a safety class, and it tells you which class, so a blocked release is a work item.",
    href: "/govern", cta: "see the gate",
  },
];

export default function Onboarding() {
  const router = useRouter();
  const pathname = usePathname();
  const [step, setStep] = useState(0);
  const [open, setOpen] = useState(false);
  const [locale, setLocaleState] = useState<Locale>("en");

  useEffect(() => {
    // Only for a signed-in user, on a screen past sign-in, and only once. The user check alone was not
    // enough: AuthBootstrap mints an admin token on mount in local dev, so by the time this ran there was
    // always a user, and the tour opened over the login form and ate its clicks.
    if (typeof window === "undefined") return;
    setLocaleState(getLocale());
    if (!shouldOpenOnboarding(pathname, !!getUser(), !!localStorage.getItem(SEEN_KEY))) return;
    setOpen(true);
  }, [pathname]);

  const dismiss = () => {
    localStorage.setItem(SEEN_KEY, new Date().toISOString());
    setOpen(false);
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") dismiss(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // The menu entries. Wired here rather than in the menu so a menu item is never a dead link: the handler
  // and the thing it opens live in the same file.
  useEffect(() => {
    const reopen = () => { if (isPreAuthRoute(pathname)) return; setStep(0); setOpen(true); };
    const language = () => { if (isPreAuthRoute(pathname)) return; setStep(0); setOpen(true); };
    window.addEventListener("lbx:tour", reopen);
    window.addEventListener("lbx:language", language);
    return () => {
      window.removeEventListener("lbx:tour", reopen);
      window.removeEventListener("lbx:language", language);
    };
  }, [pathname]);

  if (!open) return null;
  const s = STEPS[step];
  const last = step === STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-[100] bg-bg/85 flex items-center justify-center p-4">
      <div className="panel w-[min(34rem,94vw)] p-5 space-y-4">
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <div className="font-mono text-[10px] uppercase text-ink-3">
              {t("onboarding.welcome")} · {step + 1} of {STEPS.length}
            </div>
            <h2 className="font-display font-bold text-[17px] text-ink mt-1">{s.title}</h2>
          </div>
          {/* The language picker is here rather than buried in a settings page, because the person who most
              needs it is the one who cannot read the settings page. */}
          <select value={locale}
            onChange={(e) => setLocale(e.target.value as Locale)}
            className="bg-bg border border-line px-1.5 py-0.5 font-mono text-[11px] text-ink-2">
            {LOCALES.map((l) => <option key={l.code} value={l.code}>{l.native}</option>)}
          </select>
        </div>

        <p className="font-sans text-[13.5px] leading-6 text-ink-2">{s.body}</p>

        <div className="flex items-center gap-2 pt-1">
          <div className="flex gap-1">
            {STEPS.map((_, i) => (
              <span key={i} className={`w-6 h-0.5 ${i <= step ? "bg-accent" : "bg-line"}`} />
            ))}
          </div>
          <button onClick={dismiss}
            className="ml-auto font-mono text-[11px] text-ink-3 hover:text-ink">
            {t("onboarding.skip")}
          </button>
          {s.href && (
            <button onClick={() => { dismiss(); router.push(s.href!); }}
              className="border border-line px-2 py-1 font-mono text-[11px] text-ink-2 hover:border-accent">
              {s.cta}
            </button>
          )}
          <button onClick={() => (last ? dismiss() : setStep((x) => x + 1))}
            className="border border-accent px-3 py-1 font-mono text-[11px] text-accent hover:bg-accent/10">
            {last ? t("onboarding.done") : t("onboarding.next")}
          </button>
        </div>
      </div>
    </div>
  );
}

/** Re-open the tour from a menu, for somebody who skipped it and wants it back. */
export function resetOnboarding(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem(SEEN_KEY);
    window.location.reload();
  }
}
