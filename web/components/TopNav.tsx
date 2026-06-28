"use client";

import { useRouter } from "next/navigation";
import UserPicker from "./UserPicker";

// Shared app navigation. One header across every workspace page, with the active page in accent.
// Operational Materialism: color is earned; nav is grey, the current page is the only accent.

const LINKS = [
  { href: "/", label: "TRIAGE" },
  { href: "/annotate/new", label: "NEW" },
  { href: "/annotations", label: "OPEN" },
  { href: "/scenarios", label: "SCENARIOS" },
  { href: "/analytics", label: "ANALYTICS" },
  { href: "/search", label: "SEARCH" },
  { href: "/discovery", label: "DISCOVERY" },
  { href: "/curation", label: "CURATION" },
  { href: "/calibration", label: "CALIBRATION" },
  { href: "/map", label: "MAP" },
  { href: "/quality", label: "QUALITY" },
  { href: "/training", label: "TRAINING" },
  { href: "/import", label: "IMPORT" },
  { href: "/datasets", label: "DATASETS" },
  { href: "/review/queue", label: "REVIEW" },
  { href: "/govern", label: "GOVERN" },
  { href: "/collaborate", label: "COLLABORATE" },
  { href: "/jobs", label: "JOBS" },
];

export default function TopNav({ active, right }: { active: string; right?: React.ReactNode }) {
  const router = useRouter();
  return (
    <header className="flex items-center justify-between gap-4 px-4 h-12 border-b hairline shrink-0">
      <div className="flex items-center gap-3 min-w-0">
        <button onClick={() => router.push("/")} className="font-display font-bold shrink-0" title="home (triage)">
          Labelox<span className="text-accent">AV</span>
        </button>
        <nav className="flex items-center gap-3 overflow-x-auto no-scrollbar">
          {LINKS.map((l) => (
            <button key={l.href} onClick={() => router.push(l.href)}
              className={`font-mono text-xs whitespace-nowrap border-b-2 -mb-px pb-px ${
                active === l.label ? "text-accent border-accent" : "text-ink-3 border-transparent hover:text-ink-2"
              }`}>
              {l.label}
            </button>
          ))}
        </nav>
      </div>
      <div className="flex items-center gap-3 font-mono text-xs shrink-0">
        {right}
        <UserPicker />
      </div>
    </header>
  );
}
