import Link from "next/link";

// A typo in a URL, or a stale bookmark to a route that moved, previously rendered Next's default 404 with no
// way back into the app. This keeps the user oriented and points at the two entry points that always exist.
export default function NotFound() {
  return (
    <div className="p-8 max-w-2xl mx-auto space-y-4">
      <h1 className="font-display text-lg text-ink">No such page</h1>
      <p className="font-mono text-[12px] text-ink-3">
        That route does not exist. It may have moved, or the link may be stale.
      </p>
      <div className="flex items-center gap-2">
        <Link href="/" className="font-mono text-xs border border-accent text-accent px-3 py-1 hover:bg-accent/10">
          triage queue
        </Link>
        <Link href="/platforms" className="font-mono text-xs border border-line text-ink-2 px-3 py-1 hover:border-accent">
          all platforms
        </Link>
      </div>
    </div>
  );
}
