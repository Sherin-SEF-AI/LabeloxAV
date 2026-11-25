import Link from "next/link";

// A typo in a URL, or a stale bookmark to a route that moved, previously rendered Next's default 404 with no
// way back into the app. This keeps the user oriented and points at the two entry points that always exist:
// Home, and the session list. (It used to offer "triage queue" and "all platforms", neither of which is what
// either page is called any more.)
export default function NotFound() {
  return (
    <div className="min-h-[100dvh] flex items-center justify-center p-6">
      <div className="panel p-8 max-w-md w-full text-center">
        <div className="text-[13px] font-medium text-ink-3 tabular-nums">404</div>
        <h1 className="mt-1 text-[18px] font-semibold text-ink">This page does not exist</h1>
        <p className="mt-2 text-[13px] text-ink-3 leading-relaxed">
          It may have moved, or the link may be out of date.
        </p>
        <div className="mt-6 flex items-center justify-center gap-2">
          <Link href="/" className="btn btn-primary">Go to Home</Link>
          <Link href="/annotations" className="btn">Browse sessions</Link>
        </div>
      </div>
    </div>
  );
}
