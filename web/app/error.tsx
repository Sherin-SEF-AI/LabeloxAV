"use client";

import { RouteErrorBody } from "@/components/shell/RouteError";

// Route-level error boundary. Without one, a single render throw anywhere in a page unmounts the whole tree
// and the user is left staring at a blank white screen with no way back. This keeps the app usable and
// offers the two things that actually help: retry the segment, or leave for a page that works.
//
// This is the root one, the last line. The routes that load a rendering engine have their own closer in
// (app/lidar/error.tsx and friends), which keep the sidebar; this one cannot, because the sidebar may be
// what threw. The markup they share is components/shell/RouteError.tsx.
export default function RouteError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return <RouteErrorBody error={error} reset={reset} />;
}
