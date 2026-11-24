"use client";

import SegmentError from "@/components/shell/RouteError";

// This route's own error boundary, so a failure here costs this page and not the app. See
// components/shell/RouteError.tsx for why the heavy routes each have one.
export default function Error(props: { error: Error & { digest?: string }; reset: () => void }) {
  return <SegmentError {...props} name="Training" />;
}
