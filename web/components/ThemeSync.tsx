"use client";

import { useEffect } from "react";

import { watchTheme } from "@/lib/theme";

// Keeps <html data-theme> true after load. The inline script in app/layout.tsx sets it before the first
// paint; this is for what happens later: the device going dark at sunset while the choice is "System", or
// the theme being changed in another tab. Renders nothing.
export default function ThemeSync() {
  useEffect(() => watchTheme(), []);
  return null;
}
