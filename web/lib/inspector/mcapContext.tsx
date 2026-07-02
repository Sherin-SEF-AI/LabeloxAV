"use client";

import { createContext, useContext } from "react";
import type { SessionMcap } from "@/lib/inspector/mcap";

// One SessionMcap reader shared by every panel, plus the session id for the deep-link integrations.
type Ctx = { mcap: SessionMcap; sessionId: string };
const McapContext = createContext<Ctx | null>(null);

export function McapProvider({ mcap, sessionId, children }: { mcap: SessionMcap; sessionId: string; children: React.ReactNode }) {
  return <McapContext.Provider value={{ mcap, sessionId }}>{children}</McapContext.Provider>;
}

export function useMcap(): Ctx {
  const c = useContext(McapContext);
  if (!c) throw new Error("useMcap must be used inside a McapProvider");
  return c;
}
