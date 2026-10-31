"use client";

import { useEffect, useState } from "react";

import { apiGet } from "@/lib/api";

// Which optional parts this installation has (GET /api/system/features), so navigation offers only pages
// whose routes exist. Fetched once per page load and shared: the sidebar and the command palette both ask.
//
// Until the answer arrives, and if it cannot be had (signed out, API down), every optional part reads as
// absent. Showing a security console that then fails on every request is worse than showing it a moment
// late.

export type Features = { optional_packs: Record<string, boolean>; gpu_worker?: boolean; autolabel_executor?: string };

const NONE: Features = { optional_packs: {} };
let cached: Features | null = null;
let inflight: Promise<Features> | null = null;

export function loadFeatures(): Promise<Features> {
  if (cached) return Promise.resolve(cached);
  if (!inflight) {
    inflight = apiGet<Features>("/api/system/features")
      .then((f) => { cached = f; return f; })
      .catch(() => NONE)
      .finally(() => { inflight = null; });
  }
  return inflight;
}

/** The features, and whether the answer has arrived (a gate must not say "not installed" before it knows). */
export function useFeaturesState(): { features: Features; loaded: boolean } {
  const [state, setState] = useState({ features: cached ?? NONE, loaded: cached !== null });
  useEffect(() => {
    let live = true;
    void loadFeatures().then((v) => { if (live) setState({ features: v, loaded: true }); });
    return () => { live = false; };
  }, []);
  return state;
}

export function useFeatures(): Features {
  return useFeaturesState().features;
}

export function packEnabled(f: Features, pack: string): boolean {
  return Boolean(f.optional_packs[pack]);
}
