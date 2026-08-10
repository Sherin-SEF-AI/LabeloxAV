"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";

// What this operation is known to get right, shown on the control that runs it.
//
// Every batch button reported volume and nothing reported correctness, so an annotator pressing "fit 3D
// boxes" had no way to know whether that operation has ever been checked. The endpoint answers with a
// measurement or a 404, and a 404 is a real answer: this operation is unmeasured, so it must dry-run first
// and everything it produces goes to review.
export type OpState = { measured: boolean; precision?: number; n?: number; reason?: string };

const CACHE = new Map<string, { at: number; v: OpState }>();
const TTL_MS = 60_000;

export async function fetchOpState(opType: string): Promise<OpState> {
  const hit = CACHE.get(opType);
  if (hit && Date.now() - hit.at < TTL_MS) return hit.v;
  let v: OpState;
  try {
    const r = await api.opPrecisionLatest(opType);
    v = { measured: true, precision: r.precision, n: r.n };
  } catch {
    // A 404 and an unreachable harness deliberately land here together. Both mean "no measurement backs
    // this", and giving them one client path stops a network blip from being read as a good score.
    v = { measured: false };
  }
  CACHE.set(opType, { at: Date.now(), v });
  return v;
}

export function useOpState(opType: string): OpState | null {
  const [s, setS] = useState<OpState | null>(null);
  useEffect(() => {
    let alive = true;
    fetchOpState(opType).then((v) => { if (alive) setS(v); });
    return () => { alive = false; };
  }, [opType]);
  return s;
}

/** A chip stating the operation's measured precision, or that nothing has measured it. */
export function OpPrecisionChip({ opType }: { opType: string }) {
  const s = useOpState(opType);
  if (!s) return null;
  if (!s.measured) {
    return (
      <span title="no measurement backs this operation; it will dry-run and route results to review"
        className="font-mono text-[9px] uppercase tracking-wide border border-line text-ink-3 rounded px-1 leading-tight">
        unmeasured
      </span>
    );
  }
  const p = s.precision ?? 0;
  const tone = p >= 0.9 ? "border-pass/50 text-pass" : p >= 0.7 ? "border-warn/50 text-warn" : "border-block/50 text-block";
  return (
    <span title={`measured over ${s.n} reviewed outcomes; review is not a random sample`}
      className={`font-mono text-[9px] tracking-wide border ${tone} rounded px-1 leading-tight`}>
      {(p * 100).toFixed(0)}%
    </span>
  );
}
