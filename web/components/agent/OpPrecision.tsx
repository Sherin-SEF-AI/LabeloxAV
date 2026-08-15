"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";

// What this operation is known to get right, shown on the control that runs it.
//
// Every batch button reported volume and nothing reported correctness, so an annotator pressing "fit 3D
// boxes" had no way to know whether that operation has ever been checked. An unmeasured operation must
// dry-run first and everything it produces goes to review.
//
// Read from the aggregate endpoint, not per operation. The per-operation route answers an unmeasured kind
// with a 404, which is a deliberate and correct contract for an API, but a browser logs every 404 as a
// failed request no matter how the calling code handles it. All nine kinds are unmeasured today, so a page
// of these chips printed a wall of red into the console describing the expected state of the system:
//
//   :3000/api/eval/operations/cuboid/latest     404 (Not Found)
//   :3000/api/eval/operations/attribute/latest  404 (Not Found)
//   :3000/api/eval/operations/relabel/latest    404 (Not Found)
//
// Console noise that says "everything is normal" is worse than none, because it trains people to ignore the
// console, and this codebase has already had to clean that up once. The aggregate returns 200 with every
// kind's state in one response, which is the same information without the noise, and one request instead of
// one per chip.
export type OpState = { measured: boolean; precision?: number; n?: number; reason?: string;
                        runs_scored?: number; excluded_runs?: number };

type AggRow = { measured?: boolean; precision?: number; n?: number; reason?: string;
                runs_scored?: number; excluded_runs?: number };

const TTL_MS = 60_000;

// The whole map, not one entry per kind. A page mounts several chips at once, and caching per kind would
// still let each of them start its own request before any of them finished.
let cache: { at: number; ops: Record<string, AggRow> } | null = null;
// The in-flight request, shared. Without this, five chips mounting in the same tick make five identical
// calls and only the last one populates the cache.
let inflight: Promise<Record<string, AggRow>> | null = null;

async function loadAll(): Promise<Record<string, AggRow>> {
  if (cache && Date.now() - cache.at < TTL_MS) return cache.ops;
  if (inflight) return inflight;
  inflight = api.opPrecisionAll()
    .then((r) => {
      const ops = (r.operations ?? {}) as Record<string, AggRow>;
      cache = { at: Date.now(), ops };
      return ops;
    })
    .catch(() => {
      // An unreachable harness lands on the same answer as an unmeasured operation, deliberately: both mean
      // "no measurement backs this". Not cached, so a blip does not pin every chip to unmeasured for a
      // minute after the backend comes back.
      return {} as Record<string, AggRow>;
    })
    .finally(() => { inflight = null; });
  return inflight;
}

export async function fetchOpState(opType: string): Promise<OpState> {
  const ops = await loadAll();
  const row = ops[opType];
  if (!row?.measured) {
    return { measured: false, n: row?.n, reason: row?.reason, excluded_runs: row?.excluded_runs };
  }
  return { measured: true, precision: row.precision, n: row.n,
           runs_scored: row.runs_scored, excluded_runs: row.excluded_runs };
}

/** For tests and for a caller that has just run an operation and wants the next read to be fresh. */
export function resetOpStateCache(): void {
  cache = null;
  inflight = null;
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
      <span title={s.reason
        ? `no measurement backs this operation (${s.reason}); it will dry-run and route results to review`
        : "no measurement backs this operation; it will dry-run and route results to review"}
        className="font-mono text-[9px] uppercase tracking-wide border border-line text-ink-3 rounded px-1 leading-tight">
        unmeasured
      </span>
    );
  }
  const p = s.precision ?? 0;
  const tone = p >= 0.9 ? "border-pass/50 text-pass" : p >= 0.7 ? "border-warn/50 text-warn" : "border-block/50 text-block";
  // The window is part of the claim. A score over recent runs and a score over everything the operation
  // ever did are different statements, and a chip that shows only the number lets the reader assume the
  // wrong one.
  const scope = s.excluded_runs
    ? `; recent runs only (${s.excluded_runs} older runs excluded, so a fixed operation can recover)`
    : "";
  return (
    <span title={`measured over ${s.n} reviewed outcomes; review is not a random sample${scope}`}
      className={`font-mono text-[9px] tracking-wide border ${tone} rounded px-1 leading-tight`}>
      {(p * 100).toFixed(0)}%
    </span>
  );
}
