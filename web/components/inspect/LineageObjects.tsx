"use client";

import { useEffect, useState } from "react";

import { api, humanizeError } from "@/lib/api";
import LoadState from "@/components/shell/LoadState";

// The objects one class move actually produced.
//
// The lineage table shows a move and two counts: `bus -> bmtc_bus_shelter`, 1,047 objects, refused because
// it turns a countable object into uncountable stuff. The only thing it could offer was to open one example
// in the editor, because the grouped endpoint caps its examples at eight where the aggregate is built. So a
// thousand-object decision could be inspected eight deep, and the only other action on the row reverts all
// thousand.
//
// This lists them, as crops, and hands one to the evidence panel when it is clicked. It is the same
// predicate the revert uses, on purpose: a list showing a different set from the one revert would touch
// would invite somebody to check these and change those.

export default function LineageObjects({ fromName, toName, selectedId, onPick }: {
  fromName: string;
  toName: string;
  selectedId?: string | null;
  onPick: (objectId: string) => void;
}) {
  const [rows, setRows] = useState<{ object_id: string; frame_id: string; conf: number; state: string }[]>([]);
  const [total, setTotal] = useState(0);
  const [reason, setReason] = useState<string | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setErr(null);
    api.agentLineageObjects(fromName, toName, 60)
      .then((r) => { if (!live) return; setRows(r.objects); setTotal(r.total); setReason(r.reason); })
      .catch((e) => { if (live) setErr(e); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [fromName, toName]);

  return (
    <div className="p-3 space-y-2">
      <div className="font-mono text-[11px]">
        <span className="text-ink-3">{fromName}</span>
        <span className="text-ink-3"> &rarr; </span>
        <span className="text-block">{toName}</span>
      </div>
      {reason && <div className="text-xs text-ink-3 leading-relaxed">{reason}</div>}

      {err != null ? <LoadState error={err} /> : loading ? (
        <div className="font-mono text-[11px] text-ink-3/60 animate-pulse py-6 text-center">loading objects...</div>
      ) : rows.length === 0 ? (
        <div className="font-mono text-[11px] text-ink-3 py-6 text-center">
          Nothing still carries this class. The move has already been put right or reviewed.
        </div>
      ) : (
        <>
          <div className="font-mono text-[10px] text-ink-3">
            {/* The count is stated because the grid is capped: a reviewer looking at 60 of 1,047 should
                know that is what they are looking at. */}
            showing {rows.length} of {total.toLocaleString()} still carrying {toName}
          </div>
          <div className="grid grid-cols-3 gap-1">
            {rows.map((o) => (
              <button key={o.object_id} onClick={() => onPick(o.object_id)}
                title={`${o.state} · conf ${o.conf.toFixed(2)}`}
                className={`relative border overflow-hidden ${o.object_id === selectedId ? "border-accent" : "border-line hover:border-ink-3"}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`/api/objects/${o.object_id}/crop?pad=0.2`} alt="" loading="lazy"
                  onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = "hidden"; }}
                  className="h-16 w-full object-cover bg-bg-2" />
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
