"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { api } from "@/lib/api";
import type { MapProvenance } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import Inspector from "@/components/shell/Inspector";
import { StateBadge, ConfBar } from "@/components/StateBadge";
import { useQuery } from "@/lib/useQuery";

// M3.3 HD map viewer: render a fused map_commit (lanes + signs) on a MapLibre basemap; click an element to
// trace its provenance (source frames, calibration version, fusion run). OSM raster basemap, no token.
//
// The map itself is components/map/HdMapCanvas, loaded on demand so maplibre-gl is not in this route's
// first load. ssr: false because it needs a canvas and has nothing to render without one.
const HdMapCanvas = dynamic(() => import("@/components/map/HdMapCanvas"), {
  ssr: false,
  loading: () => <div className="relative flex-1 flex items-center justify-center text-[12px] text-ink-3" role="status">loading map</div>,
});

export default function MapPage() {
  const [sel, setSel] = useState<string>("");
  const [prov, setProv] = useState<MapProvenance | null>(null);

  // A bare .then() once left a rejection unhandled and the map permanently empty, which reads as a corpus
  // with no map rather than a request that failed.
  const commitsQ = useQuery(["hdmapCommits"], () => api.hdmapCommits());
  const commits = commitsQ.data ?? [];
  const loadErr = commitsQ.error;
  const first = commitsQ.data?.[0]?.commit_id;
  useEffect(() => { if (first) setSel((cur) => cur || first); }, [first]);

  const commit = commits.find((c) => c.commit_id === sel);

  return (
    <PageShell active="HD MAP" title="HD map" filters={(
      <>
        <span className="text-ink-3">commit</span>
        <select value={sel} onChange={(e) => setSel(e.target.value)} className="bg-bg border border-line px-2 py-1 text-ink">
          {commits.map((c) => <option key={c.commit_id} value={c.commit_id}>{c.commit_id} · {c.element_count} elem · {c.region}</option>)}
        </select>
        {commit && (
          <span className="text-ink-3 flex items-center gap-2">
            {Object.entries(commit.formats).map(([k, uri]) => <span key={k} title={uri}><StateBadge state={k} /></span>)}
            <span>calib {commit.calibration_version}</span>
          </span>
        )}
        {!commits.length && <span className="text-warn">no map commits yet (run hdmap fuse on a georef&apos;d session)</span>}
      </>
    )}>
      <div className="flex h-full min-h-0">
        <HdMapCanvas commitId={sel} onProvenance={setProv} />
        <Inspector title="element provenance" side="right">
          <div className="p-3 text-[12px]">
            {loadErr != null && <LoadState error={loadErr} onRetry={() => window.location.reload()} />}
            {prov?.found ? (
              <div className="space-y-1.5">
                <div className="text-ink-2">{prov.kind} <span className="text-ink-3">{prov.element_id?.slice(0, 8)}</span></div>
                <div className="text-ink-3 flex items-center gap-2">confidence {typeof prov.confidence === "number" ? <ConfBar conf={prov.confidence} /> : "-"}</div>
                <div className="text-ink-3">calibration {prov.calibration_version}</div>
                <div className="text-ink-3">commit {prov.commit_id}</div>
                <div className="text-ink-3">fusion job {prov.fusion_job_id?.slice(0, 8)}</div>
                <div className="text-ink-3 pt-1">attrs: {JSON.stringify(prov.attrs)}</div>
                <div className="text-ink-3 uppercase text-[10px] pt-2">source frames ({prov.source_frames?.length})</div>
                {prov.source_frames?.map((f) => (
                  <a key={f.frame_id} href={`/frame/${f.frame_id}`} className="block text-info hover:text-accent truncate">{f.vehicle_id} · {f.cam_id} · {f.frame_id.slice(0, 8)}</a>
                ))}
              </div>
            ) : <div className="text-ink-3">click a lane or sign to trace it to its source frames + calibration.</div>}
          </div>
        </Inspector>
      </div>
    </PageShell>
  );
}
