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
import { humanize, plural, shortId } from "@/lib/format";

// M3.3 HD map viewer: render a fused map_commit (lanes + signs) on a MapLibre basemap; click an element to
// trace its provenance (source frames, calibration version, fusion run). OSM raster basemap, no token.
//
// The map itself is components/map/HdMapCanvas, loaded on demand so maplibre-gl is not in this route's
// first load. ssr: false because it needs a canvas and has nothing to render without one.
const HdMapCanvas = dynamic(() => import("@/components/map/HdMapCanvas"), {
  ssr: false,
  loading: () => <div className="relative flex-1 flex items-center justify-center text-[12px] text-ink-3" role="status">Loading map...</div>,
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
        <label htmlFor="map-commit" className="text-ink-3">Map version</label>
        <select id="map-commit" value={sel} onChange={(e) => setSel(e.target.value)} className="field text-[12.5px]">
          {commits.map((c) => (
            <option key={c.commit_id} value={c.commit_id}>
              {c.region} · {plural(c.element_count, "element")} · {c.commit_id}
            </option>
          ))}
        </select>
        {commit && (
          <span className="text-ink-3 flex items-center gap-2">
            <span>Exports</span>
            {Object.entries(commit.formats).map(([k, uri]) => <span key={k} data-tip={uri}><StateBadge state={k} /></span>)}
            <span className="ml-1">Calibration <span className="text-ink-2">{commit.calibration_version}</span></span>
          </span>
        )}
        {commitsQ.loading && !commits.length && <span className="text-ink-3">Loading map versions...</span>}
        {!commitsQ.loading && !loadErr && !commits.length && (
          <span className="text-warn">No maps built yet. A map is fused from a session that has GPS positions.</span>
        )}
      </>
    )}>
      <div className="flex h-full min-h-0">
        <HdMapCanvas commitId={sel} onProvenance={setProv} />
        <Inspector title="Element provenance" side="right">
          <div className="p-3 text-[12.5px]">
            {loadErr != null && <LoadState error={loadErr} onRetry={() => void commitsQ.reload()} />}
            {prov?.found ? (
              <div className="space-y-3">
                <div>
                  <div className="text-ink font-medium">{humanize(prov.kind)}</div>
                  <div className="text-[11px] text-ink-3 font-mono">{shortId(prov.element_id)}</div>
                </div>
                <div className="space-y-1">
                  <div className="flex items-center justify-between gap-2"><span className="text-ink-3">Confidence</span>{typeof prov.confidence === "number" ? <ConfBar conf={prov.confidence} /> : <span>-</span>}</div>
                  <div className="flex justify-between gap-2"><span className="text-ink-3">Calibration</span><span className="text-ink-2 truncate">{prov.calibration_version}</span></div>
                  <div className="flex justify-between gap-2"><span className="text-ink-3">Map version</span><span className="text-ink-2 truncate font-mono text-[11.5px]">{prov.commit_id}</span></div>
                  <div className="flex justify-between gap-2"><span className="text-ink-3">Fusion job</span><span className="text-ink-2 font-mono text-[11.5px]">{shortId(prov.fusion_job_id)}</span></div>
                </div>
                {prov.attrs && Object.keys(prov.attrs).length > 0 && (
                  <div className="space-y-1">
                    <div className="label-caps">Attributes</div>
                    {Object.entries(prov.attrs).map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-2">
                        <span className="text-ink-3">{humanize(k)}</span>
                        <span className="text-ink-2 truncate">{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>
                      </div>
                    ))}
                  </div>
                )}
                <div className="space-y-1">
                  <div className="label-caps">Source frames ({prov.source_frames?.length ?? 0})</div>
                  {prov.source_frames?.map((f) => (
                    <a key={f.frame_id} href={`/frame/${f.frame_id}`} className="link block truncate">
                      {f.vehicle_id} · {f.cam_id} <span className="text-ink-3 font-mono text-[11px]">{shortId(f.frame_id)}</span>
                    </a>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-ink-3 leading-relaxed">
                Click a lane or sign on the map to see the frames and calibration it was built from.
              </div>
            )}
          </div>
        </Inspector>
      </div>
    </PageShell>
  );
}
