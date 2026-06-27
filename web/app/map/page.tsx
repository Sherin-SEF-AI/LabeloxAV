"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { api } from "@/lib/api";
import type { MapCommitRow, MapProvenance } from "@/lib/types";
import TopNav from "@/components/TopNav";

// M3.3 HD map viewer: render a fused map_commit (lanes + signs) on a MapLibre basemap; click an element to
// trace its provenance (source frames, calibration version, fusion run). OSM raster basemap, no token.

const STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: { osm: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256, attribution: "OSM" } },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};

export default function MapPage() {
  const wrap = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [commits, setCommits] = useState<MapCommitRow[]>([]);
  const [sel, setSel] = useState<string>("");
  const [prov, setProv] = useState<MapProvenance | null>(null);

  useEffect(() => {
    api.hdmapCommits().then((c) => { setCommits(c); if (c[0]) setSel(c[0].commit_id); });
  }, []);

  useEffect(() => {
    if (!wrap.current || map.current) return;
    const m = new maplibregl.Map({ container: wrap.current, style: STYLE, center: [77.5946, 12.9716], zoom: 15 });
    m.addControl(new maplibregl.NavigationControl(), "top-right");
    map.current = m;
  }, []);

  const render = useCallback(async (commitId: string) => {
    const m = map.current;
    if (!m || !commitId) return;
    const fc = await api.hdmapElements(`commit_id=${commitId}`);
    const draw = () => {
      for (const id of ["lanes", "signs", "lanes-hit"]) if (m.getLayer(id)) m.removeLayer(id);
      if (m.getSource("hdmap")) m.removeSource("hdmap");
      m.addSource("hdmap", { type: "geojson", data: fc as GeoJSON.FeatureCollection });
      m.addLayer({ id: "lanes", type: "line", source: "hdmap", filter: ["==", "kind", "lane"],
        paint: { "line-color": "#FF7A2F", "line-width": 3, "line-opacity": ["get", "confidence"] } });
      m.addLayer({ id: "signs", type: "circle", source: "hdmap", filter: ["==", "kind", "sign"],
        paint: { "circle-radius": 6, "circle-color": "#58A6FF", "circle-stroke-color": "#fff", "circle-stroke-width": 1 } });
      // fit to data
      const coords: number[][] = [];
      for (const f of fc.features) {
        if (!f.geometry) continue;
        if (f.geometry.type === "LineString") coords.push(...(f.geometry.coordinates as number[][]));
        else coords.push(f.geometry.coordinates as number[]);
      }
      if (coords.length) {
        const b = coords.reduce((bb, c) => bb.extend(c as [number, number]), new maplibregl.LngLatBounds(coords[0] as [number, number], coords[0] as [number, number]));
        m.fitBounds(b, { padding: 80, maxZoom: 18, duration: 600 });
      }
    };
    if (m.isStyleLoaded()) draw(); else m.once("load", draw);
    m.on("click", "lanes", (e) => { const id = e.features?.[0]?.properties?.element_id as string; if (id) api.hdmapProvenance(id).then(setProv); });
    m.on("click", "signs", (e) => { const id = e.features?.[0]?.properties?.element_id as string; if (id) api.hdmapProvenance(id).then(setProv); });
  }, []);

  useEffect(() => { if (sel) render(sel); }, [sel, render]);

  const commit = commits.find((c) => c.commit_id === sel);

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="MAP" />
      <div className="px-3 h-10 flex items-center gap-3 border-b hairline font-mono text-[11px]">
        <span className="text-ink-3">commit</span>
        <select value={sel} onChange={(e) => setSel(e.target.value)} className="bg-bg border border-line px-2 py-1 text-ink">
          {commits.map((c) => <option key={c.commit_id} value={c.commit_id}>{c.commit_id} · {c.element_count} elem · {c.region}</option>)}
        </select>
        {commit && (
          <span className="text-ink-3 flex gap-2">
            {Object.entries(commit.formats).map(([k, uri]) => <span key={k} className="text-info" title={uri}>{k}</span>)}
            <span>calib {commit.calibration_version}</span>
          </span>
        )}
        {!commits.length && <span className="text-warn">no map commits yet (run hdmap fuse on a georef'd session)</span>}
      </div>

      <div className="flex flex-1 min-h-0">
        <div ref={wrap} className="flex-1" />
        <aside className="w-72 border-l hairline p-3 font-mono text-[11px] overflow-auto">
          <div className="text-ink-3 uppercase text-[10px] mb-2">element provenance</div>
          {prov?.found ? (
            <div className="space-y-1.5">
              <div className="text-ink-2">{prov.kind} <span className="text-ink-3">{prov.element_id?.slice(0, 8)}</span></div>
              <div className="text-ink-3">confidence {prov.confidence?.toFixed(2)}</div>
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
        </aside>
      </div>
    </div>
  );
}
