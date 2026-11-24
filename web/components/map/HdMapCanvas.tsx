"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { api } from "@/lib/api";
import type { MapProvenance } from "@/lib/types";
import { OSM_STYLE } from "@/lib/basemap";
import { tryCreateMap } from "@/lib/mapgl";

// The MapLibre half of the HD map page: the basemap, the fused lanes and signs of one commit, and the click
// that traces an element to its provenance.
//
// It is its own module so the page can load it with next/dynamic. maplibre-gl is the largest single
// dependency this app has, and imported statically from app/map/page.tsx it sat in that route's first
// load; the commit picker and the provenance panel beside the map need none of it and now render while
// it arrives.

const STYLE = OSM_STYLE;

export default function HdMapCanvas({ commitId, onProvenance }: {
  commitId: string;
  onProvenance: (p: MapProvenance) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [mapErr, setMapErr] = useState<string | null>(null);
  // Read through a ref: the click handlers are registered once per render of a commit, and must reach
  // whatever callback the page has now.
  const onProv = useRef(onProvenance);
  onProv.current = onProvenance;

  useEffect(() => {
    if (!wrap.current || map.current) return;
    const { map: m, error } = tryCreateMap({ container: wrap.current, style: STYLE, center: [77.5946, 12.9716], zoom: 15 });
    if (!m) { setMapErr(error); return; }
    m.addControl(new maplibregl.NavigationControl(), "top-right");
    map.current = m;
  }, []);

  const render = useCallback(async (id: string) => {
    const m = map.current;
    if (!m || !id) return;
    const fc = await api.hdmapElements(`commit_id=${id}`);
    const draw = () => {
      for (const layer of ["lanes", "signs", "lanes-hit"]) if (m.getLayer(layer)) m.removeLayer(layer);
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
    m.on("click", "lanes", (e) => { const el = e.features?.[0]?.properties?.element_id as string; if (el) api.hdmapProvenance(el).then((p) => onProv.current(p)); });
    m.on("click", "signs", (e) => { const el = e.features?.[0]?.properties?.element_id as string; if (el) api.hdmapProvenance(el).then((p) => onProv.current(p)); });
  }, []);

  useEffect(() => { if (commitId) render(commitId); }, [commitId, render]);

  return (
    <div className="relative flex-1">
      {/* MapLibre's stylesheet sets position: relative on its container, which beats `absolute`; a plain
          wrapper takes the position and the container only fills it, or it collapses to 0px tall. */}
      <div className="absolute inset-0"><div ref={wrap} className="h-full w-full" /></div>
      {mapErr && (
        <div role="alert" className="absolute inset-0 flex items-center justify-center p-6">
          <p className="max-w-md text-[12px] text-warn">{mapErr}</p>
        </div>
      )}
    </div>
  );
}
