"use client";

import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useClock } from "@/lib/inspector/clock";
import { useMcap } from "@/lib/inspector/mcapContext";

// Map panel (MapLibre): the GNSS track with the current position advancing on the clock. Works offline with
// a plain background style (the track is the data); a raster basemap can be layered in later.

const STYLE: maplibregl.StyleSpecification = {
  version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#0F1113" } }],
};

export default function MapPanel() {
  const clock = useClock();
  const { mcap } = useMcap();
  const hostRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const markerRef = useRef<maplibregl.Marker | null>(null);
  const trackRef = useRef<{ t: number[]; lat: number[]; lon: number[] } | null>(null);
  const boundsRef = useRef<maplibregl.LngLatBounds | null>(null);
  const [note, setNote] = useState("locating GNSS...");

  useEffect(() => {
    let alive = true;
    const gnss = mcap.topics().find((t) => /locationfix/i.test(t.schema) || /gnss|gps/i.test(t.topic));
    if (!gnss) { setNote("no GNSS topic"); return; }
    const map = new maplibregl.Map({ container: hostRef.current!, style: STYLE, attributionControl: false, center: [0, 0], zoom: 1 });
    mapRef.current = map;
    // the grid cell sizes after the map is created, so a map made at 0x0 renders blank; keep it sized.
    const ro = new ResizeObserver(() => {
      map.resize();
      if (boundsRef.current) map.fitBounds(boundsRef.current, { padding: 30, maxZoom: 17, duration: 0 });
    });
    ro.observe(hostRef.current!);
    map.on("load", async () => {
      try {
        const { t, values } = await mcap.collect(gnss.topic, clock.startNs);
        if (!alive) return;
        const lat = values["latitude"] ?? [];
        const lon = values["longitude"] ?? [];
        const coords = lon.map((x, i) => [x, lat[i]] as [number, number]).filter((c) => Number.isFinite(c[0]) && Number.isFinite(c[1]) && (c[0] !== 0 || c[1] !== 0));
        if (coords.length === 0) { setNote("no valid GNSS positions"); return; }
        setNote("");
        map.addSource("track", { type: "geojson", data: { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: coords } } });
        map.addLayer({ id: "track", type: "line", source: "track", paint: { "line-color": "#FF7A2F", "line-width": 2 } });
        const el = document.createElement("div");
        el.style.cssText = "width:10px;height:10px;border-radius:50%;background:#58A6FF;border:2px solid #fff;box-shadow:0 0 6px #58A6FF";
        markerRef.current = new maplibregl.Marker({ element: el }).setLngLat(coords[0]).addTo(map);
        const b = coords.reduce((acc, c) => acc.extend(c), new maplibregl.LngLatBounds(coords[0], coords[0]));
        boundsRef.current = b;
        map.resize();
        map.fitBounds(b, { padding: 30, maxZoom: 17, duration: 0 });
        trackRef.current = { t, lat, lon };
      } catch (e) {
        if (alive) setNote("map load error: " + String(e));
      }
    });
    return () => { alive = false; ro.disconnect(); map.remove(); mapRef.current = null; };
  }, [mcap, clock.startNs]);

  useEffect(() => {
    return clock.subscribe((offsetSec) => {
      const tr = trackRef.current;
      const mk = markerRef.current;
      if (!tr || !mk || tr.t.length === 0) return;
      // nearest sample at or before the clock
      let i = 0;
      while (i < tr.t.length - 1 && tr.t[i + 1] <= offsetSec) i++;
      const lat = tr.lat[i], lon = tr.lon[i];
      if (Number.isFinite(lat) && Number.isFinite(lon)) mk.setLngLat([lon, lat]);
    });
  }, [clock]);

  return (
    <div className="relative h-full w-full">
      <div ref={hostRef} className="absolute inset-0" />
      {note && <div className="absolute inset-0 flex items-center justify-center font-mono text-[10px] text-ink-3 pointer-events-none">{note}</div>}
    </div>
  );
}
