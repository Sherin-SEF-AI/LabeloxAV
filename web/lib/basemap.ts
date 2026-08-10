/**
 * The OSM raster basemap, as data rather than as a literal inside a page component.
 *
 * `maxzoom` is the load-bearing part. OSM serves tiles to z19 and no further: z20 answers 400, and a 400
 * carries no `Access-Control-Allow-Origin`, so MapLibre's own default of 22 produced a console full of CORS
 * failures that named the symptom and not the cause. Measured against the live tile server:
 *
 *   z19 -> 200 image/png, access-control-allow-origin: *
 *   z20 -> 400, no CORS header
 *   z21 -> 400, no CORS header
 *
 * Declaring the real limit makes MapLibre upscale the z19 tile past that point, which is what "zoomed in
 * further than the basemap has detail" should look like.
 *
 * It lives here rather than in the map page because it is plain data worth testing without mounting a map,
 * and because the inspector's MapPanel is written to take a raster basemap later.
 */

import type { StyleSpecification } from "maplibre-gl";

/** The deepest zoom tile.openstreetmap.org actually serves. */
export const OSM_MAX_ZOOM = 19;

/** OSM's tile usage policy requires visible attribution from anything drawing their tiles. */
export const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

export const OSM_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      maxzoom: OSM_MAX_ZOOM,
      attribution: OSM_ATTRIBUTION,
    },
  },
  layers: [{ id: "osm", type: "raster", source: "osm" }],
};
