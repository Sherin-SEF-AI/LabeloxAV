/**
 * The basemap must not ask OSM for tiles it does not serve.
 *
 * MapLibre defaults a raster source to maxzoom 22. OSM stops at 19 and answers anything deeper with a 400,
 * and a 400 carries no `Access-Control-Allow-Origin`, so the browser reports a CORS violation. The console
 * then blames cross-origin policy for what is really a request for a tile that has never existed, which is
 * a long way from the actual cause. Verified against the live server: z19 returns 200 image/png with
 * `access-control-allow-origin: *`, z20 and z21 return 400 with no such header.
 *
 * Pinned because the failure only appears past a zoom level nobody reaches in a quick check, and because
 * removing `maxzoom` looks like deleting a redundant default.
 */

import { describe, expect, it } from "vitest";

import { OSM_ATTRIBUTION, OSM_MAX_ZOOM, OSM_STYLE } from "./basemap";

const source = OSM_STYLE.sources.osm as {
  type: string;
  tiles: string[];
  tileSize: number;
  maxzoom?: number;
  attribution?: string;
};

describe("OSM basemap style", () => {
  it("declares a maxzoom, rather than inheriting MapLibre's 22", () => {
    expect(source.maxzoom).toBeTypeOf("number");
  });

  it("stops at the deepest zoom OSM actually serves", () => {
    expect(source.maxzoom).toBe(19);
    expect(OSM_MAX_ZOOM).toBe(19);
  });

  it("never exceeds it, which is the request that 400s without a CORS header", () => {
    expect(source.maxzoom).toBeLessThanOrEqual(19);
  });

  it("carries the attribution OSM's tile policy requires", () => {
    expect(source.attribution).toBe(OSM_ATTRIBUTION);
    expect(source.attribution).toContain("OpenStreetMap");
  });

  it("still points at the OSM tile template with the standard 256px tiles", () => {
    expect(source.type).toBe("raster");
    expect(source.tiles).toEqual(["https://tile.openstreetmap.org/{z}/{x}/{y}.png"]);
    expect(source.tileSize).toBe(256);
  });

  it("draws the source it declares", () => {
    expect(OSM_STYLE.layers).toHaveLength(1);
    expect(OSM_STYLE.layers[0]).toMatchObject({ id: "osm", type: "raster", source: "osm" });
  });
});
