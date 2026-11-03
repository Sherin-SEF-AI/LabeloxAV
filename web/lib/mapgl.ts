/**
 * Creating a MapLibre map without letting a missing WebGL take the page down with it.
 *
 * `new maplibregl.Map()` throws synchronously when the browser cannot create a WebGL context: hardware
 * acceleration switched off, a GPU process that crashed, a blocklisted driver. Thrown inside an effect, that
 * error unmounted the whole HD map page, so a browser setting read as a broken application. The map is one
 * panel; the commit list and provenance beside it work without WebGL, so the failure stays inside the panel
 * and says what to change.
 */

import maplibregl from "maplibre-gl";

export const NO_WEBGL =
  "The map needs WebGL, and this browser could not start it. Turn on hardware acceleration " +
  "(Chrome: Settings, System, \"Use graphics acceleration when available\"), then reload. " +
  "Everything else on this page works without it.";

/** What to show for an error thrown while creating a map. */
export function mapErrorMessage(e: unknown): string {
  const text = e instanceof Error ? e.message : String(e);
  // MapLibre's message is the JSON of the webglcontextcreationerror event.
  if (/webglcontextcreationerror|webgl/i.test(text)) return NO_WEBGL;
  return `The map could not start: ${text}`;
}

export function tryCreateMap(
  options: maplibregl.MapOptions,
): { map: maplibregl.Map; error: null } | { map: null; error: string } {
  try {
    return { map: new maplibregl.Map(options), error: null };
  } catch (e) {
    return { map: null, error: mapErrorMessage(e) };
  }
}
