/**
 * A browser without WebGL gets a map panel that says so, not a crashed page.
 *
 * The error below is the one MapLibre threw on a machine with hardware acceleration off; it unmounted the
 * HD map page. Pinned because the failure only shows on a browser without WebGL, which no developer machine
 * running these tests is likely to be.
 */

import { describe, expect, it, vi } from "vitest";

vi.mock("maplibre-gl", () => ({
  default: {
    Map: class {
      constructor() {
        throw new Error(JSON.stringify({
          statusMessage: "Could not create a WebGL context, GL_VENDOR = Disabled, GL_RENDERER = Disabled",
          type: "webglcontextcreationerror",
          message: "Failed to initialize WebGL",
        }));
      }
    },
  },
}));

import { NO_WEBGL, mapErrorMessage, tryCreateMap } from "./mapgl";

describe("tryCreateMap", () => {
  it("returns the WebGL message instead of throwing", () => {
    const r = tryCreateMap({ container: "x" } as never);
    expect(r.map).toBeNull();
    expect(r.error).toBe(NO_WEBGL);
  });

  it("passes any other failure through with its own text", () => {
    expect(mapErrorMessage(new Error("container 'x' not found"))).toBe(
      "The map could not start: container 'x' not found");
  });
});
