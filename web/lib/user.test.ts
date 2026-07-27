import { describe, expect, it } from "vitest";

import { decodeExp } from "./user";

// Build an lbx2 token with the given payload. Only the payload segment matters to decodeExp; the signature is
// never verified client-side, so a placeholder is fine here.
function lbx2(payload: Record<string, unknown>): string {
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  return `lbx2.${body}.deadbeef`;
}

describe("decodeExp", () => {
  it("reads exp from a v2 token payload", () => {
    expect(decodeExp(lbx2({ uid: "u1", exp: 1893456000, tv: 1 }))).toBe(1893456000);
  });

  it("returns null for a legacy lbx1 token", () => {
    expect(decodeExp("lbx1.abc.def")).toBeNull();
  });

  it("returns null when the payload has no exp", () => {
    expect(decodeExp(lbx2({ uid: "u1", tv: 1 }))).toBeNull();
  });

  it("returns null for a malformed or absent token", () => {
    expect(decodeExp("garbage")).toBeNull();
    expect(decodeExp("lbx2.not-base64-@@@.sig")).toBeNull();
    expect(decodeExp(undefined)).toBeNull();
    expect(decodeExp("")).toBeNull();
  });
});
