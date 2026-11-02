// Thin JSON helpers for the Data Engine (M10-M19) plane surfaces. The api.ts get/post are module-private; these
// give the new plane pages a shared, auth-header-aware fetch without widening api.ts. Deny-by-default auth means
// every call carries the current user header.

import { userHeaders } from "./user";

export async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path, { cache: "no-store", headers: { ...userHeaders() } });
  if (!r.ok) throw new Error(`GET ${path} -> ${r.status}`);
  return r.json();
}

export async function runJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...userHeaders() },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json();
}

/** Parse a pasted JSON field, with an error a person can act on instead of "Unexpected end of JSON input". */
export function parseInput<T = unknown>(text: string, what = "this field"): T {
  if (!text.trim()) throw new Error(`Paste ${what} as JSON first. The greyed example shows the expected shape.`);
  try {
    return JSON.parse(text) as T;
  } catch (e) {
    throw new Error(`${what} is not valid JSON: ${e instanceof Error ? e.message : String(e)}`);
  }
}
