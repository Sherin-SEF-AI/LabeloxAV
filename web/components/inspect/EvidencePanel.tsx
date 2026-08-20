"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { api, humanizeError } from "@/lib/api";
import type { ObjectDetail } from "@/lib/types";
import LoadState from "@/components/shell/LoadState";
import { toast, toastError, toastSuccess } from "@/lib/toast";

// The evidence behind a row, next to the decision it is asking for.
//
// Several console tables name objects and never show them. The fix queue is the worst of them: a paragraph
// of VLM prose - "the image shows a building facade with signs and overgrown vegetation, not any type of
// vehicle" - with confirm and dismiss beside it and thousands of rows pending. The only way to check the
// claim was to believe it, and several of those verdicts turn on context the prose asserts and the reader
// could not see.
//
// So: the crop first, because it answers most rows on its own; then the whole frame with the box outlined,
// because the rows the crop cannot settle are exactly the ones that are about the surroundings; then the
// text; then the tools. Deciding while looking at the image is the entire point, which is why the actions
// live here rather than back on the row.
//
// The box overlay is an SVG in IMAGE pixel coordinates over an object-contain img, both letterboxed by the
// browser (components/editor/RigView.tsx does the same). No scaling arithmetic, so nothing drifts when the
// panel is resized. strokeWidth scales with the frame because the units are image pixels: a fixed width is
// a hairline on a 4K frame, which is the bug in components/inspector/ImagePanel.tsx.
//
// Images authenticate by the lbx_media cookie, not a Bearer header, which an img tag cannot send. That is
// handled globally (lib/user.ts mints it before any api call returns), so a plain img works here - but only
// for paths listed in services/api/media.py. Do not point this at a new image route without adding it there.

export type EvidenceAction = {
  key: string;
  label: string;
  tone?: "accept" | "reject" | "neutral";
  hint?: string;
  run: () => Promise<void>;
};

export type EvidenceSubject = {
  objectId: string;
  /** Suggested replacement class, when the row proposes one. Renders the apply button. */
  suggestion?: { class_id: number; class_name: string } | null;
  /** The row's own words: the VLM reason, the refusal, whatever the table was showing. */
  text?: string | null;
  kind?: string | null;
  score?: number | null;
  /** Extra key/value context from the row, shown verbatim under the text. */
  detail?: Record<string, unknown> | null;
};

function Tile({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3">{label}</div>
      {children}
    </div>
  );
}

export default function EvidencePanel({ subject, actions = [], onResolved }: {
  subject: EvidenceSubject | null;
  actions?: EvidenceAction[];
  /** Called after an action succeeds, so the list can drop the row and advance the cursor. */
  onResolved?: (objectId: string) => void;
}) {
  const router = useRouter();
  const [obj, setObj] = useState<ObjectDetail | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [cropFailed, setCropFailed] = useState(false);
  const [frameFailed, setFrameFailed] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<number[] | null>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);

  const objectId = subject?.objectId ?? null;

  useEffect(() => {
    setObj(null);
    setErr(null);
    setCropFailed(false);
    setFrameFailed(false);
    setEditing(false);
    setDraft(null);
    if (!objectId) return;
    let live = true;
    setLoading(true);
    api.object(objectId)
      .then((o) => { if (live) setObj(o); })
      .catch((e) => { if (live) setErr(e); })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [objectId]);

  const runAction = useCallback(async (a: EvidenceAction) => {
    if (!objectId) return;
    setBusy(a.key);
    try {
      await a.run();
      onResolved?.(objectId);
    } catch (e) {
      toastError(`${a.label} failed: ${humanizeError(e)}`);
    } finally {
      setBusy(null);
    }
  }, [objectId, onResolved]);

  // Apply the class the row proposes. Through the same review path the editor uses, so the role clamp and
  // the optimistic-concurrency check apply: two people acting on one object cannot silently overwrite each
  // other, and an annotator cannot write an accepted-grade row.
  const applySuggestion = useCallback(async () => {
    if (!obj || !subject?.suggestion) return;
    const was = obj.class_name;
    const to = subject.suggestion.class_name;
    setBusy("apply");
    try {
      const next = await api.review(obj.object_id, {
        action: "reclassify", class_name: to, state: "accepted",
        expected_version: obj.version,
      });
      setObj(next as ObjectDetail);
      toast(`${was} -> ${to}`, "success", 12000, {
        label: "undo",
        run: async () => {
          try {
            const back = await api.review(obj.object_id, {
              action: "reclassify", class_name: was, state: "accepted",
              expected_version: (next as ObjectDetail).version,
            });
            setObj(back as ObjectDetail);
            toastSuccess(`restored to ${was}`);
          } catch (e) {
            toastError(`undo failed: ${humanizeError(e)}`);
          }
        },
      });
      onResolved?.(obj.object_id);
    } catch (e) {
      // A stale version is the interesting failure and reads as a plain 409 otherwise: somebody else moved
      // this object while it was open, so saying so beats "request failed".
      toastError(`could not apply ${to}: ${humanizeError(e)}`);
    } finally {
      setBusy(null);
    }
  }, [obj, subject, onResolved]);

  const saveBox = useCallback(async () => {
    if (!obj || !draft) return;
    setBusy("box");
    try {
      const before = obj.bbox;
      const next = await api.review(obj.object_id, {
        action: "adjust", bbox: draft, state: "accepted", expected_version: obj.version,
      });
      setObj(next as ObjectDetail);
      setEditing(false);
      setDraft(null);
      toast("box adjusted", "success", 12000, {
        label: "undo",
        run: async () => {
          try {
            setObj(await api.review(obj.object_id, {
              action: "adjust", bbox: before, state: "accepted",
              expected_version: (next as ObjectDetail).version,
            }) as ObjectDetail);
            toastSuccess("box restored");
          } catch (e) { toastError(`undo failed: ${humanizeError(e)}`); }
        },
      });
    } catch (e) {
      toastError(`could not save the box: ${humanizeError(e)}`);
    } finally {
      setBusy(null);
    }
  }, [obj, draft]);

  if (!subject) {
    return (
      <div className="p-4 font-mono text-[11px] text-ink-3 leading-relaxed">
        Select a row to see the object it is about.
        <div className="mt-2 text-ink-3">
          The crop and the frame it came from appear here, with the tools to act on it.
        </div>
      </div>
    );
  }

  const box = draft ?? obj?.bbox ?? null;

  // Pointer drag on the frame: convert client coords to image pixels through the rendered rect. The image is
  // object-contain, so the drawn area is letterboxed inside the element and the offsets have to come out.
  const toImage = (e: React.PointerEvent<SVGSVGElement>): [number, number] | null => {
    if (!obj) return null;
    const r = e.currentTarget.getBoundingClientRect();
    const scale = Math.min(r.width / obj.width, r.height / obj.height);
    const ox = (r.width - obj.width * scale) / 2;
    const oy = (r.height - obj.height * scale) / 2;
    return [(e.clientX - r.left - ox) / scale, (e.clientY - r.top - oy) / scale];
  };

  return (
    <div className="p-3 space-y-3">
      {err != null ? (
        <LoadState error={err} onRetry={() => { if (objectId) { setErr(null); setLoading(true); api.object(objectId).then(setObj).catch(setErr).finally(() => setLoading(false)); } }} />
      ) : loading || !obj ? (
        <div className="font-mono text-[11px] text-ink-3/60 animate-pulse py-6 text-center">loading object...</div>
      ) : (
        <>
          <Tile label="the object">
            {cropFailed ? (
              <div className="h-32 grid place-items-center border hairline bg-bg-2 font-mono text-[10px] text-ink-3 text-center px-2">
                the crop could not be loaded
                <br />
                <span className="text-ink-3">the image may be missing from storage, or your role may not reach it</span>
              </div>
            ) : (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={`/api/objects/${obj.object_id}/crop?pad=0.25`} alt={obj.class_name}
                onError={() => setCropFailed(true)}
                className="w-full max-h-48 object-contain bg-bg-2 border hairline" />
            )}
            <div className="flex items-center gap-2 font-mono text-[10px] text-ink-3">
              <span className="text-ink">{obj.class_name}</span>
              <span>conf {Number(obj.conf ?? 0).toFixed(2)}</span>
              <span>{obj.state}</span>
              <span className="ml-auto">{obj.source}</span>
            </div>
          </Tile>

          <Tile label={editing ? "drag on the frame to redraw the box" : "in its frame"}>
            <div className="relative w-full bg-bg-2 border hairline">
              {frameFailed ? (
                <div className="h-40 grid place-items-center font-mono text-[10px] text-ink-3">
                  the frame could not be loaded
                </div>
              ) : (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={obj.image_url} alt="frame" onError={() => setFrameFailed(true)}
                    className="w-full max-h-56 object-contain" />
                  <svg viewBox={`0 0 ${obj.width} ${obj.height}`} preserveAspectRatio="xMidYMid meet"
                    onPointerDown={(e) => {
                      if (!editing) return;
                      const p = toImage(e);
                      if (!p) return;
                      e.currentTarget.setPointerCapture(e.pointerId);
                      drag.current = { x: p[0], y: p[1] };
                      setDraft([p[0], p[1], p[0], p[1]]);
                    }}
                    onPointerMove={(e) => {
                      if (!editing || !drag.current) return;
                      const p = toImage(e);
                      if (!p) return;
                      const { x, y } = drag.current;
                      setDraft([Math.min(x, p[0]), Math.min(y, p[1]), Math.max(x, p[0]), Math.max(y, p[1])]);
                    }}
                    onPointerUp={() => { drag.current = null; }}
                    className={`absolute inset-0 h-full w-full ${editing ? "cursor-crosshair" : "pointer-events-none"}`}>
                    {box && (
                      <rect x={box[0]} y={box[1]} width={Math.max(0, box[2] - box[0])} height={Math.max(0, box[3] - box[1])}
                        fill="none" stroke={editing ? "#5b86c7" : "#e0a63f"}
                        // Image-pixel units, so a constant width would vanish on a large frame.
                        strokeWidth={Math.max(obj.width, obj.height) / 320} opacity={0.95} />
                    )}
                  </svg>
                </>
              )}
            </div>
          </Tile>

          {subject.text && (
            <Tile label={subject.kind ? `why (${subject.kind})` : "why"}>
              <p className="text-xs text-ink-2 leading-relaxed">{subject.text}</p>
              {subject.score != null && (
                <div className="font-mono text-[10px] text-ink-3">score {subject.score.toFixed(2)}</div>
              )}
            </Tile>
          )}

          <div className="space-y-1.5 pt-1">
            {subject.suggestion && (
              <button onClick={applySuggestion} disabled={!!busy || obj.class_name === subject.suggestion.class_name}
                className="w-full border border-accent/50 bg-accent/10 text-accent px-3 py-1.5 rounded font-mono text-[11px] hover:bg-accent/20 disabled:opacity-40">
                {obj.class_name === subject.suggestion.class_name
                  ? `already ${subject.suggestion.class_name}`
                  : busy === "apply" ? "applying..." : `relabel as ${subject.suggestion.class_name}`}
              </button>
            )}

            <div className="flex gap-1.5">
              {actions.map((a) => (
                <button key={a.key} onClick={() => void runAction(a)} disabled={!!busy} title={a.hint}
                  className={`flex-1 border px-2 py-1 rounded font-mono text-[11px] disabled:opacity-40 ${
                    a.tone === "accept" ? "border-pass text-pass hover:bg-pass/10"
                      : a.tone === "reject" ? "border-line text-ink-3 hover:border-block hover:text-block"
                        : "border-line text-ink-3 hover:border-accent"}`}>
                  {busy === a.key ? "..." : a.label}
                </button>
              ))}
            </div>

            <div className="flex gap-1.5">
              {editing ? (
                <>
                  <button onClick={() => void saveBox()} disabled={!!busy || !draft}
                    className="flex-1 border border-accent/50 bg-accent/10 text-accent px-2 py-1 rounded font-mono text-[11px] disabled:opacity-40">
                    {busy === "box" ? "saving..." : "save box"}
                  </button>
                  <button onClick={() => { setEditing(false); setDraft(null); }} disabled={!!busy}
                    className="flex-1 border border-line text-ink-3 px-2 py-1 rounded font-mono text-[11px] hover:border-accent disabled:opacity-40">
                    cancel
                  </button>
                </>
              ) : (
                <button onClick={() => setEditing(true)} disabled={!!busy || frameFailed}
                  className="flex-1 border border-line text-ink-3 px-2 py-1 rounded font-mono text-[11px] hover:border-accent disabled:opacity-40">
                  adjust box
                </button>
              )}
              <button onClick={() => router.push(`/frame/${obj.frame_id}?focus=${obj.object_id}`)}
                title="everything this panel cannot do"
                className="flex-1 border border-line text-ink-3 px-2 py-1 rounded font-mono text-[11px] hover:border-accent">
                open editor
              </button>
            </div>
          </div>

          <div className="font-mono text-[9px] text-ink-3/70 pt-1 break-all">
            object {obj.object_id.slice(0, 8)} &middot; frame {obj.frame_id.slice(0, 8)} &middot; cam {obj.cam_id}
          </div>
        </>
      )}
    </div>
  );
}
