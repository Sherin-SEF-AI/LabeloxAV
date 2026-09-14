"use client";

// Everything the panel knows about the one object under the cursor: what it is, how sure the model was,
// who last wrote it, what geometry it carries, why it was labelled that way, what it is linked to, and its
// ontology attributes.
//
// The provenance block exists because the panel used to name a class and a confidence and nothing else. An
// object's version and which of the six geometry kinds it actually holds are the two facts that decide
// whether an edit will collide with someone else's and whether the shape on canvas is the shape on the
// server, and neither was visible anywhere.

import { useState } from "react";
import { useRouter } from "next/navigation";

import { api } from "@/lib/api";
import { attrsForClass } from "@/lib/attrScope";
import { classColor } from "@/lib/colors";
import { trackOp } from "@/lib/canvasOps";
import { toast } from "@/lib/toast";
import type { Ontology, Relationship } from "@/lib/types";
import ExplainPanel from "@/components/ExplainPanel";
import { ConfBar, StateBadge } from "@/components/StateBadge";

import AttrControl from "./AttrControl";
import { QUALITY_GOOD, QUALITY_WEAK, qualityTone } from "./panelStats";
import type { EdObject } from "../useEditor";

// Directed object-relationship kinds offered in the editor. rider_of is the India two-wheeler case: a
// person on a motorcycle is two objects and one road user, and every downstream count is wrong if the
// relationship is not recorded.
const RELATION_KINDS = ["rider_of", "towed_by", "part_of", "member_of", "occludes"];

const QUALITY_CLASS = {
  good: "border-pass/50 text-pass", weak: "border-warn/50 text-warn", bad: "border-block/50 text-block",
} as const;

export default function SelectedObjectCard({
  object, onto, relationships, linkKind, linkFrom,
  onLinkKind, onToggleLink, onDeleteRelationship, onSetAttr, onFitCuboid, onSetYaw, onClearAmodal,
}: {
  object: EdObject;
  onto: Ontology;
  relationships: Relationship[];
  linkKind: string;
  linkFrom: string | null;
  onLinkKind: (k: string) => void;
  onToggleLink: () => void;
  onDeleteRelationship: (rid: string) => void;
  onSetAttr: (name: string, val: unknown) => void;
  /** Fit a cuboid to this object's 2D box, using the monocular solve. */
  onFitCuboid?: () => void;
  /** Set the cuboid's yaw, in radians. */
  onSetYaw?: (yaw: number) => void;
  /** Take back the whole-extent box. */
  onClearAmodal?: () => void;
}) {
  const router = useRouter();
  const [explainOpen, setExplainOpen] = useState(false);

  const geometry = ([
    ["box", object.bbox?.length === 4],
    ["mask", object.mask.length > 0],
    ["polyline", !!object.polyline?.length],
    ["pose", !!object.keypoints],
    ["3D", !!object.cuboid_3d],
    ["whole extent", !!object.bbox_amodal?.length],
    ["rotated", !!object.rot],
  ] as [string, boolean][]).filter(([, on]) => on);

  // Shared with the server's attrs_for_class rather than reimplemented: l1 alone stopped being the whole
  // answer once per-class extras existed, and an offered attribute the server rejects is a 400 with no
  // explanation an annotator can act on.
  const allowedAttrs = attrsForClass(onto, object.class_id) ?? undefined;

  return (
    <div className="p-2">
      <div className="flex items-center justify-between mb-1">
        <span className="font-mono text-[10px] uppercase tracking-wide text-ink-3">attributes</span>
        {object.track_id && (
          <button onClick={() => router.push(`/track/${object.track_id}`)}
            className="font-mono text-[10px] text-info hover:text-accent">view track &rarr;</button>
        )}
      </div>

      <div className="flex items-center gap-2 mb-1.5">
        <span className="w-2.5 h-2.5 inline-block shrink-0 rounded-sm" style={{ background: classColor(object.class_id) }} />
        <span className="font-mono text-[11px] text-ink truncate flex-1">{object.class_name}</span>
        {object.quality_score != null && (
          <span title={`M-F.1 label quality score (0-1): calibrated correctness, penalised for geometric/consistency defects. Good at ${QUALITY_GOOD}, weak at ${QUALITY_WEAK}.`}
            className={`font-mono text-[10px] px-1 rounded border ${QUALITY_CLASS[qualityTone(object.quality_score)]}`}>
            Q {object.quality_score.toFixed(2)}
          </span>
        )}
        <ConfBar conf={object.conf} />
        <StateBadge state={object.state} />
      </div>

      {/* Real identity, version and geometry. No fabricated detector names: what is not known is not shown. */}
      <div className="flex flex-col gap-1.5 bg-bg-2 border border-line rounded p-2 mb-1.5 font-mono text-[10px]">
        <div className="flex items-center">
          <span className="text-ink-3 w-16 shrink-0">object</span>
          <span className="text-ink-2 truncate">{object.isNew ? "new (unsaved)" : object.id.slice(0, 12)}</span>
        </div>
        <div className="flex items-center">
          <span className="text-ink-3 w-16 shrink-0">track</span>
          {object.track_id
            ? <button onClick={() => router.push(`/track/${object.track_id}`)}
                className="text-info hover:text-accent truncate">{object.track_id.slice(0, 12)} &rarr;</button>
            : <span className="text-ink-3">none</span>}
        </div>
        <div className="flex items-center">
          <span className="text-ink-3 w-16 shrink-0">version</span>
          <span className="text-ink-2">{object.version ?? "-"}</span>
        </div>
        <div className="flex items-start gap-1">
          <span className="text-ink-3 w-16 shrink-0 pt-0.5">geometry</span>
          <div className="flex flex-wrap gap-1">
            {geometry.map(([k]) => (
              <span key={k} className="text-ink-2 bg-line/40 border border-line rounded px-1.5 py-0.5">{k}</span>
            ))}
          </div>
        </div>
      </div>

      {/* The whole extent of a partly hidden object, when somebody has judged one.
          Shown separately from the geometry chips above because it is the one piece of geometry here
          that is a claim about what CANNOT be seen, and it is worth being able to take back. */}
      {object.bbox_amodal?.length === 4 && (
        <div className="flex items-center gap-2 bg-bg-2 border border-line rounded p-2 mb-1.5 font-mono text-[10px]">
          <span className="text-ink-3 w-16 shrink-0">extent</span>
          <span className="text-ink-2 truncate">
            {Math.round(object.bbox_amodal[2] - object.bbox_amodal[0])} x{" "}
            {Math.round(object.bbox_amodal[3] - object.bbox_amodal[1])} px
            {object.bbox?.length === 4 && (
              <span className="text-ink-3">
                {" "}({Math.round(
                  (100 * ((object.bbox_amodal[2] - object.bbox_amodal[0]) *
                          (object.bbox_amodal[3] - object.bbox_amodal[1]))) /
                  Math.max(1, (object.bbox[2] - object.bbox[0]) * (object.bbox[3] - object.bbox[1])) - 100,
                )}% larger than what is visible)
              </span>
            )}
          </span>
          {onClearAmodal && (
            <button onClick={onClearAmodal}
              title="take back the whole-extent box; the visible box is unaffected"
              className="ml-auto shrink-0 border border-line rounded px-1.5 py-0.5 text-ink-3 hover:border-block hover:text-block">
              clear
            </button>
          )}
        </div>
      )}

      {/* The 3D box, and the way to get one.
          The monocular solve has existed since cuboids were added and was reachable only as a frame-wide
          agent run, so the one place an annotator works on a single object could not ask for it. */}
      {(onFitCuboid || object.cuboid_3d) && (
        <div className="flex flex-col gap-1.5 bg-bg-2 border border-line rounded p-2 mb-1.5 font-mono text-[10px]">
          <div className="flex items-center gap-2">
            <span className="text-ink-3 w-16 shrink-0">cuboid</span>
            {object.cuboid_3d ? (
              <span className="text-ink-2 truncate">
                {object.cuboid_3d.size.map((v) => v.toFixed(1)).join(" x ")} m at{" "}
                {object.cuboid_3d.center[0].toFixed(1)}, {object.cuboid_3d.center[1].toFixed(1)}
              </span>
            ) : (
              <span className="text-ink-3">none</span>
            )}
            {onFitCuboid && (
              <button onClick={onFitCuboid}
                title="lift this box to a 3D cuboid: ground contact, class size prior, yaw fitted then snapped to the lane"
                className="ml-auto shrink-0 border border-line rounded px-1.5 py-0.5 text-ink-2 hover:border-accent hover:text-accent">
                {object.cuboid_3d ? "re-fit" : "fit"}
              </button>
            )}
          </div>
          {object.cuboid_3d && onSetYaw && (
            <div className="flex items-center gap-2">
              <span className="text-ink-3 w-16 shrink-0">yaw</span>
              <input type="range" min={-Math.PI} max={Math.PI} step={0.01}
                value={object.cuboid_3d.yaw}
                onChange={(e) => onSetYaw(Number(e.target.value))}
                aria-label="cuboid yaw"
                className="flex-1" />
              <span className="text-ink-2 tabular-nums w-10 text-right">
                {Math.round((object.cuboid_3d.yaw * 180) / Math.PI)}&deg;
              </span>
            </div>
          )}
          {/* Where the yaw came from. A reprojection-fitted yaw is an inference from a silhouette and the
              image box of a car seen head-on is nearly the box of one seen from behind, so it is worth
              knowing which yaws were measured from the road instead. */}
          {object.cuboid_3d?.yaw_source && (
            <div className="flex items-center">
              <span className="text-ink-3 w-16 shrink-0">yaw from</span>
              <span className={object.cuboid_3d.yaw_source === "lane" ? "text-accent" : "text-ink-2"}>
                {object.cuboid_3d.yaw_source === "lane" ? "the lane direction"
                  : object.cuboid_3d.yaw_source === "reprojection" ? "reprojection (weak: a head-on and a rear view look alike)"
                  : "not set"}
              </span>
            </div>
          )}
        </div>
      )}

      {/* M-F.0: why this label was decided the way it was, from real provenance. A new object has none. */}
      {!object.isNew && (
        <div className="mb-1.5">
          <button onClick={() => setExplainOpen((v) => !v)} aria-expanded={explainOpen}
            className="w-full flex items-center justify-between font-mono text-[10px] uppercase tracking-wide text-ink-3 border border-line rounded px-1.5 py-1 hover:border-accent">
            <span>why this label</span><span aria-hidden>{explainOpen ? "−" : "+"}</span>
          </button>
          {explainOpen && (
            <div className="mt-1.5 bg-bg-2 border border-line rounded p-2">
              <ExplainPanel objectId={object.id} />
            </div>
          )}
        </div>
      )}

      <button
        disabled={object.isNew}
        title={object.isNew
          ? "save the frame first, then propagate"
          : "optical-flow propagate this box across the next 12 frames as a track to confirm"}
        onClick={async () => {
          const r = await trackOp("propagate", "propagate object", () => api.propagateObject(object.id, 12));
          toast(r.created
            ? `propagated forward ${r.created} frames (track ${r.track_id?.slice(0, 8)}). Open the track to review/confirm.`
            : `could not propagate: ${r.reason || "no motion"}`);
        }}
        className="w-full mb-1 font-mono text-[10px] border border-line rounded text-ink-2 px-1.5 py-1 hover:border-accent disabled:opacity-40">
        propagate forward 12 frames &rarr;
      </button>

      {/* Relationships: pick a kind, press link, then click the target object on the canvas. */}
      <div className="mb-1 space-y-1">
        <div className="flex items-center gap-1">
          <select value={linkKind} onChange={(e) => onLinkKind(e.target.value)} aria-label="relationship kind"
            className="flex-1 bg-bg border border-line rounded px-1 py-0.5 font-mono text-[10px] text-ink">
            {RELATION_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
          <button onClick={onToggleLink}
            className={`font-mono text-[10px] border rounded px-1.5 py-0.5 ${linkFrom === object.id ? "border-accent text-accent" : "border-line text-ink-2 hover:border-accent"}`}>
            {linkFrom === object.id ? "click target" : "link"}
          </button>
        </div>
        {relationships
          .filter((r) => r.from_object_id === object.id || r.to_object_id === object.id)
          .map((r) => (
            <div key={r.relationship_id} className="flex items-center gap-1 font-mono text-[10px] text-ink-3">
              <span className="flex-1 truncate">
                {r.from_object_id === object.id
                  ? `${r.kind} ${r.to_object_id.slice(0, 8)}`
                  : `${r.from_object_id.slice(0, 8)} ${r.kind}`}
              </span>
              <button onClick={() => onDeleteRelationship(r.relationship_id)}
                className="hover:text-block" title="remove" aria-label="remove relationship">x</button>
            </div>
          ))}
      </div>

      <div className="space-y-1">
        {Object.entries(onto.attributes)
          // Only the attributes applicable to this object's class, by its l1 subclass. A subclass with no
          // scope entry shows all of them rather than none, so a gap in the ontology does not read as an
          // object that has no attributes.
          .filter(([name]) => !allowedAttrs || allowedAttrs.includes(name))
          .map(([name, spec]) => (
            <AttrControl key={name} name={name} spec={spec} value={object.attrs[name]}
              onChange={(val) => onSetAttr(name, val)} />
          ))}
      </div>
    </div>
  );
}
