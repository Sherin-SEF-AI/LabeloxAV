"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import dynamic from "next/dynamic";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { api, lidarCloudPoints, type Cuboid3D, type LidarCloud, type LidarPoints , humanizeError } from "@/lib/api";
import type { ColorBy } from "@/components/lidar/PointCloudViewer";
import type { AdverseRegion, AlItem, DrivingEvent, ErrorCandidateRow, FrameMeta, LaneRow, ObjectDynamicsRow, Ontology, OntologyClass, ProjectedCuboid, Relationship } from "@/lib/types";
import { classColor } from "@/lib/colors";
import { acceptState, getUser, setUser } from "@/lib/user";
import { beginOp, endOp, resetOps, trackOp } from "@/lib/canvasOps";
import { openConsole } from "@/components/console/ConsoleModal";
import { simplifyMask, simplifyPolygon } from "@/lib/simplify";
import CanvasConsole from "@/components/editor/CanvasConsole";
import { isDirty, tmpId, useEditor, type EdObject, type Tool } from "@/components/editor/useEditor";
import { PERSON_17 } from "@/lib/skeleton";
import BackButton from "@/components/BackButton";
import { useConfirm } from "@/components/ConfirmProvider";
import { ObjectSourceBadge } from "@/components/SourceBadge";
import CorrectionModal, { type CorrectionChange } from "@/components/CorrectionModal";
import ToolStrip from "@/components/shell/ToolStrip";
import ModeRail from "@/components/shell/ModeRail";
import FloatingLayers from "@/components/shell/FloatingLayers";
import AgentPanel from "@/components/agent/AgentPanel";
import BulkEditBar from "@/components/agent/BulkEditBar";
import SceneGraphPanel from "@/components/editor/SceneGraphPanel";
import PanelSection from "@/components/editor/PanelSection";
import { StateBadge, ConfBar } from "@/components/StateBadge";
import ScoreBar from "@/components/shell/ScoreBar";
import Icon, { MODE_ICON } from "@/components/shell/Icon";
import ShortcutOverlay from "@/components/shell/ShortcutOverlay";
import IssuePanel from "@/components/labelops/IssuePanel";
import CloudControl from "@/components/shell/CloudControl";
import { MODES, type ToolGroup } from "@/lib/editor/registry";
import type { SelectHow } from "@/components/editor/useEditor";
import Filmstrip from "@/components/editor/Filmstrip";
import HistoryPanel from "@/components/editor/HistoryPanel";
import CursorReadout from "@/components/editor/CursorReadout";
import { camLabel } from "@/lib/editor/camLabel";
import { IS_DEMO_BUILD } from "@/lib/demoFlag";
import { setCursor as publishCursor } from "@/lib/editor/cursorStore";

// Frame-centric professional annotation editor. Pan/zoom canvas, draw + edit boxes, SAM-assisted masks,
// layers panel, class palette, attributes, keyboard-driven, batched save. Operational Materialism tokens.

// Wrap the import so next/dynamic's convertModule always gets a clean { default } and cannot mistake the
// module for a react-konva export on a StrictMode re-mount.
const EditorCanvas = dynamic(() => import("@/components/editor/EditorCanvas").then((m) => ({ default: m.default })), { ssr: false });
const RigView = dynamic(() => import("@/components/editor/RigView"), { ssr: false });
const RigIdentityPanel = dynamic(() => import("@/components/editor/RigIdentityPanel"), { ssr: false });
const ExplainPanel = dynamic(() => import("@/components/ExplainPanel"), { ssr: false });
const RigTrackPanel = dynamic(() => import("@/components/editor/RigTrackPanel"), { ssr: false });
// Lanes mode swaps to this fit-to-width Konva stage (the folded-in lane editor). Loaded once, ssr off.
const LaneCanvas = dynamic(() => import("@/components/lane/LaneCanvas"), { ssr: false });
// 3D and LiDAR mode swaps to the three.js point cloud (the folded-in cuboid workspace). Loaded once, ssr off.
const PointCloudViewer = dynamic(() => import("@/components/lidar/PointCloudViewer"), { ssr: false });
const LANE_TYPES = ["solid", "dashed", "double", "road_edge", "implicit", "fallback"];
const CUBOID_DIMS: Record<string, number[]> = {
  sedan: [4.2, 1.8, 1.5], suv: [4.6, 1.9, 1.7], truck: [7.0, 2.5, 3.0], bus: [11.0, 2.6, 3.2],
  motorcycle: [2.0, 0.8, 1.4], pedestrian: [0.6, 0.6, 1.7], autorickshaw: [2.6, 1.4, 1.8],
};
const LANE_COLOR: Record<string, string> = { proposed: "#58A6FF", human: "#FF7A2F", propagated: "#E3B341" };

// Editor tools grouped so the strip renders one button per group (not 14 peers in a row). Tool keys match
// the editor's dispatch keys. The groups are split across modes: switching mode swaps which groups show,
// so each mode's strip stays short and one row. A new tool is one entry in a group's flyout.
const G = {
  select: { key: "select", label: "Select", tools: [{ key: "select", label: "select", hotkey: "V" }] },
  draw: { key: "draw", label: "Draw", tools: [
    { key: "box", label: "box", hotkey: "B" },
    { key: "polygon", label: "polygon", hotkey: "G" },
    { key: "polyline", label: "polyline", hotkey: "L" },
  ] },
  ai: { key: "ai", label: "AI assist", tools: [
    { key: "sam-point", label: "sam point", hotkey: "S" },
    { key: "sam-box", label: "sam box", hotkey: "M" },
    { key: "magic-wand", label: "wand", hotkey: "W" },
  ] },
  mask: { key: "mask", label: "Mask edit", tools: [
    { key: "brush", label: "brush", hotkey: "P" },
    { key: "eraser", label: "eraser", hotkey: "E" },
    { key: "superpixel", label: "cells", hotkey: "U" },
  ] },
  pose: { key: "pose", label: "Pose", tools: [{ key: "keypoint", label: "pose", hotkey: "K" }] },
  region: { key: "region", label: "Region", tools: [{ key: "adverse", label: "adverse", hotkey: "D" }] },
  cuboid: { key: "cuboid", label: "3D box", tools: [{ key: "cuboid", label: "cuboid", hotkey: "C" }] },
  measure: { key: "measure", label: "Measure", tools: [{ key: "measure", label: "measure", hotkey: "R" }] },
  // Semantic paints a dense class raster rather than creating objects. It reuses the polygon and brush the
  // object canvas already has, because a region drawn for a class and a region drawn for an instance are
  // the same gesture and teaching the annotator two of them would be gratuitous.
  semantic: { key: "semantic", label: "Semantic", tools: [
    { key: "sem-region", label: "region", hotkey: "G" },
    { key: "sem-erase", label: "erase", hotkey: "E" },
  ] },
} satisfies Record<string, ToolGroup>;

// Per-mode tool strips. The mode rail picks one; the strip renders only that mode's groups.
// Each mode lists only tools its canvas actually honors. Objects/Pose/Review use the Konva EditorCanvas
// (select, draw, AI, mask, region, the 2D cuboid placement, measure, keypoint all work there). Lanes
// (LaneCanvas) and 3D (PointCloudViewer) are driven by their panel/options controls, not st.tool, so their
// strip is just Select; showing draw/measure/cuboid there would be inert buttons.
const MODE_GROUPS: Record<string, ToolGroup[]> = {
  objects: [G.select, G.draw, G.ai, G.mask, G.region, G.cuboid, G.measure],
  pose: [G.select, G.pose, G.measure],
  lidar3d: [G.select],
  lanes: [G.select],
  semantic: [G.select, G.semantic],
  events: [G.select],
  review: [G.select],
};
const MODE_TOOLS: Record<string, string[]> = Object.fromEntries(
  Object.entries(MODE_GROUPS).map(([m, gs]) => [m, gs.flatMap((g) => g.tools.map((t) => t.key))]));

// Ways to pick a set that are not "drag a box round them". A dense frame holds forty vehicles and the
// useful selections are almost never contiguous: every autorickshaw, everything the model was unsure
// about, everything nobody has looked at yet.
const SELECTIONS: { how: SelectHow; label: string; value?: string | number; hint: string; key?: string }[] = [
  { how: "all", label: "all", hint: "every visible, unlocked object", key: "⌘A" },
  { how: "none", label: "none", hint: "clear the selection", key: "Esc" },
  { how: "invert", label: "invert", hint: "everything not currently selected", key: "⌘I" },
  { how: "sameClass", label: "same class", hint: "everything of the selected object's class", key: "⌘⇧A" },
  { how: "unreviewed", label: "unreviewed", hint: "still in review: the queue you are working" },
  { how: "new", label: "new", hint: "drawn here and not yet saved" },
  { how: "lowConf", label: "conf < 0.5", value: 0.5, hint: "the model was unsure about these" },
  { how: "state", label: "rejected", value: "rejected", hint: "already rejected" },
];

// The three surface classes the ternary drivable mask carries. Fallback is the unpaved shoulder India
// actually drives on, which is why it is a first-class surface rather than a kind of non-drivable.
const SURFACE_CLASSES = [
  { key: "drivable", label: "drivable", tone: "border-pass text-pass" },
  { key: "fallback", label: "fallback", tone: "border-warn text-warn" },
  { key: "non_drivable", label: "non-drv", tone: "border-block text-block" },
];

// directed object-relationship kinds offered in the editor (rider_of is the India two-wheeler case)
const RELATION_KINDS = ["rider_of", "towed_by", "part_of", "member_of", "occludes"];

// ray-casting point-in-polygon for a flattened [x,y,x,y,...] polygon, used to pick a clicked superpixel
function pointInPoly(pt: number[], poly: number[]): boolean {
  let inside = false;
  const n = poly.length / 2;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = poly[2 * i], yi = poly[2 * i + 1], xj = poly[2 * j], yj = poly[2 * j + 1];
    if (((yi > pt[1]) !== (yj > pt[1])) && (pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi || 1e-9) + xi)) inside = !inside;
  }
  return inside;
}

// client-side mirror of the server's class-name normalization (snake_case, ascii)
const normClass = (s: string) => s.trim().toLowerCase().replace(/[\s-]+/g, "_").replace(/[^a-z0-9_]/g, "");

function bboxOfPolys(polys: number[][]): number[] {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of polys)
    for (let i = 0; i < p.length; i += 2) {
      x0 = Math.min(x0, p[i]); x1 = Math.max(x1, p[i]);
      y0 = Math.min(y0, p[i + 1]); y1 = Math.max(y1, p[i + 1]);
    }
  return [x0, y0, x1, y1];
}

// Fraction of `box` that lies inside `ref` (0..1). Used to decide whether a SAM mask refines the
// selected object (high overlap) or is a different object that should become its own (low overlap).
function overlapFrac(box: number[], ref: number[]): number {
  if (ref.length < 4) return 0;
  const ix = Math.max(0, Math.min(box[2], ref[2]) - Math.max(box[0], ref[0]));
  const iy = Math.max(0, Math.min(box[3], ref[3]) - Math.max(box[1], ref[1]));
  const area = Math.max(1, (box[2] - box[0]) * (box[3] - box[1]));
  return (ix * iy) / area;
}

// Every mask that comes back from a model is traced at pixel resolution: a segmented car is several hundred
// vertices, most of them a fraction of a pixel apart. That size is carried in every payload and every
// export, and it is what put a draggable handle every two pixels along the outline. One image pixel of
// tolerance removes what cannot be seen at 100% zoom and keeps every corner that can.
const MASK_TOLERANCE_PX = 1;
const trim = (polys: number[][]) => simplifyMask(polys, MASK_TOLERANCE_PX);

export default function FrameEditor() {
  const router = useRouter();
  const confirm = useConfirm();
  const { id } = useParams<{ id: string }>();
  const focus = useSearchParams().get("focus");
  const rigParam = useSearchParams().get("rig");   // M-MC.1 deep link: keep rig view + layout across camera focus
  // The job this frame was opened from, when it was opened from one. It is what attributes a drawn box to
  // an annotator and a job, which is the whole basis of measuring how much two annotators agree; and for a
  // blind replica job it is also what tells the server to withhold the existing labels.
  const jobParam = useSearchParams().get("job");

  const [st, dispatch] = useEditor();
  const [meta, setMeta] = useState<FrameMeta | null>(null);
  const [onto, setOnto] = useState<Ontology | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [currentClass, setCurrentClass] = useState<OntologyClass | null>(null);
  const [panning, setPanning] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [autosave, setAutosave] = useState(true);
  const [search, setSearch] = useState("");
  const loadedRef = useRef(false);
  // inline class-edit popup anchored on the clicked box (quick relabel of a wrong annotation)
  const [editOpen, setEditOpen] = useState(false);
  const [editSearch, setEditSearch] = useState("");
  const canvasWrapRef = useRef<HTMLDivElement>(null);
  // P3 derived dynamics (distance/speed/heading/ttc/risk) keyed by object_id
  const [dynamics, setDynamics] = useState<Record<string, ObjectDynamicsRow>>({});
  // P4 segmentation overlays: lanes (M2.1) + drivable area (M2.2), with per-layer visibility
  const [lanes, setLanes] = useState<(LaneRow & { dirty?: boolean })[]>([]);
  const [drivable, setDrivable] = useState<Record<string, number[][]> | null>(null);
  // Lanes-mode editing (canvas swap): selected lane, the in-progress add path, the raster image + fit scale
  const [laneSel, setLaneSel] = useState<string | null>(null);
  const [laneAdding, setLaneAdding] = useState<number[][] | null>(null);
  // Drivable region editing. Separate buffers from the lane ones: a lane is an open polyline and a surface
  // region is a closed area, and sharing a buffer would render one of them wrongly mid-draw.
  const [areaAdding, setAreaAdding] = useState<number[][] | null>(null);
  const [areaClass, setAreaClass] = useState<string>("drivable");
  const [areaSel, setAreaSel] = useState<string | null>(null);
  const [laneImg, setLaneImg] = useState<HTMLImageElement | null>(null);
  const [laneScale, setLaneScale] = useState(1);
  // 3D and LiDAR mode (canvas swap): the cloud nearest this frame, its points, the 3D cuboids, edit state
  const [cloud3d, setCloud3d] = useState<LidarCloud | null>(null);
  const [pts3d, setPts3d] = useState<LidarPoints | null>(null);
  const [cub3d, setCub3d] = useState<Cuboid3D[]>([]);
  const [cubSel, setCubSel] = useState<string | null>(null);
  const [colorBy3d, setColorBy3d] = useState<ColorBy>("height");
  const [lidarMsg, setLidarMsg] = useState<string | null>(null);
  // Review mode (canvas stays Konva, the rail becomes the value queue): highest-value items + error candidates
  const [alItems, setAlItems] = useState<AlItem[]>([]);
  const [errItems, setErrItems] = useState<ErrorCandidateRow[]>([]);
  const [reviewLoaded, setReviewLoaded] = useState(false);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [linkFrom, setLinkFrom] = useState<string | null>(null); // active "relate" mode: the source object id
  const [linkKind, setLinkKind] = useState("rider_of");
  const [adverse, setAdverse] = useState<AdverseRegion[]>([]);
  const [adverseCond, setAdverseCond] = useState("glare");
  const [cuboids, setCuboids] = useState<ProjectedCuboid[]>([]);
  const [superpixels, setSuperpixels] = useState<number[][]>([]);
  const [brushRadius, setBrushRadius] = useState(14);
  const [segUrl, setSegUrl] = useState<string | null>(null); // dense-segmentation overlay png url
  const [segKind, setSegKind] = useState<"semantic" | "panoptic">("semantic");
  // provenance of the machine-produced overlays (drivable/seg), so the layers panel can show who made each
  const [drivableMeta, setDrivableMeta] = useState<{ source: string; model?: string | null } | null>(null);
  const [segMeta, setSegMeta] = useState<{ source: string; model?: string | null } | null>(null);
  const [objSearch, setObjSearch] = useState("");
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());
  const [mode, setMode] = useState("objects");
  // M-MC.1 rig view: a canvas view state (not a mode). rigGroup is the synchronized frame group this frame
  // belongs to; rigView toggles the multi-camera layout with the current camera focused.
  const [rigView, setRigView] = useState(false);
  const [rigLayout, setRigLayout] = useState<import("@/components/editor/RigView").RigLayout>("focus");
  const [rigGroup, setRigGroup] = useState<{ groupId: string; cameras: string[]; frameIds: Record<string, string>; missingCams: string[]; confirmed: boolean } | null>(null);
  const [rigPanel, setRigPanel] = useState(false);   // M-MC.2 rig-identity panel visibility
  const [explainOpen, setExplainOpen] = useState(false);  // M-F.0 "why this label" rationale
  const [rigCalibrated, setRigCalibrated] = useState<boolean | null>(null);  // M-MC.3 Tier 2 eligibility
  const [rigRefresh, setRigRefresh] = useState(0);   // bump to refetch the identity panel after a propagate
  const [rigTracks, setRigTracks] = useState(false);  // M-MC.4 rig-track timeline panel visibility
  const [scaleNoteOpen, setScaleNoteOpen] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [rightTab, setRightTab] = useState<"objects" | "tools">("objects");  // right panel: annotation vs AI tools
  // Responsive: on a narrow screen the properties panel collapses first (the design's degradation order),
  // giving the canvas and tool strip room. One-time on mount; the user can expand it manually after.
  useEffect(() => { if (typeof window !== "undefined" && window.innerWidth < 1100) setRightCollapsed(true); }, []);
  // switching mode swaps the tool strip; reset the active tool to the mode's first tool if it does not carry over
  const switchMode = (m: string) => {
    setMode(m);
    const tools = MODE_TOOLS[m] ?? [];
    if (!tools.includes(stRef.current.tool)) dispatch({ t: "tool", tool: (tools[0] ?? "select") as Tool });
  };
  const [layers, setLayers] = useState({ boxes: true, masks: true, labels: true, lanes: true, drivable: true, adverse: true, cuboids: true, seg: true });
  // Driving events on this frame's session (lane changes, signal phases), shown in the events mode.
  const [frameEvents, setFrameEvents] = useState<DrivingEvent[]>([]);

  const flash = (m: string) => {
    setNotice(m);
    setTimeout(() => setNotice(null), 3500);
  };

  // load frame + objects + ontology. Wrapped so a transient backend failure surfaces a retry state instead of
  // an uncaught promise rejection and a page stuck forever on "loading" (the three fetches share a Promise.all,
  // so any one 500 rejects the whole load).
  useEffect(() => {
    let live = true;
    // The operation board belongs to this frame. Carrying the previous frame's segmentations into the next
    // one would attribute work to an image it never touched.
    resetOps();
    (async () => {
      setLoadError(null);
      try {
        const [m, objs, o] = await Promise.all([
          api.frame(id), api.frameObjects(id, jobParam ?? undefined), api.ontology()]);
        if (!live) return;
        setMeta(m);
        setOnto(o);
        const eds: EdObject[] = objs.map((x) => ({
          id: x.object_id, track_id: x.track_id, class_id: x.class_id, class_name: x.class_name, bbox: x.bbox,
          mask: x.mask_polygons || [], attrs: {}, conf: x.conf, quality_score: x.quality_score, state: x.state, visible: true, version: x.version,
          rot: x.rot_deg, keypoints: x.keypoints ?? undefined, polyline: x.polyline ?? undefined,
          cuboid_3d: x.cuboid_3d ?? undefined,
        }));
        dispatch({ t: "load", objects: eds, viewport: { scale: 0, ox: 0, oy: 0 }, selectedId: focus });
        const fc = (focus && eds.find((e) => e.id === focus)) || null;
        setCurrentClass(fc ? o.classes.find((c) => c.id === fc.class_id) || o.classes[0] : o.classes[0]);
        loadedRef.current = true;
      } catch (e) {
        if (live) setLoadError(humanizeError(e));
      }
    })();
    return () => { live = false; };
  }, [id, focus, dispatch, reloadKey]);

  // M-MC.1: resolve the synchronized frame group this frame belongs to, so the rig view can show the sibling
  // cameras. Uses the persisted groups; if none exist yet for the session it builds them once on demand.
  useEffect(() => {
    if (!meta?.session_id || meta.ts_ns == null) return;
    let live = true;
    (async () => {
      const load = async () => api.multicamGroupAt(meta.session_id, meta.ts_ns);
      try {
        let g = await load().catch(() => null);
        if (!g) { await api.multicamBuild(meta.session_id).catch(() => undefined); g = await load().catch(() => null); }
        if (!live || !g) { setRigGroup(null); return; }
        const cams = Object.keys(g.frame_ids).concat(g.missing_cams);
        setRigGroup({ groupId: g.group_id, cameras: Array.from(new Set(cams)), frameIds: g.frame_ids, missingCams: g.missing_cams, confirmed: g.confirmed });
        if (rigParam && (rigParam === "grid" || rigParam === "strip" || rigParam === "focus")) {
          setRigLayout(rigParam as import("@/components/editor/RigView").RigLayout);
          setRigView(true);
        }
      } catch { if (live) setRigGroup(null); }
    })();
    return () => { live = false; };
  }, [meta?.session_id, meta?.ts_ns, rigParam]);

  const rigMulti = !!rigGroup && rigGroup.cameras.length > 1;
  const focusCamOnce = useCallback((cam: string, frameId: string) => {
    router.push(`/frame/${frameId}?rig=${rigLayout}`);
  }, [router, rigLayout]);
  const confirmRigGroup = async () => {
    if (!rigGroup) return;
    try { const g = await api.multicamGroupConfirm(rigGroup.groupId); setRigGroup((s) => s && { ...s, confirmed: g.confirmed }); flash("group confirmed"); }
    catch (e) { flash("confirm group failed: " + humanizeError(e)); }
  };
  // group-aware prev/next: jump to the adjacent synchronized group, keeping the same camera focused when it has
  // a frame there (else the group's first available camera), preserving the rig layout.
  const navGroup = async (direction: "prev" | "next") => {
    if (!meta?.session_id || !rigGroup) return;
    try {
      const r = await api.multicamGroupNav(meta.session_id, rigGroup.groupId, direction);
      if (!r.group) { flash(`no ${direction} group`); return; }
      const fids = r.group.frame_ids;
      const target = fids[meta.cam_id] ?? Object.values(fids)[0];
      if (target) router.push(`/frame/${target}?rig=${rigLayout}`);
    } catch (e) { flash("group nav failed: " + humanizeError(e)); }
  };
  const rigEditable = mode !== "lanes" && mode !== "lidar3d";

  // M-MC.3: is this session calibrated? Determines Tier 2 (projection) vs Tier 1 (manual link only).
  useEffect(() => {
    if (!rigMulti || !meta?.session_id) return;
    let live = true;
    api.calibrationDetail(meta.session_id)
      .then((d) => live && setRigCalibrated(d.validations.length > 0 && d.overall !== "fail"))
      .catch(() => live && setRigCalibrated(false));
    return () => { live = false; };
  }, [rigMulti, meta?.session_id]);

  const propagateSelected = async () => {
    if (!selected) return;
    try {
      const r = await api.multicamPropagate(selected.id, false);
      if (r.gated) { flash("session not calibrated: Tier 1 manual linking only"); setRigCalibrated(false); return; }
      const inView = (r.targets || []).filter((t) => t.in_view).length;
      flash(`propagated to ${r.created?.length ?? 0} view(s)${r.metric ? ` · ${r.metric.range_m}m` : ""}${inView ? "" : " (out of view)"}`);
      setRigPanel(true); setRigRefresh((n) => n + 1);
    } catch (e) { flash("propagate failed: " + humanizeError(e)); }
  };

  const selected = st.objects.find((o) => o.id === st.selectedId) || null;
  const dirty = isDirty(st);

  // object relationships: in link mode, the next clicked object becomes the target of the relationship
  const relate = async (toId: string) => {
    if (!linkFrom || toId === linkFrom) { setLinkFrom(null); return; }
    try {
      await api.relateObject(linkFrom, { to_object_id: toId, kind: linkKind });
      setRelationships(await api.frameRelationships(id).catch(() => []));
      flash(`linked: ${linkKind}`);
    } catch (e) { flash("link failed: " + humanizeError(e)); }
    setLinkFrom(null);
  };
  const doSelect = (oid: string | null) => {
    if (linkFrom && oid && oid !== linkFrom) { relate(oid); return; }
    dispatch({ t: "select", id: oid });
  };
  const delRelationship = async (rid: string) => {
    await api.deleteRelationship(rid).catch(() => {});
    setRelationships(await api.frameRelationships(id).catch(() => []));
  };
  // cuboid tool: click the ground in the image, lift to an ego ground point, drop a default 3D box there
  const placeCuboid = async (pt: number[]) => {
    if (!currentClass) return;
    try {
      const { ego, reason } = await api.liftGround(id, pt[0], pt[1]);
      if (!ego) { flash(reason || "click on the road ahead to place a cuboid"); return; }
      const cub = { center: [ego[0], ego[1], 0.75], size: [1.8, 4.2, 1.5], yaw: 0 };
      dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name,
        bbox: [pt[0] - 40, pt[1] - 40, pt[0] + 40, pt[1] + 40], mask: [], cuboid_3d: cub, attrs: {},
        conf: 1, state: "accepted", visible: true, isNew: true } });
    } catch (e) { flash("could not place cuboid: " + humanizeError(e)); }
  };
  // Auto-detect the class of a freshly-drawn object from its crop (SigLIP2 zero-shot over the ontology), so
  // a SAM box or wand click picks the class for you. Overrides the palette class; you can still relabel.
  const autoClassify = async (objId: string, box: number[]) => {
    if (box.length !== 4) return;
    try {
      const { predictions } = await api.classifyObject(id, box);
      const top = predictions?.[0];
      if (top && top.conf >= 0.15) {
        dispatch({ t: "update", id: objId, patch: { class_id: top.class_id, class_name: top.class_name } });
        flash(`detected ${top.class_name} (${Math.round(top.conf * 100)}%)`);
      }
    } catch { /* keep the palette class if classification is unavailable */ }
  };
  // magic-wand: a single SAM point click that auto-creates (or refines) the object, no accept step
  const runMagicWand = async (pt: number[]) => {
    try {
      const r = await trackOp("sam", "magic wand",
        () => api.segmentPrompt(id, { points: [pt], labels: [1], precise: segKind === "panoptic" }),
        (res) => `${res.polygons.length} region${res.polygons.length === 1 ? "" : "s"}`);
      if (!r.polygons.length) { flash("magic-wand found nothing here"); return; }
      const polys = trim(r.polygons);
      const box = bboxOfPolys(polys);
      if (selected && overlapFrac(box, selected.bbox) > 0.5) {
        dispatch({ t: "update", id: selected.id, patch: { mask: polys, bbox: box } });
      } else if (currentClass) {
        const nid = tmpId();
        dispatch({ t: "add", obj: { id: nid, class_id: currentClass.id, class_name: currentClass.name,
          bbox: box, mask: polys, attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } });
        autoClassify(nid, box);
      }
    } catch (e) { flash(humanizeError(e).includes("503") ? "GPU busy (training)" : "magic-wand failed"); }
  };
  // brush/eraser: compose the stroke stamps into the selected object's mask (or a new object)
  const onBrushStroke = async (ops: { op: string; center: number[]; radius: number }[]) => {
    if (!meta) return;
    try {
      const r = await trackOp("mask", "brush stroke",
        () => api.composeMask({ polygons: selected?.mask ?? [], ops, width: meta.width, height: meta.height }));
      if (selected) {
        dispatch({ t: "update", id: selected.id, patch: { mask: trim(r.polygons), bbox: r.polygons.length ? bboxOfPolys(r.polygons) : selected.bbox } });
      } else if (currentClass && r.polygons.length) {
        dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name,
          bbox: bboxOfPolys(r.polygons), mask: trim(r.polygons), attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } });
      }
    } catch (e) { flash("brush failed: " + humanizeError(e)); }
  };
  // superpixel: add the clicked SLIC cell to the active mask
  const pickSuperpixel = (pt: number[]) => {
    const found = superpixels.find((pp) => pointInPoly(pt, pp));
    if (!found) return;
    const poly = simplifyPolygon(found, MASK_TOLERANCE_PX);
    if (selected) {
      const next = [...selected.mask, poly];
      dispatch({ t: "update", id: selected.id, patch: { mask: next, bbox: bboxOfPolys(next) } });
    } else if (currentClass) {
      dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name,
        bbox: bboxOfPolys([poly]), mask: [poly], attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } });
    }
  };

  // P3 derived dynamics: fetch this frame's readout, and a recompute over the session
  const loadDynamics = useCallback(async () => {
    const r = await api.frameDynamics(id).catch(() => null);
    if (r) setDynamics(Object.fromEntries(r.dynamics.map((d) => [d.object_id, d])));
  }, [id]);
  useEffect(() => { loadDynamics(); }, [loadDynamics]);
  const recomputeDynamics = useCallback(async () => {
    if (!meta) return;
    flash("computing dynamics...");
    await api.computeDynamics(meta.session_id);
    await loadDynamics();
    flash("dynamics updated");
  }, [meta, loadDynamics]);

  // P4 layers: fetch lane + drivable overlays, and inline generators
  const loadLayers = useCallback(async () => {
    const [ls, dr, rel, adv, cub] = await Promise.all([api.framesLanes(id).catch(() => []), api.getDrivable(id).catch(() => null), api.frameRelationships(id).catch(() => []), api.listAdverse(id).catch(() => []), api.frameCuboids(id).catch(() => [])]);
    setLanes(ls);
    setDrivable(dr && dr.found ? dr.classes ?? null : null);
    setDrivableMeta(dr && dr.found ? { source: dr.source ?? "", model: dr.model_version } : null);
    setRelationships(rel);
    setAdverse(adv);
    setCuboids(cub);
    const seg = await api.getSegment(id, segKind).catch(() => ({ found: false, has_overlay: false, source: "", model_version: null }));
    setSegUrl(seg.found && seg.has_overlay ? `/api/frames/${id}/segment/overlay?kind=${segKind}&t=${Date.now()}` : null);
    setSegMeta(seg.found ? { source: seg.source ?? "", model: seg.model_version } : null);
  }, [id, segKind]);
  useEffect(() => { loadLayers(); }, [loadLayers]);
  // Compact provenance per overlay layer ("proposed - mask2former-mapillary"), so the layers panel shows
  // who produced each overlay: a model on the pod, a human, or an import. Object layers (boxes/masks/
  // cuboids) already carry per-object source badges in the object list, so they are not repeated here.
  const layerMeta = useMemo(() => {
    const clean = (m?: string | null) => (m ? m.split(":")[0] : ""); // drop the ":pod"/":local" runtime tag
    // The model is only meaningful for machine-produced overlays; a human-owned layer reads plainly as
    // "human" (showing a stale proposing-model there was misleading, e.g. "human - clrernet").
    const MACHINE = new Set(["proposed", "propagated", "fused", "auto_accept", "interpolated"]);
    const fmt = (source?: string, model?: string | null) =>
      [source || "", source && MACHINE.has(source) ? clean(model) : ""].filter(Boolean).join(" · ");
    const out: Record<string, string> = {};
    if (lanes.length) {
      const srcs = Array.from(new Set(lanes.map((l) => l.source)));
      out.lanes = srcs.length === 1 ? fmt(srcs[0], lanes.find((l) => l.model_version)?.model_version) : "mixed sources";
    }
    if (drivableMeta) out.drivable = fmt(drivableMeta.source, drivableMeta.model);
    if (adverse.length) {
      const srcs = Array.from(new Set(adverse.map((a) => a.source)));
      out.adverse = srcs.length === 1 ? srcs[0] || "" : "mixed sources";
    }
    if (segMeta) out.seg = fmt(segMeta.source, segMeta.model);
    return out;
  }, [lanes, drivableMeta, adverse, segMeta]);
  // fetch SLIC superpixels lazily, the first time the superpixel tool is used on this frame
  useEffect(() => {
    if (st.tool === "superpixel" && !superpixels.length) {
      api.superpixels(id).then((r) => setSuperpixels(r.superpixels)).catch(() => flash("superpixels unavailable"));
    }
  }, [st.tool, id, superpixels.length]);
  // The editor has its own header (no TopNav/UserPicker), so guarantee a valid identity or every mutation
  // 401s. Drop a stale/deleted cached user and auto-pick one, mirroring the UserPicker.
  useEffect(() => {
    api.users().then((us) => {
      const cur = getUser();
      if ((!cur || !us.some((u) => u.user_id === cur.user_id)) && us.length) {
        setUser(us.find((u) => u.role === "admin") ?? us[0]);
      }
    }).catch(() => {});
  }, []);
  const segRoad = useCallback(async () => {
    flash("segmenting road surface...");
    try {
      await trackOp("drivable", "drivable surface", () => api.segmentDrivable(id));
      await loadLayers();
      flash("drivable area updated");
    }
    catch (e) { flash("segment road failed: " + humanizeError(e)); }
  }, [id, loadLayers]);
  const genLanes = useCallback(async () => {
    try { const r = await api.proposeLanes(id); await loadLayers(); flash(`proposed ${r.proposed} lanes (${r.model})`); }
    catch (e) { flash("propose lanes failed: " + humanizeError(e)); }
  }, [id, loadLayers]);

  // Lanes mode mounts LaneCanvas (a fit-to-width Konva stage), so it needs the raster image and a scale.
  useEffect(() => {
    if (mode !== "lanes" || !meta || laneImg) return;
    const im = new window.Image();
    im.src = meta.image_url;
    im.onload = () => setLaneImg(im);
  }, [mode, meta, laneImg]);
  useEffect(() => {
    if (mode !== "lanes" || !meta || !canvasWrapRef.current) return;
    setLaneScale(canvasWrapRef.current.clientWidth / meta.width);
  }, [mode, meta, laneImg]);

  const laneToImg = (e: { evt: MouseEvent }): number[] => {
    const r = canvasWrapRef.current!.getBoundingClientRect();
    return [(e.evt.clientX - r.left) / laneScale, (e.evt.clientY - r.top) / laneScale];
  };
  const laneStageClick = (e: { evt: MouseEvent }) => {
    if (laneAdding) { setLaneAdding([...laneAdding, laneToImg(e)]); return; }
    if (areaAdding) setAreaAdding([...areaAdding, laneToImg(e)]);
  };
  const laneDragPoint = (laneId: string, i: number, x: number, y: number) =>
    setLanes((ls) => ls.map((l) => l.lane_id === laneId
      ? { ...l, dirty: true, control_points: l.control_points.map((p, j) => (j === i ? [x, y] : p)) } : l));
  const saveLanes = async () => {
    const dirty = lanes.filter((x) => x.dirty);
    if (!dirty.length) { flash("no lane edits"); return; }
    for (const l of dirty) await api.updateLane(l.lane_id, { control_points: l.control_points, lane_type: l.lane_type, is_ego: l.is_ego });
    await loadLayers(); flash(`saved ${dirty.length} lane${dirty.length === 1 ? "" : "s"}`);
  };
  const finishAddLane = async (type: string) => {
    if (!laneAdding || laneAdding.length < 2) { setLaneAdding(null); return; }
    await api.createLane(id, { control_points: laneAdding, lane_type: type, is_ego: false });
    setLaneAdding(null); await loadLayers(); flash("lane added");
  };
  const setLaneType = (t: string) => { if (laneSel) setLanes((ls) => ls.map((l) => l.lane_id === laneSel ? { ...l, lane_type: t, dirty: true } : l)); };
  const toggleLaneEgo = () => { if (laneSel) setLanes((ls) => ls.map((l) => ({ ...l, is_ego: l.lane_id === laneSel ? !l.is_ego : l.is_ego, dirty: l.lane_id === laneSel ? true : l.dirty }))); };
  const delLane = async () => {
    if (!laneSel) return;
    if (!(await confirm({ title: "Delete this lane?", danger: true, confirmLabel: "Delete" }))) return;
    await api.deleteLane(laneSel); setLaneSel(null); await loadLayers(); flash("lane deleted");
  };
  const propagateLanes = async () => {
    const r = await trackOp("propagate", "propagate lanes", () => api.propagateLanes(id, 8),
      (res) => `${res.created} frames`);
    flash(`propagated to ${r.created} lane-frames`);
  };

  // ---- drivable surface editing --------------------------------------------------------------------
  // The mask is stored as flat [x,y,x,y,...] rings per class, which is what the API takes and returns, so
  // the editor speaks that shape rather than converting back and forth and risking a mismatch on save.
  const saveDrivable = async (next: Record<string, number[][]>) => {
    if (!meta) return;
    try {
      const r = await api.refineDrivable(id, next, meta.width, meta.height);
      setDrivable(next);
      setDrivableMeta({ source: "human", model: "human" });
      const pct = Math.round((r.coverage.drivable ?? 0) * 100);
      flash(`surface saved, ${pct}% drivable`);
    } catch (e) { flash("surface save failed: " + humanizeError(e)); }
  };

  const finishArea = async () => {
    // Three points is the minimum that encloses anything. Fewer is a stray click, and closing it would
    // write a degenerate sliver into the coverage statistics.
    if (!areaAdding || areaAdding.length < 3) { setAreaAdding(null); return; }
    const flat = areaAdding.flat();
    const next = { ...(drivable ?? {}) };
    next[areaClass] = [...(next[areaClass] ?? []), flat];
    setAreaAdding(null);
    await saveDrivable(next);
  };

  const dragAreaPoint = (key: string, i: number, x: number, y: number) => {
    const [cls, idx] = key.split(":");
    setDrivable((d) => {
      if (!d?.[cls]?.[Number(idx)]) return d;
      const polys = d[cls].map((p2, j) => {
        if (j !== Number(idx)) return p2;
        const copy = [...p2];
        copy[i * 2] = x; copy[i * 2 + 1] = y;
        return copy;
      });
      return { ...d, [cls]: polys };
    });
  };

  const deleteArea = async () => {
    if (!areaSel || !drivable) return;
    const [cls, idx] = areaSel.split(":");
    if (!(await confirm({ title: `Delete this ${cls.replace("_", " ")} region?`, danger: true,
                          confirmLabel: "Delete" }))) return;
    const next = { ...drivable, [cls]: drivable[cls].filter((_p, j) => j !== Number(idx)) };
    setAreaSel(null);
    await saveDrivable(next);
  };

  // Dense semantic editing. The raster had a `human` source it could never be set to, because there was no
  // write path: the layer was machine output a person could look at and not correct.
  const paintSemantic = async (pts: number[]) => {
    if (!currentClass) { flash("pick a class first"); return; }
    try {
      const r = await api.editFrameSegmentation(id, {
        kind: "semantic",
        classes: [{ class_name: currentClass.name, polygons: [pts] }],
      });
      if (r.unknown_classes.length) flash(`ontology does not know ${r.unknown_classes.join(", ")}`);
      else flash(`painted ${currentClass.name}, ${(r.labelled_fraction * 100).toFixed(1)}% of the frame labelled`);
      await loadLayers();
    } catch (e) { flash("semantic paint failed: " + humanizeError(e)); }
  };

  // Driving events for this frame's session, so behaviour is reviewable next to the frame that shows it.
  const loadFrameEvents = async () => {
    if (!meta) return;
    try {
      const r = await api.drivingEvents(meta.session_id, { limit: 200 });
      setFrameEvents(r.events);
      if (!r.events.length) flash("no driving events yet, derive them from the events page");
    } catch (e) { flash("events load failed: " + humanizeError(e)); }
  };

  // 3D mode: load the session cloud nearest this frame's timestamp, then its points and 3D cuboids.
  useEffect(() => {
    if (mode !== "lidar3d" || !meta || cloud3d) return;
    let cancelled = false;
    (async () => {
      setLidarMsg("loading point cloud...");
      try {
        const r = await api.lidarClouds(meta.session_id);
        if (!r.clouds.length) { if (!cancelled) setLidarMsg("no point cloud for this session"); return; }
        const near = r.clouds.reduce((a, b) => (Math.abs(b.ts_ns - meta.ts_ns) < Math.abs(a.ts_ns - meta.ts_ns) ? b : a));
        const [pts, objs] = await Promise.all([lidarCloudPoints(near.cloud_id, { variant: "raw", max: 300000 }), api.lidarObjects3d(near.cloud_id)]);
        if (cancelled) return;
        setCloud3d(near); setPts3d(pts); setCub3d(objs.objects); setLidarMsg(null);
      } catch (e) { if (!cancelled) setLidarMsg("cloud load failed: " + humanizeError(e)); }
    })();
    return () => { cancelled = true; };
  }, [mode, meta, cloud3d]);

  const cubSelected = cub3d.find((c) => c.object_3d_id === cubSel) || null;
  const patchCub = (cid: string, patch: Partial<Cuboid3D>) => setCub3d((cs) => cs.map((c) => (c.object_3d_id === cid ? { ...c, ...patch } : c)));
  const saveCub = async (cid: string, fields: Partial<Cuboid3D>) => {
    const cur = cub3d.find((c) => c.object_3d_id === cid); if (!cur) return;
    try {
      const saved = await api.lidarPatchCuboid(cid, {
        class_id: (fields.class_id as number) ?? cur.class_id, center: (fields.center as number[]) ?? cur.center,
        dims: (fields.dims as number[]) ?? cur.dims, yaw: (fields.yaw as number) ?? cur.yaw,
        pitch: (fields.pitch as number) ?? cur.pitch, roll: (fields.roll as number) ?? cur.roll,
        ground_snap: Boolean(fields.attrs && (fields.attrs as Record<string, unknown>).ground_snap), expected_version: cur.version,
      });
      patchCub(cid, saved);
    } catch (e) { setLidarMsg("save failed: " + humanizeError(e)); }
  };
  const moveCub = (cid: string, x: number, y: number, commit: boolean) => {
    const cur = cub3d.find((c) => c.object_3d_id === cid); if (!cur) return;
    const center = [x, y, cur.center[2]]; patchCub(cid, { center }); if (commit) saveCub(cid, { center });
  };
  const addCub = async () => {
    if (!cloud3d || !onto) return;
    const cls = onto.classes.find((c) => c.name === "sedan") || onto.classes[0]; if (!cls) return;
    try {
      const created = await api.lidarCreateCuboid(cloud3d.cloud_id, { class_id: cls.id, center: [12, 0, 1], dims: CUBOID_DIMS[cls.name] || [4, 1.8, 1.5], yaw: 0, ground_snap: true });
      setCub3d((cs) => [...cs, created]); setCubSel(created.object_3d_id);
    } catch (e) { setLidarMsg("add failed: " + humanizeError(e)); }
  };
  const delCub = async (cid: string) => {
    // A 3D cuboid is not covered by the frame's checkpoints, which snapshot 2D objects, so this deletion is
    // genuinely irreversible and asks first.
    if (!(await confirm({ title: "Delete this 3D box?", danger: true, confirmLabel: "Delete" }))) return;
    try { await api.lidarDeleteCuboid(cid); setCub3d((cs) => cs.filter((c) => c.object_3d_id !== cid)); setCubSel(null); }
    catch (e) { setLidarMsg("delete failed: " + humanizeError(e)); }
  };
  const aiLift3d = async () => {
    if (!cloud3d) return;
    setLidarMsg("lifting 2D objects to 3D...");
    try { const r = await api.lidarLiftCloud(cloud3d.cloud_id); const objs = await api.lidarObjects3d(cloud3d.cloud_id); setCub3d(objs.objects); setLidarMsg(r.cuboids ? `lifted ${r.cuboids} cuboids` : "no 2D objects to lift"); }
    catch (e) { setLidarMsg("lift failed: " + humanizeError(e)); }
  };

  // Review mode: lazily load the value queue (this frame's items ranked first) and the error candidates.
  useEffect(() => {
    if (mode !== "review" || !meta || reviewLoaded) return;
    setReviewLoaded(true);
    (async () => {
      const [al, ec] = await Promise.all([
        api.alScore(meta.session_id, 80).then((r) => r.items).catch(() => [] as AlItem[]),
        api.errorCandidates("pending", 80).catch(() => [] as ErrorCandidateRow[]),
      ]);
      al.sort((a, b) => (a.frame_id === id ? 0 : 1) - (b.frame_id === id ? 0 : 1) || b.value - a.value);
      setAlItems(al);
      setErrItems(ec);
    })();
  }, [mode, meta, reviewLoaded, id]);

  // Accept or reject the selected object (persisted directly with an explicit state), then advance the queue.
  //
  // The accepted state comes from the role, not from the verb. An annotator's "accept" means submitted, so
  // the work lands in the QA queue the triage page's `submitted` band exists to serve. This file's own
  // `save()` already used `acceptState(role)`; these two paths hardcoded `accepted`, so the same person
  // pressing A skipped the review step that pressing Cmd+S respected. `/review/rapid` was always right.
  const reviewObject = async (verdict: "accept" | "reject") => {
    const newState = verdict === "accept" ? acceptState(getUser()?.role) : "rejected";
    const o = selected;
    if (!o) { flash("select an object to review"); return; }
    if (o.isNew) { flash("save the new object first"); return; }
    // api.review carries only the state, not geometry, and the reviewed reducer clears dirty; reviewing on
    // top of unsaved edits would silently drop them, so require an explicit save first.
    if (o.dirty) { flash("save your edits first (Cmd S), then accept or reject"); return; }
    try {
      // No expected_version here on purpose: this is a pure accept/reject decision, and unsaved edits are
      // already blocked above, so there is nothing to clobber. Gating the human's decision on a version that
      // a background re-autolabel or embed pass may have bumped just produced spurious 409s. The optimistic
      // lock still guards the geometry-edit path (adjust_geometry), where a concurrent box edit does matter.
      const r = await api.review(o.id, { action: verdict, state: newState });
      dispatch({ t: "reviewed", id: o.id, state: newState, version: r.version });
      setAlItems((s) => s.filter((it) => it.object_id !== o.id)); // drop the handled item so the queue advances
      flash(newState);
      advanceReview(o.id);
    } catch (e) { flash("review failed: " + humanizeError(e)); }
  };
  // Move to the next value-queue item (excluding the one just handled): select it if it is on this frame,
  // else jump to its frame. Never re-selects an already-reviewed item.
  const advanceReview = (currentObjId: string) => {
    const rest = alItems.filter((it) => it.object_id !== currentObjId);
    const onFrame = rest.find((it) => it.frame_id === id);
    if (onFrame) doSelect(onFrame.object_id);
    else { const off = rest.find((it) => it.frame_id !== id); if (off) gotoFrame(off.frame_id); }
  };

  // each new selection starts with the compact chip (class name + edit), not the open picker
  useEffect(() => { setEditOpen(false); setEditSearch(""); }, [st.selectedId]);
  const editClasses = useMemo(
    () => (onto ? onto.classes.filter((c) => c.name.includes(editSearch.toLowerCase().replace(/\s/g, "_"))) : []),
    [onto, editSearch],
  );

  // ---- interactive AI correction: a deliberate relabel / attribute change on an EXISTING object opens
  // the "fix similar" modal (debounced so rapid reclass settles on the final value before searching) ----
  type Corr = { objectId: string; kind: "class" | "attr"; change: CorrectionChange };
  const [pendingCorr, setPendingCorr] = useState<Corr | null>(null);
  const [activeCorr, setActiveCorr] = useState<Corr | null>(null);

  const recordCorrection = useCallback(
    (objectId: string, kind: "class" | "attr", oldVal: CorrectionChange["old"], newVal: CorrectionChange["new"], attrKey?: string) => {
      setPendingCorr((prev) =>
        prev && prev.objectId === objectId && prev.kind === kind && prev.change.attrKey === attrKey
          ? { ...prev, change: { ...prev.change, new: newVal } } // keep original old, update to final new
          : { objectId, kind, change: { old: oldVal, new: newVal, attrKey } });
    },
    [],
  );

  const relabelSelected = useCallback(
    (c: OntologyClass) => {
      setCurrentClass(c);
      if (!selected) return;
      const old = selected.class_name;
      dispatch({ t: "update", id: selected.id, patch: { class_id: c.id, class_name: c.name } });
      if (!selected.isNew && c.name !== old) recordCorrection(selected.id, "class", old, c.name);
    },
    [selected, dispatch, recordCorrection],
  );

  // create a new custom class on the fly, then apply it (to the selected object, or as the new-object class)
  const addAndRelabel = useCallback(
    async (rawName: string) => {
      try {
        const cls = await api.addClass(rawName);
        const o = await api.ontology();   // refresh so the new class shows in every picker
        setOnto(o);
        const full = o.classes.find((c) => c.id === cls.id) || cls;
        relabelSelected(full as OntologyClass);
        setEditOpen(false); setEditSearch(""); setSearch("");
        flash(cls.existed ? `class "${cls.name}" already existed, applied` : `added custom class "${cls.name}"`);
      } catch (e) {
        flash("could not add class: " + humanizeError(e));
      }
    },
    [relabelSelected],
  );

  const setAttrSelected = useCallback(
    (name: string, val: unknown) => {
      if (!selected) return;
      const old = selected.attrs[name];
      dispatch({ t: "update", id: selected.id, patch: { attrs: { ...selected.attrs, [name]: val } } });
      if (!selected.isNew && val !== old)
        recordCorrection(selected.id, "attr", (old ?? null) as CorrectionChange["old"], val as CorrectionChange["new"], name);
    },
    [selected, dispatch, recordCorrection],
  );

  useEffect(() => {
    if (!pendingCorr) return;
    const t = setTimeout(() => { setActiveCorr(pendingCorr); setPendingCorr(null); }, 800);
    return () => clearTimeout(t);
  }, [pendingCorr]);

  // ---- SAM ----
  // Each mask is COMMITTED before the next SAM click runs, so clicking the next object never loses the
  // one before. Enter still commits the latest, Esc discards it. With nothing selected, every click
  // makes a new object; with an object selected, SAM refines that object's mask.
  const acceptCandidate = useCallback(() => {
    if (!st.candidate?.length) return;
    const box = bboxOfPolys(st.candidate);
    // Refine the selected object only if the mask overlaps it; otherwise it's a different object.
    if (selected && overlapFrac(box, selected.bbox) > 0.5) {
      dispatch({ t: "update", id: selected.id, patch: { mask: st.candidate, bbox: selected.bbox.length === 4 ? selected.bbox : box } });
    } else if (currentClass) {
      const nid = tmpId();
      dispatch({ t: "add", obj: { id: nid, class_id: currentClass.id, class_name: currentClass.name,
        bbox: box, mask: st.candidate, attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } });
      autoClassify(nid, box);
      if (selected) dispatch({ t: "select", id: null }); // moved off the old object -> keep creating new
    }
    dispatch({ t: "candidate", polys: null });
  }, [st.candidate, selected, currentClass, dispatch]);  // eslint-disable-line react-hooks/exhaustive-deps

  const runSam = useCallback(
    async (prompt: { points?: number[][]; labels?: number[]; box?: number[] }) => {
      if (st.candidate?.length) acceptCandidate(); // commit the pending mask before starting the next
      try {
        const r = await trackOp("sam", segKind === "panoptic" ? "SAM (precise)" : "SAM segment",
          () => api.segmentPrompt(id, { ...prompt, precise: segKind === "panoptic" }),
          (res) => `${res.polygons.length} region${res.polygons.length === 1 ? "" : "s"}`);
        dispatch({ t: "candidate", polys: trim(r.polygons) });
        if (!r.polygons.length) flash("SAM found nothing here");
      } catch (e) {
        const msg = humanizeError(e);
        flash(msg.includes("503") ? "GPU busy (training). Box tools still work." : "segment failed");
      }
    },
    [id, dispatch, st.candidate, acceptCandidate, segKind],
  );

  // Leaving a SAM tool (or switching away) commits any uncommitted mask instead of dropping it.
  const acceptRef = useRef(acceptCandidate);
  acceptRef.current = acceptCandidate;
  useEffect(() => {
    if (st.tool !== "sam-point" && st.tool !== "sam-box") acceptRef.current();
  }, [st.tool]);

  // ---- save (diff vs server) ----
  // A synchronous ref mutex (not the async `saving` state) guarantees two saves never overlap. Without
  // it, the unmount flush or a fast second autosave could re-run before dispatch({t:"saved"}) clears the
  // isNew flags, creating the same object twice on the server. The idem_key is belt-and-suspenders: the
  // server de-dupes a create that still slips through (network retry, multi-tab).
  const savingRef = useRef(false);
  // Signature of the pending (deleted + new/dirty) set that last failed to save. Autosave will not retry the
  // exact same set, so a persistent per-object error (a malformed object, a stale id) cannot become an
  // infinite 700ms retry storm; any real edit changes the signature and resumes autosave.
  const lastFailRef = useRef("");
  const pendingSig = () => JSON.stringify([st.deleted,
    st.objects.filter((o) => o.isNew || o.dirty).map((o) => `${o.id}:${o.version ?? 0}`)]);
  const save = useCallback(async () => {
    if (!dirty || savingRef.current) return;
    savingRef.current = true;
    setSaving(true);
    const tgt = acceptState(getUser()?.role);  // annotator -> submitted (QA), reviewer/admin -> accepted
    // Tracked rather than wrapped in trackOp: this function already owns a try/catch/finally with a
    // savingRef guard, and threading a wrapper through it would have meant restructuring the one path in
    // this file that must not grow another way to leak.
    const pending = st.objects.filter((o) => o.isNew || o.dirty).length + st.deleted.length;
    const opId = beginOp("save", pending === 1 ? "saving 1 object" : `saving ${pending} objects`);
    try {
      // Delete is idempotent: a 404 means the object is already gone, which is the desired end state. Without
      // this, deleting an already-removed object throws, aborts the save before the "saved" dispatch clears
      // st.deleted, and the autosave effect retries the same failing delete every 700ms forever.
      for (const oid of st.deleted) {
        try { await api.deleteObject(oid); }
        catch (e) { if (!humanizeError(e).includes("404")) throw e; }
      }
      const remap: Record<string, string> = {};
      const versions: Record<string, number> = {};
      for (const o of st.objects) {
        // A tmp- id is a client-only object never persisted, so it must be CREATED, never reviewed: sending
        // a tmp id to /objects/{id}/review is a 422 (not a UUID), and if it loops it floods like the delete
        // case did. createObject de-dupes via idem_key, so a redundant create is safe.
        if (o.isNew || o.id.startsWith("tmp-")) {
          const created = await api.createObject(id, {
            class_name: o.class_name, bbox: o.bbox, attrs: o.attrs,
            mask_polygons: o.mask.length ? o.mask : undefined, state: tgt, idem_key: o.id, rot_deg: o.rot ?? 0,
            keypoints: o.keypoints ?? null, polyline: o.polyline, cuboid_3d: o.cuboid_3d ?? undefined,
            job_id: jobParam ?? undefined,
          });
          remap[o.id] = created.object_id;
          if (created.version != null) versions[o.id] = created.version;
        } else if (o.dirty) {
          // One atomic request: geometry, mask, rotation, and keypoints persist together (no separate
          // updateMask that could leave the mask out of sync on a partial failure).
          const r = await api.review(o.id, { action: "adjust_geometry",
            class_name: o.class_name, bbox: o.bbox, attrs: o.attrs, state: tgt, expected_version: o.version,
            rot_deg: o.rot ?? 0, keypoints: o.keypoints ?? null, polyline: o.polyline, cuboid_3d: o.cuboid_3d ?? undefined,
            mask_polygons: o.mask.length ? o.mask : undefined });
          if (r.version != null) versions[o.id] = r.version;
        }
      }
      dispatch({ t: "saved", remap, versions });
      lastFailRef.current = "";                  // succeeded: clear the no-retry guard
      flash("saved");
      setCuboids(await api.frameCuboids(id).catch(() => [])); // refresh projected cuboid wireframes
      endOp(opId, "ok", pending === 1 ? "1 object" : `${pending} objects`);
    } catch (e) {
      const msg = humanizeError(e);
      lastFailRef.current = pendingSig();         // do not auto-retry this exact set until something changes
      endOp(opId, "failed", msg.includes("409") ? "another annotator changed this object" : msg);
      flash(msg.includes("409") ? "conflict: another annotator changed this object; reload to continue" : "save failed: " + msg);
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  }, [dirty, st.deleted, st.objects, id, meta, dispatch]);

  // ---- autosave: persist edits ~700ms after the last change settles (covers move/resize/relabel/
  // attribute/mask/delete). The debounce waits out an active drag, so we never save mid-gesture. ----
  // Save-as: a named state on the server. Distinct from save, which writes the working copy back; this one
  // is the thing you can return to after an hour of work went a different way.
  const saveAs = useCallback(async () => {
    const suggested = `working state ${new Date().toLocaleTimeString()}`;
    const name = window.prompt("Name this save", suggested);
    if (name === null) return;
    try {
      await save();
      const c = await api.saveCheckpoint(id, name.trim() || suggested);
      flash(`saved "${c.name}" with ${c.object_count} objects`);
    } catch (e) {
      flash("save as failed: " + humanizeError(e));
    }
  }, [id, save]);

  const saveRef = useRef(save);
  saveRef.current = save;
  const stRef = useRef(st);
  stRef.current = st;
  useEffect(() => {
    if (!autosave || !loadedRef.current || !dirty || saving) return;
    if (pendingSig() === lastFailRef.current) return;  // this exact set already failed; wait for a real edit
    const t = setTimeout(() => saveRef.current(), 700);
    return () => clearTimeout(t);
  }, [autosave, dirty, saving, st.objects, st.deleted]);
  // flush a still-pending edit when leaving the editor (back button / route change)
  useEffect(() => () => { if (isDirty(stRef.current)) saveRef.current(); }, []);

  // ---- viewport helpers ----
  const fit = useCallback(() => dispatch({ t: "viewport", viewport: { scale: 0, ox: 0, oy: 0 } }), [dispatch]);
  const zoomBy = useCallback((f: number) => dispatch({ t: "viewport", viewport: { ...st.viewport, scale: Math.max(0.05, Math.min(20, st.viewport.scale * f)) } }), [st.viewport, dispatch]);
  const gotoFrame = useCallback(async (fid: string | null) => {
    if (!fid) return;
    if (isDirty(st)) await save();  // flush before leaving so no edit is lost
    router.push(`/frame/${fid}`);
  }, [router, st, save]);

  // ---- keypoint pose tool + object clipboard ----
  const [kpDraft, setKpDraft] = useState<number[][] | null>(null);
  const kpDraftRef = useRef<number[][] | null>(null);
  kpDraftRef.current = kpDraft;
  const clipboardRef = useRef<EdObject | null>(null);
  useEffect(() => { if (st.tool !== "keypoint") setKpDraft(null); }, [st.tool]);

  const finishKeypoints = useCallback((pts: number[][]) => {
    if (!currentClass || !pts.length) { setKpDraft(null); return; }
    const full = pts.slice(0, PERSON_17.points.length);
    while (full.length < PERSON_17.points.length) full.push([0, 0, 0]);
    const vis = full.filter((q) => q[2] > 0);
    const xs = vis.map((q) => q[0]), ys = vis.map((q) => q[1]);
    const bbox = vis.length ? [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] : [0, 0, 1, 1];
    dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name, bbox,
      mask: [], attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true,
      keypoints: { skeleton: PERSON_17.name, points: full } } });
    setKpDraft(null);
  }, [currentClass, dispatch]);

  const onPlaceKeypoint = useCallback((pt: number[]) => {
    const next = [...(kpDraftRef.current ?? []), [pt[0], pt[1], 2]];
    if (next.length >= PERSON_17.points.length) finishKeypoints(next);
    else setKpDraft(next);
  }, [finishKeypoints]);

  const onUpdateKeypoints = useCallback((oid: string, points: number[][]) => {
    const o = stRef.current.objects.find((x) => x.id === oid);
    dispatch({ t: "update", id: oid, patch: { keypoints: { skeleton: o?.keypoints?.skeleton ?? PERSON_17.name, points } } });
  }, [dispatch]);

  // ---- keyboard ----
  useEffect(() => {
    const typing = (t: EventTarget | null) => {
      const el = t as HTMLElement;
      return el && (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA");
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space" && !typing(e.target)) { e.preventDefault(); setPanning(true); return; }
      if (typing(e.target)) return;
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "z") { e.preventDefault(); dispatch(e.shiftKey ? { t: "redo" } : { t: "undo" }); return; }
      if (mod && e.key.toLowerCase() === "s") {
        e.preventDefault();
        // Shift makes it save-as: a named, server-side state that survives a refresh, which the ordinary
        // save does not create and undo cannot get back to.
        if (e.shiftKey) void saveAs(); else save();
        return;
      }
      // Selection, on the modifiers people already expect from every other editor.
      if (mod && e.key.toLowerCase() === "a") {
        e.preventDefault();
        dispatch({ t: "selectBy", how: e.shiftKey ? "sameClass" : "all" });
        return;
      }
      if (mod && e.key.toLowerCase() === "i") {
        e.preventDefault(); dispatch({ t: "selectBy", how: "invert" }); return;
      }
      // Backslash: near Enter on every layout, and unclaimed. Toggling the console has to be reachable
      // without leaving the drawing hand, or it is a panel people open once.
      if (!mod && e.key === "\\") {
        e.preventDefault();
        window.dispatchEvent(new Event("lbx:canvas-console"));
        return;
      }
      if (mod && e.key.toLowerCase() === "c" && st.selectedId) {
        const o = stRef.current.objects.find((x) => x.id === st.selectedId);
        if (o) { clipboardRef.current = o; flash("copied object"); }
        return;
      }
      if (mod && e.key.toLowerCase() === "v" && clipboardRef.current) {
        e.preventDefault();
        const c = clipboardRef.current; const D = 14;
        dispatch({ t: "add", obj: { id: tmpId(), class_id: c.class_id, class_name: c.class_name,
          bbox: [c.bbox[0] + D, c.bbox[1] + D, c.bbox[2] + D, c.bbox[3] + D],
          mask: c.mask.map((poly) => poly.map((val) => val + D)), rot: c.rot,
          attrs: { ...c.attrs }, conf: 1, state: "accepted", visible: true, isNew: true,
          keypoints: c.keypoints ? { skeleton: c.keypoints.skeleton,
            points: c.keypoints.points.map((pt) => (pt[2] > 0 ? [pt[0] + D, pt[1] + D, pt[2]] : pt)) } : undefined } });
        flash("pasted object");
        return;
      }
      if (mod) return;
      // Shift+1..5 switches mode (plain 1..9 stays the quick-relabel shortcut)
      if (e.shiftKey && /^Digit[1-9]$/.test(e.code)) {
        const target = MODES[Number(e.code.slice(5)) - 1];
        if (target) { e.preventDefault(); switchMode(target.key); }
        return;
      }
      const k = e.key.toLowerCase();
      // Review mode rebinds a/x to accept/reject the selected object (and advance the queue).
      if (mode === "review") {
        if (k === "a") { reviewObject("accept"); return; }
        if (k === "x") { reviewObject("reject"); return; }
      }
      if (k === "a") dispatch({ t: "acceptAll" });
      else if (k === "v") dispatch({ t: "tool", tool: "select" });
      else if (k === "b") dispatch({ t: "tool", tool: "box" });
      else if (k === "g") dispatch({ t: "tool", tool: "polygon" });
      else if (k === "l") dispatch({ t: "tool", tool: "polyline" });
      else if (k === "d") dispatch({ t: "tool", tool: "adverse" });
      else if (k === "c") dispatch({ t: "tool", tool: "cuboid" });
      else if (k === "k") dispatch({ t: "tool", tool: "keypoint" });
      else if (k === "r") dispatch({ t: "tool", tool: "measure" });
      else if (k === "s") dispatch({ t: "tool", tool: "sam-point" });
      else if (k === "m") dispatch({ t: "tool", tool: "sam-box" });
      else if (k === "w") dispatch({ t: "tool", tool: "magic-wand" });
      else if (k === "p") dispatch({ t: "tool", tool: "brush" });
      else if (k === "e") dispatch({ t: "tool", tool: "eraser" });
      else if (k === "u") dispatch({ t: "tool", tool: "superpixel" });
      else if (k === "f") fit();
      else if (e.key === "=" || e.key === "+") zoomBy(1.2);
      else if (e.key === "-") zoomBy(1 / 1.2);
      else if (e.key === "Enter") {
        if (stRef.current.tool === "keypoint" && kpDraftRef.current?.length) finishKeypoints(kpDraftRef.current);
        else acceptCandidate();
      }
      else if (e.key === "Escape") {
        // Escape clears in the order a person means it: the in-progress thing first, then the selection.
        // Dropping both at once loses a selection somebody was still using.
        if (st.candidate || kpDraft) { dispatch({ t: "candidate", polys: null }); setKpDraft(null); }
        else if (st.selectedIds.length) dispatch({ t: "selectBy", how: "none" });
      }
      else if ((e.key === "Delete" || e.key === "Backspace") && st.selectedId) dispatch({ t: "delete", id: st.selectedId });
      else if (e.key === "[") gotoFrame(meta?.prev_frame_id ?? null);
      else if (e.key === "]") gotoFrame(meta?.next_frame_id ?? null);
      else if (/^[1-9]$/.test(e.key) && onto) {
        const c = onto.classes[parseInt(e.key, 10) - 1];
        if (c) relabelSelected(c);
      }
    };
    const onUp = (e: KeyboardEvent) => { if (e.code === "Space") setPanning(false); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onUp);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("keyup", onUp); };
  }, [st.selectedId, selected, onto, meta, dispatch, save, fit, zoomBy, acceptCandidate, gotoFrame, relabelSelected, finishKeypoints, mode, alItems]);

  const filteredClasses = useMemo(
    () => (onto ? onto.classes.filter((c) => c.name.includes(search.toLowerCase().replace(/\s/g, "_"))) : []),
    [onto, search],
  );

  if (loadError && (!meta || !onto)) return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="panel px-5 py-4 text-center space-y-3 max-w-sm">
        <div className="text-ink font-medium">Couldn&apos;t load this frame</div>
        <div className="font-mono text-[11px] text-ink-3">{loadError}</div>
        <div className="flex items-center justify-center gap-2">
          <button onClick={() => setReloadKey((k) => k + 1)}
            className="border border-accent text-accent px-3 py-1.5 font-mono text-xs hover:bg-accent/10">Retry</button>
          <button onClick={() => router.push("/")}
            className="border border-line text-ink-2 px-3 py-1.5 font-mono text-xs hover:border-accent">Home</button>
        </div>
      </div>
    </div>
  );
  if (!meta || !onto) return <div className="min-h-screen flex items-center justify-center font-mono text-ink-3">loading frame...</div>;

  // The focused annotation canvas, hoisted so single-frame and rig (M-MC.1) views share one instance and every
  // tool behaves identically in both. In rig view this element renders inside the focused camera tile.
  const editorCanvasEl = (
    <EditorCanvas
      imageUrl={meta.image_url} imgW={meta.width} imgH={meta.height}
      objects={st.objects} selectedId={st.selectedId} tool={st.tool} candidate={st.candidate}
      viewport={st.viewport} panning={panning}
      lanes={lanes} drivable={drivable} layers={layers}
      onViewport={(viewport) => dispatch({ t: "viewport", viewport })}
      onSelect={doSelect}
      selectedIds={st.selectedIds}
      // Marquee selection. The reducer already excluded locked objects and the canvas already knew how to
      // draw a rubber band; what was missing was the gesture on empty canvas that connects the two, so
      // picking six of forty vehicles meant six trips to the object list while they were all on screen.
      onSelectMany={(ids, additive) => dispatch({ t: "selectMany", ids, additive })}
      relationships={relationships}
      onUpdateBbox={(oid, bbox, rot) => dispatch({ t: "update", id: oid, patch: rot !== undefined ? { bbox, rot } : { bbox } })}
      onDrawBox={(bbox) => { if (currentClass) { const nid = tmpId(); dispatch({ t: "add", obj: { id: nid, class_id: currentClass.id, class_name: currentClass.name, bbox, mask: [], attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } }); autoClassify(nid, bbox); } }}
      onDrawPolygon={(pts) => {
        if (!currentClass) return;
        // In semantic mode the same gesture means a class region rather than an instance. Routed here rather
        // than through a second canvas because the drawing is identical and only the destination differs.
        if (mode === "semantic") { void paintSemantic(pts); return; }
        const nid = tmpId(); const bb = bboxOfPolys([pts]);
        dispatch({ t: "add", obj: { id: nid, class_id: currentClass.id, class_name: currentClass.name, bbox: bb, mask: [pts], attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } });
        autoClassify(nid, bb);
      }}
      onDrawPolyline={(pts) => currentClass && dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name, bbox: bboxOfPolys([pts]), mask: [], polyline: Array.from({ length: pts.length / 2 }, (_, i) => [pts[2 * i], pts[2 * i + 1]]), attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } })}
      adverse={adverse}
      onDrawAdverse={async (pts) => { try { await api.createAdverse(id, { geometry: pts, condition: adverseCond }); setAdverse(await api.listAdverse(id).catch(() => [])); flash(`tagged ${adverseCond}`); } catch (e) { flash("region failed: " + humanizeError(e)); } }}
      cuboids={layers.cuboids ? cuboids : []}
      onPlaceCuboid={placeCuboid}
      onMagicWand={runMagicWand}
      brushRadius={brushRadius}
      onBrushStroke={onBrushStroke}
      superpixels={superpixels}
      onPickSuperpixel={pickSuperpixel}
      segOverlayUrl={layers.seg ? segUrl : null}
      keypointDraft={kpDraft} skeletonEdges={PERSON_17.edges as unknown as number[][]}
      onPlaceKeypoint={onPlaceKeypoint} onUpdateKeypoints={onUpdateKeypoints}
      mPerPx={meta.lidar_res ?? undefined}
      onSamPoint={(pt, label) => runSam({ points: [pt], labels: [label] })}
      onSamBox={(box) => runSam({ box })}
      onUpdateMask={(oid, polys) =>
        // Keep the bbox in sync with the edited mask so geometry and segmentation never diverge.
        dispatch({ t: "update", id: oid, patch: polys.length ? { mask: polys, bbox: bboxOfPolys(polys) } : { mask: polys } })}
      onCursor={publishCursor}
    />
  );

  return (
    // min-w-0 and overflow-hidden on the root: a flex column otherwise adopts the widest child's intrinsic
    // width, so the unwrappable top bar widened the entire page and the body scrolled sideways instead of
    // the bar scrolling inside itself. On a tablet that pushed the canvas off screen entirely.
    <div className="h-[100dvh] flex flex-col min-w-0 overflow-hidden">
      {/* The editor has no room to print a page title, but the page still needs a name in an outline view
          or a screen reader, where "untitled document" is the alternative. */}
      <h1 className="sr-only">Frame editor</h1>
      {/* TOP BAR: identity, frame context, global actions, confirm (the design's 46px top bar) */}
      <header className="flex items-center gap-3 px-3 h-[46px] border-b hairline shrink-0
                         min-w-0 overflow-x-auto no-scrollbar">
        <BackButton />
        <button onClick={() => router.push("/")} className="flex items-baseline gap-px" title="home (triage)">
          <span className="font-display font-bold text-[15px] tracking-tight text-ink">Labelox</span>
          <span className="font-mono font-semibold text-[12px] text-accent tracking-tight">AV</span>
        </button>
        <span className="w-px h-5 bg-line" />
        <div className="flex flex-col leading-tight">
          <span className="font-mono text-[11px] text-ink">FRAME {String(id).slice(0, 8)}</span>
          <span className="font-mono text-[9.5px] text-ink-3">{st.objects.length} objects{meta.is_lidar ? " · lidar" : ""}</span>
        </div>
        {meta.annotation_source ? (
          <span title={meta.annotation_source === "imported"
            ? "these labels were imported from a public dataset, not created in this app"
            : "these labels were produced in this app"}>
            <ObjectSourceBadge source={meta.annotation_source} importFormat={meta.import_format} />
          </span>
        ) : null}
        <button onClick={() => router.push(`/search?frame=${id}`)} title="find visually similar frames (DINOv3)"
          className="flex items-center justify-center w-[30px] h-[30px] rounded-md text-ink-3 hover:bg-line/50 hover:text-ink"><Icon name="search" size={16} /></button>
        {meta.has_mcap && (
          <button onClick={() => router.push(`/inspect/${meta.session_id}?ts=${meta.ts_ns}`)} title="inspect this moment in the Session Inspector (MCAP timeline)"
            className="flex items-center justify-center h-[30px] px-2 rounded-md text-ink-3 hover:bg-line/50 hover:text-ink font-mono text-[11px]">inspect</button>
        )}

        <div className="ml-auto flex items-center gap-1.5">
          <button onClick={() => gotoFrame(meta.prev_frame_id)} disabled={!meta.prev_frame_id} title="previous frame ( [ )"
            className="flex items-center justify-center w-[30px] h-[30px] rounded-md text-ink-2 hover:bg-line/50 hover:text-ink disabled:opacity-30"><Icon name="prev" size={18} /></button>
          <button onClick={() => gotoFrame(meta.next_frame_id)} disabled={!meta.next_frame_id} title="next frame ( ] )"
            className="flex items-center justify-center w-[30px] h-[30px] rounded-md text-ink-2 hover:bg-line/50 hover:text-ink disabled:opacity-30"><Icon name="next" size={18} /></button>
          <span className="w-px h-5 bg-line mx-0.5" />
          <button onClick={() => dispatch({ t: "undo" })} disabled={!st.past.length} title="undo (Cmd Z)"
            className="flex items-center justify-center w-[30px] h-[30px] rounded-md text-ink-2 hover:bg-line/50 hover:text-ink disabled:opacity-30"><Icon name="undo" size={17} /></button>
          <button onClick={() => dispatch({ t: "redo" })} disabled={!st.future.length} title="redo (Cmd Shift Z)"
            className="flex items-center justify-center w-[30px] h-[30px] rounded-md text-ink-2 hover:bg-line/50 hover:text-ink disabled:opacity-30"><Icon name="redo" size={17} /></button>
          <button onClick={save} disabled={!dirty || saving} title="save now (Cmd S)"
            className="flex items-center justify-center w-[30px] h-[30px] rounded-md text-ink-2 hover:bg-line/50 hover:text-ink disabled:opacity-30"><Icon name="save" size={17} /></button>
          <button onClick={() => setAutosave((v) => !v)} title="autosave: persist edits a moment after you stop"
            className="flex items-center gap-1.5 px-1.5 h-[30px]">
            <span className={`w-1.5 h-1.5 rounded-full ${saving ? "bg-warn" : dirty ? "bg-ink-3" : "bg-pass"}`} />
            <span className="font-mono text-[10px] text-ink-3">{saving ? "saving" : dirty ? (autosave ? "autosave on" : "unsaved") : "saved"}</span>
          </button>
          {/* The console, over the frame. The editor has its own top bar and so never carried the global
              activity chip, which left the busiest page in the application as the one with no way to ask
              what the machine was doing. */}
          <button onClick={() => openConsole()} title="console: jobs, GPU, host, background work"
            className="flex items-center justify-center w-[30px] h-[30px] rounded-md text-ink-2 hover:bg-line/50 hover:text-ink"><Icon name="activity" size={17} /></button>
          <span className="w-px h-5 bg-line mx-0.5" />
          <CloudControl />
          <span className="w-px h-5 bg-line mx-0.5" />
          {IS_DEMO_BUILD && (
            <button onClick={() => setScaleNoteOpen(true)} title="how this layout scales"
              className="flex items-center gap-1.5 h-[30px] px-2.5 rounded-md border border-line text-ink-2 hover:bg-line/50 hover:text-ink text-[11.5px]"><Icon name="info" size={15} /><span>How it scales</span></button>
          )}
          <button onClick={() => window.dispatchEvent(new Event("lbx:shortcuts"))} title="keyboard shortcuts ( ? )"
            className="flex items-center justify-center w-[30px] h-[30px] rounded-md text-ink-2 hover:bg-line/50 hover:text-ink"><Icon name="keyboard" size={17} /></button>
          <button onClick={() => dispatch({ t: "acceptAll" })} disabled={!st.objects.length} title="confirm every object as human-verified gold (A)"
            className="flex items-center gap-1.5 h-[30px] px-3.5 rounded-md bg-accent text-bg font-display font-semibold text-[12.5px] hover:bg-accent/90 disabled:opacity-40"><Icon name="confirm" size={15} /><span>Confirm frame</span></button>
        </div>
      </header>

      <div className="flex-1 flex min-h-0">
        <ModeRail mode={mode} onMode={switchMode} />
        {/* CENTER: tool strip row above the canvas */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0">
          {/* No overflow-x here: the strip collapses its own tail into an overflow flyout, and a scroll
              container would hide the tail again with no affordance, which is the bug it just fixed. */}
          <div className="h-[50px] shrink-0 flex items-center gap-1.5 px-2.5 border-b hairline overflow-hidden">
          <ToolStrip groups={MODE_GROUPS[mode] ?? MODE_GROUPS.objects} tool={st.tool}
            modeIcon={MODE_ICON[mode]} modeLabel={MODES.find((m) => m.key === mode)?.label}
            onSelect={(t) => dispatch({ t: "tool", tool: t as Tool })}
            options={
              <>
                {mode === "objects" && st.tool === "adverse" && (
                  <select value={adverseCond} onChange={(e) => setAdverseCond(e.target.value)} title="adverse condition to tag"
                    className="bg-bg border border-accent text-accent px-1 py-1 font-mono text-[11px]">
                    {["glare", "reflection", "shadow", "rain", "fog", "lowlight"].map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                )}
                {mode === "objects" && (st.tool === "brush" || st.tool === "eraser") && (
                  <input type="range" min={4} max={60} value={brushRadius} title={`brush radius ${brushRadius}px`}
                    onChange={(e) => setBrushRadius(Number(e.target.value))} className="w-20" />
                )}
                {mode === "lanes" && (
                  <div className="flex items-center gap-1 font-mono text-[11px]">
                    <button onClick={() => setLaneAdding(laneAdding ? null : [])} title="draw a new lane spline (click to add control points)"
                      className={`border px-2 py-1 ${laneAdding ? "border-accent text-accent" : "border-line text-ink-3 hover:border-accent"}`}>+ lane</button>
                    {laneAdding ? (
                      <>
                        <span className="text-ink-3">{laneAdding.length} pts:</span>
                        {["solid", "implicit"].map((t) => (
                          <button key={t} onClick={() => finishAddLane(t)} className="border border-line text-ink-2 px-2 py-1 hover:border-accent">{t}</button>
                        ))}
                        <button onClick={() => setLaneAdding(null)} className="text-ink-3 hover:text-block px-1">cancel</button>
                      </>
                    ) : (
                      <>
                        <button onClick={segRoad} className="border border-line text-ink-3 px-2 py-1 hover:border-accent">drivable</button>
                        <button onClick={genLanes} className="border border-line text-ink-3 px-2 py-1 hover:border-accent">propose</button>
                        <button onClick={propagateLanes} className="border border-line text-ink-3 px-2 py-1 hover:border-accent">propagate</button>
                        <button onClick={saveLanes} className="border border-pass text-pass px-2 py-1 hover:bg-pass/10">save lanes</button>
                      </>
                    )}
                  </div>
                )}
              </>
            } />
          </div>
        <div ref={canvasWrapRef} className="flex-1 min-w-0 relative">
          {/* What the canvas is doing, in the canvas. Everything here ran fire-and-forget with a one-line
              flash, so a slow call and a call that never went looked the same from the image. */}
          <CanvasConsole />
          {mode === "events" ? (
            // Opaque: this panel replaces the image rather than floating over it, and inheriting the
            // transparent canvas let what is behind print through the controls.
            <div className="absolute inset-0 overflow-auto bg-bg p-3 space-y-2">
              <div className="flex items-center gap-2">
                <button onClick={() => void loadFrameEvents()}
                  className="border border-line text-ink-2 px-2 py-1 font-mono text-[11px] hover:border-accent">
                  load session events
                </button>
                <button onClick={() => router.push("/events")}
                  className="border border-line text-ink-3 px-2 py-1 font-mono text-[11px] hover:border-accent">
                  derive and review &rarr;
                </button>
                <span className="font-mono text-[10px] text-ink-3">{frameEvents.length} events</span>
              </div>
              <p className="font-mono text-[10px] text-ink-3 max-w-[70ch]">
                Behaviour, not boxes: what an actor did over time. A lane change is a track crossing a lane
                boundary and staying across it; a signal phase is one contiguous state of one light. Every
                one is a candidate until somebody rules on it.
              </p>
              {frameEvents.length > 0 && (
                <table className="w-full font-mono text-[11px]">
                  <thead className="text-ink-3 text-left">
                    <tr><th className="py-1">kind</th><th>severity</th><th>start</th><th>conf</th><th>state</th></tr>
                  </thead>
                  <tbody>
                    {frameEvents.map((ev) => (
                      <tr key={ev.event_id} className="border-t border-line">
                        <td className="py-1">{ev.kind}</td>
                        <td className={ev.severity === "violation" ? "text-block" : ev.severity === "notable" ? "text-warn" : "text-ink-3"}>{ev.severity}</td>
                        <td className="tabular-nums">{(ev.t_start_ns / 1e9).toFixed(2)}s</td>
                        <td className="tabular-nums">{ev.conf == null ? "-" : ev.conf.toFixed(2)}</td>
                        <td className={ev.state === "confirmed" ? "text-pass" : ev.state === "review" ? "text-warn" : "text-ink-3"}>{ev.state}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          ) : mode === "lanes" ? (
            laneImg && meta ? (
              <LaneCanvas img={laneImg} meta={{ width: meta.width, height: meta.height }} scale={laneScale}
                lanes={lanes} sel={laneSel} drivable={layers.drivable ? drivable : null} adding={laneAdding}
                addingArea={areaAdding} addingAreaClass={areaClass}
                areaSel={areaSel} onSelectArea={(k) => { setAreaSel(k); setLaneSel(null); }}
                onDragAreaPoint={dragAreaPoint}
                onStageClick={laneStageClick} onSelect={(lid) => { setLaneSel(lid); setAreaSel(null); }}
                onDragPoint={laneDragPoint} />
            ) : <div className="absolute inset-0 grid place-items-center font-mono text-[11px] text-ink-3">loading lanes...</div>
          ) : mode === "lidar3d" ? (
            pts3d ? (
              <div className="absolute inset-0 flex flex-col">
                <div className="relative flex-1 min-h-0">
                  <span className="absolute left-2 top-1 z-10 font-mono text-[10px] text-ink-3 uppercase">3d</span>
                  <PointCloudViewer points={pts3d.points} count={pts3d.count} colorBy={colorBy3d}
                    intensityRange={[pts3d.intensityMin, pts3d.intensityMax]} source={pts3d.source} mode="perspective"
                    cuboids={cub3d} selectedId={cubSel} onSelectCuboid={setCubSel} />
                </div>
                <div className="relative h-2/5 min-h-[200px] border-t hairline">
                  <span className="absolute left-2 top-1 z-10 font-mono text-[10px] text-ink-3 uppercase">bev (drag to move)</span>
                  <PointCloudViewer points={pts3d.points} count={pts3d.count} colorBy={colorBy3d}
                    intensityRange={[pts3d.intensityMin, pts3d.intensityMax]} source={pts3d.source} mode="bev" pointSize={0.4}
                    cuboids={cub3d} selectedId={cubSel} onSelectCuboid={setCubSel} onMoveCuboid={moveCub} />
                </div>
              </div>
            ) : <div className="absolute inset-0 grid place-items-center font-mono text-[11px] text-ink-3">{lidarMsg ?? "loading point cloud..."}</div>
          ) : rigView && rigMulti && rigGroup ? (
            <RigView cameras={rigGroup.cameras} focusedCam={meta.cam_id} frameIds={rigGroup.frameIds}
              missingCams={rigGroup.missingCams} layout={rigLayout} onFocus={focusCamOnce}>
              {editorCanvasEl}
            </RigView>
          ) : (
            editorCanvasEl
          )}
          {/* Nearby frames, under the canvas where a scrubber belongs. Hidden in the 3D and rig views, whose
              canvases are not a single camera's timeline. */}
          {mode !== "lidar3d" && !rigView && (
            <div className="absolute bottom-0 left-0 right-0 z-10">
              <Filmstrip frameId={id} onPick={gotoFrame} />
            </div>
          )}
          {mode !== "lidar3d" && <FloatingLayers layers={layers} meta={layerMeta} onToggle={(k) => setLayers((s) => ({ ...s, [k]: !s[k as keyof typeof s] }))}
            extra={
              <>
                <select value={segKind} onChange={(e) => setSegKind(e.target.value as "semantic" | "panoptic")}
                  title="dense segmentation kind" className="w-full bg-bg border border-line px-1 py-0.5 text-ink-3">
                  <option value="semantic">semantic</option>
                  <option value="panoptic">panoptic</option>
                </select>
                <button title="run dense segmentation (SAM-everything + VLM) on this frame"
                  onClick={async () => { flash("segmenting..."); try { const r = await api.autoSegment(id, segKind); setSegUrl(`/api/frames/${id}/segment/overlay?kind=${segKind}&t=${Date.now()}`); flash(`segmented ${segKind} (${Object.keys(r.coverage).length} classes${r.n_instances ? ", " + r.n_instances + " instances" : ""})`); } catch (e) { flash("segment failed: " + humanizeError(e)); } }}
                  className="w-full border border-line px-1 py-0.5 text-ink-3 hover:border-accent">auto-seg</button>
              </>
            } />}
          {/* Review mode: a quiet bottom-center action bar so the reviewer accepts/rejects without leaving the canvas */}
          {mode === "review" && selected && (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 panel px-3 py-1.5 flex items-center gap-3 font-mono text-[11px]">
              <span className="text-ink-2">{selected.class_name}</span>
              <ConfBar conf={selected.conf} />
              <button onClick={() => reviewObject("accept")} className="border border-pass text-pass px-2 py-0.5 hover:bg-pass/10">accept (A)</button>
              <button onClick={() => reviewObject("reject")} className="border border-block text-block px-2 py-0.5 hover:bg-block/10">reject (X)</button>
              <button onClick={() => advanceReview(selected.id)} className="text-ink-3 hover:text-ink">skip</button>
            </div>
          )}
          {st.candidate?.length ? (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 panel px-3 py-1.5 font-mono text-[11px] text-ink-2">
              mask ready, click the next object to keep this one &amp; continue, <span className="text-pass">Enter</span> to finish, <span className="text-ink-3">Esc</span> to discard
            </div>
          ) : null}

          {/* inline edit popup: click a (wrong) annotation to fix its class right where it sits */}
          {mode !== "lanes" && mode !== "lidar3d" && selected && st.tool === "select" && st.viewport.scale > 0 && (() => {
            const wrap = canvasWrapRef.current;
            const v = st.viewport;
            const sx = selected.bbox[0] * v.scale + v.ox;
            const sy = selected.bbox[1] * v.scale + v.oy;
            const left = Math.max(4, Math.min(sx, (wrap?.clientWidth ?? 9999) - 232));
            const top = Math.max(4, Math.min(sy - 6, (wrap?.clientHeight ?? 9999) - (editOpen ? 240 : 44)));
            return (
              <div className="absolute z-20" style={{ left, top }}>
                <div className="panel border border-line shadow-xl">
                  {!editOpen ? (
                    <div className="flex items-center gap-2 px-2 py-1 font-mono text-[11px]">
                      <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: classColor(selected.class_id) }} />
                      <span className="text-ink-2 truncate max-w-[120px]" title={selected.class_name}>{selected.class_name}</span>
                      <button onClick={() => { setEditOpen(true); setEditSearch(""); }}
                        className="text-accent hover:underline">edit</button>
                      <button onClick={() => dispatch({ t: "select", id: null })}
                        className="text-ink-3 hover:text-ink" title="close">x</button>
                    </div>
                  ) : (
                    <div className="w-56 p-1.5">
                      <div className="flex items-center justify-between mb-1 px-0.5">
                        <span className="font-mono text-[10px] uppercase text-ink-3">fix class</span>
                        <button onClick={() => setEditOpen(false)} className="font-mono text-[10px] text-ink-3 hover:text-ink">close</button>
                      </div>
                      <input autoFocus value={editSearch} onChange={(e) => setEditSearch(e.target.value)} placeholder="search or add class..."
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            const norm = normClass(editSearch);
                            const exact = editClasses.find((c) => c.name === norm);
                            if (exact) { relabelSelected(exact); setEditOpen(false); }
                            else if (norm) addAndRelabel(editSearch);
                          } else if (e.key === "Escape") setEditOpen(false);
                        }}
                        className="w-full bg-panel border border-line px-2 py-1 font-mono text-[11px] text-ink mb-1" />
                      <div className="max-h-40 overflow-auto space-y-0.5">
                        {editSearch.trim() && normClass(editSearch) && !editClasses.some((c) => c.name === normClass(editSearch)) && (
                          <button onClick={() => addAndRelabel(editSearch)}
                            className="w-full flex items-center gap-1.5 px-1 py-0.5 font-mono text-[11px] text-left text-accent hover:bg-line">
                            <span className="shrink-0">+</span>
                            <span className="truncate">add &quot;{normClass(editSearch)}&quot;</span>
                          </button>
                        )}
                        {editClasses.slice(0, 50).map((c) => (
                          <button key={c.id} onClick={() => { relabelSelected(c); setEditOpen(false); }}
                            className={`w-full flex items-center gap-1.5 px-1 py-0.5 font-mono text-[11px] text-left ${c.id === selected.class_id ? "text-ink" : "text-ink-3"} hover:text-ink`}>
                            <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: classColor(c.id) }} />
                            <span className="truncate">{c.name}</span>
                            {c.india && <span className="ml-auto text-accent">*</span>}
                          </button>
                        ))}
                        {!editClasses.length && <div className="text-ink-3 text-center py-2 font-mono text-[10px]">no match</div>}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })()}
          {notice && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 panel px-3 py-1.5 font-mono text-[11px] text-warn">{notice}</div>
          )}
          {/* HUD: frame time and camera, a quiet overlay top-left (the design's HUD). Suppressed in events
              mode, where the canvas is a table rather than an image: a frame timestamp and a cursor position
              describe something that is not on screen, and the overlay printed through the controls. */}
          {meta && mode !== "events" && (
            <div className="absolute top-3 left-3 z-10 flex flex-col gap-1 pointer-events-none">
              <span className="font-mono text-[11px] text-ink-2 bg-bg/60 px-1.5 py-0.5 rounded w-fit">{new Date(Number(meta.ts_ns) / 1e6).toISOString().replace("T", " ").replace("Z", "")}</span>
              <span className="font-mono text-[11px] text-ink-3 bg-bg/60 px-1.5 py-0.5 rounded w-fit">{camLabel(meta.cam_id)}{meta.is_lidar ? " · lidar" : ""}<CursorReadout /></span>
            </div>
          )}

          {/* M-MC.1 rig view control: a canvas view-state cluster (NOT a mode or tool). Only when this frame is
              part of a multi-camera group and the current mode uses the annotation canvas. */}
          {rigMulti && rigEditable && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-20 flex items-center gap-1 font-mono text-[10px] panel/80 bg-bg/70 px-1.5 py-1 rounded">
              <button onClick={() => setRigView((v) => !v)} title="toggle the multi-camera rig view (this camera stays focused)"
                className={`border px-2 py-0.5 rounded ${rigView ? "border-accent text-accent bg-accent/10" : "border-line text-ink-3 hover:border-accent"}`}>
                rig {rigGroup ? `${Object.keys(rigGroup.frameIds).length}/${rigGroup.cameras.length}` : ""}
              </button>
              {rigView && (
                <>
                  <div className="flex border border-line rounded overflow-hidden">
                    {(["focus", "grid", "strip"] as const).map((l) => (
                      <button key={l} onClick={() => setRigLayout(l)} title={`${l} layout`}
                        className={`px-1.5 py-0.5 ${rigLayout === l ? "bg-accent/20 text-accent" : "text-ink-3 hover:text-ink"}`}>{l}</button>
                    ))}
                  </div>
                  <button onClick={() => navGroup("prev")} title="previous synchronized group" className="border border-line px-1.5 py-0.5 rounded text-ink-3 hover:border-accent">◂</button>
                  <button onClick={() => navGroup("next")} title="next synchronized group" className="border border-line px-1.5 py-0.5 rounded text-ink-3 hover:border-accent">▸</button>
                  {rigGroup && rigGroup.missingCams.length > 0 && (
                    <span className="border border-block/50 text-block px-1.5 py-0.5 rounded" title="cameras with no frame in this group">
                      {rigGroup.missingCams.length} dropped
                    </span>
                  )}
                  {/* M-MC.3 tier chip: Tier 2 projects across views (calibrated); Tier 1 is manual link only */}
                  {rigCalibrated !== null && (
                    <span title={rigCalibrated ? "calibrated: annotate once and project into the other views" : "not calibrated: use manual linking (Tier 1). Run calibration validation to enable projection."}
                      className={`border px-1.5 py-0.5 rounded uppercase ${rigCalibrated ? "border-pass/60 text-pass" : "border-warn/60 text-warn"}`}>
                      {rigCalibrated ? "tier 2" : "tier 1"}
                    </span>
                  )}
                  {rigCalibrated && selected && (
                    <button onClick={propagateSelected} title="annotate once, project the selected object into the other views"
                      className="border border-info/60 text-info bg-info/10 px-2 py-0.5 rounded hover:bg-info/20">propagate</button>
                  )}
                  <button onClick={() => setRigPanel((v) => !v)} title="rig identities: link the same object across views"
                    className={`border px-2 py-0.5 rounded ${rigPanel ? "border-accent text-accent bg-accent/10" : "border-line text-ink-3 hover:border-accent"}`}>identities</button>
                  <button onClick={() => setRigTracks((v) => !v)} title="rig tracks over time + cross-view consistency check"
                    className={`border px-2 py-0.5 rounded ${rigTracks ? "border-accent text-accent bg-accent/10" : "border-line text-ink-3 hover:border-accent"}`}>tracks</button>
                  <button onClick={confirmRigGroup} title="confirm the whole group at once"
                    className={`border px-2 py-0.5 rounded ${rigGroup?.confirmed ? "border-pass text-pass bg-pass/10" : "border-line text-ink-3 hover:border-pass"}`}>
                    {rigGroup?.confirmed ? "confirmed" : "confirm group"}
                  </button>
                </>
              )}
            </div>
          )}

          {/* M-MC.2 rig identity panel: rig-first object list, manual link, appearance-suggest, unlink */}
          {rigView && rigMulti && rigPanel && rigGroup && (
            <RigIdentityPanel key={rigRefresh} sessionId={meta.session_id} groupId={rigGroup.groupId} onClose={() => setRigPanel(false)}
              onSelectObject={(oid, cam) => {
                if (cam === meta.cam_id) doSelect(oid);
                else if (rigGroup.frameIds[cam]) router.push(`/frame/${rigGroup.frameIds[cam]}?rig=${rigLayout}&focus=${oid}`);
              }} />
          )}

          {/* M-MC.4 rig track timeline + consistency check */}
          {rigView && rigMulti && rigTracks && (
            <RigTrackPanel sessionId={meta.session_id} onClose={() => setRigTracks(false)}
              onOpenInstant={(oid, cam) => {
                if (cam === meta.cam_id) doSelect(oid);
                else if (rigGroup?.frameIds[cam]) router.push(`/frame/${rigGroup.frameIds[cam]}?rig=${rigLayout}&focus=${oid}`);
              }} />
          )}
        </div>
        </div>

        {/* right rail: contextual properties panel, collapsible to give the canvas the full width */}
        {rightCollapsed ? (
          <div className="w-8 shrink-0 border-l hairline flex flex-col items-center pt-2 bg-bg">
            <button onClick={() => setRightCollapsed(false)} title="expand panel"
              className="w-6 h-6 flex items-center justify-center rounded text-ink-3 hover:bg-line/50 hover:text-ink"><Icon name="chevL" size={14} /></button>
            <span className="mt-2 [writing-mode:vertical-rl] font-display text-[10px] uppercase tracking-wider text-ink-3">Properties</span>
          </div>
        ) : (
          // Fixed width on a desktop, an overlay on anything narrower. At 340px fixed, a 768px tablet is
          // left with 428px of canvas, which is not an annotation surface; the collapse toggle above
          // already exists, so below the breakpoint the panel floats over the canvas instead of eating it.
          <aside className="w-[340px] shrink-0 border-l hairline flex flex-col min-h-0
                          max-lg:absolute max-lg:right-0 max-lg:top-0 max-lg:bottom-0 max-lg:z-30
                          max-lg:bg-panel max-lg:shadow-xl">
          <div className="h-[38px] shrink-0 flex items-center gap-2 px-3 border-b hairline">
            <span className="font-display font-semibold text-[12.5px] text-ink">
              {selected ? selected.class_name : mode === "review" ? "Review" : mode === "lanes" ? "Lanes" : mode === "lidar3d" ? "Cuboids" : "Properties"}
            </span>
            <span className="font-mono text-[10px] text-ink-3">{mode === "lanes" ? `${lanes.length} lanes` : mode === "lidar3d" ? `${cub3d.length} cuboids` : `${st.objects.length} objects`}</span>
            <button onClick={() => setRightCollapsed(true)} title="collapse panel"
              className="ml-auto w-6 h-6 flex items-center justify-center rounded text-ink-3 hover:bg-line/50 hover:text-ink"><Icon name="chevR" size={14} /></button>
          </div>
          {/* Lanes mode: the panel routes to lane content (list + selected lane props) instead of objects */}
          {mode === "lanes" && (
            <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2 font-mono text-[11px]">
              <div className="text-ink-3 uppercase text-[10px]">lanes ({lanes.length})</div>
              {lanes.map((l) => (
                <div key={l.lane_id} onClick={() => setLaneSel(l.lane_id)}
                  className={`flex items-center gap-1.5 cursor-pointer ${l.lane_id === laneSel ? "text-ink" : "text-ink-3 hover:text-ink-2"}`}>
                  <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: l.is_ego ? "#56D364" : (LANE_COLOR[l.source] || "#A0A6AD") }} />
                  <span className="truncate flex-1">{l.lane_type}{l.is_ego ? " (ego)" : ""}</span>
                  <span className="text-ink-3">{l.source[0]}</span>
                </div>
              ))}
              {!lanes.length && <div className="text-ink-3 py-4 text-center">no lanes. propose, or + lane to draw.</div>}
              {(() => {
                const sl = lanes.find((l) => l.lane_id === laneSel);
                if (!sl) return null;
                return (
                  <div className="border-t hairline pt-2 space-y-2">
                    <div className="text-ink-3 uppercase text-[10px]">selected lane</div>
                    <select value={sl.lane_type} onChange={(e) => setLaneType(e.target.value)} className="w-full bg-bg border border-line px-1 py-0.5 text-ink">
                      {LANE_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                    <button onClick={toggleLaneEgo} className={`w-full border px-2 py-1 ${sl.is_ego ? "border-pass text-pass" : "border-line text-ink-3"}`}>{sl.is_ego ? "ego lane set" : "mark ego"}</button>
                    <button onClick={delLane} className="w-full border border-line text-ink-3 px-2 py-1 hover:border-block hover:text-block">delete lane</button>
                  </div>
                );
              })()}

              {/* Drivable surface. The PUT behind this has existed since M2.2 and nothing could reach it,
                  so `source` was `proposed` on 2,478 of 2,479 masks: the layer was machine output a person
                  could look at and not correct. */}
              <div className="border-t hairline pt-2 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-ink-3 uppercase text-[10px]">drivable surface</span>
                  <span className="text-ink-3">
                    {drivable
                      ? `${Object.values(drivable).reduce((n, ps) => n + ps.length, 0)} regions`
                      : "none"}
                  </span>
                </div>

                {areaAdding ? (
                  <>
                    <div className="text-ink-2">
                      click to add points ({areaAdding.length}), then close the region
                    </div>
                    <div className="grid grid-cols-2 gap-1">
                      <button onClick={finishArea} disabled={areaAdding.length < 3}
                        className="border border-pass text-pass px-2 py-1 disabled:opacity-40">close region</button>
                      <button onClick={() => setAreaAdding(null)}
                        className="border border-line text-ink-3 px-2 py-1 hover:border-block">cancel</button>
                    </div>
                  </>
                ) : (
                  <>
                    <div className="grid grid-cols-3 gap-1">
                      {SURFACE_CLASSES.map((c) => (
                        <button key={c.key}
                          onClick={() => { setAreaClass(c.key); setAreaAdding([]); setAreaSel(null); }}
                          title={`draw a ${c.label} region`}
                          className={`border px-1 py-1 text-[10px] ${c.tone}`}>
                          + {c.label}
                        </button>
                      ))}
                    </div>
                    <button onClick={segRoad}
                      className="w-full border border-line text-ink-3 px-2 py-1 hover:border-accent">
                      re-run the model
                    </button>
                  </>
                )}

                {areaSel && (
                  <div className="border-t hairline pt-2 space-y-1">
                    <div className="text-ink-3 uppercase text-[10px]">
                      selected {areaSel.split(":")[0].replace("_", " ")} region
                    </div>
                    <div className="text-ink-3 text-[10px]">
                      drag a handle to pull the boundary onto the kerb, then save
                    </div>
                    <div className="grid grid-cols-2 gap-1">
                      <button onClick={() => drivable && void saveDrivable(drivable)}
                        className="border border-pass text-pass px-2 py-1">save edits</button>
                      <button onClick={deleteArea}
                        className="border border-line text-ink-3 px-2 py-1 hover:border-block hover:text-block">delete</button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
          {/* 3D and LiDAR mode: the panel routes to the cuboid list plus selected-cuboid geometry */}
          {mode === "lidar3d" && (
            <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2 font-mono text-[11px]">
              <div className="flex items-center justify-between">
                <span className="text-ink-3 uppercase text-[10px]">cuboids ({cub3d.length})</span>
                <button onClick={addCub} disabled={!cloud3d} className="border border-line text-ink-3 px-1.5 py-0.5 hover:border-accent disabled:opacity-40">+ box</button>
              </div>
              <button onClick={aiLift3d} disabled={!cloud3d} className="w-full border border-line text-ink-2 px-2 py-1 hover:border-accent disabled:opacity-40">AI lift 2D to 3D</button>
              {cub3d.map((c) => (
                <div key={c.object_3d_id} onClick={() => setCubSel(c.object_3d_id)}
                  className={`flex items-center justify-between gap-1.5 cursor-pointer ${c.object_3d_id === cubSel ? "text-ink" : "text-ink-3 hover:text-ink-2"}`}>
                  <span className="truncate flex-1">{c.class_name}</span>
                  <span className="text-ink-3">{c.state} {c.box_source}</span>
                </div>
              ))}
              {!cub3d.length && <div className="text-ink-3 py-4 text-center">{lidarMsg ?? "no cuboids. lift or + box."}</div>}
              {cubSelected && (
                <div className="border-t hairline pt-2 space-y-2">
                  <div className="text-ink-3 uppercase text-[10px]">selected ({cubSelected.source})</div>
                  <select value={cubSelected.class_id}
                    onChange={(e) => { const cid = Number(e.target.value); patchCub(cubSelected.object_3d_id, { class_id: cid, class_name: onto?.classes.find((x) => x.id === cid)?.name }); saveCub(cubSelected.object_3d_id, { class_id: cid }); }}
                    className="w-full bg-bg border border-line px-1 py-0.5 text-ink">
                    {onto?.classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                  </select>
                  {["L", "W", "H"].map((lab, i) => (
                    <label key={lab} className="flex items-center gap-2">
                      <span className="w-3 text-ink-3">{lab}</span>
                      <input type="range" min={0.3} max={14} step={0.1} value={cubSelected.dims[i]}
                        onChange={(e) => patchCub(cubSelected.object_3d_id, { dims: cubSelected.dims.map((d, j) => (j === i ? Number(e.target.value) : d)) })}
                        onMouseUp={() => saveCub(cubSelected.object_3d_id, { dims: cubSelected.dims })} className="flex-1" />
                      <span className="w-9 text-right text-ink-3">{cubSelected.dims[i].toFixed(1)}</span>
                    </label>
                  ))}
                  <label className="flex items-center gap-2">
                    <span className="w-3 text-ink-3">yaw</span>
                    <input type="range" min={-3.14159} max={3.14159} step={0.01} value={cubSelected.yaw}
                      onChange={(e) => patchCub(cubSelected.object_3d_id, { yaw: Number(e.target.value) })}
                      onMouseUp={() => saveCub(cubSelected.object_3d_id, { yaw: cubSelected.yaw })} className="flex-1" />
                    <span className="w-9 text-right text-ink-3">{(cubSelected.yaw * 57.3).toFixed(0)}</span>
                  </label>
                  {/* 9-DOF: pitch (up/down a ramp) and roll (side bank). Auto-estimated from the ground on
                      lift; editable here for slopes the estimator clamped or missed. */}
                  {([["pitch", cubSelected.pitch ?? 0], ["roll", cubSelected.roll ?? 0]] as const).map(([axis, val]) => (
                    <label key={axis} className="flex items-center gap-2">
                      <span className="w-3 text-ink-3" title={axis === "pitch" ? "tilt up/down a ramp" : "side bank"}>{axis === "pitch" ? "pit" : "rol"}</span>
                      <input type="range" min={-0.6} max={0.6} step={0.01} value={val}
                        onChange={(e) => patchCub(cubSelected.object_3d_id, { [axis]: Number(e.target.value) })}
                        onMouseUp={() => saveCub(cubSelected.object_3d_id, { [axis]: axis === "pitch" ? cubSelected.pitch : cubSelected.roll })} className="flex-1" />
                      <span className="w-9 text-right text-ink-3">{(val * 57.3).toFixed(0)}</span>
                    </label>
                  ))}
                  <div className="flex items-center gap-1 pt-0.5">
                    {(["height", "intensity", "source", "segment"] as ColorBy[]).map((kb) => (
                      <button key={kb} onClick={() => setColorBy3d(kb)} className={`px-1.5 py-0.5 border ${colorBy3d === kb ? "border-accent text-accent" : "border-line text-ink-3"}`}>{kb}</button>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <button onClick={() => saveCub(cubSelected.object_3d_id, { attrs: { ground_snap: true } })} className="flex-1 border border-line text-ink-2 px-2 py-1 hover:border-accent">ground snap</button>
                    <button onClick={() => delCub(cubSelected.object_3d_id)} className="border border-line text-ink-3 px-2 py-1 hover:border-block hover:text-block">delete</button>
                  </div>
                  <div className="text-ink-3">drag the box in the BEV view to move it.</div>
                </div>
              )}
            </div>
          )}
          {/* Review mode: the panel becomes the value queue (highest-value items + error candidates) with
              per-object accept/reject. Canvas stays Konva; reviewer pans/zooms and clicks objects. */}
          {mode === "review" && (
            <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2 font-mono text-[11px]">
              {selected ? (
                <div className="border-b hairline pb-2 space-y-1.5">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: classColor(selected.class_id) }} />
                    <span className="truncate flex-1">{selected.class_name}</span>
                    <StateBadge state={selected.state} />
                  </div>
                  <ConfBar conf={selected.conf} />
                  <div className="flex gap-1">
                    <button onClick={() => reviewObject("accept")} className="flex-1 border border-pass text-pass px-2 py-1 hover:bg-pass/10">accept (A)</button>
                    <button onClick={() => reviewObject("reject")} className="flex-1 border border-block text-block px-2 py-1 hover:bg-block/10">reject (X)</button>
                  </div>
                </div>
              ) : <div className="text-ink-3 border-b hairline pb-2">click an object on the canvas to accept or reject it.</div>}
              <div className="text-ink-3 uppercase text-[10px]">value queue ({alItems.length})</div>
              {alItems.slice(0, 60).map((it) => (
                <button key={it.object_id} onClick={() => (it.frame_id === id ? doSelect(it.object_id) : gotoFrame(it.frame_id))}
                  className={`block w-full text-left border-b hairline pb-1.5 ${it.object_id === st.selectedId ? "text-ink" : "text-ink-3 hover:text-ink-2"}`}>
                  <div className="flex items-center gap-1.5">
                    <span className="truncate flex-1">{it.class_name}</span>
                    {it.frame_id !== id && <span className="text-info text-[10px]">other frame</span>}
                    <span className="text-accent">{it.value.toFixed(3)}</span>
                  </div>
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-0.5">
                    <ConfBar conf={it.conf} />
                    <ScoreBar label="u" value={it.scores.uncertainty} showValue={false} />
                    <ScoreBar label="d" value={it.scores.diversity} showValue={false} />
                    <ScoreBar label="r" value={it.scores.rarity} showValue={false} />
                    <ScoreBar label="e" value={it.scores.error_prone} showValue={false} tone="warn" />
                  </div>
                </button>
              ))}
              {!alItems.length && <div className="text-ink-3 py-2 text-center">{reviewLoaded ? "value queue empty" : "loading value queue..."}</div>}
              {errItems.length > 0 && (
                <>
                  <div className="text-ink-3 uppercase text-[10px] pt-1">error candidates ({errItems.length})</div>
                  {errItems.slice(0, 40).map((ec) => (
                    <div key={ec.candidate_id} className="border-b hairline pb-1.5 space-y-0.5">
                      <div className="flex items-center gap-1.5">
                        <span className="truncate flex-1">{ec.kind}</span>
                        <span className="text-info">{ec.proposed_label?.class_name || "(review)"}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <ScoreBar value={ec.score} tone="warn" />
                        <button onClick={async () => { await api.errorConfirm(ec.candidate_id); setErrItems((s) => s.filter((x) => x.candidate_id !== ec.candidate_id)); flash("confirmed error"); }} className="text-block hover:text-accent">confirm</button>
                        <button onClick={async () => { await api.errorDismiss(ec.candidate_id); setErrItems((s) => s.filter((x) => x.candidate_id !== ec.candidate_id)); }} className="text-ink-3 hover:text-ink">dismiss</button>
                      </div>
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
          {mode !== "lanes" && mode !== "lidar3d" && mode !== "review" && (<>
          {/* right-panel tabs: the object-annotation workflow vs the AI/automation tools, so neither buries the
              other in one long scroll */}
          <div className="flex border-b hairline shrink-0 font-mono text-[10px] uppercase tracking-wide">
            {(["objects", "tools"] as const).map((t) => (
              <button key={t} onClick={() => setRightTab(t)}
                className={`flex-1 py-2 ${rightTab === t ? "text-accent border-b-2 border-accent -mb-px" : "text-ink-3 hover:text-ink-2"}`}>{t}</button>
            ))}
          </div>

          {/* class palette (objects tab) */}
          {rightTab === "objects" && (
          <div className="border-b hairline p-2">
            <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">class for new / selected</div>
            <div className="font-mono text-xs text-ink mb-1 flex items-center gap-1.5">
              <span className="w-3 h-3 inline-block" style={{ background: currentClass ? classColor(currentClass.id) : "#333" }} />
              {currentClass?.name ?? "-"}
            </div>
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="search or add class..."
              onKeyDown={(e) => { if (e.key === "Enter") { const n = normClass(search); const ex = filteredClasses.find((c) => c.name === n); if (ex) relabelSelected(ex); else if (n) addAndRelabel(search); } }}
              className="w-full bg-panel border border-line px-2 py-1 font-mono text-[11px] text-ink mb-1" />
            <div className="max-h-32 overflow-auto space-y-0.5">
              {search.trim() && normClass(search) && !filteredClasses.some((c) => c.name === normClass(search)) && (
                <button onClick={() => addAndRelabel(search)}
                  className="w-full flex items-center gap-1.5 px-1 py-0.5 font-mono text-[11px] text-left text-accent hover:text-ink">
                  <span className="shrink-0">+</span><span className="truncate">add &quot;{normClass(search)}&quot; as custom class</span>
                </button>
              )}
              {filteredClasses.slice(0, 40).map((c, i) => (
                <button key={c.id} onClick={() => relabelSelected(c)}
                  className={`w-full flex items-center gap-1.5 px-1 py-0.5 font-mono text-[11px] text-left ${currentClass?.id === c.id ? "text-ink" : "text-ink-3"} hover:text-ink`}>
                  <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: classColor(c.id) }} />
                  <span className="truncate">{c.name}</span>
                  {search === "" && i < 9 && <span className="ml-auto text-ink-3">{i + 1}</span>}
                  {c.india && <span className="text-accent">*</span>}
                </button>
              ))}
            </div>
          </div>
          )}

          {/* scroll body: shows the objects-tab content or the tools-tab content depending on the active tab.
              Keyed by tab so switching fades the new content in. */}
          <div key={rightTab} className="flex-1 min-h-0 overflow-y-auto reveal">

          {/* attributes of selected (objects tab) */}
          {rightTab === "objects" && selected && (
            <div className="border-b hairline p-2">
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-[10px] uppercase text-ink-3">attributes</span>
                {selected.track_id && (
                  <button onClick={() => router.push(`/track/${selected.track_id}`)}
                    className="font-mono text-[10px] text-info hover:text-accent">view track &rarr;</button>
                )}
              </div>
              {/* the selected object's identity at a glance: class, calibrated confidence, state */}
              <div className="flex items-center gap-2 mb-1.5">
                <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: classColor(selected.class_id) }} />
                <span className="font-mono text-[11px] text-ink truncate flex-1">{selected.class_name}</span>
                {selected.quality_score != null && (
                  <span title="M-F.1 label quality score (0-1): calibrated correctness, penalised for geometric/consistency defects"
                    className={`font-mono text-[10px] px-1 rounded border ${selected.quality_score >= 0.4 ? "border-pass/50 text-pass" : selected.quality_score >= 0.25 ? "border-warn/50 text-warn" : "border-block/50 text-block"}`}>
                    Q {selected.quality_score.toFixed(2)}
                  </span>
                )}
                <ConfBar conf={selected.conf} />
                <StateBadge state={selected.state} />
              </div>
              {/* provenance: real identity, version, and which geometry this object carries (no fabricated detector names) */}
              <div className="flex flex-col gap-1.5 bg-bg-2 border border-line rounded p-2 mb-1.5 font-mono text-[10px]">
                <div className="flex items-center"><span className="text-ink-3 w-16 shrink-0">object</span><span className="text-ink-2 truncate">{selected.isNew ? "new (unsaved)" : selected.id.slice(0, 12)}</span></div>
                <div className="flex items-center"><span className="text-ink-3 w-16 shrink-0">track</span>{selected.track_id
                  ? <button onClick={() => router.push(`/track/${selected.track_id}`)} className="text-info hover:text-accent truncate">{selected.track_id.slice(0, 12)} &rarr;</button>
                  : <span className="text-ink-3">none</span>}</div>
                <div className="flex items-center"><span className="text-ink-3 w-16 shrink-0">version</span><span className="text-ink-2">{selected.version ?? "-"}</span></div>
                <div className="flex items-start gap-1"><span className="text-ink-3 w-16 shrink-0 pt-0.5">geometry</span>
                  <div className="flex flex-wrap gap-1">
                    {([["box", selected.bbox?.length === 4], ["mask", selected.mask.length > 0], ["polyline", !!selected.polyline?.length], ["pose", !!selected.keypoints], ["3D", !!selected.cuboid_3d], ["rotated", !!selected.rot]] as [string, boolean][])
                      .filter(([, on]) => on).map(([k]) => <span key={k} className="text-ink-2 bg-line/40 border border-line rounded px-1.5 py-0.5">{k}</span>)}
                  </div>
                </div>
              </div>
              {/* M-F.0: why this label was decided the way it was, from real provenance */}
              {!selected.isNew && (
                <div className="mb-1.5">
                  <button onClick={() => setExplainOpen((v) => !v)}
                    className="w-full flex items-center justify-between font-mono text-[10px] uppercase text-ink-3 border border-line rounded px-1.5 py-1 hover:border-accent">
                    <span>why this label</span><span>{explainOpen ? "−" : "+"}</span>
                  </button>
                  {explainOpen && (
                    <div className="mt-1.5 bg-bg-2 border border-line rounded p-2">
                      <ExplainPanel objectId={selected.id} />
                    </div>
                  )}
                </div>
              )}
              <button
                disabled={selected.isNew}
                title={selected.isNew ? "save the frame first, then propagate" : "optical-flow propagate this box across the next 12 frames as a track to confirm"}
                onClick={async () => {
                  const r = await trackOp("propagate", "propagate object",
                    () => api.propagateObject(selected.id, 12));
                  toast(r.created ? `propagated forward ${r.created} frames (track ${r.track_id?.slice(0, 8)}). Open the track to review/confirm.` : `could not propagate: ${r.reason || "no motion"}`);
                }}
                className="w-full mb-1 font-mono text-[10px] border border-line text-ink-2 px-1.5 py-1 hover:border-accent disabled:opacity-40">
                propagate forward 12 frames →
              </button>
              {/* relationships / grouping: pick a kind, click "link", then click the target object */}
              <div className="mb-1 space-y-1">
                <div className="flex items-center gap-1">
                  <select value={linkKind} onChange={(e) => setLinkKind(e.target.value)}
                    className="flex-1 bg-bg border border-line px-1 py-0.5 font-mono text-[10px] text-ink">
                    {RELATION_KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
                  </select>
                  <button onClick={() => setLinkFrom(linkFrom === selected.id ? null : selected.id)}
                    className={`font-mono text-[10px] border px-1.5 py-0.5 ${linkFrom === selected.id ? "border-accent text-accent" : "border-line text-ink-2 hover:border-accent"}`}>
                    {linkFrom === selected.id ? "click target" : "link"}
                  </button>
                </div>
                {relationships.filter((r) => r.from_object_id === selected.id || r.to_object_id === selected.id).map((r) => (
                  <div key={r.relationship_id} className="flex items-center gap-1 font-mono text-[10px] text-ink-3">
                    <span className="flex-1 truncate">{r.from_object_id === selected.id ? `${r.kind} ${r.to_object_id.slice(0, 8)}` : `${r.from_object_id.slice(0, 8)} ${r.kind}`}</span>
                    <button onClick={() => delRelationship(r.relationship_id)} className="hover:text-block" title="remove">x</button>
                  </div>
                ))}
              </div>
              <div className="space-y-1">
                {Object.entries(onto.attributes)
                  .filter(([name]) => {
                    // show only attributes applicable to the selected object's class (by its l1 subclass);
                    // a subclass without a scope entry shows all attributes
                    const l1 = onto.classes.find((c) => c.id === selected.class_id)?.l1;
                    const allowed = l1 ? onto.attribute_scope?.[l1] : undefined;
                    return !allowed || allowed.includes(name);
                  })
                  .map(([name, spec]) => (
                    <AttrControl key={name} name={name} spec={spec} value={selected.attrs[name]}
                      onChange={(val) => setAttrSelected(name, val)} />
                  ))}
              </div>
            </div>
          )}

          {/* P3 derived dynamics readout for the selected object (planning/prediction signals) */}
          {rightTab === "objects" && selected && (
            <PanelSection title="dynamics">
              <div className="flex justify-end mb-1">
                <button onClick={recomputeDynamics} title="compute distance/speed/heading/TTC/risk for this session"
                  className="font-mono text-[10px] text-info hover:text-accent">recompute</button>
              </div>
              {(() => {
                const d = dynamics[selected.id];
                if (!d) return <div className="font-mono text-[10px] text-ink-3">no dynamics yet (save the object, then recompute)</div>;
                const rc = d.risk_level === "high" ? "text-block" : d.risk_level === "medium" ? "text-warn" : "text-pass";
                const row = (label: string, val: string, cls = "text-ink-2") => (
                  <div className="flex justify-between"><span className="text-ink-3">{label}</span><span className={cls}>{val}</span></div>
                );
                return (
                  <div className="font-mono text-[10px] space-y-0.5">
                    {row("distance", d.distance_m != null ? `${d.distance_m} m` : "-")}
                    {row("speed", d.speed_kmh != null ? `${d.speed_kmh} km/h` : "-")}
                    {row("closing", d.closing_speed_kmh != null ? `${d.closing_speed_kmh} km/h` : "-")}
                    {row("heading", d.heading_deg != null ? `${d.heading_deg}°` : "-")}
                    {row("TTC", d.ttc_s != null ? `${d.ttc_s} s` : "-")}
                    {row("risk", d.risk_level ?? "-", rc)}
                    {d.track_id && row("track", d.track_id.slice(0, 8))}
                    <div className="text-ink-3 pt-0.5">estimate · IPM monocular</div>
                  </div>
                );
              })()}
            </PanelSection>
          )}

          {/* LiDAR BEV: draw oriented boxes on the bird's-eye view, then lift them to metric 3D cuboids */}
          {rightTab === "tools" && meta?.is_lidar && (
            <PanelSection title="lidar bev" badge={`${(meta.lidar_points ?? 0).toLocaleString()} pts`}>
              <button
                onClick={async () => {
                  if (isDirty(st)) await save();
                  const r = await api.computeLidarCuboids(id);
                  flash(`lifted ${r.cuboids} oriented box${r.cuboids === 1 ? "" : "es"} to 3D cuboids`);
                }}
                title="draw oriented boxes (select + rotate handle), then lift each to a metric 3D cuboid using the enclosed points"
                className="w-full font-mono text-[10px] border border-line text-ink-2 px-1.5 py-1 hover:border-accent">
                compute 3D cuboids from boxes &rarr;
              </button>
            </PanelSection>
          )}

          {/* Tools tab: the AI/automation clusters, each collapsible, in their own tab so they never bury the
              object list. The AI panels are no longer nested inside road segmentation. */}
          {rightTab === "tools" && (<>
          <PanelSection title="history and saves" defaultOpen
            badge={`${st.past.length} step${st.past.length === 1 ? "" : "s"}`}>
            <HistoryPanel
              frameId={id}
              past={st.past}
              future={st.future}
              onJump={(at) => dispatch({ t: "jump", at })}
              onRestored={() => setReloadKey((k) => k + 1)}
              flash={flash}
            />
          </PanelSection>

          <PanelSection title="agent · auto-label" defaultOpen>
            <AgentPanel frameId={id} selectedId={st.selectedId} onApplied={loadLayers} embedded />
          </PanelSection>

          <PanelSection title="natural-language bulk edit" defaultOpen>
            <BulkEditBar frameId={id} sessionId={meta.session_id} onApplied={() => flash("bulk edit applied (routed to review)")} embedded />
          </PanelSection>

          <PanelSection title="scene graph + vlm dataset">
            <SceneGraphPanel frameId={id} embedded />
          </PanelSection>

          <PanelSection title="road segmentation" badge={`${lanes.length} lanes${drivable ? " · drivable" : ""}`}>
            <div className="grid grid-cols-2 gap-1 font-mono text-[10px]">
              <button onClick={segRoad} className="border border-line text-ink-2 px-1.5 py-1 hover:border-accent">segment road</button>
              <button onClick={genLanes} className="border border-line text-ink-2 px-1.5 py-1 hover:border-accent">propose lanes</button>
              <button onClick={() => router.push(`/annotate/lane/${id}`)} className="border border-line text-ink-2 px-1.5 py-1 hover:border-accent col-span-2">edit lanes + drivable &rarr;</button>
            </div>
          </PanelSection>
          </>)}

          {/* object list (objects tab): grouped by class, searchable, collapsible, with a confidence bar per
              row; selection is bidirectional with the canvas. */}
          {rightTab === "objects" && (
          <div className="p-2">
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono text-[10px] uppercase text-ink-3">objects ({st.objects.length})</span>
            </div>
            <input value={objSearch} onChange={(e) => setObjSearch(e.target.value)} placeholder="search objects..."
              className="w-full bg-bg border border-line px-1.5 py-1 font-mono text-[11px] text-ink mb-1" />
            <div className="space-y-1">
              {(() => {
                const q = objSearch.toLowerCase();
                const filtered = st.objects.filter((o) => !q || o.class_name.toLowerCase().includes(q) || o.id.toLowerCase().includes(q));
                if (!filtered.length) return <div className="text-ink-3 text-center py-4 font-mono text-[11px]">no objects. draw a box (B).</div>;
                const groups: Record<string, EdObject[]> = {};
                for (const o of filtered) (groups[o.class_name] ??= []).push(o);
                return Object.entries(groups).sort((a, b) => a[0].localeCompare(b[0])).map(([cls, objs]) => {
                  const collapsed = collapsedGroups.has(cls);
                  return (
                    <div key={cls}>
                      <button onClick={() => setCollapsedGroups((s) => { const n = new Set(s); if (n.has(cls)) n.delete(cls); else n.add(cls); return n; })}
                        className="flex items-center gap-1.5 w-full font-mono text-[10px] text-ink-3 hover:text-ink-2 py-0.5">
                        <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: classColor(objs[0].class_id) }} />
                        <span className="flex-1 text-left truncate uppercase">{cls}</span>
                        <span>{objs.length}</span>
                        <span className="w-3 text-right">{collapsed ? "+" : "−"}</span>
                      </button>
                      {!collapsed && objs.map((o) => (
                        <div key={o.id}
                          onClick={(e) => {
                            // Ctrl/Cmd or Shift extends the selection, matching every other list in every
                            // other tool; a plain click still selects exactly one.
                            if (e.ctrlKey || e.metaKey || e.shiftKey) dispatch({ t: "toggleSelect", id: o.id });
                            else dispatch({ t: "select", id: o.id });
                          }}
                          className={`flex items-center gap-1.5 pl-3 pr-1 py-0.5 cursor-pointer font-mono text-[11px] ${st.selectedIds.includes(o.id) ? "bg-line text-ink" : "text-ink-3 hover:text-ink-2"}`}>
                          <button title={o.visible ? "hide" : "show"} aria-label={o.visible ? "hide object" : "show object"}
                            onClick={(e) => { e.stopPropagation(); dispatch({ t: "setVisible", ids: [o.id], visible: !o.visible }); }}
                            className={o.visible ? "text-ink-2" : "text-ink-3"}>{o.visible ? "●" : "○"}</button>
                          <button title={o.locked ? "unlock" : "lock"} aria-label={o.locked ? "unlock object" : "lock object"}
                            onClick={(e) => { e.stopPropagation(); dispatch({ t: "setLocked", ids: [o.id], locked: !o.locked }); }}
                            className={o.locked ? "text-warn" : "text-ink-3 hover:text-ink-2"}>{o.locked ? "L" : "l"}</button>
                          <span className="truncate flex-1">{o.id.startsWith("tmp-") ? "new" : o.id.slice(0, 8)}{o.isNew ? " *" : ""}</span>
                          {o.quality_score != null && <span title={`label quality ${o.quality_score.toFixed(2)}`}
                            className={`w-1.5 h-1.5 rounded-full shrink-0 ${o.quality_score >= 0.4 ? "bg-pass" : o.quality_score >= 0.25 ? "bg-warn" : "bg-block"}`} />}
                          <ConfBar conf={o.conf} />
                          {o.mask.length > 0 && <span className="text-info" title="has mask">&#9670;</span>}
                          <button onClick={(e) => { e.stopPropagation(); dispatch({ t: "delete", id: o.id }); }}
                            disabled={o.locked} aria-label="delete object"
                            className={o.locked ? "text-line cursor-not-allowed" : "text-ink-3 hover:text-block"}>x</button>
                        </div>
                      ))}
                    </div>
                  );
                });
              })()}
            </div>
          </div>
          )}
          </div>
          </>)}

          {/* Ways to pick a set, sitting directly above the bulk actions they feed. A dense frame holds
              forty vehicles and the useful selections are almost never contiguous, so a marquee alone leaves
              "every autorickshaw" and "everything nobody has reviewed" as manual work. */}
          {rightTab === "objects" && (
            <div className="panel px-2 py-1.5">
              <div className="flex flex-wrap gap-1">
                {SELECTIONS.map((sel) => (
                  <button
                    key={sel.how + String(sel.value ?? "")}
                    onClick={() => dispatch({ t: "selectBy", how: sel.how, value: sel.value })}
                    title={sel.hint + (sel.key ? ` (${sel.key})` : "")}
                    className="border border-line text-ink-3 px-1.5 py-0.5 hover:border-accent font-mono text-[10px]"
                  >
                    {sel.label}
                  </button>
                ))}
              </div>
              <p className="font-mono text-[10px] text-ink-3 mt-1">
                Locked and hidden objects are never picked: a bulk action must not reach the thing somebody
                locked to protect it.
              </p>
            </div>
          )}

          {/* Bulk actions on a multi-selection. Before this, selection was a single id, so every batch
              operation had to go through the natural-language agent bar: there was no way to pick three
              boxes and delete or reclassify them directly. */}
          {st.selectedIds.length > 1 && (
            <div className="panel px-2 py-1.5 flex items-center gap-2 font-mono text-[11px]" role="toolbar"
                 aria-label="bulk actions">
              <span className="text-ink-2">{st.selectedIds.length} selected</span>
              <button onClick={() => dispatch({ t: "setVisible", ids: st.selectedIds, visible: false })}
                className="px-1.5 py-0.5 border border-line hover:border-accent">hide</button>
              <button onClick={() => dispatch({ t: "setVisible", ids: st.selectedIds, visible: true })}
                className="px-1.5 py-0.5 border border-line hover:border-accent">show</button>
              <button onClick={() => dispatch({ t: "setLocked", ids: st.selectedIds, locked: true })}
                className="px-1.5 py-0.5 border border-line hover:border-accent">lock</button>
              <button
                onClick={async () => {
                  // Confirm before a destructive batch: a mis-swept marquee can hold far more than intended,
                  // and this is not a single undoable box.
                  if (!(await confirm({ title: `Delete ${st.selectedIds.length} objects?`, danger: true,
                                        confirmLabel: "Delete" }))) return;
                  for (const id of st.selectedIds) dispatch({ t: "delete", id });
                  dispatch({ t: "select", id: null });
                }}
                className="px-1.5 py-0.5 border border-line text-block hover:border-block ml-auto">delete</button>
            </div>
          )}

          {/* Issue threads for this frame, anchored to the selected object when there is one. Frame-scoped
              so it stays available in every mode: a problem worth reporting does not depend on which tool
              happens to be active. */}
          <div className="shrink-0 border-t hairline max-h-64 overflow-y-auto">
            <IssuePanel frameId={id} objectId={selected?.id ?? null} />
          </div>
        </aside>
        )}
      </div>

      {/* BOTTOM BAR: zoom, shortcut hints, counts, save status (the design's 28px bottom bar) */}
      <footer className="h-7 shrink-0 flex items-center border-t hairline font-mono text-[10.5px] text-ink-3
                         min-w-0 overflow-x-auto no-scrollbar">
        <div className="flex items-center h-full border-r hairline px-1">
          <button onClick={() => zoomBy(1 / 1.2)} title="zoom out" className="w-6 h-5 flex items-center justify-center rounded text-ink-2 hover:bg-line/50"><Icon name="zoomOut" size={14} /></button>
          <span className="min-w-[38px] text-center text-ink-2">{Math.round(st.viewport.scale * 100) || 0}%</span>
          <button onClick={() => zoomBy(1.2)} title="zoom in" className="w-6 h-5 flex items-center justify-center rounded text-ink-2 hover:bg-line/50"><Icon name="zoomIn" size={14} /></button>
          <button onClick={fit} title="fit to view" className="w-6 h-5 flex items-center justify-center rounded text-ink-2 hover:bg-line/50 ml-0.5"><Icon name="fit" size={14} /></button>
        </div>
        <div className="flex-1 px-3 overflow-hidden whitespace-nowrap text-ink-3/80">
          <span className="text-ink-2">V</span> select &middot; <span className="text-ink-2">B</span> box &middot; <span className="text-ink-2">G</span> polygon &middot; <span className="text-ink-2">K</span> pose &middot; <span className="text-ink-2">R</span> measure &middot; <span className="text-ink-2">[ ]</span> frame &middot; <span className="text-ink-2">Cmd Z</span> undo &middot; <span className="text-ink-2">Cmd A</span> select all &middot; <span className="text-ink-2">Cmd Shift S</span> save as &middot; <span className="text-ink-2">?</span> shortcuts
        </div>
        <div className="flex items-center gap-1.5 h-full border-l hairline px-3">
          <span className="text-ink-2">{st.objects.length} objects</span>
          <span className="text-line">&middot;</span>
          <span className="text-pass">{st.objects.filter((o) => o.state === "accepted").length} confirmed</span>
        </div>
        <div className="flex items-center gap-1.5 h-full border-l hairline px-3">
          <span className={`w-1.5 h-1.5 rounded-full ${dirty ? "bg-warn" : "bg-pass"}`} />
          <span>{dirty ? "unsaved" : "saved"}</span>
        </div>
      </footer>

      {/* "How it scales": the add-a-feature explainer, the layout absorbs features by grouping and mode */}
      {scaleNoteOpen && (
        <div className="fixed inset-0 z-50 bg-bg/60 flex items-center justify-center" onClick={() => setScaleNoteOpen(false)}>
          <div className="w-[360px] panel" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 px-4 py-3 border-b hairline">
              <span className="flex text-accent"><Icon name="plus" size={16} /></span>
              <span className="font-display font-semibold text-[13.5px] text-ink">Adding a capability, by organization</span>
              <button onClick={() => setScaleNoteOpen(false)} className="ml-auto flex text-ink-3 hover:text-ink"><Icon name="close" size={16} /></button>
            </div>
            <div className="p-4 flex flex-col gap-3 text-[12.5px] text-ink-2 leading-relaxed">
              <p>A new detector does not get a new toolbar button. It joins a tool group as one more flyout item, or a new mode is one rail icon. The layout absorbs features through grouping and mode, never by widening a row.</p>
              <div className="flex flex-col gap-1.5 bg-bg-2 border border-line rounded-md p-3 text-[11.5px]">
                <div className="flex items-center gap-2"><span className="flex text-pass"><Icon name="check" size={14} /></span>Tool strip stays one row. No wrap, no clip.</div>
                <div className="flex items-center gap-2"><span className="flex text-pass"><Icon name="check" size={14} /></span>Left rail width unchanged. No new mode needed.</div>
                <div className="flex items-center gap-2"><span className="flex text-pass"><Icon name="check" size={14} /></span>A 3D tool slots into the 3D mode the same way.</div>
              </div>
              <p className="text-[11.5px] text-ink-3">That is the whole rule: no region grows unbounded.</p>
              <button onClick={() => setScaleNoteOpen(false)} className="self-start h-7 px-3 rounded-md border border-line bg-bg-2 text-ink hover:bg-line text-[11.5px]">Got it</button>
            </div>
          </div>
        </div>
      )}
      <ShortcutOverlay />

      {activeCorr && (
        <CorrectionModal
          objectId={activeCorr.objectId}
          kind={activeCorr.kind}
          change={activeCorr.change}
          onClose={() => setActiveCorr(null)}
          onApplied={(n) => flash(`applied "${String(activeCorr.change.new)}" to ${n} similar objects`)}
        />
      )}
    </div>
  );
}

function AttrControl({ name, spec, value, onChange }: {
  name: string; spec: { type: string; values: unknown[] | null; range: number[] | null }; value: unknown; onChange: (v: unknown) => void;
}) {
  const label = <span className="w-24 shrink-0 font-mono text-[11px] text-ink-3 truncate">{name}</span>;
  if (spec.type === "enum")
    return (
      <label className="flex items-center gap-2">{label}
        <select value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} className="flex-1 bg-panel border border-line px-1 py-0.5 font-mono text-[11px] text-ink">
          <option value="">-</option>
          {(spec.values || []).map((v) => <option key={String(v)} value={String(v)}>{String(v)}</option>)}
        </select>
      </label>
    );
  if (spec.type === "bool")
    return <label className="flex items-center gap-2">{label}<input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} /></label>;
  return (
    <label className="flex items-center gap-2">{label}
      <input type="number" step={spec.type === "float" ? 0.01 : 1} value={value == null ? "" : Number(value)}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        className="flex-1 bg-panel border border-line px-1 py-0.5 font-mono text-[11px] text-ink" />
    </label>
  );
}
