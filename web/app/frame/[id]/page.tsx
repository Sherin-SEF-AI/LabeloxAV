"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { api, lidarCloudPoints, type Cuboid3D, type LidarCloud, type LidarPoints } from "@/lib/api";
import type { ColorBy } from "@/components/lidar/PointCloudViewer";
import type { AdverseRegion, AlItem, ErrorCandidateRow, FrameMeta, LaneRow, ObjectDynamicsRow, Ontology, OntologyClass, ProjectedCuboid, Relationship } from "@/lib/types";
import { classColor } from "@/lib/colors";
import { acceptState, getUser, setUser } from "@/lib/user";
import { isDirty, tmpId, useEditor, type EdObject, type Tool } from "@/components/editor/useEditor";
import { PERSON_17 } from "@/lib/skeleton";
import BackButton from "@/components/BackButton";
import CorrectionModal, { type CorrectionChange } from "@/components/CorrectionModal";
import ToolStrip from "@/components/shell/ToolStrip";
import ModeRail from "@/components/shell/ModeRail";
import FloatingLayers from "@/components/shell/FloatingLayers";
import { StateBadge, ConfBar } from "@/components/StateBadge";
import ScoreBar from "@/components/shell/ScoreBar";
import Icon, { MODE_ICON } from "@/components/shell/Icon";
import ShortcutOverlay from "@/components/shell/ShortcutOverlay";
import { MODES, type ToolGroup } from "@/lib/editor/registry";

// Frame-centric professional annotation editor. Pan/zoom canvas, draw + edit boxes, SAM-assisted masks,
// layers panel, class palette, attributes, keyboard-driven, batched save. Operational Materialism tokens.

// Wrap the import so next/dynamic's convertModule always gets a clean { default } and cannot mistake the
// module for a react-konva export on a StrictMode re-mount.
const EditorCanvas = dynamic(() => import("@/components/editor/EditorCanvas").then((m) => ({ default: m.default })), { ssr: false });
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
} satisfies Record<string, ToolGroup>;

// Per-mode tool strips. The mode rail picks one; the strip renders only that mode's groups.
const MODE_GROUPS: Record<string, ToolGroup[]> = {
  objects: [G.select, G.draw, G.ai, G.mask, G.region, G.measure],
  pose: [G.select, G.pose, G.measure],
  lidar3d: [G.select, G.cuboid, G.measure],
  lanes: [G.select, G.measure],
  review: [G.select],
};
const MODE_TOOLS: Record<string, string[]> = Object.fromEntries(
  Object.entries(MODE_GROUPS).map(([m, gs]) => [m, gs.flatMap((g) => g.tools.map((t) => t.key))]));

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

export default function FrameEditor() {
  const router = useRouter();
  const { id } = useParams<{ id: string }>();
  const focus = useSearchParams().get("focus");

  const [st, dispatch] = useEditor();
  const [meta, setMeta] = useState<FrameMeta | null>(null);
  const [onto, setOnto] = useState<Ontology | null>(null);
  const [currentClass, setCurrentClass] = useState<OntologyClass | null>(null);
  const [panning, setPanning] = useState(false);
  const [cursor, setCursor] = useState<number[] | null>(null);
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
  const [objSearch, setObjSearch] = useState("");
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());
  const [mode, setMode] = useState("objects");
  const [scaleNoteOpen, setScaleNoteOpen] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  // switching mode swaps the tool strip; reset the active tool to the mode's first tool if it does not carry over
  const switchMode = (m: string) => {
    setMode(m);
    const tools = MODE_TOOLS[m] ?? [];
    if (!tools.includes(st.tool)) dispatch({ t: "tool", tool: (tools[0] ?? "select") as Tool });
  };
  const [layers, setLayers] = useState({ boxes: true, masks: true, lanes: true, drivable: true, adverse: true, cuboids: true, seg: true });

  const flash = (m: string) => {
    setNotice(m);
    setTimeout(() => setNotice(null), 3500);
  };

  // load frame + objects + ontology
  useEffect(() => {
    (async () => {
      const [m, objs, o] = await Promise.all([api.frame(id), api.frameObjects(id), api.ontology()]);
      setMeta(m);
      setOnto(o);
      const eds: EdObject[] = objs.map((x) => ({
        id: x.object_id, track_id: x.track_id, class_id: x.class_id, class_name: x.class_name, bbox: x.bbox,
        mask: x.mask_polygons || [], attrs: {}, conf: x.conf, state: x.state, visible: true, version: x.version,
        rot: x.rot_deg, keypoints: x.keypoints ?? undefined, polyline: x.polyline ?? undefined,
        cuboid_3d: x.cuboid_3d ?? undefined,
      }));
      dispatch({ t: "load", objects: eds, viewport: { scale: 0, ox: 0, oy: 0 }, selectedId: focus });
      const fc = (focus && eds.find((e) => e.id === focus)) || null;
      setCurrentClass(fc ? o.classes.find((c) => c.id === fc.class_id) || o.classes[0] : o.classes[0]);
      loadedRef.current = true;
    })();
  }, [id, focus, dispatch]);

  const selected = st.objects.find((o) => o.id === st.selectedId) || null;
  const dirty = isDirty(st);

  // object relationships: in link mode, the next clicked object becomes the target of the relationship
  const relate = async (toId: string) => {
    if (!linkFrom || toId === linkFrom) { setLinkFrom(null); return; }
    try {
      await api.relateObject(linkFrom, { to_object_id: toId, kind: linkKind });
      setRelationships(await api.frameRelationships(id).catch(() => []));
      flash(`linked: ${linkKind}`);
    } catch (e) { flash("link failed: " + String(e)); }
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
      const { ego } = await api.liftGround(id, pt[0], pt[1]);
      const cub = { center: [ego[0], ego[1], 0.75], size: [1.8, 4.2, 1.5], yaw: 0 };
      dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name,
        bbox: [pt[0] - 40, pt[1] - 40, pt[0] + 40, pt[1] + 40], mask: [], cuboid_3d: cub, attrs: {},
        conf: 1, state: "accepted", visible: true, isNew: true } });
    } catch (e) { flash("could not place cuboid: " + String(e)); }
  };
  // magic-wand: a single SAM point click that auto-creates (or refines) the object, no accept step
  const runMagicWand = async (pt: number[]) => {
    try {
      const r = await api.segmentPrompt(id, { points: [pt], labels: [1] });
      if (!r.polygons.length) { flash("magic-wand found nothing here"); return; }
      const box = bboxOfPolys(r.polygons);
      if (selected && overlapFrac(box, selected.bbox) > 0.5) {
        dispatch({ t: "update", id: selected.id, patch: { mask: r.polygons, bbox: box } });
      } else if (currentClass) {
        dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name,
          bbox: box, mask: r.polygons, attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } });
      }
    } catch (e) { flash(String(e).includes("503") ? "GPU busy (training)" : "magic-wand failed"); }
  };
  // brush/eraser: compose the stroke stamps into the selected object's mask (or a new object)
  const onBrushStroke = async (ops: { op: string; center: number[]; radius: number }[]) => {
    if (!meta) return;
    try {
      const r = await api.composeMask({ polygons: selected?.mask ?? [], ops, width: meta.width, height: meta.height });
      if (selected) {
        dispatch({ t: "update", id: selected.id, patch: { mask: r.polygons, bbox: r.polygons.length ? bboxOfPolys(r.polygons) : selected.bbox } });
      } else if (currentClass && r.polygons.length) {
        dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name,
          bbox: bboxOfPolys(r.polygons), mask: r.polygons, attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } });
      }
    } catch (e) { flash("brush failed: " + String(e)); }
  };
  // superpixel: add the clicked SLIC cell to the active mask
  const pickSuperpixel = (pt: number[]) => {
    const poly = superpixels.find((pp) => pointInPoly(pt, pp));
    if (!poly) return;
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
    setRelationships(rel);
    setAdverse(adv);
    setCuboids(cub);
    const seg = await api.getSegment(id, segKind).catch(() => ({ found: false, has_overlay: false }));
    setSegUrl(seg.found && seg.has_overlay ? `/api/frames/${id}/segment/overlay?kind=${segKind}&t=${Date.now()}` : null);
  }, [id, segKind]);
  useEffect(() => { loadLayers(); }, [loadLayers]);
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
    try { await api.segmentDrivable(id); await loadLayers(); flash("drivable area updated"); }
    catch (e) { flash("segment road failed: " + String(e)); }
  }, [id, loadLayers]);
  const genLanes = useCallback(async () => {
    try { const r = await api.proposeLanes(id); await loadLayers(); flash(`proposed ${r.proposed} lanes (${r.model})`); }
    catch (e) { flash("propose lanes failed: " + String(e)); }
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
  const laneStageClick = (e: { evt: MouseEvent }) => { if (laneAdding) setLaneAdding([...laneAdding, laneToImg(e)]); };
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
  const delLane = async () => { if (laneSel) { await api.deleteLane(laneSel); setLaneSel(null); await loadLayers(); flash("lane deleted"); } };
  const propagateLanes = async () => { const r = await api.propagateLanes(id, 8); flash(`propagated to ${r.created} lane-frames`); };

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
      } catch (e) { if (!cancelled) setLidarMsg("cloud load failed: " + String(e)); }
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
        ground_snap: Boolean(fields.attrs && (fields.attrs as Record<string, unknown>).ground_snap), expected_version: cur.version,
      });
      patchCub(cid, saved);
    } catch (e) { setLidarMsg("save failed: " + String(e)); }
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
    } catch (e) { setLidarMsg("add failed: " + String(e)); }
  };
  const delCub = async (cid: string) => { try { await api.lidarDeleteCuboid(cid); setCub3d((cs) => cs.filter((c) => c.object_3d_id !== cid)); setCubSel(null); } catch (e) { setLidarMsg("delete failed: " + String(e)); } };
  const aiLift3d = async () => {
    if (!cloud3d) return;
    setLidarMsg("lifting 2D objects to 3D...");
    try { const r = await api.lidarLiftCloud(cloud3d.cloud_id); const objs = await api.lidarObjects3d(cloud3d.cloud_id); setCub3d(objs.objects); setLidarMsg(r.cuboids ? `lifted ${r.cuboids} cuboids` : "no 2D objects to lift"); }
    catch (e) { setLidarMsg("lift failed: " + String(e)); }
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
  const reviewObject = async (newState: "accepted" | "rejected") => {
    const o = selected;
    if (!o) { flash("select an object to review"); return; }
    if (o.isNew) { flash("save the new object first"); return; }
    try {
      const r = await api.review(o.id, { action: newState === "accepted" ? "accept" : "reject", state: newState, expected_version: o.version });
      dispatch({ t: "reviewed", id: o.id, state: newState, version: r.version });
      flash(newState);
      advanceReview(o.id);
    } catch (e) { flash("review failed: " + String(e)); }
  };
  // Move to the next value-queue item: select it if it is on this frame, else navigate to its frame.
  const advanceReview = (currentObjId: string) => {
    const here = alItems.filter((it) => it.frame_id === id);
    const idx = here.findIndex((it) => it.object_id === currentObjId);
    const next = here[idx + 1] ?? here.find((it) => it.object_id !== currentObjId);
    if (next) doSelect(next.object_id);
    else { const off = alItems.find((it) => it.frame_id !== id); if (off) gotoFrame(off.frame_id); }
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
        flash("could not add class: " + String(e));
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
      dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name,
        bbox: box, mask: st.candidate, attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } });
      if (selected) dispatch({ t: "select", id: null }); // moved off the old object -> keep creating new
    }
    dispatch({ t: "candidate", polys: null });
  }, [st.candidate, selected, currentClass, dispatch]);

  const runSam = useCallback(
    async (prompt: { points?: number[][]; labels?: number[]; box?: number[] }) => {
      if (st.candidate?.length) acceptCandidate(); // commit the pending mask before starting the next
      try {
        const r = await api.segmentPrompt(id, prompt);
        dispatch({ t: "candidate", polys: r.polygons });
        if (!r.polygons.length) flash("SAM found nothing here");
      } catch (e) {
        const msg = String(e);
        flash(msg.includes("503") ? "GPU busy (training). Box tools still work." : "segment failed");
      }
    },
    [id, dispatch, st.candidate, acceptCandidate],
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
  const save = useCallback(async () => {
    if (!dirty || savingRef.current) return;
    savingRef.current = true;
    setSaving(true);
    const tgt = acceptState(getUser()?.role);  // annotator -> submitted (QA), reviewer/admin -> accepted
    try {
      for (const oid of st.deleted) await api.deleteObject(oid);
      const remap: Record<string, string> = {};
      const versions: Record<string, number> = {};
      for (const o of st.objects) {
        if (o.isNew) {
          const created = await api.createObject(id, {
            class_name: o.class_name, bbox: o.bbox, attrs: o.attrs,
            mask_polygons: o.mask.length ? o.mask : undefined, state: tgt, idem_key: o.id, rot_deg: o.rot ?? 0,
            keypoints: o.keypoints ?? null, polyline: o.polyline, cuboid_3d: o.cuboid_3d ?? undefined,
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
      flash("saved");
      setCuboids(await api.frameCuboids(id).catch(() => [])); // refresh projected cuboid wireframes
    } catch (e) {
      const msg = String(e);
      flash(msg.includes("409") ? "conflict: another annotator changed this object; reload to continue" : "save failed: " + msg);
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  }, [dirty, st.deleted, st.objects, id, meta, dispatch]);

  // ---- autosave: persist edits ~700ms after the last change settles (covers move/resize/relabel/
  // attribute/mask/delete). The debounce waits out an active drag, so we never save mid-gesture. ----
  const saveRef = useRef(save);
  saveRef.current = save;
  const stRef = useRef(st);
  stRef.current = st;
  useEffect(() => {
    if (!autosave || !loadedRef.current || !dirty || saving) return;
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
      if (mod && e.key.toLowerCase() === "s") { e.preventDefault(); save(); return; }
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
      if (e.shiftKey && /^Digit[1-5]$/.test(e.code)) { e.preventDefault(); switchMode(MODES[Number(e.code.slice(5)) - 1].key); return; }
      const k = e.key.toLowerCase();
      // Review mode rebinds a/x to accept/reject the selected object (and advance the queue).
      if (mode === "review") {
        if (k === "a") { reviewObject("accepted"); return; }
        if (k === "x") { reviewObject("rejected"); return; }
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
      else if (e.key === "Escape") { dispatch({ t: "candidate", polys: null }); setKpDraft(null); }
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

  if (!meta || !onto) return <div className="min-h-screen flex items-center justify-center font-mono text-ink-3">loading frame...</div>;

  return (
    <div className="h-screen flex flex-col">
      {/* TOP BAR: identity, frame context, global actions, confirm (the design's 46px top bar) */}
      <header className="flex items-center gap-3 px-3 h-[46px] border-b hairline shrink-0">
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
        <button onClick={() => router.push(`/search?frame=${id}`)} title="find visually similar frames (DINOv3)"
          className="flex items-center justify-center w-[30px] h-[30px] rounded-md text-ink-3 hover:bg-line/50 hover:text-ink"><Icon name="search" size={16} /></button>

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
          <span className="w-px h-5 bg-line mx-0.5" />
          <button onClick={() => setScaleNoteOpen(true)} title="how this layout scales"
            className="flex items-center gap-1.5 h-[30px] px-2.5 rounded-md border border-line text-ink-2 hover:bg-line/50 hover:text-ink text-[11.5px]"><Icon name="info" size={15} /><span>How it scales</span></button>
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
          <div className="h-[50px] shrink-0 flex items-center gap-1.5 px-2.5 border-b hairline overflow-x-auto no-scrollbar">
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
          {mode === "lanes" ? (
            laneImg && meta ? (
              <LaneCanvas img={laneImg} meta={{ width: meta.width, height: meta.height }} scale={laneScale}
                lanes={lanes} sel={laneSel} drivable={layers.drivable ? drivable : null} adding={laneAdding}
                onStageClick={laneStageClick} onSelect={setLaneSel} onDragPoint={laneDragPoint} />
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
          ) : (
          <EditorCanvas
            imageUrl={meta.image_url} imgW={meta.width} imgH={meta.height}
            objects={st.objects} selectedId={st.selectedId} tool={st.tool} candidate={st.candidate}
            viewport={st.viewport} panning={panning}
            lanes={lanes} drivable={drivable} layers={layers}
            onViewport={(viewport) => dispatch({ t: "viewport", viewport })}
            onSelect={doSelect}
            relationships={relationships}
            onUpdateBbox={(oid, bbox, rot) => dispatch({ t: "update", id: oid, patch: rot !== undefined ? { bbox, rot } : { bbox } })}
            onDrawBox={(bbox) => currentClass && dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name, bbox, mask: [], attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } })}
            onDrawPolygon={(pts) => currentClass && dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name, bbox: bboxOfPolys([pts]), mask: [pts], attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } })}
            onDrawPolyline={(pts) => currentClass && dispatch({ t: "add", obj: { id: tmpId(), class_id: currentClass.id, class_name: currentClass.name, bbox: bboxOfPolys([pts]), mask: [], polyline: Array.from({ length: pts.length / 2 }, (_, i) => [pts[2 * i], pts[2 * i + 1]]), attrs: {}, conf: 1, state: "accepted", visible: true, isNew: true } })}
            adverse={adverse}
            onDrawAdverse={async (pts) => { try { await api.createAdverse(id, { geometry: pts, condition: adverseCond }); setAdverse(await api.listAdverse(id).catch(() => [])); flash(`tagged ${adverseCond}`); } catch (e) { flash("region failed: " + String(e)); } }}
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
            onCursor={setCursor}
          />
          )}
          {mode !== "lidar3d" && <FloatingLayers layers={layers} onToggle={(k) => setLayers((s) => ({ ...s, [k]: !s[k as keyof typeof s] }))}
            extra={
              <>
                <select value={segKind} onChange={(e) => setSegKind(e.target.value as "semantic" | "panoptic")}
                  title="dense segmentation kind" className="w-full bg-bg border border-line px-1 py-0.5 text-ink-3">
                  <option value="semantic">semantic</option>
                  <option value="panoptic">panoptic</option>
                </select>
                <button title="run dense segmentation (SAM-everything + VLM) on this frame"
                  onClick={async () => { flash("segmenting..."); try { const r = await api.autoSegment(id, segKind); setSegUrl(`/api/frames/${id}/segment/overlay?kind=${segKind}&t=${Date.now()}`); flash(`segmented ${segKind} (${Object.keys(r.coverage).length} classes${r.n_instances ? ", " + r.n_instances + " instances" : ""})`); } catch (e) { flash("segment failed: " + String(e)); } }}
                  className="w-full border border-line px-1 py-0.5 text-ink-3 hover:border-accent">auto-seg</button>
              </>
            } />}
          {/* Review mode: a quiet bottom-center action bar so the reviewer accepts/rejects without leaving the canvas */}
          {mode === "review" && selected && (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 panel px-3 py-1.5 flex items-center gap-3 font-mono text-[11px]">
              <span className="text-ink-2">{selected.class_name}</span>
              <ConfBar conf={selected.conf} />
              <button onClick={() => reviewObject("accepted")} className="border border-pass text-pass px-2 py-0.5 hover:bg-pass/10">accept (A)</button>
              <button onClick={() => reviewObject("rejected")} className="border border-block text-block px-2 py-0.5 hover:bg-block/10">reject (X)</button>
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
          {/* HUD: frame time and camera, a quiet overlay top-left (the design's HUD) */}
          {meta && (
            <div className="absolute top-3 left-3 z-10 flex flex-col gap-1 pointer-events-none">
              <span className="font-mono text-[11px] text-ink-2 bg-bg/60 px-1.5 py-0.5 rounded w-fit">{new Date(Number(meta.ts_ns) / 1e6).toISOString().replace("T", " ").replace("Z", "")}</span>
              <span className="font-mono text-[11px] text-ink-3 bg-bg/60 px-1.5 py-0.5 rounded w-fit">cam {meta.cam_id}{meta.is_lidar ? " · lidar" : ""}{cursor ? `  ·  ${Math.round(cursor[0])}, ${Math.round(cursor[1])}` : ""}</span>
            </div>
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
        <aside className="w-[340px] shrink-0 border-l hairline flex flex-col min-h-0">
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
                    <button onClick={() => reviewObject("accepted")} className="flex-1 border border-pass text-pass px-2 py-1 hover:bg-pass/10">accept (A)</button>
                    <button onClick={() => reviewObject("rejected")} className="flex-1 border border-block text-block px-2 py-1 hover:bg-block/10">reject (X)</button>
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
          {/* class palette */}
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

          {/* Everything below the class picker scrolls together, so a long attributes list plus the
              dynamics and road-segmentation panels stay reachable on short screens. */}
          <div className="flex-1 min-h-0 overflow-y-auto">

          {/* attributes of selected */}
          {selected && (
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
              <button
                disabled={selected.isNew}
                title={selected.isNew ? "save the frame first, then propagate" : "optical-flow propagate this box across the next 12 frames as a track to confirm"}
                onClick={async () => {
                  const r = await api.propagateObject(selected.id, 12);
                  alert(r.created ? `propagated forward ${r.created} frames (track ${r.track_id?.slice(0, 8)}). Open the track to review/confirm.` : `could not propagate: ${r.reason || "no motion"}`);
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
          {selected && (
            <div className="border-b hairline p-2">
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-[10px] uppercase text-ink-3">dynamics</span>
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
            </div>
          )}

          {/* LiDAR BEV: draw oriented boxes on the bird's-eye view, then lift them to metric 3D cuboids */}
          {meta?.is_lidar && (
            <div className="border-b hairline p-2">
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-[10px] uppercase text-ink-3">lidar bev</span>
                <span className="font-mono text-[10px] text-ink-3">{(meta.lidar_points ?? 0).toLocaleString()} pts</span>
              </div>
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
            </div>
          )}

          {/* P4 road segmentation: generate + edit the lane and drivable layers in place */}
          <div className="border-b hairline p-2">
            <div className="flex items-center justify-between mb-1">
              <span className="font-mono text-[10px] uppercase text-ink-3">road segmentation</span>
              <span className="font-mono text-[10px] text-ink-3">{lanes.length} lanes{drivable ? " · drivable" : ""}</span>
            </div>
            <div className="grid grid-cols-2 gap-1 font-mono text-[10px]">
              <button onClick={segRoad} className="border border-line text-ink-2 px-1.5 py-1 hover:border-accent">segment road</button>
              <button onClick={genLanes} className="border border-line text-ink-2 px-1.5 py-1 hover:border-accent">propose lanes</button>
              <button onClick={() => router.push(`/annotate/lane/${id}`)} className="border border-line text-ink-2 px-1.5 py-1 hover:border-accent col-span-2">edit lanes + drivable &rarr;</button>
            </div>
          </div>

          {/* object list: grouped by class, searchable, collapsible, with a confidence bar per row. Scales
              to many objects without a flat scroll; selection is bidirectional with the canvas. */}
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
                        <div key={o.id} onClick={() => dispatch({ t: "select", id: o.id })}
                          className={`flex items-center gap-1.5 pl-3 pr-1 py-0.5 cursor-pointer font-mono text-[11px] ${o.id === st.selectedId ? "bg-line text-ink" : "text-ink-3 hover:text-ink-2"}`}>
                          <button onClick={(e) => { e.stopPropagation(); dispatch({ t: "update", id: o.id, patch: { visible: !o.visible } }); }}
                            className={o.visible ? "text-ink-2" : "text-ink-3"}>{o.visible ? "●" : "○"}</button>
                          <span className="truncate flex-1">{o.id.startsWith("tmp-") ? "new" : o.id.slice(0, 8)}{o.isNew ? " *" : ""}</span>
                          <ConfBar conf={o.conf} />
                          {o.mask.length > 0 && <span className="text-info" title="has mask">&#9670;</span>}
                          <button onClick={(e) => { e.stopPropagation(); dispatch({ t: "delete", id: o.id }); }} className="text-ink-3 hover:text-block">x</button>
                        </div>
                      ))}
                    </div>
                  );
                });
              })()}
            </div>
          </div>
          </div>
          </>)}
        </aside>
        )}
      </div>

      {/* BOTTOM BAR: zoom, shortcut hints, counts, save status (the design's 28px bottom bar) */}
      <footer className="h-7 shrink-0 flex items-center border-t hairline font-mono text-[10.5px] text-ink-3">
        <div className="flex items-center h-full border-r hairline px-1">
          <button onClick={() => zoomBy(1 / 1.2)} title="zoom out" className="w-6 h-5 flex items-center justify-center rounded text-ink-2 hover:bg-line/50"><Icon name="zoomOut" size={14} /></button>
          <span className="min-w-[38px] text-center text-ink-2">{Math.round(st.viewport.scale * 100) || 0}%</span>
          <button onClick={() => zoomBy(1.2)} title="zoom in" className="w-6 h-5 flex items-center justify-center rounded text-ink-2 hover:bg-line/50"><Icon name="zoomIn" size={14} /></button>
          <button onClick={fit} title="fit to view" className="w-6 h-5 flex items-center justify-center rounded text-ink-2 hover:bg-line/50 ml-0.5"><Icon name="fit" size={14} /></button>
        </div>
        <div className="flex-1 px-3 overflow-hidden whitespace-nowrap text-ink-3/80">
          <span className="text-ink-2">V</span> select &middot; <span className="text-ink-2">B</span> box &middot; <span className="text-ink-2">G</span> polygon &middot; <span className="text-ink-2">K</span> pose &middot; <span className="text-ink-2">R</span> measure &middot; <span className="text-ink-2">[ ]</span> frame &middot; <span className="text-ink-2">Cmd Z</span> undo &middot; <span className="text-ink-2">?</span> shortcuts
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
