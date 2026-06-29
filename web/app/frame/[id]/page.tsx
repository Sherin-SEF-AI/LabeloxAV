"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import type { AdverseRegion, FrameMeta, LaneRow, ObjectDynamicsRow, Ontology, OntologyClass, Relationship } from "@/lib/types";
import { classColor } from "@/lib/colors";
import { acceptState, getUser, setUser } from "@/lib/user";
import { isDirty, tmpId, useEditor, type EdObject, type Tool } from "@/components/editor/useEditor";
import { PERSON_17 } from "@/lib/skeleton";
import BackButton from "@/components/BackButton";
import CorrectionModal, { type CorrectionChange } from "@/components/CorrectionModal";

// Frame-centric professional annotation editor. Pan/zoom canvas, draw + edit boxes, SAM-assisted masks,
// layers panel, class palette, attributes, keyboard-driven, batched save. Operational Materialism tokens.

// Wrap the import so next/dynamic's convertModule always gets a clean { default } and cannot mistake the
// module for a react-konva export on a StrictMode re-mount.
const EditorCanvas = dynamic(() => import("@/components/editor/EditorCanvas").then((m) => ({ default: m.default })), { ssr: false });

const TOOLS: { key: Tool; label: string; hot: string }[] = [
  { key: "select", label: "select", hot: "V" },
  { key: "box", label: "box", hot: "B" },
  { key: "polygon", label: "polygon", hot: "G" },
  { key: "polyline", label: "polyline", hot: "L" },
  { key: "adverse", label: "adverse", hot: "D" },
  { key: "keypoint", label: "pose", hot: "K" },
  { key: "measure", label: "measure", hot: "R" },
  { key: "sam-point", label: "sam pt", hot: "S" },
  { key: "sam-box", label: "sam box", hot: "M" },
];

// directed object-relationship kinds offered in the editor (rider_of is the India two-wheeler case)
const RELATION_KINDS = ["rider_of", "towed_by", "part_of", "member_of", "occludes"];

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
  const [lanes, setLanes] = useState<LaneRow[]>([]);
  const [drivable, setDrivable] = useState<Record<string, number[][]> | null>(null);
  const [relationships, setRelationships] = useState<Relationship[]>([]);
  const [linkFrom, setLinkFrom] = useState<string | null>(null); // active "relate" mode: the source object id
  const [linkKind, setLinkKind] = useState("rider_of");
  const [adverse, setAdverse] = useState<AdverseRegion[]>([]);
  const [adverseCond, setAdverseCond] = useState("glare");
  const [layers, setLayers] = useState({ boxes: true, masks: true, lanes: true, drivable: true, adverse: true });

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
    const [ls, dr, rel, adv] = await Promise.all([api.framesLanes(id).catch(() => []), api.getDrivable(id).catch(() => null), api.frameRelationships(id).catch(() => []), api.listAdverse(id).catch(() => [])]);
    setLanes(ls);
    setDrivable(dr && dr.found ? dr.classes ?? null : null);
    setRelationships(rel);
    setAdverse(adv);
  }, [id]);
  useEffect(() => { loadLayers(); }, [loadLayers]);
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
            keypoints: o.keypoints ?? null, polyline: o.polyline,
          });
          remap[o.id] = created.object_id;
          if (created.version != null) versions[o.id] = created.version;
        } else if (o.dirty) {
          // One atomic request: geometry, mask, rotation, and keypoints persist together (no separate
          // updateMask that could leave the mask out of sync on a partial failure).
          const r = await api.review(o.id, { action: "adjust_geometry",
            class_name: o.class_name, bbox: o.bbox, attrs: o.attrs, state: tgt, expected_version: o.version,
            rot_deg: o.rot ?? 0, keypoints: o.keypoints ?? null, polyline: o.polyline,
            mask_polygons: o.mask.length ? o.mask : undefined });
          if (r.version != null) versions[o.id] = r.version;
        }
      }
      dispatch({ t: "saved", remap, versions });
      flash("saved");
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
      const k = e.key.toLowerCase();
      if (k === "a") dispatch({ t: "acceptAll" });
      else if (k === "v") dispatch({ t: "tool", tool: "select" });
      else if (k === "b") dispatch({ t: "tool", tool: "box" });
      else if (k === "g") dispatch({ t: "tool", tool: "polygon" });
      else if (k === "l") dispatch({ t: "tool", tool: "polyline" });
      else if (k === "d") dispatch({ t: "tool", tool: "adverse" });
      else if (k === "k") dispatch({ t: "tool", tool: "keypoint" });
      else if (k === "r") dispatch({ t: "tool", tool: "measure" });
      else if (k === "s") dispatch({ t: "tool", tool: "sam-point" });
      else if (k === "m") dispatch({ t: "tool", tool: "sam-box" });
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
  }, [st.selectedId, selected, onto, meta, dispatch, save, fit, zoomBy, acceptCandidate, gotoFrame, relabelSelected, finishKeypoints]);

  const filteredClasses = useMemo(
    () => (onto ? onto.classes.filter((c) => c.name.includes(search.toLowerCase().replace(/\s/g, "_"))) : []),
    [onto, search],
  );

  if (!meta || !onto) return <div className="min-h-screen flex items-center justify-center font-mono text-ink-3">loading frame...</div>;

  return (
    <div className="h-screen flex flex-col">
      {/* header / toolbar */}
      <header className="flex items-center gap-3 px-3 h-11 border-b hairline shrink-0">
        <BackButton />
        <button onClick={() => router.push("/")} className="font-display font-bold text-sm" title="home (triage)">
          Labelox<span className="text-accent">AV</span>
        </button>
        <span className="font-mono text-[11px] text-ink-3">/ FRAME <span className="text-ink-2">{String(id).slice(0, 8)}</span></span>
        <button onClick={() => router.push(`/search?frame=${id}`)} title="find visually similar frames (DINOv3)"
          className="font-mono text-[11px] text-ink-3 hover:text-accent border border-line hover:border-accent px-2 py-0.5">find similar</button>
        <div className="flex items-center gap-1 ml-2">
          {TOOLS.map((t) => (
            <button key={t.key} onClick={() => dispatch({ t: "tool", tool: t.key })} title={`${t.label} (${t.hot})`}
              className={`font-mono text-[11px] px-2 py-1 border ${st.tool === t.key ? "border-accent text-ink" : "border-line text-ink-3"}`}>
              {t.label}
            </button>
          ))}
          {st.tool === "adverse" && (
            <select value={adverseCond} onChange={(e) => setAdverseCond(e.target.value)} title="adverse condition to tag"
              className="bg-bg border border-accent text-accent px-1 py-1 font-mono text-[11px]">
              {["glare", "reflection", "shadow", "rain", "fog", "lowlight"].map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          )}
        </div>
        <div className="flex items-center gap-1 ml-2 font-mono text-[11px]">
          <button onClick={() => zoomBy(1 / 1.2)} className="border border-line px-2 py-1 hover:border-accent">-</button>
          <span className="w-12 text-center text-ink-2">{Math.round(st.viewport.scale * 100) || ""}%</span>
          <button onClick={() => zoomBy(1.2)} className="border border-line px-2 py-1 hover:border-accent">+</button>
          <button onClick={fit} className="border border-line px-2 py-1 hover:border-accent">fit</button>
        </div>
        {/* P4 layer visibility toggles: show that the road IS segmented (lanes + drivable + masks) */}
        <div className="flex items-center gap-1 ml-1 font-mono text-[11px]" title="toggle annotation layers">
          {(["boxes", "masks", "lanes", "drivable", "adverse"] as const).map((k) => (
            <button key={k} onClick={() => setLayers((s) => ({ ...s, [k]: !s[k] }))}
              className={`px-1.5 py-1 border ${layers[k] ? "border-accent text-ink" : "border-line text-ink-3"}`}>{k}</button>
          ))}
        </div>
        <div className="flex items-center gap-1 font-mono text-[11px]">
          <button onClick={() => dispatch({ t: "undo" })} disabled={!st.past.length} className="border border-line px-2 py-1 disabled:opacity-40 hover:border-accent">undo</button>
          <button onClick={() => dispatch({ t: "redo" })} disabled={!st.future.length} className="border border-line px-2 py-1 disabled:opacity-40 hover:border-accent">redo</button>
        </div>
        <div className="ml-auto flex items-center gap-2 font-mono text-[11px]">
          <button onClick={() => gotoFrame(meta.prev_frame_id)} disabled={!meta.prev_frame_id} className="border border-line px-2 py-1 disabled:opacity-40 hover:border-accent">[ prev</button>
          <button onClick={() => gotoFrame(meta.next_frame_id)} disabled={!meta.next_frame_id} className="border border-line px-2 py-1 disabled:opacity-40 hover:border-accent">next ]</button>
          <button onClick={() => dispatch({ t: "acceptAll" })} disabled={!st.objects.length}
            title="confirm every object as human-verified gold (A)"
            className="border border-pass text-pass px-2 py-1 disabled:opacity-40 hover:bg-pass/10">
            confirm frame (A)
          </button>
          <button onClick={() => setAutosave((v) => !v)}
            title="autosave: persist every edit automatically a moment after you stop"
            className={`border px-2 py-1 ${autosave ? "border-accent text-accent" : "border-line text-ink-3"}`}>
            autosave {autosave ? "on" : "off"}
          </button>
          <span className="w-16 text-center" title="save status">
            {saving ? <span className="text-warn">saving…</span>
              : dirty ? <span className="text-ink-3">{autosave ? "editing…" : "unsaved"}</span>
              : <span className="text-pass">✓ saved</span>}
          </span>
          <button onClick={save} disabled={!dirty || saving}
            title="save now (Cmd/Ctrl+S)"
            className={`px-3 py-1 border ${dirty ? "border-pass text-pass" : "border-line text-ink-3"} disabled:opacity-50`}>
            save
          </button>
        </div>
      </header>

      <div className="flex-1 flex min-h-0">
        {/* canvas */}
        <div ref={canvasWrapRef} className="flex-1 min-w-0 relative">
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
          {st.candidate?.length ? (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 panel px-3 py-1.5 font-mono text-[11px] text-ink-2">
              mask ready, click the next object to keep this one &amp; continue, <span className="text-pass">Enter</span> to finish, <span className="text-ink-3">Esc</span> to discard
            </div>
          ) : null}

          {/* inline edit popup: click a (wrong) annotation to fix its class right where it sits */}
          {selected && st.tool === "select" && st.viewport.scale > 0 && (() => {
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
          {/* status bar */}
          <div className="absolute bottom-0 left-0 right-0 h-6 bg-bg/80 border-t hairline flex items-center gap-4 px-3 font-mono text-[10px] text-ink-3">
            <span>{Math.round(st.viewport.scale * 100)}%</span>
            <span>{cursor ? `${Math.round(cursor[0])}, ${Math.round(cursor[1])}` : "-, -"}</span>
            <span>{st.objects.length} objects</span>
            <span className={dirty ? "text-warn" : "text-pass"}>{dirty ? "unsaved" : "saved"}</span>
            <span className="ml-auto">V select · B box · G polygon · K pose (Enter to finish) · R measure · S sam · M sam-box · Cmd+C/V copy · space pan · Del delete · [ ] frame · Cmd+Z undo · Cmd+S save</span>
          </div>
        </div>

        {/* right rail */}
        <aside className="w-72 shrink-0 border-l hairline flex flex-col min-h-0">
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
                    className="font-mono text-[10px] text-info hover:text-accent">view track →</button>
                )}
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

          {/* layers / object list */}
          <div className="p-2">
            <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">objects ({st.objects.length})</div>
            <div className="space-y-0.5">
              {st.objects.map((o) => (
                <div key={o.id} onClick={() => dispatch({ t: "select", id: o.id })}
                  className={`flex items-center gap-1.5 px-1 py-0.5 cursor-pointer font-mono text-[11px] ${o.id === st.selectedId ? "bg-line text-ink" : "text-ink-3 hover:text-ink-2"}`}>
                  <button onClick={(e) => { e.stopPropagation(); dispatch({ t: "update", id: o.id, patch: { visible: !o.visible } }); }}
                    className={o.visible ? "text-ink-2" : "text-ink-3"}>{o.visible ? "●" : "○"}</button>
                  <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: classColor(o.class_id) }} />
                  <span className="truncate flex-1">{o.class_name}{o.isNew ? " *" : ""}</span>
                  {o.mask.length > 0 && <span className="text-info" title="has mask">◆</span>}
                  <button onClick={(e) => { e.stopPropagation(); dispatch({ t: "delete", id: o.id }); }} className="text-ink-3 hover:text-block">x</button>
                </div>
              ))}
              {!st.objects.length && <div className="text-ink-3 text-center py-4">no objects. draw a box (B).</div>}
            </div>
          </div>
          </div>
        </aside>
      </div>
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
