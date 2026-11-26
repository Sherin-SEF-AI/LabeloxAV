"use client";

// The point-labelling canvas for /lidar/segment. three.js renders the cloud (top-down BEV or an orbit view)
// coloured by class, with the current selection highlighted. Selection is done in screen space over the
// projected points, so it works the same in both views: brush (drag), lasso (drag a loop), box (drag a
// rectangle), polygon (click vertices, double-click or Enter to close), and pick (click a point; the page
// asks the server for the DBSCAN cluster under it). Shift adds to the selection, Alt subtracts.

import { useCallback, useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import {
  brushInto, classRgb, selectPolygon, selectRect, type Combine, type XY,
} from "@/lib/tools/scene3dGeom";

export type SegTool = "orbit" | "brush" | "lasso" | "box" | "polygon" | "pick";

type Props = {
  points: Float32Array | null;        // interleaved x, y, z, intensity
  count: number;
  labels: Int16Array | null;
  labelsRev: number;                  // bump to recolour after an in-place label edit
  selection: Uint8Array | null;
  view: "bev" | "perspective";
  tool: SegTool;
  brushPx: number;
  pointSize?: number;
  onSelect: (mask: Uint8Array, how: Combine) => void;
  onPick?: (p: [number, number, number]) => void;
};

const combineOf = (e: { shiftKey: boolean; altKey: boolean }): Combine => (e.altKey ? "subtract" : e.shiftKey ? "add" : "replace");

export default function SegmentViewer({
  points, count, labels, labelsRev, selection, view, tool, brushPx, pointSize, onSelect, onPick,
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const st = useRef<{
    renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.Camera; controls: OrbitControls;
    cloud: THREE.Points | null; zmin: number; zmax: number;
  } | null>(null);
  const draw = useRef<{ path: XY[]; mask: Uint8Array | null; screen: Float32Array | null; how: Combine; active: boolean }>(
    { path: [], mask: null, screen: null, how: "replace", active: false });

  // renderer, camera, controls
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const w = mount.clientWidth || 800, h = mount.clientHeight || 600;
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.setClearColor(0x0a0e14, 1);
    mount.insertBefore(renderer.domElement, mount.firstChild);
    const scene = new THREE.Scene();
    let camera: THREE.Camera;
    if (view === "bev") {
      const span = 40;
      const o = new THREE.OrthographicCamera(-span, span, span * (h / w), -span * (h / w), 0.1, 5000);
      o.position.set(15, 0, 200);
      o.up.set(1, 0, 0);
      o.lookAt(15, 0, 0);
      camera = o;
    } else {
      const p = new THREE.PerspectiveCamera(60, w / h, 0.1, 5000);
      p.up.set(0, 0, 1);
      p.position.set(-15, -18, 14);
      camera = p;
    }
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    if (view === "bev") {
      controls.enableRotate = false;
      controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
      controls.target.set(15, 0, 0);
    }
    const grid = new THREE.GridHelper(200, 40, 0x1e2a38, 0x141d27);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    let alive = true;
    const loop = () => {
      if (!alive) return;
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(loop);
    };
    loop();
    const ro = new ResizeObserver(() => {
      const nw = mount.clientWidth || 800, nh = mount.clientHeight || 600;
      renderer.setSize(nw, nh);
      const ov = overlayRef.current;
      if (ov) { ov.width = nw; ov.height = nh; }
      if (camera instanceof THREE.PerspectiveCamera) {
        camera.aspect = nw / nh;
      } else if (camera instanceof THREE.OrthographicCamera) {
        const span = (camera.right - camera.left) / 2;
        camera.top = span * (nh / nw); camera.bottom = -span * (nh / nw);
      }
      (camera as THREE.PerspectiveCamera).updateProjectionMatrix();
    });
    ro.observe(mount);
    st.current = { renderer, scene, camera, controls, cloud: null, zmin: -2, zmax: 2 };
    return () => {
      alive = false;
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      st.current = null;
    };
  }, [view]);

  // geometry
  useEffect(() => {
    const s = st.current;
    if (!s || !points || !count) return;
    if (s.cloud) {
      s.scene.remove(s.cloud);
      s.cloud.geometry.dispose();
      (s.cloud.material as THREE.Material).dispose();
    }
    const pos = new Float32Array(count * 3);
    const zs: number[] = [];
    for (let i = 0; i < count; i++) {
      pos[i * 3] = points[i * 4]; pos[i * 3 + 1] = points[i * 4 + 1]; pos[i * 3 + 2] = points[i * 4 + 2];
      if (i % 50 === 0) zs.push(points[i * 4 + 2]);
    }
    zs.sort((a, b) => a - b);
    s.zmin = zs[Math.floor(zs.length * 0.02)] ?? -2;
    s.zmax = zs[Math.floor(zs.length * 0.98)] ?? 2;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    g.computeBoundingSphere();
    const m = new THREE.PointsMaterial({ size: pointSize ?? (view === "bev" ? 2.2 : 0.06), vertexColors: true,
                                         sizeAttenuation: view !== "bev" });
    s.cloud = new THREE.Points(g, m);
    s.scene.add(s.cloud);
  }, [points, count, view, pointSize]);

  // colours: selection on top, then class, then a dim height ramp for unlabelled points
  useEffect(() => {
    const s = st.current;
    if (!s || !s.cloud || !points) return;
    const attr = s.cloud.geometry.getAttribute("color") as THREE.BufferAttribute;
    const col = attr.array as Float32Array;
    const span = s.zmax - s.zmin || 1;
    const cache = new Map<number, [number, number, number] | null>();
    for (let i = 0; i < count; i++) {
      if (selection && selection[i]) { col[i * 3] = 1; col[i * 3 + 1] = 0.95; col[i * 3 + 2] = 0.35; continue; }
      const c = labels ? labels[i] : -1;
      let rgb = cache.get(c);
      if (rgb === undefined) { rgb = classRgb(c); cache.set(c, rgb); }
      if (rgb) { col[i * 3] = rgb[0]; col[i * 3 + 1] = rgb[1]; col[i * 3 + 2] = rgb[2]; continue; }
      const t = Math.max(0, Math.min(1, (points[i * 4 + 2] - s.zmin) / span));
      const v = 0.22 + 0.4 * t;
      col[i * 3] = v; col[i * 3 + 1] = v + 0.03; col[i * 3 + 2] = v + 0.08;
    }
    attr.needsUpdate = true;
  }, [points, count, labels, labelsRev, selection]);

  useEffect(() => {
    const s = st.current;
    if (s) s.controls.enabled = tool === "orbit" || tool === "pick";
    draw.current = { path: [], mask: null, screen: null, how: "replace", active: false };
    clearOverlay();
  }, [tool, view]);

  const project = useCallback((): Float32Array => {
    const s = st.current!;
    const el = s.renderer.domElement;
    const w = el.clientWidth, h = el.clientHeight;
    s.camera.updateMatrixWorld();
    const m = new THREE.Matrix4().multiplyMatrices(
      (s.camera as THREE.PerspectiveCamera).projectionMatrix, s.camera.matrixWorldInverse).elements;
    const out = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      const x = points![i * 4], y = points![i * 4 + 1], z = points![i * 4 + 2];
      const cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw <= 1e-6) { out[i * 2] = NaN; out[i * 2 + 1] = NaN; continue; }
      const nx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / cw;
      const ny = (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw;
      const nz = (m[2] * x + m[6] * y + m[10] * z + m[14]) / cw;
      if (nz < -1 || nz > 1) { out[i * 2] = NaN; out[i * 2 + 1] = NaN; continue; }
      out[i * 2] = (nx + 1) / 2 * w;
      out[i * 2 + 1] = (1 - ny) / 2 * h;
    }
    return out;
  }, [points, count]);

  function clearOverlay() {
    const ov = overlayRef.current;
    const ctx = ov?.getContext("2d");
    if (ov && ctx) ctx.clearRect(0, 0, ov.width, ov.height);
  }

  function paintOverlay(cursor?: XY) {
    const ov = overlayRef.current;
    const ctx = ov?.getContext("2d");
    if (!ov || !ctx) return;
    ctx.clearRect(0, 0, ov.width, ov.height);
    const d = draw.current;
    ctx.strokeStyle = d.how === "subtract" ? "#f87171" : "#fde047";
    ctx.fillStyle = "rgba(253, 224, 71, 0.08)";
    ctx.lineWidth = 1.5;
    if ((tool === "lasso" || tool === "polygon") && d.path.length) {
      ctx.beginPath();
      ctx.moveTo(d.path[0][0], d.path[0][1]);
      for (const p of d.path.slice(1)) ctx.lineTo(p[0], p[1]);
      if (tool === "polygon" && cursor) ctx.lineTo(cursor[0], cursor[1]);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    } else if (tool === "box" && d.path.length === 2) {
      const [a, b] = d.path;
      ctx.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
      ctx.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]);
    }
    if (tool === "brush" && cursor) {
      ctx.beginPath();
      ctx.arc(cursor[0], cursor[1], brushPx, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // pointer handling for the selection tools
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !points || tool === "orbit") return;
    const xy = (e: PointerEvent | MouseEvent): XY => {
      const r = mount.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    const finishPolygon = () => {
      const d = draw.current;
      if (d.path.length >= 3) onSelect(selectPolygon(project(), d.path), d.how);
      draw.current = { path: [], mask: null, screen: null, how: "replace", active: false };
      clearOverlay();
    };
    const down = (e: PointerEvent) => {
      if (e.button !== 0) return;
      const p = xy(e);
      const d = draw.current;
      if (tool === "pick") {
        const s = st.current;
        if (!s?.cloud) return;
        const r = mount.getBoundingClientRect();
        const ray = new THREE.Raycaster();
        ray.params.Points = { threshold: view === "bev" ? 0.4 : 0.25 };
        ray.setFromCamera(new THREE.Vector2((p[0] / r.width) * 2 - 1, -(p[1] / r.height) * 2 + 1), s.camera);
        const hit = ray.intersectObject(s.cloud)[0];
        if (hit && hit.index != null && onPick) {
          const i = hit.index;
          onPick([points[i * 4], points[i * 4 + 1], points[i * 4 + 2]]);
        }
        return;
      }
      if (tool === "polygon") {
        if (!d.path.length) d.how = combineOf(e);
        d.path.push(p);
        paintOverlay(p);
        return;
      }
      mount.setPointerCapture(e.pointerId);
      draw.current = { path: [p], mask: tool === "brush" ? new Uint8Array(count) : null,
                       screen: tool === "brush" ? project() : null, how: combineOf(e), active: true };
      if (tool === "brush") brushInto(draw.current.mask!, draw.current.screen!, p[0], p[1], brushPx);
      paintOverlay(p);
    };
    const move = (e: PointerEvent) => {
      const p = xy(e);
      const d = draw.current;
      if (tool === "polygon") { if (d.path.length) paintOverlay(p); return; }
      if (!d.active) { if (tool === "brush") paintOverlay(p); return; }
      if (tool === "brush") brushInto(d.mask!, d.screen!, p[0], p[1], brushPx);
      else if (tool === "lasso") d.path.push(p);
      else if (tool === "box") d.path = [d.path[0], p];
      paintOverlay(p);
    };
    const up = () => {
      const d = draw.current;
      if (!d.active) return;
      if (tool === "brush" && d.mask) onSelect(d.mask, d.how);
      else if (tool === "lasso" && d.path.length >= 3) onSelect(selectPolygon(project(), d.path), d.how);
      else if (tool === "box" && d.path.length === 2) onSelect(selectRect(project(), d.path[0], d.path[1]), d.how);
      draw.current = { path: [], mask: null, screen: null, how: "replace", active: false };
      clearOverlay();
    };
    const dbl = () => { if (tool === "polygon") finishPolygon(); };
    const key = (e: KeyboardEvent) => {
      if (tool !== "polygon") return;
      if (e.key === "Enter") finishPolygon();
      if (e.key === "Escape") { draw.current = { path: [], mask: null, screen: null, how: "replace", active: false }; clearOverlay(); }
    };
    mount.addEventListener("pointerdown", down);
    mount.addEventListener("pointermove", move);
    mount.addEventListener("pointerup", up);
    mount.addEventListener("dblclick", dbl);
    window.addEventListener("keydown", key);
    return () => {
      mount.removeEventListener("pointerdown", down);
      mount.removeEventListener("pointermove", move);
      mount.removeEventListener("pointerup", up);
      mount.removeEventListener("dblclick", dbl);
      window.removeEventListener("keydown", key);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tool, points, count, brushPx, view, onSelect, onPick, project]);

  return (
    <div ref={mountRef} className="relative h-full w-full" style={{ cursor: tool === "orbit" ? "grab" : "crosshair" }}>
      <canvas ref={overlayRef} className="pointer-events-none absolute inset-0" />
    </div>
  );
}
