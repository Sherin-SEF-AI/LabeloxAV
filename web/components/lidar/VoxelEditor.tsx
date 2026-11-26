"use client";

// The occupancy voxel canvas for /lidar/occupancy. Draws the merged grid (generated plus a person's edits) as
// instanced cubes, the voxels a person cleared as red markers, and the grid's outline. In the bird's-eye view
// a drag paints whole columns (the z range chosen on the page) or spans a box; the page turns that into edit
// ops for the server. The orbit view is for looking, not editing.

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { voxelBox, voxelColumn, type XY } from "@/lib/tools/scene3dGeom";
import type { OccGrid, OccOp } from "@/lib/tools/scene3dApi";

export type VoxTool = "orbit" | "paint" | "box";
export type VoxValue = "occupied" | "free" | "reset";

type Props = {
  grid: OccGrid | null;
  view: "bev" | "perspective";
  tool: VoxTool;
  value: VoxValue;
  zLo: number;
  zHi: number;
  onOps: (ops: OccOp[]) => void;
};

export default function VoxelEditor({ grid, view, tool, value, zLo, zHi, onOps }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const st = useRef<{ renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.Camera; controls: OrbitControls;
    layer: THREE.Group | null; framedFor: string | null } | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const w = mount.clientWidth || 800, h = mount.clientHeight || 600;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.setClearColor(0x0a0e14, 1);
    mount.insertBefore(renderer.domElement, mount.firstChild);
    const scene = new THREE.Scene();
    let camera: THREE.Camera;
    if (view === "bev") {
      const o = new THREE.OrthographicCamera(-45, 45, 45 * (h / w), -45 * (h / w), 0.1, 5000);
      o.up.set(1, 0, 0);
      o.position.set(0, 0, 300);
      o.lookAt(0, 0, 0);
      camera = o;
    } else {
      const p = new THREE.PerspectiveCamera(55, w / h, 0.1, 5000);
      p.up.set(0, 0, 1);
      p.position.set(-30, -40, 30);
      camera = p;
    }
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    if (view === "bev") {
      controls.enableRotate = false;
      controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    }
    scene.add(new THREE.AmbientLight(0xffffff, 1));
    let alive = true;
    const loop = () => { if (!alive) return; controls.update(); renderer.render(scene, camera); requestAnimationFrame(loop); };
    loop();
    const ro = new ResizeObserver(() => {
      const nw = mount.clientWidth || 800, nh = mount.clientHeight || 600;
      renderer.setSize(nw, nh);
      const ov = overlayRef.current;
      if (ov) { ov.width = nw; ov.height = nh; }
      if (camera instanceof THREE.PerspectiveCamera) camera.aspect = nw / nh;
      else if (camera instanceof THREE.OrthographicCamera) {
        const span = (camera.right - camera.left) / 2;
        camera.top = span * (nh / nw); camera.bottom = -span * (nh / nw);
      }
      (camera as THREE.PerspectiveCamera).updateProjectionMatrix();
    });
    ro.observe(mount);
    st.current = { renderer, scene, camera, controls, layer: null, framedFor: null };
    return () => {
      alive = false; ro.disconnect(); controls.dispose(); renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      st.current = null;
    };
  }, [view]);

  // the voxels
  useEffect(() => {
    const s = st.current;
    if (!s) return;
    if (s.layer) {
      s.scene.remove(s.layer);
      s.layer.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose();
        (m.material as THREE.Material | undefined)?.dispose?.();
      });
      s.layer = null;
    }
    if (!grid) return;
    const { origin, voxel_m: vm, dims } = grid;
    const group = new THREE.Group();
    const shown = grid.voxels.map((v, i) => [v, i] as const).filter(([v]) => v[2] >= zLo && v[2] <= zHi);
    const geo = new THREE.BoxGeometry(vm * 0.92, vm * 0.92, vm * 0.92);
    const mesh = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.55 }),
      Math.max(1, shown.length));
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    shown.forEach(([v, i], k) => {
      m.makeTranslation(origin[0] + (v[0] + 0.5) * vm, origin[1] + (v[1] + 0.5) * vm, origin[2] + (v[2] + 0.5) * vm);
      mesh.setMatrixAt(k, m);
      if (grid.edited[i]) c.set(0xf59e0b);
      else c.setHSL((1 - v[2] / Math.max(1, dims[2] - 1)) * 0.6, 0.75, 0.5);
      mesh.setColorAt(k, c);
    });
    mesh.count = shown.length;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    group.add(mesh);

    const freed = grid.set_free.filter((v) => v[2] >= zLo && v[2] <= zHi);
    if (freed.length) {
      const pos: number[] = [];
      for (const v of freed) {
        const x = origin[0] + (v[0] + 0.5) * vm, y = origin[1] + (v[1] + 0.5) * vm, z = origin[2] + (v[2] + 0.5) * vm;
        const d = vm * 0.35;
        pos.push(x - d, y - d, z, x + d, y + d, z, x - d, y + d, z, x + d, y - d, z);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      group.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xef4444 })));
    }
    // grid outline at the floor
    const x0 = origin[0], y0 = origin[1], x1 = origin[0] + dims[0] * vm, y1 = origin[1] + dims[1] * vm, z0 = origin[2];
    const ol = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x0, y0, z0), new THREE.Vector3(x1, y0, z0), new THREE.Vector3(x1, y1, z0),
      new THREE.Vector3(x0, y1, z0), new THREE.Vector3(x0, y0, z0)]);
    group.add(new THREE.Line(ol, new THREE.LineBasicMaterial({ color: 0x334155 })));
    s.scene.add(group);
    s.layer = group;

    // frame once per session, so stepping through grids keeps the view where the person put it
    if (s.framedFor !== grid.session_id) {
      s.framedFor = grid.session_id;
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      s.controls.target.set(cx, cy, z0);
      if (s.camera instanceof THREE.OrthographicCamera) s.camera.position.set(cx, cy, 300);
      else s.camera.position.set(cx - 35, cy - 45, z0 + 35);
      s.controls.update();
    }
  }, [grid, zLo, zHi]);

  useEffect(() => {
    const s = st.current;
    if (s) s.controls.enabled = tool === "orbit" || view !== "bev";
  }, [tool, view]);

  // editing, bird's-eye only
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || !grid || tool === "orbit" || view !== "bev") return;
    const ray = new THREE.Raycaster();
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    const ground = (e: PointerEvent): number[] | null => {
      const s = st.current;
      if (!s) return null;
      const r = mount.getBoundingClientRect();
      ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), s.camera);
      const p = new THREE.Vector3();
      return ray.ray.intersectPlane(plane, p) ? [p.x, p.y] : null;
    };
    const cell = (p: number[]) => [Math.floor((p[0] - grid.origin[0]) / grid.voxel_m), Math.floor((p[1] - grid.origin[1]) / grid.voxel_m)];
    let start: number[] | null = null;
    let startPx: XY | null = null;
    const cells = new Map<string, number[]>();
    const ctx = () => overlayRef.current?.getContext("2d") ?? null;
    const px = (e: PointerEvent): XY => { const r = mount.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    const colour = value === "free" ? "#f87171" : value === "reset" ? "#94a3b8" : "#fbbf24";
    const down = (e: PointerEvent) => {
      if (e.button !== 0) return;
      const g = ground(e);
      if (!g) return;
      mount.setPointerCapture(e.pointerId);
      start = g; startPx = px(e); cells.clear();
      if (tool === "paint") { const c = cell(g); cells.set(c.join(","), c); }
    };
    const move = (e: PointerEvent) => {
      if (!start || !startPx) return;
      const g = ground(e);
      const c2 = ctx();
      const p = px(e);
      if (!g || !c2) return;
      c2.clearRect(0, 0, c2.canvas.width, c2.canvas.height);
      c2.strokeStyle = colour; c2.lineWidth = 1.5;
      if (tool === "paint") {
        const c = cell(g); cells.set(c.join(","), c);
        c2.beginPath(); c2.arc(p[0], p[1], 6, 0, Math.PI * 2); c2.stroke();
      } else {
        c2.strokeRect(startPx[0], startPx[1], p[0] - startPx[0], p[1] - startPx[1]);
      }
    };
    const up = (e: PointerEvent) => {
      if (!start) return;
      const g = ground(e) ?? start;
      const c2 = ctx();
      if (c2) c2.clearRect(0, 0, c2.canvas.width, c2.canvas.height);
      if (tool === "paint") {
        const vox: number[][] = [];
        for (const [ix, iy] of cells.values()) {
          if (ix < 0 || iy < 0 || ix >= grid.dims[0] || iy >= grid.dims[1]) continue;
          vox.push(...voxelColumn(ix, iy, zLo, zHi));
        }
        if (vox.length) onOps([{ kind: "paint", value, voxels: vox }]);
      } else {
        const b = voxelBox(start, g, zLo, zHi, grid.origin, grid.voxel_m, grid.dims);
        if (b) onOps([{ kind: "box", value, lo: b.lo, hi: b.hi }]);
      }
      start = null; startPx = null; cells.clear();
    };
    mount.addEventListener("pointerdown", down);
    mount.addEventListener("pointermove", move);
    mount.addEventListener("pointerup", up);
    return () => {
      mount.removeEventListener("pointerdown", down);
      mount.removeEventListener("pointermove", move);
      mount.removeEventListener("pointerup", up);
    };
  }, [grid, tool, view, value, zLo, zHi, onOps]);

  return (
    <div ref={mountRef} className="relative h-full w-full" style={{ cursor: tool === "orbit" || view !== "bev" ? "grab" : "crosshair" }}>
      <canvas ref={overlayRef} className="pointer-events-none absolute inset-0" />
    </div>
  );
}
