"use client";

// A three.js point cloud canvas. One instance renders a perspective orbit view, another a top-down BEV.
// Points arrive as a packed Float32 [x, y, z, intensity] buffer; colour is computed on the CPU through a
// 256-entry lookup table so re-colouring 400k points stays instant. The ego frame is x forward, y left,
// z up, so the camera up is set to +z. Click two points to measure the distance between them.

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export type ColorBy = "height" | "intensity" | "source";

export type ViewerCuboid = {
  object_3d_id: string;
  center: number[];
  dims: number[];
  yaw: number;
  pitch?: number;
  roll?: number;
  state: string;
  class_name?: string;
};

type Props = {
  points: Float32Array | null; // interleaved [x, y, z, intensity]
  count: number;
  colorBy: ColorBy;
  intensityRange: [number, number];
  source: string;
  mode: "perspective" | "bev";
  pointSize?: number;
  onMeasure?: (meters: number | null) => void;
  showEgo?: boolean;
  trajectory?: { x: number; y: number }[];
  cuboids?: ViewerCuboid[];
  selectedId?: string | null;
  onSelectCuboid?: (id: string | null) => void;
  onMoveCuboid?: (id: string, x: number, y: number, commit: boolean) => void;
};

const SOURCE_RGB: Record<string, [number, number, number]> = {
  pseudo: [0.2, 0.85, 0.95],
  lidar: [0.45, 0.95, 0.55],
  dataset: [0.98, 0.78, 0.32],
};

const STATE_COLOR: Record<string, number> = {
  auto_accept: 0x4ade80,
  accepted: 0x22d3ee,
  review: 0xfbbf24,
  annotate: 0xf87171,
};

function buildLUT(colorBy: ColorBy, source: string): Float32Array {
  // 256 RGB triples spanning the normalized value 0..1 for the active colour mode.
  const lut = new Float32Array(256 * 3);
  const c = new THREE.Color();
  const fixed = SOURCE_RGB[source] || [0.7, 0.7, 0.8];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let r: number, g: number, b: number;
    if (colorBy === "height") {
      // blue (low) to red (high), the standard elevation ramp
      c.setHSL((1 - t) * 0.66, 0.9, 0.5);
      r = c.r; g = c.g; b = c.b;
    } else if (colorBy === "intensity") {
      // dark to bright, so high-reflectance road markings stand out
      r = 0.12 + 0.88 * t; g = 0.12 + 0.86 * t; b = 0.14 + 0.7 * t;
    } else {
      const k = 0.45 + 0.55 * t; // shade the fixed source colour by height for depth cue
      r = fixed[0] * k; g = fixed[1] * k; b = fixed[2] * k;
    }
    lut[i * 3] = r; lut[i * 3 + 1] = g; lut[i * 3 + 2] = b;
  }
  return lut;
}

export default function PointCloudViewer({
  points, count, colorBy, intensityRange, source, mode, pointSize = 0.06, onMeasure,
  showEgo = false, trajectory, cuboids, selectedId, onSelectCuboid, onMoveCuboid,
}: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.Camera;
    controls: OrbitControls; cloud: THREE.Points | null; raycaster: THREE.Raycaster;
    measure: THREE.Vector3[]; marks: THREE.Object3D | null; dispose: () => void;
  } | null>(null);
  const cubeRef = useRef<{ group: THREE.Group; meshes: THREE.Mesh[] } | null>(null);

  // mount: renderer, camera, controls, render loop
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const w = mount.clientWidth || 800;
    const h = mount.clientHeight || 600;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.setClearColor(0x0a0e14, 1);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();

    let camera: THREE.Camera;
    if (mode === "bev") {
      const span = 60;
      const ortho = new THREE.OrthographicCamera(-span, span, span * (h / w), -span * (h / w), 0.1, 5000);
      ortho.position.set(0, 0, 200);
      ortho.up.set(1, 0, 0); // vehicle forward (+x) points up on screen
      ortho.lookAt(0, 0, 0);
      camera = ortho;
    } else {
      const persp = new THREE.PerspectiveCamera(60, w / h, 0.1, 5000);
      persp.up.set(0, 0, 1);
      persp.position.set(-18, -22, 14);
      camera = persp;
    }

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    if (mode === "bev") {
      controls.enableRotate = false;
      controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    }

    // a faint ground grid in the road plane (z = 0)
    const grid = new THREE.GridHelper(120, 24, 0x1e2a38, 0x141d27);
    grid.rotation.x = Math.PI / 2; // GridHelper is xz by default; rotate into the xy (ground) plane
    scene.add(grid);

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.35 };

    let alive = true;
    const animate = () => {
      if (!alive) return;
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    animate();

    const onResize = () => {
      const nw = mount.clientWidth || 800;
      const nh = mount.clientHeight || 600;
      renderer.setSize(nw, nh);
      if (camera instanceof THREE.PerspectiveCamera) {
        camera.aspect = nw / nh;
        camera.updateProjectionMatrix();
      } else if (camera instanceof THREE.OrthographicCamera) {
        const span = (camera.top - camera.bottom) / 2 / (nh / nw) || 60;
        camera.left = -span; camera.right = span;
        camera.top = span * (nh / nw); camera.bottom = -span * (nh / nw);
        camera.updateProjectionMatrix();
      }
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);

    const dispose = () => {
      alive = false;
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };

    stateRef.current = { renderer, scene, camera, controls, cloud: null, raycaster, measure: [], marks: null, dispose };
    return dispose;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  // data: rebuild geometry + colours when the points or colour mode change
  useEffect(() => {
    const st = stateRef.current;
    if (!st || !points || count === 0) return;

    if (st.cloud) {
      st.scene.remove(st.cloud);
      st.cloud.geometry.dispose();
      (st.cloud.material as THREE.Material).dispose();
      st.cloud = null;
    }

    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const lut = buildLUT(colorBy, source);
    let zmin = Infinity, zmax = -Infinity;
    for (let i = 0; i < count; i++) zmin = Math.min(zmin, points[i * 4 + 2]), zmax = Math.max(zmax, points[i * 4 + 2]);
    const zspan = zmax - zmin || 1;
    const [imin, imax] = intensityRange;
    const ispan = imax - imin || 1;

    for (let i = 0; i < count; i++) {
      const x = points[i * 4], y = points[i * 4 + 1], z = points[i * 4 + 2], it = points[i * 4 + 3];
      pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
      let t: number;
      if (colorBy === "intensity") t = (it - imin) / ispan;
      else if (colorBy === "height") t = (z - zmin) / zspan;
      else t = (z - zmin) / zspan;
      const bin = Math.max(0, Math.min(255, Math.round(t * 255)));
      col[i * 3] = lut[bin * 3]; col[i * 3 + 1] = lut[bin * 3 + 1]; col[i * 3 + 2] = lut[bin * 3 + 2];
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geom.computeBoundingSphere();
    const mat = new THREE.PointsMaterial({ size: pointSize, vertexColors: true, sizeAttenuation: mode !== "bev" });
    const cloud = new THREE.Points(geom, mat);
    st.scene.add(cloud);
    st.cloud = cloud;

    // frame the camera on the cloud
    const bs = geom.boundingSphere;
    if (bs && st.camera instanceof THREE.OrthographicCamera) {
      const span = Math.max(bs.radius * 1.1, 10);
      const mount = mountRef.current!;
      const ar = (mount.clientHeight || 600) / (mount.clientWidth || 800);
      st.camera.left = -span; st.camera.right = span;
      st.camera.top = span * ar; st.camera.bottom = -span * ar;
      st.camera.position.set(bs.center.x, bs.center.y, 200);
      st.controls.target.set(bs.center.x, bs.center.y, 0);
      st.camera.updateProjectionMatrix();
    } else if (bs && st.camera instanceof THREE.PerspectiveCamera) {
      const r = Math.max(bs.radius, 8);
      st.controls.target.copy(bs.center);
      st.camera.position.set(bs.center.x - r * 0.9, bs.center.y - r * 1.1, bs.center.z + r * 0.7);
    }
    st.controls.update();
  }, [points, count, colorBy, intensityRange, source, pointSize, mode]);

  // overlays: the ego vehicle footprint, sensor positions, turning radius, and the GNSS trajectory
  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    const group = new THREE.Group();
    const z = 0.03;

    if (showEgo) {
      const L0 = -0.9, L1 = 3.1, wy = 0.85, camH = 1.5;
      const rect = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(L0, -wy, z), new THREE.Vector3(L1, -wy, z),
        new THREE.Vector3(L1, wy, z), new THREE.Vector3(L0, wy, z), new THREE.Vector3(L0, -wy, z),
      ]);
      group.add(new THREE.Line(rect, new THREE.LineBasicMaterial({ color: 0x4ade80 })));
      const arrow = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(L1, 0, z), new THREE.Vector3(L1 + 1.6, 0, z),
      ]);
      group.add(new THREE.Line(arrow, new THREE.LineBasicMaterial({ color: 0x4ade80 })));

      const mounts: [number, number][] = [[3.0, 0], [1.0, wy], [1.0, -wy], [L0, 0]];
      const sphere = new THREE.SphereGeometry(0.13, 8, 8);
      const sensorMat = new THREE.MeshBasicMaterial({ color: 0x22d3ee });
      for (const [mx, my] of mounts) {
        const s = new THREE.Mesh(sphere, sensorMat);
        s.position.set(mx, my, camH);
        group.add(s);
      }

      const R = 5;
      const arc = (sign: number) => {
        const pts: THREE.Vector3[] = [];
        for (let a = 0; a <= 40; a++) {
          const th = (a / 40) * (Math.PI / 2);
          pts.push(new THREE.Vector3(Math.sin(th) * R, sign * (R - Math.cos(th) * R), z));
        }
        return new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
          new THREE.LineBasicMaterial({ color: 0x334155 }));
      };
      group.add(arc(1), arc(-1));
    }

    if (trajectory && trajectory.length > 1) {
      const tp = trajectory.map((p) => new THREE.Vector3(p.x, p.y, 0.06));
      group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(tp),
        new THREE.LineBasicMaterial({ color: 0xfbbf24 })));
    }

    st.scene.add(group);
    return () => {
      st.scene.remove(group);
      group.traverse((o) => {
        const m = o as THREE.Mesh | THREE.Line;
        if (m.geometry) m.geometry.dispose();
      });
    };
  }, [showEgo, trajectory, mode]);

  // cuboids: oriented wireframe boxes with a faint fill for picking, the selected one highlighted white
  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    const group = new THREE.Group();
    const meshes: THREE.Mesh[] = [];
    for (const cub of cuboids || []) {
      const [length, width, height] = cub.dims;
      const geo = new THREE.BoxGeometry(length, width, height);
      const sel = cub.object_3d_id === selectedId;
      const color = sel ? 0xffffff : (STATE_COLOR[cub.state] ?? 0x60a5fa);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo),
        new THREE.LineBasicMaterial({ color }));
      const fill = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: sel ? 0.18 : 0.07, depthWrite: false }));
      fill.userData = { id: cub.object_3d_id };
      const box = new THREE.Group();
      box.add(fill);
      box.add(edges);
      box.position.set(cub.center[0], cub.center[1], cub.center[2]);
      box.rotation.set(cub.roll || 0, cub.pitch || 0, cub.yaw);
      group.add(box);
      meshes.push(fill);
    }
    st.scene.add(group);
    cubeRef.current = { group, meshes };
    return () => {
      st.scene.remove(group);
      group.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
        const mat = m.material as THREE.Material | undefined;
        if (mat && mat.dispose) mat.dispose();
      });
      cubeRef.current = null;
    };
  }, [cuboids, selectedId]);

  // cuboid selection (both views) and drag-to-move on the ground plane (BEV)
  useEffect(() => {
    const st = stateRef.current;
    const mount = mountRef.current;
    if (!st || !mount || (!onSelectCuboid && !onMoveCuboid)) return;
    const ray = new THREE.Raycaster();
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    let dragging: string | null = null;

    const ndc = (e: PointerEvent) => {
      const r = mount.getBoundingClientRect();
      return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
        -((e.clientY - r.top) / r.height) * 2 + 1);
    };
    const pick = (e: PointerEvent): string | null => {
      ray.setFromCamera(ndc(e), st.camera as THREE.Camera);
      const hit = ray.intersectObjects(cubeRef.current?.meshes || [], false)[0];
      return hit ? (hit.object.userData.id as string) : null;
    };
    const groundXY = (e: PointerEvent): [number, number] | null => {
      ray.setFromCamera(ndc(e), st.camera as THREE.Camera);
      const p = new THREE.Vector3();
      return ray.ray.intersectPlane(plane, p) ? [p.x, p.y] : null;
    };
    const onDown = (e: PointerEvent) => {
      const id = pick(e);
      onSelectCuboid?.(id);
      if (id && mode === "bev" && onMoveCuboid) {
        dragging = id;
        st.controls.enabled = false;
      }
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging || !onMoveCuboid) return;
      const xy = groundXY(e);
      if (xy) onMoveCuboid(dragging, xy[0], xy[1], false);
    };
    const onUp = (e: PointerEvent) => {
      if (dragging && onMoveCuboid) {
        const xy = groundXY(e);
        if (xy) onMoveCuboid(dragging, xy[0], xy[1], true);
      }
      dragging = null;
      st.controls.enabled = true;
    };
    mount.addEventListener("pointerdown", onDown);
    mount.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      mount.removeEventListener("pointerdown", onDown);
      mount.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [mode, onSelectCuboid, onMoveCuboid]);

  // measurement: click two points, report the distance and draw a segment
  useEffect(() => {
    const st = stateRef.current;
    const mount = mountRef.current;
    if (!st || !mount || !onMeasure) return;

    const onClick = (e: MouseEvent) => {
      if (!st.cloud) return;
      const rect = mount.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      st.raycaster.setFromCamera(ndc, st.camera as THREE.Camera);
      const hit = st.raycaster.intersectObject(st.cloud)[0];
      if (!hit || hit.index == null) return;
      const attr = st.cloud.geometry.getAttribute("position");
      const p = new THREE.Vector3(attr.getX(hit.index), attr.getY(hit.index), attr.getZ(hit.index));
      st.measure.push(p);
      if (st.measure.length > 2) st.measure = [p];
      if (st.marks) { st.scene.remove(st.marks); st.marks = null; }
      if (st.measure.length === 2) {
        const d = st.measure[0].distanceTo(st.measure[1]);
        const g = new THREE.BufferGeometry().setFromPoints(st.measure);
        const line = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xffd166 }));
        st.scene.add(line);
        st.marks = line;
        onMeasure(d);
      } else {
        onMeasure(null);
      }
    };
    mount.addEventListener("click", onClick);
    return () => mount.removeEventListener("click", onClick);
  }, [onMeasure]);

  return <div ref={mountRef} className="h-full w-full" />;
}
