"use client";

// The reconstruction as coloured points in a free 3D view: splat centres (or map points) in their own colour,
// the cameras that took the window as a line, and the cuboid being placed as a wireframe. Points arrive
// relative to `anchor` so float32 stays exact; everything else is passed in world coordinates and shifted here.

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { CUBOID_EDGES, cuboidCorners } from "@/lib/tools/sensorsGeom";

export type DraftBox = { center: number[]; dims: number[]; yaw: number };

export default function ReconCloud({ points, count, anchor, cameras, draft, pointSize = 0.08 }: {
  points: Float32Array; count: number; anchor: number[]; cameras: number[][]; draft: DraftBox | null;
  pointSize?: number;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{ scene: THREE.Scene; box: THREE.LineSegments | null } | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const w = mount.clientWidth || 800, h = mount.clientHeight || 480;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    renderer.setClearColor(0x0a0e14, 1);
    mount.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 5000);
    camera.up.set(0, 0, 1);

    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = points[i * 6]; pos[i * 3 + 1] = points[i * 6 + 1]; pos[i * 3 + 2] = points[i * 6 + 2];
      col[i * 3] = points[i * 6 + 3]; col[i * 3 + 1] = points[i * 6 + 4]; col[i * 3 + 2] = points[i * 6 + 5];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const cloud = new THREE.Points(geo, new THREE.PointsMaterial({ size: pointSize, vertexColors: true }));
    scene.add(cloud);

    if (cameras.length > 1) {
      const tr = new THREE.BufferGeometry().setFromPoints(
        cameras.map((c) => new THREE.Vector3(c[0] - anchor[0], c[1] - anchor[1], c[2] - anchor[2])));
      scene.add(new THREE.Line(tr, new THREE.LineBasicMaterial({ color: 0x38bdf8 })));
    }

    // Start behind and above the first camera, looking down the drive.
    const first = cameras[0] ?? [anchor[0], anchor[1], anchor[2]];
    const last = cameras[cameras.length - 1] ?? first;
    const dir = new THREE.Vector3(last[0] - first[0], last[1] - first[1], 0);
    if (dir.length() < 1) dir.set(1, 0, 0);
    dir.normalize();
    const start = new THREE.Vector3(first[0] - anchor[0], first[1] - anchor[1], first[2] - anchor[2]);
    camera.position.copy(start.clone().addScaledVector(dir, -12).add(new THREE.Vector3(0, 0, 8)));
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(start.clone().addScaledVector(dir, 15));
    controls.update();

    sceneRef.current = { scene, box: null };
    let raf = 0;
    const loop = () => { raf = requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); };
    loop();
    const ro = new ResizeObserver(() => {
      const ww = mount.clientWidth || w, hh = mount.clientHeight || h;
      renderer.setSize(ww, hh);
      camera.aspect = ww / hh;
      camera.updateProjectionMatrix();
    });
    ro.observe(mount);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      geo.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, [points, count, anchor, cameras, pointSize]);

  // The draft cuboid, redrawn without rebuilding the scene.
  useEffect(() => {
    const s = sceneRef.current;
    if (!s) return;
    if (s.box) { s.scene.remove(s.box); s.box.geometry.dispose(); s.box = null; }
    if (!draft) return;
    const c = cuboidCorners(draft.center, draft.dims, draft.yaw)
      .map((p) => new THREE.Vector3(p[0] - anchor[0], p[1] - anchor[1], p[2] - anchor[2]));
    const verts: THREE.Vector3[] = [];
    for (const [a, b] of CUBOID_EDGES) verts.push(c[a], c[b]);
    const g = new THREE.BufferGeometry().setFromPoints(verts);
    s.box = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xfbbf24 }));
    s.scene.add(s.box);
  }, [draft, anchor, points]);

  return <div ref={mountRef} className="w-full h-full min-h-[360px] rounded overflow-hidden" />;
}
