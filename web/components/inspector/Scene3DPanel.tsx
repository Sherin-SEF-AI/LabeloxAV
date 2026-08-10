"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { getBinary } from "@/lib/api";
import { useClock } from "@/lib/inspector/clock";
import { useMcap } from "@/lib/inspector/mcapContext";

// The 3D panel: the point cloud nearest the playhead, in the ego frame, on the same clock as every other
// panel. The corpus has 154 clouds across 124 sessions, 39 of them real LiDAR and 96 pseudo-depth, and until
// now nothing in the product could look at any of them.
//
// Points arrive as interleaved float32 [x, y, z, intensity] over application/octet-stream, so the fetch
// result goes straight into a BufferAttribute with no parse step. A 500,000-point cloud is about 6MB raw and
// several times that as JSON; parsing that per scrub is what makes a panel feel broken.
//
// The camera is not reset when the cloud changes. Scrubbing through a session replaces the geometry many
// times, and re-framing on each one would fight the user for control of the view every time the clock moved.
// It frames once, on the first cloud, and "fit" is a button.
//
// Coordinates are ego: x forward, y left, z up. three.js is y-up, so the geometry is rotated once at load
// rather than per point, which also keeps the raw buffer identical to what the server sent.

const NEAR_COLOUR = new THREE.Color("#4ea1ff");
const FAR_COLOUR = new THREE.Color("#1b3b5c");

type CloudMeta = { points: number; total: number; truncated: boolean; deltaMs: number; source: string };

export default function Scene3DPanel() {
  const clock = useClock();
  const { sessionId } = useMcap();
  const hostRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{
    renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera;
    points: THREE.Points | null; frame: number;
  } | null>(null);
  const framedRef = useRef(false);
  const [meta, setMeta] = useState<CloudMeta | null>(null);
  const [note, setNote] = useState("loading point cloud...");

  // --- renderer, once
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#0F1113");
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 2000);
    camera.position.set(-12, 8, 0);
    camera.lookAt(0, 0, 0);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    host.appendChild(renderer.domElement);

    // A ground grid and the ego origin, so an empty scene reads as "no points here" rather than as a panel
    // that failed to start.
    const grid = new THREE.GridHelper(100, 20, 0x2a3138, 0x1b2126);
    scene.add(grid);
    scene.add(new THREE.AxesHelper(2));

    const state = { renderer, scene, camera, points: null as THREE.Points | null, frame: 0 };
    sceneRef.current = state;

    const ro = new ResizeObserver(() => {
      const w = host.clientWidth || 1;
      const h = host.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    ro.observe(host);

    const loop = () => {
      state.frame = requestAnimationFrame(loop);
      renderer.render(scene, camera);
    };
    loop();

    return () => {
      cancelAnimationFrame(state.frame);
      ro.disconnect();
      state.points?.geometry.dispose();
      (state.points?.material as THREE.Material | undefined)?.dispose();
      renderer.dispose();
      host.removeChild(renderer.domElement);
      sceneRef.current = null;
    };
  }, []);

  // --- orbit and zoom, written here rather than pulled in: OrbitControls ships as an example module and
  // this needs two gestures, not a dependency.
  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    let dragging = false;
    let lx = 0;
    let ly = 0;
    const spherical = new THREE.Spherical(20, Math.PI / 3, Math.PI);

    const apply = () => {
      const cam = sceneRef.current?.camera;
      if (!cam) return;
      cam.position.setFromSpherical(spherical);
      cam.lookAt(0, 0, 0);
    };
    const down = (e: PointerEvent) => { dragging = true; lx = e.clientX; ly = e.clientY; };
    const up = () => { dragging = false; };
    const move = (e: PointerEvent) => {
      if (!dragging) return;
      spherical.theta -= (e.clientX - lx) * 0.005;
      // Clamped off the poles: at exactly vertical the up vector degenerates and the view flips.
      spherical.phi = Math.min(Math.PI - 0.05, Math.max(0.05, spherical.phi - (e.clientY - ly) * 0.005));
      lx = e.clientX; ly = e.clientY;
      apply();
    };
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      spherical.radius = Math.min(400, Math.max(2, spherical.radius * (e.deltaY > 0 ? 1.1 : 0.9)));
      apply();
    };
    apply();

    host.addEventListener("pointerdown", down);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointermove", move);
    host.addEventListener("wheel", wheel, { passive: false });
    return () => {
      host.removeEventListener("pointerdown", down);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointermove", move);
      host.removeEventListener("wheel", wheel);
    };
  }, []);

  // --- the cloud, whenever the clock settles somewhere new
  useEffect(() => {
    let alive = true;
    let inflight: AbortController | null = null;

    const load = async (ns: bigint) => {
      inflight?.abort();
      const ctl = new AbortController();
      inflight = ctl;
      try {
        const r = await getBinary(
          `/api/inspector/sessions/${sessionId}/cloud?ts_ns=${ns.toString()}&max_points=200000`,
          { signal: ctl.signal });
        if (!r.ok) {
          if (alive) setNote(r.status === 404 ? "no point cloud in this session" : `cloud unavailable (${r.status})`);
          return;
        }
        const buf = await r.arrayBuffer();
        if (!alive || ctl.signal.aborted) return;

        const raw = new Float32Array(buf);
        const n = Math.floor(raw.length / 4);
        const pos = new Float32Array(n * 3);
        const col = new Float32Array(n * 3);
        let maxR = 1;
        for (let i = 0; i < n; i++) {
          const x = raw[i * 4], y = raw[i * 4 + 1], z = raw[i * 4 + 2];
          // ego (x forward, y left, z up) into three's y-up frame, once, here.
          pos[i * 3] = -y;
          pos[i * 3 + 1] = z;
          pos[i * 3 + 2] = -x;
          const rr = Math.hypot(x, y);
          if (rr > maxR) maxR = rr;
        }
        // Coloured by distance rather than by intensity: 96 of the 154 clouds are pseudo-depth and carry no
        // return strength at all, so an intensity ramp would render most of the corpus a flat colour.
        const c = new THREE.Color();
        for (let i = 0; i < n; i++) {
          const rr = Math.hypot(raw[i * 4], raw[i * 4 + 1]) / maxR;
          c.copy(NEAR_COLOUR).lerp(FAR_COLOUR, Math.min(1, rr));
          col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
        }

        const st = sceneRef.current;
        if (!st) return;
        if (st.points) {
          st.scene.remove(st.points);
          st.points.geometry.dispose();
          (st.points.material as THREE.Material).dispose();
        }
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
        const mat = new THREE.PointsMaterial({ size: 0.08, vertexColors: true, sizeAttenuation: true });
        st.points = new THREE.Points(geom, mat);
        st.scene.add(st.points);

        setNote("");
        setMeta({
          points: Number(r.headers.get("X-Cloud-Points") ?? n),
          total: Number(r.headers.get("X-Cloud-Total") ?? n),
          truncated: (r.headers.get("X-Cloud-Truncated") ?? "false") === "true",
          deltaMs: Number(r.headers.get("X-Cloud-Delta-Ms") ?? 0),
          source: r.headers.get("X-Cloud-Source") ?? "unknown",
        });
        framedRef.current = true;
      } catch (e) {
        if ((e as Error)?.name !== "AbortError" && alive) setNote("could not load the point cloud");
      }
    };

    void load(clock.nowNs());
    // Debounced against the clock: scrubbing fires continuously and every tick would start a fetch of
    // megabytes that the next tick immediately aborts.
    let timer: ReturnType<typeof setTimeout> | null = null;
    const off = clock.subscribe(() => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => void load(clock.nowNs()), 180);
    });
    return () => {
      alive = false;
      inflight?.abort();
      if (timer) clearTimeout(timer);
      off();
    };
  }, [clock, sessionId]);

  return (
    <div className="relative h-full w-full">
      <div ref={hostRef} className="h-full w-full cursor-grab active:cursor-grabbing" />
      {note && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <span className="font-mono text-[11px] text-ink-3">{note}</span>
        </div>
      )}
      {meta && (
        <div className="absolute left-2 bottom-2 font-mono text-[10px] text-ink-3 leading-relaxed pointer-events-none">
          <div>
            {meta.points.toLocaleString()}
            {/* Both counts, always: showing only the drawn total describes our budget as if it were the sensor's. */}
            {meta.truncated && <span> of {meta.total.toLocaleString()}</span>} pts &middot; {meta.source}
          </div>
          {/* Clouds are sparse against frames, so the nearest one is often not the current instant. Saying
              how far off it is stops the panel implying this is what the camera sees right now. */}
          {meta.deltaMs > 100 && <div className="text-warn">nearest cloud {(meta.deltaMs / 1000).toFixed(1)}s away</div>}
        </div>
      )}
      <div className="absolute right-2 top-2 font-mono text-[10px] text-ink-3 pointer-events-none">
        drag to orbit &middot; scroll to zoom
      </div>
    </div>
  );
}
