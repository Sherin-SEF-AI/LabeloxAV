"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type SimilarObject } from "@/lib/api";
import type { ObjectDetail, Ontology } from "@/lib/types";
import { StateBadge, ConfBar } from "@/components/StateBadge";
import BackButton from "@/components/BackButton";
import PageHeaderBar from "@/components/shell/PageHeaderBar";
import Inspector from "@/components/shell/Inspector";

const AnnotationCanvas = dynamic(() => import("@/components/AnnotationCanvas"), { ssr: false });

export default function ObjectPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const [onto, setOnto] = useState<Ontology | null>(null);
  const [obj, setObj] = useState<ObjectDetail | null>(null);
  const [cls, setCls] = useState<string>("");
  const [attrs, setAttrs] = useState<Record<string, unknown>>({});
  const [candidate, setCandidate] = useState<number[][]>([]);
  const [clickPoint, setClickPoint] = useState<number[] | null>(null);
  const [flash, setFlash] = useState(false);
  const [similar, setSimilar] = useState<SimilarObject[] | null>(null);
  const [t0] = useState<number>(() => Date.now());

  useEffect(() => {
    api.ontology().then(setOnto);
  }, []);
  useEffect(() => {
    api.object(params.id).then((o) => {
      setObj(o);
      setCls(o.class_name);
      setAttrs(o.attrs ?? {});
    });
  }, [params.id]);

  const shortlist = useMemo(() => {
    if (!onto || !obj) return [];
    const cur = onto.classes.find((c) => c.name === obj.class_name);
    const sibs = onto.classes.filter((c) => c.l1 === cur?.l1).map((c) => c.name);
    return Array.from(new Set([obj.class_name, ...sibs, "object_fallback"])).slice(0, 9);
  }, [onto, obj]);

  const onPointClick = useCallback(
    async (x: number, y: number) => {
      if (!obj) return;
      setClickPoint([x, y]);
      const res = await api.segment(obj.frame_id, [x, y]);
      setCandidate(res.polygons ?? []);
    },
    [obj],
  );

  const save = useCallback(async () => {
    if (!obj) return;
    const changedClass = cls !== obj.class_name;
    await api.review(obj.object_id, {
      reviewer: "reviewer",
      action: changedClass ? "reclassify" : "confirm",
      class_name: cls,
      attrs,
      state: "accepted",
      time_spent_ms: Date.now() - t0,
    });
    setFlash(true);
    setTimeout(() => router.push("/"), 140);
  }, [obj, cls, attrs, router, t0]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT") return;
      if (e.key === "Enter") save();
      else if (e.key === " ") {
        e.preventDefault();
        router.push("/");
      } else if (/^[1-9]$/.test(e.key)) {
        const idx = parseInt(e.key, 10) - 1;
        if (shortlist[idx]) setCls(shortlist[idx]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [save, shortlist, router]);

  if (!obj || !onto) return <div className="p-6 text-ink-3">loading…</div>;

  return (
    <div className={`min-h-screen flex flex-col ${flash ? "confirm-flash" : ""}`}>
      <PageHeaderBar
        title="Object"
        subtitle={obj.object_id.slice(0, 8)}
        meta={
          <>
            <BackButton />
            <span>{obj.cam_id} · ts {String(obj.ts_ns).slice(0, 13)}</span>
            <span className="text-ink-3">{cls}</span>
            <ConfBar conf={obj.conf} />
            <StateBadge state={obj.state} />
          </>
        }
        primaryAction={
          <div className="flex gap-2">
            <button onClick={save} className="border border-pass text-pass px-3 py-1 font-mono hover:bg-pass/10">
              SAVE ⏎
            </button>
            <button onClick={() => router.push("/")} className="border border-line text-ink-2 px-3 py-1 font-mono hover:text-ink">
              SKIP ␣
            </button>
          </div>
        }
      />

      <div className="flex flex-1 min-h-0">
        <main className="flex-1 min-w-0 p-4 overflow-auto bg-bg-2">
          <AnnotationCanvas
            imageUrl={obj.image_url}
            imgWidth={obj.width}
            imgHeight={obj.height}
            bbox={obj.bbox}
            maskPolygons={obj.mask_polygons}
            candidatePolygons={candidate}
            clickPoint={clickPoint}
            onPointClick={onPointClick}
          />
          <div className="font-mono text-[11px] text-ink-3 mt-2">
            click-to-segment (SAM) · click the object to propose a mask
          </div>
        </main>

        <Inspector title="object" side="right">
          <div className="p-4 space-y-5">
          <section>
            <div className="font-mono text-[11px] text-ink-3 uppercase mb-2">class</div>
            <div className="space-y-1">
              {shortlist.map((name, i) => (
                <button
                  key={name}
                  onClick={() => setCls(name)}
                  className={`block w-full text-left text-sm px-2 py-1 font-mono ${
                    cls === name ? "text-accent border-l-2 border-accent" : "text-ink-2 hover:text-ink"
                  }`}
                >
                  <span className="text-ink-3">[{i + 1}]</span> {name}
                </button>
              ))}
            </div>
            <select
              value={cls}
              onChange={(e) => setCls(e.target.value)}
              className="mt-2 w-full bg-panel border hairline text-ink text-xs px-2 py-1 font-mono"
            >
              {onto.classes.map((c) => (
                <option key={c.id} value={c.name}>
                  {c.name}
                  {c.india ? " *" : ""}
                </option>
              ))}
            </select>
          </section>

          <section>
            <div className="font-mono text-[11px] text-ink-3 uppercase mb-2">attributes</div>
            <div className="space-y-2">
              {Object.entries(onto.attributes).map(([name, spec]) => (
                <AttrControl
                  key={name}
                  name={name}
                  spec={spec}
                  value={attrs[name]}
                  onChange={(v) => setAttrs((a) => ({ ...a, [name]: v }))}
                />
              ))}
            </div>
          </section>

          <section>
            <div className="font-mono text-[11px] text-ink-3 uppercase mb-2">provenance</div>
            <div className="font-mono text-[11px] space-y-1">
              {(obj.provenance?.proposals as { path: string; verdict: string }[] | undefined)?.map(
                (p, i) => (
                  <div key={i} className="flex justify-between">
                    <span className="text-ink-2">{p.path.replace("path_", "")}</span>
                    <span className="text-ink-3">{p.verdict}</span>
                  </div>
                ),
              ) ?? <span className="text-ink-3">none</span>}
            </div>
          </section>

          <section>
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[11px] text-ink-3 uppercase">similar (CLIP)</span>
              <button
                onClick={async () => setSimilar(await api.objectSimilar(obj.object_id).then((r) => r.results))}
                className="font-mono text-[11px] text-ink-3 hover:text-accent"
              >
                find
              </button>
            </div>
            {similar && (
              <div className="grid grid-cols-3 gap-1">
                {similar.map((s) => (
                  <button
                    key={s.object_id}
                    onClick={() => router.push(`/object/${s.object_id}`)}
                    className="relative aspect-square border hairline overflow-hidden"
                    title={`${s.class_name} ${s.score.toFixed(2)}`}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={s.image_url} alt={s.class_name} className="w-full h-full object-cover" />
                    <span className="absolute bottom-0 right-0 bg-bg/80 font-mono text-[9px] px-0.5">
                      {s.score.toFixed(2)}
                    </span>
                  </button>
                ))}
                {!similar.length && <span className="text-ink-3 text-[11px] col-span-3">no embeddings yet (run make embed)</span>}
              </div>
            )}
          </section>
          </div>
        </Inspector>
      </div>
    </div>
  );
}

function AttrControl({
  name,
  spec,
  value,
  onChange,
}: {
  name: string;
  spec: { type: string; values: unknown[] | null };
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const label = <span className="font-mono text-xs text-ink-2">{name}</span>;
  if (spec.type === "enum") {
    return (
      <label className="flex items-center justify-between gap-2">
        {label}
        <select
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value === "" ? undefined : isNaN(Number(e.target.value)) ? e.target.value : Number(e.target.value))}
          className="bg-panel border hairline text-ink text-xs px-1 py-0.5 font-mono"
        >
          <option value="">–</option>
          {spec.values?.map((v) => (
            <option key={String(v)} value={String(v)}>
              {String(v)}
            </option>
          ))}
        </select>
      </label>
    );
  }
  if (spec.type === "bool") {
    return (
      <label className="flex items-center justify-between gap-2">
        {label}
        <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
      </label>
    );
  }
  if (spec.type === "int" || spec.type === "float") {
    return (
      <label className="flex items-center justify-between gap-2">
        {label}
        <input
          type="number"
          value={value === undefined ? "" : Number(value)}
          step={spec.type === "float" ? 0.01 : 1}
          onChange={(e) => onChange(e.target.value === "" ? undefined : Number(e.target.value))}
          className="w-20 bg-panel border hairline text-ink text-xs px-1 py-0.5 font-mono"
        />
      </label>
    );
  }
  return (
    <div className="flex items-center justify-between gap-2">
      {label}
      <span className="font-mono text-[11px] text-ink-3">per-rider</span>
    </div>
  );
}
