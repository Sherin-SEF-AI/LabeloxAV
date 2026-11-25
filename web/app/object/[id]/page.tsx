"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useSmartBack } from "@/lib/nav";
import { api, humanizeError, type SimilarObject } from "@/lib/api";
import type { ObjectDetail, Ontology } from "@/lib/types";
import { StateBadge, ConfBar } from "@/components/StateBadge";
import BackButton from "@/components/BackButton";
import PageHeaderBar from "@/components/shell/PageHeaderBar";
import Inspector from "@/components/shell/Inspector";
import LoadState from "@/components/shell/LoadState";
import { fromNs, humanize, shortId } from "@/lib/format";

// Ontology class names read as words ("two wheeler") but keep their case: "suv" is not "Suv".
const classLabel = (n: string | null | undefined) => (n ?? "").replace(/_/g, " ");

const AnnotationCanvas = dynamic(() => import("@/components/AnnotationCanvas"), { ssr: false });

export default function ObjectPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const goBack = useSmartBack();
  const [onto, setOnto] = useState<Ontology | null>(null);
  const [obj, setObj] = useState<ObjectDetail | null>(null);
  const [cls, setCls] = useState<string>("");
  const [attrs, setAttrs] = useState<Record<string, unknown>>({});
  const [candidate, setCandidate] = useState<number[][]>([]);
  const [clickPoint, setClickPoint] = useState<number[] | null>(null);
  const [flash, setFlash] = useState(false);
  const [similar, setSimilar] = useState<SimilarObject[] | null>(null);
  const [t0] = useState<number>(() => Date.now());
  const [loadErr, setLoadErr] = useState<unknown>(null);
  const [similarBusy, setSimilarBusy] = useState(false);
  const [similarErr, setSimilarErr] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    api.ontology().then(setOnto).catch(setLoadErr);
  }, [reload]);
  useEffect(() => {
    api.object(params.id).then((o) => {
      setObj(o);
      setCls(o.class_name);
      setAttrs(o.attrs ?? {});
    }).catch(setLoadErr);
  }, [params.id, reload]);

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
    setTimeout(goBack, 140);
  }, [obj, cls, attrs, router, t0]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT") return;
      if (e.key === "Enter") save();
      else if (e.key === " ") {
        e.preventDefault();
        goBack();
      } else if (/^[1-9]$/.test(e.key)) {
        const idx = parseInt(e.key, 10) - 1;
        if (shortlist[idx]) setCls(shortlist[idx]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [save, shortlist, router]);

  if (!obj || !onto) {
    return (
      <div className="min-h-screen flex flex-col">
        <PageHeaderBar title="Object" subtitle={shortId(params.id)} meta={<BackButton label="Back" />} />
        <div className="page-wrap-narrow">
          <LoadState loading={!loadErr} error={loadErr} rows={4}
            onRetry={() => { setLoadErr(null); setReload((n) => n + 1); }} />
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen flex flex-col ${flash ? "confirm-flash" : ""}`}>
      <PageHeaderBar
        title="Object"
        subtitle={shortId(obj.object_id)}
        meta={
          <>
            <BackButton label="Back" />
            <span title={`ts_ns ${obj.ts_ns}`}><span className="font-mono">{obj.cam_id}</span> · {fromNs(obj.ts_ns)}</span>
            <span className="text-ink-2">{classLabel(cls)}</span>
            <ConfBar conf={obj.conf} />
            <StateBadge state={obj.state} />
          </>
        }
        primaryAction={
          <div className="flex gap-2">
            <button type="button" onClick={goBack} className="btn" title="skip without saving (Space)">
              Skip <kbd className="text-ink-3 font-normal">Space</kbd>
            </button>
            <button type="button" onClick={save} className="btn btn-primary" title="accept with this class and these attributes (Enter)">
              Save <kbd className="opacity-80 font-normal">Enter</kbd>
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
          <div className="text-[12px] text-ink-3 mt-2">
            Click the object to propose a mask (SAM click-to-segment).
          </div>
        </main>

        <Inspector title="Object" side="right">
          <div className="p-4 space-y-5">
          <section>
            <h2 className="section-title mb-2">Class</h2>
            <div className="space-y-1">
              {shortlist.map((name, i) => (
                <button
                  key={name}
                  onClick={() => setCls(name)}
                  type="button"
                  aria-pressed={cls === name}
                  className={`flex items-center gap-2 w-full text-left text-[13px] px-2 py-1 rounded ${
                    cls === name ? "text-accent bg-accent/10 border-l-2 border-accent" : "text-ink-2 hover:text-ink hover:bg-head"
                  }`}
                >
                  <kbd className="w-4 text-center text-[11px] text-ink-3 font-mono">{i + 1}</kbd> {classLabel(name)}
                </button>
              ))}
            </div>
            <select
              value={cls}
              onChange={(e) => setCls(e.target.value)}
              aria-label="any class"
              className="field mt-2 w-full"
            >
              {onto.classes.map((c) => (
                <option key={c.id} value={c.name}>
                  {classLabel(c.name)}
                  {c.india ? " *" : ""}
                </option>
              ))}
            </select>
          </section>

          <section>
            <h2 className="section-title mb-2">Attributes</h2>
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

          {/* Sign typing, shown only where it applies. These columns have existed since migration 0016 and
              nothing served them, so a wrong sign_type was invisible to the only person who could catch it.
              A sign the classifier examined and declined to type reads as declined, not as blank, because
              "no type" and "not looked at" are different facts. */}
          {(obj.sign_type || obj.ocr_text || obj.class_name === "traffic_sign") && (
            <section>
              <h2 className="section-title mb-2">Sign</h2>
              <div className="text-[12.5px] space-y-1">
                <div className="flex justify-between">
                  <span className="text-ink-2">Type</span>
                  {obj.sign_type
                    ? <span className="text-ink">{humanize(obj.sign_type)}</span>
                    : <span className="text-ink-3">
                        {(obj.provenance?.sign as { rejected?: boolean } | undefined)?.rejected
                          ? "Declined" : "Not typed"}
                      </span>}
                </div>
                {obj.sign_category && (
                  <div className="flex justify-between">
                    <span className="text-ink-2">Category</span><span className="text-ink">{humanize(obj.sign_category)}</span>
                  </div>
                )}
                {(obj.provenance?.sign as { margin?: number } | undefined)?.margin != null && (
                  <div className="flex justify-between">
                    <span className="text-ink-2">Margin over &quot;not a sign&quot;</span>
                    <span className="text-ink tabular-nums">
                      {((obj.provenance!.sign as { margin: number }).margin).toFixed(3)}
                    </span>
                  </div>
                )}
                {(obj.provenance?.sign as { reason?: string } | undefined)?.reason && (
                  <div className="text-ink-3 pt-1">
                    {(obj.provenance!.sign as { reason: string }).reason}
                  </div>
                )}
                {obj.ocr_text && (
                  <div className="pt-1 border-t hairline mt-1">
                    <div className="text-ink-2">Text read {obj.ocr_lang ? `(${obj.ocr_lang})` : ""}</div>
                    <div className="text-ink">{obj.ocr_text}</div>
                    {/* Null conf is unmeasured, not zero. The local VLM returns no calibrated score, and
                        showing 0.00 would read as a bad result rather than an absent one. */}
                    <div className="text-ink-3">
                      {obj.ocr_conf == null ? "Confidence unmeasured, needs verification"
                        : `Confidence ${obj.ocr_conf.toFixed(2)}`}
                    </div>
                  </div>
                )}
              </div>
            </section>
          )}

          <section>
            <h2 className="section-title mb-2">Where it came from</h2>
            <div className="text-[12.5px] space-y-1">
              {(obj.provenance?.proposals as { path: string; verdict: string }[] | undefined)?.map(
                (p, i) => (
                  <div key={i} className="flex justify-between">
                    <span className="text-ink-2">Path {p.path.replace("path_", "").toUpperCase()}</span>
                    <span className="text-ink-3">{humanize(p.verdict)}</span>
                  </div>
                ),
              ) ?? <span className="text-ink-3">No model proposals recorded for this object.</span>}
            </div>
          </section>

          <section>
            <div className="flex items-center justify-between mb-2">
              <h2 className="section-title">Similar objects</h2>
              <button
                type="button"
                disabled={similarBusy}
                title="find objects that look like this one (CLIP embeddings)"
                onClick={async () => {
                  setSimilarBusy(true);
                  setSimilarErr(null);
                  try { setSimilar(await api.objectSimilar(obj.object_id).then((r) => r.results)); }
                  catch (e) { setSimilar(null); setSimilarErr(humanizeError(e)); }
                  finally { setSimilarBusy(false); }
                }}
                className="btn"
              >
                {similarBusy ? "Searching..." : "Find"}
              </button>
            </div>
            {similarErr && <div role="alert" className="text-[12px] text-block">Could not search: {similarErr}</div>}
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
                    <img src={s.image_url} alt={s.class_name} className="w-full h-full object-cover bg-bg-2" />
                    <span className="absolute bottom-0 right-0 bg-bg/80 text-[11px] px-1 tabular-nums">
                      {s.score.toFixed(2)}
                    </span>
                  </button>
                ))}
                {!similar.length && (
                  <div className="col-span-3 text-[12px] text-ink-3 space-y-1.5">
                    <p>No similar objects found. Object embeddings have not been computed for this corpus yet.</p>
                    <details>
                      <summary className="cursor-pointer hover:text-ink-2">For administrators</summary>
                      <p className="mt-1">Compute them with <code className="font-mono text-ink-2">make embed</code> on the server.</p>
                    </details>
                  </div>
                )}
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
  const label = <span className="text-[12.5px] text-ink-2" title={name}>{humanize(name)}</span>;
  if (spec.type === "enum") {
    return (
      <label className="flex items-center justify-between gap-2">
        {label}
        <select
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value === "" ? undefined : isNaN(Number(e.target.value)) ? e.target.value : Number(e.target.value))}
          className="field py-1 max-w-[9.5rem]"
        >
          <option value="">–</option>
          {spec.values?.map((v) => (
            <option key={String(v)} value={String(v)}>
              {typeof v === "string" ? humanize(v) : String(v)}
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
          className="field py-1 w-20 tabular-nums"
        />
      </label>
    );
  }
  return (
    <div className="flex items-center justify-between gap-2">
      {label}
      <span className="text-[12px] text-ink-3">Set per rider</span>
    </div>
  );
}
