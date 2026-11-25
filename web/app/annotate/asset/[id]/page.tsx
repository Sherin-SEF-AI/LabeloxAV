"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api , humanizeError } from "@/lib/api";
import type { AssetDetail } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import { FieldsForm, LabelChips } from "@/components/labelui/LabelPicker";
import TextSpanEditor from "@/components/labelui/TextSpanEditor";
import AudioRegionEditor from "@/components/labelui/AudioRegionEditor";
import TimeSeriesEditor from "@/components/labelui/TimeSeriesEditor";
import DocumentEditor from "@/components/labelui/DocumentEditor";
import PreferenceEditor from "@/components/labelui/PreferenceEditor";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import Icon from "@/components/shell/Icon";
import { humanize, plural, shortId } from "@/lib/format";

// A selected annotation's payload and fields as label/value rows. Numbers are rounded for reading; the
// exact JSON stays one click away.
function readable(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  if (Array.isArray(v)) return v.map(readable).join(", ");
  if (v && typeof v === "object") return Object.entries(v as Record<string, unknown>).map(([k, x]) => `${humanize(k)} ${readable(x)}`).join(" · ");
  return v == null ? "-" : String(v);
}

function SideTitle({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between mb-2">
      <h2 className="section-title">{children}</h2>
      {right}
    </div>
  );
}

// One editor, five modalities. The canvas is chosen by the asset's media_type and everything around it, the
// label picker, the typed fields, the annotation list, is driven by the project's label_config. That is the
// whole point of the config: a new project type is a config change, not a new page.

export default function AssetEditor() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [asset, setAsset] = useState<AssetDetail | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const [fields, setFields] = useState<Record<string, unknown>>({});
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // The first load's failure, kept apart from the transient flash: without it a bad id or a dropped request
  // left "loading..." on screen for good.
  const [loadErr, setLoadErr] = useState<unknown>(null);

  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(null), 4000); };

  const load = useCallback(async () => {
    try { setAsset(await api.asset(id)); setLoadErr(null); } catch (e) { setLoadErr(e); flash(humanizeError(e)); }
  }, [id]);

  useEffect(() => { load(); }, [load]);

  const config = asset?.label_config ?? {};
  const anns = asset?.annotations ?? [];

  // Default to the first label the project declares, so a single-label project needs no click at all.
  useEffect(() => {
    if (!label && config.labels?.length) setLabel(config.labels[0].name);
  }, [config, label]);

  const add = useCallback(async (kind: string, payload: Record<string, unknown>) => {
    setBusy(true);
    try {
      await api.createAnnotation(id, { kind, label, payload, fields });
      await load();
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  }, [id, label, fields, load]);

  const remove = async (annotationId: string) => {
    setBusy(true);
    try {
      await api.deleteAnnotation(annotationId);
      if (selectedId === annotationId) setSelectedId(null);
      await load();
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const markLabeled = async () => {
    setBusy(true);
    try { await api.setAssetState(id, "labeled"); await load(); flash("Marked as labelled"); }
    catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const series = useMemo(() => {
    const raw = (asset?.meta?.series ?? []) as { name: string; values: number[] }[];
    return Array.isArray(raw) ? raw : [];
  }, [asset]);

  const candidates = useMemo(() => {
    const c = asset?.meta?.candidates;
    return Array.isArray(c) ? (c as string[]) : [];
  }, [asset]);

  const duration = Number(asset?.meta?.duration_s ?? 0);

  // What the browser can actually load.
  //
  // These editors pointed straight at `asset.uri`, and every asset here carries a `file://` uri, which a
  // browser refuses to load from an http page: "Not allowed to load local resource". So the whole surface
  // showed nothing. The bytes come through the API now, the way a frame's image always has.
  const mediaUrl = asset ? `/api/assets/${asset.asset_id}/media` : null;

  const canvas = () => {
    if (!asset) {
      return (
        <div className="p-4">
          <LoadState loading={!loadErr} error={loadErr} onRetry={() => void load()} rows={6} />
        </div>
      );
    }
    switch (asset.media_type) {
      case "text":
        return <TextSpanEditor text={asset.text ?? ""} annotations={anns} config={config}
          activeLabel={label} selectedId={selectedId} onSelect={setSelectedId}
          onCreate={(start, end) => add("span", { start, end })}
          onCreateRelation={(from_annotation_id, to_annotation_id) =>
            add("relation", { from_annotation_id, to_annotation_id })} />;
      case "audio":
        return asset.uri
          ? <AudioRegionEditor uri={mediaUrl!} annotations={anns} config={config} activeLabel={label}
              selectedId={selectedId} onSelect={setSelectedId}
              onCreate={(t_start, t_end) => add("region", { t_start, t_end })} />
          : <EmptyState icon="info" title="No audio file">This audio asset has no file attached, so there is nothing to play or label.</EmptyState>;
      case "timeseries":
        return <TimeSeriesEditor series={series} duration={duration || series[0]?.values.length || 1}
          annotations={anns} config={config} activeLabel={label}
          selectedId={selectedId} onSelect={setSelectedId}
          onCreate={(t_start, t_end) => add("region", { t_start, t_end })} />;
      case "document":
      case "image":
        return asset.uri
          ? <DocumentEditor uri={mediaUrl!} annotations={anns} config={config} activeLabel={label}
              selectedId={selectedId} onSelect={setSelectedId}
              onCreate={(bbox) => add("bbox", { bbox })}
              onTranscribe={async (annotationId, bbox) => {
                try {
                  const r = await api.ocrRegion(asset.uri!, bbox);
                  if (!r.text.trim()) { flash("The reader found no text in that region"); return; }
                  await api.updateAnnotation(annotationId, { fields: { text: r.text } });
                  await load();
                  // The unmeasured case is said out loud. The local reader returns no calibrated score, and
                  // a transcription that looks verified when nobody verified it is the failure the OCR
                  // path was already fixed once for.
                  flash(r.measured ? `Read: ${r.text}` : `Read (unscored): ${r.text}`);
                } catch (e) { flash(humanizeError(e)); }
              }} />
          : <EmptyState icon="info" title="No file attached">This asset has no file attached, so there is nothing to show or label.</EmptyState>;
      case "dialogue":
        return <PreferenceEditor candidates={candidates} prompt={asset.text ?? ""} annotations={anns}
          onPreference={(chosen) => add("preference", { candidates, chosen })}
          onRubric={(scores) => add("rubric", { scores })}
          onRanking={(order) => add("ranking", { order })} />;
      default:
        return <EmptyState icon="info" title={`No editor for ${humanize(asset.media_type).toLowerCase()} assets`}>
          This project&apos;s media type has no labelling canvas yet.
        </EmptyState>;
    }
  };

  const sel = anns.find((a) => a.annotation_id === selectedId) ?? null;

  return (
    <PageShell
      active="ANNOTATE"
      title={asset ? `${humanize(asset.media_type)} asset` : "Asset"}
      subtitle={asset?.external_id ?? shortId(id)}
      meta={<span className="text-[12px] text-ink-3">
        {plural(anns.length, "annotation")}{asset?.state ? ` · ${humanize(asset.state)}` : ""}
      </span>}
      primaryAction={
        <button onClick={markLabeled} disabled={busy || !asset}
          className="btn btn-primary">
          {asset?.state === "labeled" ? "Labelled" : "Mark as labelled"}
        </button>
      }
      right={msg ? <span className="text-[12px] text-accent-2" role="status">{msg}</span> : null}
    >
      <div className="flex h-full min-h-0">
        <section className="flex-1 min-w-0 overflow-auto">{canvas()}</section>

        <aside className="w-80 shrink-0 border-l hairline overflow-auto no-scrollbar p-3 space-y-5" aria-label="Labels and annotations">
          <div>
            <SideTitle>Label</SideTitle>
            <LabelChips config={config} value={label} onChange={setLabel} />
          </div>

          {!!(config.fields ?? []).length && (
            <div>
              <SideTitle>Fields</SideTitle>
              <FieldsForm fields={config.fields ?? []} value={fields} onChange={setFields} />
              <div className="text-[12px] text-ink-3 mt-1.5">
                Applied to the next annotation you create.
              </div>
            </div>
          )}

          <div>
            <SideTitle right={<span className="text-[12px] text-ink-3 tabular-nums">{anns.length}</span>}>Annotations</SideTitle>
            <div className="space-y-0.5">
              {anns.map((a) => (
                // Two buttons side by side, not a clickable row with a button inside it: select, and remove.
                <div key={a.annotation_id}
                  className={`group flex items-center gap-2 px-2 py-1.5 rounded text-[12.5px] ${
                    a.annotation_id === selectedId ? "bg-accent/10 text-ink ring-1 ring-accent/40" : "text-ink-2 hover:bg-head/60 hover:text-ink"}`}>
                  <button type="button" aria-pressed={a.annotation_id === selectedId}
                    onClick={() => setSelectedId(a.annotation_id === selectedId ? null : a.annotation_id)}
                    className="flex items-center gap-2 flex-1 min-w-0 text-left">
                    <span className="text-[11px] text-ink-3 px-1.5 py-px rounded border border-line shrink-0">{humanize(a.kind)}</span>
                    <span className="truncate flex-1">{a.label ?? <span className="text-ink-3">No label</span>}</span>
                  </button>
                  <button type="button" onClick={() => remove(a.annotation_id)}
                    aria-label={`remove ${a.kind}${a.label ? ` ${a.label}` : ""}`}
                    data-tip="Delete this annotation"
                    disabled={busy}
                    className="text-ink-3 hover:text-block p-0.5 rounded shrink-0 disabled:opacity-40"><Icon name="x" size={13} /></button>
                </div>
              ))}
              {!anns.length && (
                <div className="px-2 py-2 text-[12px] text-ink-3 leading-relaxed">
                  No annotations yet. Pick a label above, then mark up the asset on the left.
                </div>
              )}
            </div>
          </div>

          {sel && (
            <div>
              <SideTitle>Selected</SideTitle>
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-[12px]">
                {Object.entries({ ...(sel.payload ?? {}), ...(sel.fields ?? {}) }).map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-ink-3">{humanize(k)}</dt>
                    <dd className="text-ink-2 break-words tabular-nums">{readable(v)}</dd>
                  </div>
                ))}
              </dl>
              <details className="mt-2 text-[12px]">
                <summary className="cursor-pointer text-ink-3 hover:text-ink-2">Raw JSON</summary>
                <pre className="mt-1.5 p-2 rounded bg-bg-2 border hairline text-[11.5px] text-ink-2 font-mono whitespace-pre-wrap break-all">
                  {JSON.stringify({ payload: sel.payload, fields: sel.fields }, null, 1)}
                </pre>
              </details>
            </div>
          )}

          <button onClick={() => router.push("/projects")}
            className="btn w-full">
            <Icon name="chevL" size={14} /> Back to projects
          </button>
        </aside>
      </div>
    </PageShell>
  );
}
