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

  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(null), 4000); };

  const load = useCallback(async () => {
    try { setAsset(await api.asset(id)); } catch (e) { flash(humanizeError(e)); }
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
    try { await api.setAssetState(id, "labeled"); await load(); flash("marked labeled"); }
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

  const canvas = () => {
    if (!asset) return <div className="p-4 font-mono text-[11px] text-ink-3">loading...</div>;
    switch (asset.media_type) {
      case "text":
        return <TextSpanEditor text={asset.text ?? ""} annotations={anns} config={config}
          activeLabel={label} selectedId={selectedId} onSelect={setSelectedId}
          onCreate={(start, end) => add("span", { start, end })} />;
      case "audio":
        return asset.uri
          ? <AudioRegionEditor uri={asset.uri} annotations={anns} config={config} activeLabel={label}
              selectedId={selectedId} onSelect={setSelectedId}
              onCreate={(t_start, t_end) => add("region", { t_start, t_end })} />
          : <div className="p-4 font-mono text-[11px] text-ink-3">audio asset has no uri</div>;
      case "timeseries":
        return <TimeSeriesEditor series={series} duration={duration || series[0]?.values.length || 1}
          annotations={anns} config={config} activeLabel={label}
          selectedId={selectedId} onSelect={setSelectedId}
          onCreate={(t_start, t_end) => add("region", { t_start, t_end })} />;
      case "document":
      case "image":
        return asset.uri
          ? <DocumentEditor uri={asset.uri} annotations={anns} config={config} activeLabel={label}
              selectedId={selectedId} onSelect={setSelectedId}
              onCreate={(bbox) => add("bbox", { bbox })} />
          : <div className="p-4 font-mono text-[11px] text-ink-3">asset has no uri</div>;
      case "dialogue":
        return <PreferenceEditor candidates={candidates} prompt={asset.text ?? ""} annotations={anns}
          onPreference={(chosen) => add("preference", { candidates, chosen })}
          onRubric={(scores) => add("rubric", { scores })}
          onRanking={(order) => add("ranking", { order })} />;
      default:
        return <div className="p-4 font-mono text-[11px] text-ink-3">
          no editor for media type {asset.media_type}
        </div>;
    }
  };

  const sel = anns.find((a) => a.annotation_id === selectedId) ?? null;

  return (
    <PageShell
      active="ANNOTATE"
      title={asset ? `${asset.media_type} asset` : "asset"}
      subtitle={asset?.external_id ?? id.slice(0, 8)}
      meta={<span className="font-mono text-[11px] text-ink-3">
        {anns.length} annotations - {asset?.state ?? "-"}
      </span>}
      primaryAction={
        <button onClick={markLabeled} disabled={busy || !asset}
          className="h-[30px] px-3 rounded-md bg-accent text-bg font-display font-semibold text-[12.5px] disabled:opacity-40">
          Mark labeled
        </button>
      }
      right={msg ? <span className="font-mono text-[11px] text-accent">{msg}</span> : null}
    >
      <div className="flex h-full min-h-0">
        <section className="flex-1 min-w-0 overflow-auto">{canvas()}</section>

        <aside className="w-72 shrink-0 border-l hairline overflow-auto no-scrollbar p-2 space-y-3">
          <div>
            <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">label</div>
            <LabelChips config={config} value={label} onChange={setLabel} />
          </div>

          {!!(config.fields ?? []).length && (
            <div>
              <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">fields</div>
              <FieldsForm fields={config.fields ?? []} value={fields} onChange={setFields} />
              <div className="font-mono text-[10px] text-ink-3 mt-1">
                applied to the next annotation you create
              </div>
            </div>
          )}

          <div>
            <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">
              annotations ({anns.length})
            </div>
            <div className="space-y-0.5">
              {anns.map((a) => (
                <div key={a.annotation_id}
                  onClick={() => setSelectedId(a.annotation_id === selectedId ? null : a.annotation_id)}
                  className={`flex items-center gap-1.5 px-1.5 py-0.5 font-mono text-[11px] cursor-pointer ${
                    a.annotation_id === selectedId ? "bg-bg-2 text-ink" : "text-ink-3 hover:text-ink-2"}`}>
                  <span className="text-ink-2">{a.kind}</span>
                  <span className="truncate flex-1">{a.label ?? ""}</span>
                  <button onClick={(e) => { e.stopPropagation(); remove(a.annotation_id); }}
                    className="text-ink-3 hover:text-block">x</button>
                </div>
              ))}
              {!anns.length && <div className="px-1.5 text-ink-3">none yet</div>}
            </div>
          </div>

          {sel && (
            <div>
              <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">selected</div>
              <pre className="font-mono text-[10px] text-ink-3 whitespace-pre-wrap break-all">
                {JSON.stringify({ payload: sel.payload, fields: sel.fields }, null, 1)}
              </pre>
            </div>
          )}

          <button onClick={() => router.push("/projects")}
            className="w-full border border-line px-2 py-1 font-mono text-[11px] text-ink-3 hover:border-accent">
            back to projects
          </button>
        </aside>
      </div>
    </PageShell>
  );
}
