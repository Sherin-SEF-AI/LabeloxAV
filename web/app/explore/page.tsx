"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { T } from "@/lib/i18n";
import { useRouter } from "next/navigation";
import { api , humanizeError } from "@/lib/api";
import type {
  ExplorePredicate,
  Facets,
  ProjectionPoint,
  ProjectionRow,
  SavedView,
} from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import EmbeddingMap, { type ColorBy } from "@/components/explore/EmbeddingMap";
import FacetRail from "@/components/explore/FacetRail";
import { useQueryParam } from "@/lib/useQueryParam";
import { invalidate, useQuery } from "@/lib/useQuery";
import EmptyState from "@/components/ui/EmptyState";
import { humanize, num, shortId } from "@/lib/format";

// The Explore workspace: see the corpus as structure, cut it by facets, lasso what looks alike, and act on
// the selection in bulk. The filter you build here IS a saved view and an export spec, because it is the same
// predicate a CurationSlice stores.

const COLOR_MODES: { key: ColorBy; label: string }[] = [
  { key: "cluster", label: "Cluster" },
  { key: "class", label: "Class" },
  { key: "state", label: "State" },
  { key: "source", label: "Source" },
  { key: "conf", label: "Confidence" },
  { key: "tag", label: "Tag" },
];

// A saved view's predicate as a sentence fragment for its tooltip, rather than the raw JSON it is stored as.
function describePredicate(p: ExplorePredicate | null | undefined): string {
  if (!p) return "Everything";
  const parts: string[] = [];
  for (const [k, v] of Object.entries(p)) {
    if (v == null) continue;
    if (k === "object_ids" && Array.isArray(v)) { parts.push(`${v.length} lassoed objects`); continue; }
    if (k === "min_conf" || k === "max_conf") continue;
    parts.push(`${humanize(k)}: ${Array.isArray(v) ? v.join(", ") : String(v)}`);
  }
  if (p.min_conf != null || p.max_conf != null) parts.push(`Confidence ${p.min_conf ?? 0}-${p.max_conf ?? 1}`);
  return parts.length ? parts.join(" · ") : "Everything";
}

export default function ExplorePage() {
  // useSearchParams (via useQueryParam) forces a CSR bailout Next requires be under Suspense.
  return <Suspense fallback={null}><ExploreBody /></Suspense>;
}

function ExploreBody() {
  const router = useRouter();
  const [predicate, setPredicate] = useState<ExplorePredicate>({});
  // Facets recompute whenever the predicate changes: the counts must always describe the current cut.
  // Keyed by the predicate, so going back to a cut already seen shows its counts at once; the previous
  // cut's counts stay up while the next ones load, and a failure reads as no facets, as it did.
  const facetsQ = useQuery(["exploreFacets", predicate], () => api.exploreFacets(predicate),
    { keepPrevious: true });
  const facets: Facets | null = facetsQ.error ? null : facetsQ.data ?? null;
  const projQ = useQuery(["exploreProjections"], () => api.exploreProjections());
  const projections: ProjectionRow[] = useMemo(() => projQ.data?.projections ?? [], [projQ.data]);
  const [projId, setProjId] = useState<string>("");
  const pointsQ = useQuery(projId ? ["exploreProjectionPoints", projId] : null,
    () => api.exploreProjectionPoints(projId), { keepPrevious: true });
  const points: ProjectionPoint[] = useMemo(
    () => (projId && !pointsQ.error ? pointsQ.data?.points ?? [] : []),
    [projId, pointsQ.data, pointsQ.error]);
  const [colorBy, setColorBy] = useState<ColorBy>("cluster");
  const [tagFilter, setTagFilter] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [hover, setHover] = useState<ProjectionPoint | null>(null);
  const viewsQ = useQuery(["exploreViews"], () => api.exploreViews());
  const views: SavedView[] = viewsQ.data?.views ?? [];
  // Edit > Saved views deep-links here; highlight the rail so the menu entry lands somewhere visible
  const focusViews = useQueryParam("panel") === "views";
  const [tagText, setTagText] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(null), 3500); };

  // Open on the first fitted projection once the list arrives, unless one is already chosen.
  const firstProjId = projections[0]?.projection_id;
  useEffect(() => {
    if (firstProjId) setProjId((cur) => cur || firstProjId);
  }, [firstProjId]);

  // A lasso belongs to the layout it was drawn on.
  useEffect(() => { setSelected(new Set()); }, [projId]);

  const activeProj = useMemo(
    () => projections.find((p) => p.projection_id === projId) ?? null, [projections, projId]);

  // The map shows the fitted layout, filtered by the same predicate the facets describe, so the picture and
  // the counts are one view of one thing rather than two disconnected panels. This mirrors the server-side
  // predicate for the clauses a point carries; the authoritative counts still come from the facets endpoint.
  const matchesPredicate = useCallback((p: ProjectionPoint) => {
    if (predicate.states?.length && !predicate.states.includes(p.state ?? "")) return false;
    if (predicate.sources?.length && !predicate.sources.includes(p.source ?? "")) return false;
    if (predicate.class_names?.length && facets) {
      const ids = facets.classes.filter((c) => predicate.class_names!.includes(c.value)).map((c) => c.class_id);
      if (ids.length && !ids.includes(p.class_id)) return false;
    }
    if (predicate.tags?.length && !(p.tags ?? []).some((t) => predicate.tags!.includes(t))) return false;
    if (predicate.min_conf != null && (p.conf ?? 0) < predicate.min_conf) return false;
    if (predicate.max_conf != null && (p.conf ?? 0) > predicate.max_conf) return false;
    return true;
  }, [predicate, facets]);

  const shownPoints = useMemo(() => points.filter(matchesPredicate), [points, matchesPredicate]);

  const onSelect = (ids: string[], additive: boolean) => {
    setSelected((prev) => {
      const next = additive ? new Set(prev) : new Set<string>();
      for (const id of ids) next.add(id);
      return next;
    });
  };

  // The selection predicate: an explicit id list when a lasso is active, otherwise the facet filter. This is
  // what every bulk action operates on, so what you see selected is exactly what gets changed.
  const actionPredicate = (): ExplorePredicate =>
    selected.size ? { ...predicate, object_ids: Array.from(selected) } : predicate;

  const applyTags = async (remove: boolean) => {
    const tags = tagText.split(",").map((t) => t.trim()).filter(Boolean);
    if (!tags.length) { flash("Type a tag first"); return; }
    setBusy("tag");
    try {
      const r = await api.exploreTag({
        level: "object", predicate: actionPredicate(),
        add: remove ? [] : tags, remove: remove ? tags : [],
      });
      flash(`${remove ? "Removed the tag from" : "Tagged"} ${num(r.matched)} objects`);
      setTagText("");
      // Tags are a facet and a point attribute, so every cached cut and layout is stale, not only the
      // ones on screen.
      await invalidate("exploreFacets", "exploreProjectionPoints");
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(null); }
  };

  const saveView = async () => {
    const name = window.prompt("Save this filter as a view named:");
    if (!name) return;
    setBusy("view");
    try {
      await api.exploreSaveView(name, actionPredicate());
      await viewsQ.reload();
      flash(`Saved view "${name}"`);
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(null); }
  };

  const fitProjection = async () => {
    setBusy("fit");
    flash("Fitting the projection; this can take a minute on a large corpus...");
    try {
      const r = await api.exploreFitProjection({
        kind: "object", space: "dino",
        session_id: predicate.session_id ?? null, method: "umap",
      });
      if (r.error) { flash(r.error); return; }
      await projQ.reload();
      if (r.projection_id) setProjId(r.projection_id);
      flash(`Fitted ${num(r.n)} points with ${(r.method ?? "").toUpperCase()}${r.clusters ? `, ${r.clusters} clusters` : ""}`);
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(null); }
  };

  const sendToReview = async () => {
    const ids = Array.from(selected);
    if (!ids.length) { flash("Lasso a region on the map first"); return; }
    setBusy("review");
    try {
      // Reuse the existing bulk-review path rather than inventing a second one. "confirm" with state=review
      // routes the selection into the review queue without changing any label.
      await api.bulkReview(ids, "confirm", undefined, "review");
      flash(`Sent ${num(ids.length)} objects to review`);
      // Their state changed, which is a facet here and a row in every review queue.
      void invalidate("exploreFacets", "exploreProjectionPoints", "triage", "/api/home");
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(null); }
  };

  const selCount = selected.size || facets?.total || 0;

  const filtered = Object.keys(predicate).length > 0;
  const pointsLoading = !!projId && pointsQ.loading;
  const mapEmpty = projQ.loading || pointsLoading ? (
    <span className="inline-flex items-center gap-2"><span className="running-dot" />Loading the projection...</span>
  ) : pointsQ.error ? (
    <EmptyState icon="info" title="Could not load the projection"
      action={<button type="button" className="btn" onClick={() => void pointsQ.reload()}>Retry</button>}>
      {humanizeError(pointsQ.error)}
    </EmptyState>
  ) : !projId ? (
    <EmptyState icon="scan" title="No projection yet"
      action={<button type="button" className="btn btn-primary" onClick={fitProjection} disabled={busy === "fit"}>
        {busy === "fit" ? "Fitting..." : "Fit projection"}</button>}>
      A projection lays the corpus out by visual similarity, so look-alike objects sit together and odd ones
      stand apart. Fitting one takes about a minute.
    </EmptyState>
  ) : points.length > 0 ? (
    <EmptyState icon="scan" title="No points match these filters"
      action={<button type="button" className="btn" onClick={() => setPredicate({})}>Clear filters</button>}>
      The facets on the left exclude every point in this projection.
    </EmptyState>
  ) : undefined;

  return (
    <PageShell
      active="EXPLORE"
      title="Explore"
      subtitle="See the corpus by similarity, cut it by facets, act on what you lasso."
      meta={<span className="text-[12px] text-ink-3">
        {activeProj ? `${activeProj.method.toUpperCase()} · ${num(activeProj.n)} points` : "No projection"}
      </span>}
      primaryAction={
        <button onClick={fitProjection} disabled={busy === "fit"}
          className="btn btn-primary"
          data-tip="Lay out the current objects by visual similarity (UMAP over DINO embeddings)">
          {busy === "fit" && <span className="running-dot mr-1" />}
          {busy === "fit" ? "Fitting..." : "Fit projection"}
        </button>
      }
    >
      <div className="flex h-full min-h-0">
        {/* facet rail */}
        <aside className="w-64 shrink-0 border-r hairline overflow-auto no-scrollbar bg-rail/40" aria-label="Filters">
          <FacetRail facets={facets} predicate={predicate} onChange={setPredicate} />
        </aside>

        {/* map + actions */}
        <section className="flex-1 min-w-0 flex flex-col">
          <div className="flex items-center gap-3 px-3 py-2 border-b hairline text-[12px] overflow-x-auto no-scrollbar">
            <label className="flex items-center gap-2 shrink-0">
              <span className="text-ink-3">Projection</span>
              <select value={projId} onChange={(e) => setProjId(e.target.value)}
                className="field py-0.5 max-w-[260px]">
                <option value="">None</option>
                {projections.map((p) => (
                  <option key={p.projection_id} value={p.projection_id}>
                    {humanize(p.kind)}s · {p.space.toUpperCase()} {p.method.toUpperCase()} · {num(p.n)} points
                  </option>
                ))}
              </select>
            </label>

            <span className="w-px h-5 bg-line shrink-0" aria-hidden />
            <span className="text-ink-3 shrink-0">Colour by</span>
            <div className="flex items-center gap-1 shrink-0" role="group" aria-label="Colour points by">
              {COLOR_MODES.map((m) => (
                <button key={m.key} onClick={() => setColorBy(m.key)} aria-pressed={colorBy === m.key}
                  className={`chip ${colorBy === m.key ? "chip-on" : ""}`}>
                  {m.label}
                </button>
              ))}
            </div>
            {colorBy === "tag" && (
              <select value={tagFilter ?? ""} onChange={(e) => setTagFilter(e.target.value || null)}
                aria-label="Tag to highlight"
                className="field py-0.5 shrink-0">
                <option value="">Pick a tag</option>
                {(facets?.tags ?? []).map((t) => <option key={t.value} value={t.value}>{t.value}</option>)}
              </select>
            )}

            <span className="ml-auto text-ink-2 tabular-nums shrink-0">
              {selected.size ? `${num(selected.size)} lassoed` : `${num(shownPoints.length)} shown`}
            </span>
            {selected.size > 0 && (
              <button onClick={() => setSelected(new Set())}
                className="btn shrink-0"><T k="action.clear">Clear</T></button>
            )}
          </div>

          <div className="flex-1 min-h-0 relative">
            <EmbeddingMap points={shownPoints} colorBy={colorBy} tagFilter={tagFilter}
              selected={selected} onSelect={onSelect} onHover={setHover} empty={mapEmpty} />
            {hover && (
              <div className="absolute top-2 left-2 panel px-2.5 py-1.5 text-[12px] text-ink-2 pointer-events-none space-y-0.5">
                <div className="font-mono text-ink">{shortId(hover.id)}</div>
                {hover.state && <div className="text-ink-3">{humanize(hover.state)} · {humanize(hover.source)}</div>}
                {hover.conf != null && <div className="text-ink-3">Confidence {Number(hover.conf).toFixed(2)}</div>}
                {!!(hover.tags ?? []).length && <div className="text-accent-2">{(hover.tags ?? []).join(", ")}</div>}
              </div>
            )}
            {shownPoints.length > 0 && !selected.size && (
              <div className="absolute bottom-2 left-2 text-[11.5px] text-ink-3 bg-panel/80 border hairline rounded px-2 py-1 pointer-events-none">
                Drag to lasso points · Shift to add to the selection
              </div>
            )}
          </div>

          {/* selection action bar */}
          <div className="border-t hairline px-3 py-2.5 flex items-center gap-2 text-[12.5px] flex-wrap bg-panel">
            <span className="text-ink-3">Acting on</span>
            <span className="text-ink font-semibold tabular-nums">{num(selCount)}</span>
            <span className="text-ink-3">{selected.size ? "lassoed" : filtered ? "matching" : ""} objects</span>

            <span className="w-px h-5 bg-line mx-1" aria-hidden />
            <input value={tagText} onChange={(e) => setTagText(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") void applyTags(false); }}
              aria-label="Tags to add or remove, comma separated"
              placeholder="Tags, comma separated"
              className="field py-0.5 min-w-[180px]" />
            <button onClick={() => applyTags(false)} disabled={!!busy}
              className="btn">Add tag</button>
            <button onClick={() => applyTags(true)} disabled={!!busy}
              className="btn"><T k="action.remove">Remove</T></button>

            <span className="w-px h-5 bg-line mx-1" aria-hidden />
            <button onClick={saveView} disabled={!!busy}
              className="btn">Save view</button>
            <button onClick={sendToReview} disabled={!!busy || !selected.size}
              data-tip={!selected.size ? "Lasso a region on the map first" : undefined}
              className="btn btn-on">Send to review</button>
            <button onClick={() => router.push("/analytics")}
              className="link ml-1">Evaluation drill-down →</button>

            {msg && <span className="ml-auto text-accent-2" role="status">{msg}</span>}
          </div>
        </section>

        {/* saved views */}
        <aside className={`w-56 shrink-0 border-l hairline overflow-auto no-scrollbar p-2 ${
          focusViews ? "ring-1 ring-accent bg-head/30" : ""}`} aria-label="Saved views">
          <div className={`section-title px-2 py-1.5 ${focusViews ? "text-accent-2" : ""}`}>Saved views</div>
          {viewsQ.loading ? (
            <div className="px-2 py-1 space-y-2">
              <span className="skeleton h-3 w-3/4 block" /><span className="skeleton h-3 w-1/2 block" />
            </div>
          ) : views.length === 0 && (
            <div className="px-2 py-2 text-[12px] text-ink-3 leading-relaxed">
              No saved views yet. Pick facets or lasso a region, then use <span className="text-ink-2">Save view</span> to
              come back to the same cut later.
            </div>
          )}
          {views.map((v) => (
            <button key={v.slice_id} onClick={() => { setPredicate(v.predicate ?? {}); setSelected(new Set()); }}
              title={describePredicate(v.predicate)}
              className="w-full text-left px-2 py-1.5 rounded text-[12.5px] text-ink-2 hover:text-ink hover:bg-head/60 truncate">
              {v.name}
            </button>
          ))}
        </aside>
      </div>
    </PageShell>
  );
}
