"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
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

// The Explore workspace: see the corpus as structure, cut it by facets, lasso what looks alike, and act on
// the selection in bulk. The filter you build here IS a saved view and an export spec, because it is the same
// predicate a CurationSlice stores.

const COLOR_MODES: { key: ColorBy; label: string }[] = [
  { key: "cluster", label: "cluster" },
  { key: "class", label: "class" },
  { key: "state", label: "state" },
  { key: "source", label: "source" },
  { key: "conf", label: "conf" },
  { key: "tag", label: "tag" },
];

export default function ExplorePage() {
  // useSearchParams (via useQueryParam) forces a CSR bailout Next requires be under Suspense.
  return <Suspense fallback={null}><ExploreBody /></Suspense>;
}

function ExploreBody() {
  const router = useRouter();
  const [predicate, setPredicate] = useState<ExplorePredicate>({});
  const [facets, setFacets] = useState<Facets | null>(null);
  const [projections, setProjections] = useState<ProjectionRow[]>([]);
  const [projId, setProjId] = useState<string>("");
  const [points, setPoints] = useState<ProjectionPoint[]>([]);
  const [colorBy, setColorBy] = useState<ColorBy>("cluster");
  const [tagFilter, setTagFilter] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [hover, setHover] = useState<ProjectionPoint | null>(null);
  const [views, setViews] = useState<SavedView[]>([]);
  // Edit > Saved views deep-links here; highlight the rail so the menu entry lands somewhere visible
  const focusViews = useQueryParam("panel") === "views";
  const [tagText, setTagText] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(null), 3500); };

  // Facets recompute whenever the predicate changes: the counts must always describe the current cut.
  useEffect(() => {
    api.exploreFacets(predicate).then(setFacets).catch(() => setFacets(null));
  }, [predicate]);

  useEffect(() => {
    api.exploreProjections().then((r) => {
      setProjections(r.projections);
      if (!projId && r.projections.length) setProjId(r.projections[0].projection_id);
    }).catch(() => {});
    api.exploreViews().then((r) => setViews(r.views)).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!projId) { setPoints([]); return; }
    api.exploreProjectionPoints(projId).then((r) => setPoints(r.points ?? []))
      .catch(() => setPoints([]));
    setSelected(new Set());
  }, [projId]);

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
    if (!tags.length) { flash("type a tag first"); return; }
    setBusy("tag");
    try {
      const r = await api.exploreTag({
        level: "object", predicate: actionPredicate(),
        add: remove ? [] : tags, remove: remove ? tags : [],
      });
      flash(`${remove ? "removed" : "tagged"} ${r.matched} objects`);
      setTagText("");
      const f = await api.exploreFacets(predicate);
      setFacets(f);
      if (projId) setPoints((await api.exploreProjectionPoints(projId)).points ?? []);
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(null); }
  };

  const saveView = async () => {
    const name = window.prompt("save this filter as a view named:");
    if (!name) return;
    setBusy("view");
    try {
      await api.exploreSaveView(name, actionPredicate());
      setViews((await api.exploreViews()).views);
      flash(`saved view "${name}"`);
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(null); }
  };

  const fitProjection = async () => {
    setBusy("fit");
    flash("fitting projection, this can take a minute on a large corpus...");
    try {
      const r = await api.exploreFitProjection({
        kind: "object", space: "dino",
        session_id: predicate.session_id ?? null, method: "umap",
      });
      if (r.error) { flash(r.error); return; }
      const list = await api.exploreProjections();
      setProjections(list.projections);
      if (r.projection_id) setProjId(r.projection_id);
      flash(`fitted ${r.n} points with ${r.method}${r.clusters ? `, ${r.clusters} clusters` : ""}`);
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(null); }
  };

  const sendToReview = async () => {
    const ids = Array.from(selected);
    if (!ids.length) { flash("lasso a region first"); return; }
    setBusy("review");
    try {
      // Reuse the existing bulk-review path rather than inventing a second one. "confirm" with state=review
      // routes the selection into the review queue without changing any label.
      await api.bulkReview(ids, "confirm", undefined, "review");
      flash(`sent ${ids.length} objects to review`);
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(null); }
  };

  const selCount = selected.size || facets?.total || 0;

  return (
    <PageShell
      active="EXPLORE"
      title="Explore"
      subtitle="see the corpus by similarity, cut it by facets, act on what you lasso"
      meta={<span className="font-mono text-[11px] text-ink-3">
        {activeProj ? `${activeProj.method} - ${activeProj.n.toLocaleString()} pts` : "no projection"}
      </span>}
      primaryAction={
        <button onClick={fitProjection} disabled={busy === "fit"}
          className="h-[30px] px-3 rounded-md bg-accent text-bg font-display font-semibold text-[12.5px] disabled:opacity-40">
          {busy === "fit" ? "fitting..." : "Fit projection"}
        </button>
      }
    >
      <div className="flex h-full min-h-0">
        {/* facet rail */}
        <aside className="w-60 shrink-0 border-r hairline overflow-auto no-scrollbar">
          <FacetRail facets={facets} predicate={predicate} onChange={setPredicate} />
        </aside>

        {/* map + actions */}
        <section className="flex-1 min-w-0 flex flex-col">
          <div className="flex items-center gap-2 px-3 py-1.5 border-b hairline font-mono text-[11px] overflow-x-auto no-scrollbar">
            <select value={projId} onChange={(e) => setProjId(e.target.value)}
              className="bg-bg border border-line px-1.5 py-0.5 text-ink max-w-[240px]">
              <option value="">no projection</option>
              {projections.map((p) => (
                <option key={p.projection_id} value={p.projection_id}>
                  {p.kind}/{p.space} {p.method} - {p.n} pts
                </option>
              ))}
            </select>

            <span className="text-ink-3">colour</span>
            {COLOR_MODES.map((m) => (
              <button key={m.key} onClick={() => setColorBy(m.key)}
                className={`px-1.5 py-0.5 border ${colorBy === m.key ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}>
                {m.label}
              </button>
            ))}
            {colorBy === "tag" && (
              <select value={tagFilter ?? ""} onChange={(e) => setTagFilter(e.target.value || null)}
                className="bg-bg border border-line px-1.5 py-0.5 text-ink">
                <option value="">pick a tag</option>
                {(facets?.tags ?? []).map((t) => <option key={t.value} value={t.value}>{t.value}</option>)}
              </select>
            )}

            <span className="ml-auto text-ink-3">
              {selected.size ? `${selected.size} lassoed` : `${shownPoints.length} shown`}
            </span>
            {selected.size > 0 && (
              <button onClick={() => setSelected(new Set())}
                className="px-1.5 py-0.5 border border-line text-ink-3 hover:text-ink-2">clear</button>
            )}
          </div>

          <div className="flex-1 min-h-0 relative">
            <EmbeddingMap points={shownPoints} colorBy={colorBy} tagFilter={tagFilter}
              selected={selected} onSelect={onSelect} onHover={setHover} />
            {hover && (
              <div className="absolute top-2 left-2 panel px-2 py-1 font-mono text-[10px] text-ink-2 pointer-events-none">
                <div>{hover.id.slice(0, 8)}</div>
                {hover.state && <div className="text-ink-3">{hover.state} - {hover.source}</div>}
                {hover.conf != null && <div className="text-ink-3">conf {hover.conf}</div>}
                {!!(hover.tags ?? []).length && <div className="text-accent">{(hover.tags ?? []).join(", ")}</div>}
              </div>
            )}
          </div>

          {/* selection action bar */}
          <div className="border-t hairline px-3 py-2 flex items-center gap-2 font-mono text-[11px] flex-wrap">
            <span className="text-ink-3">acting on</span>
            <span className="text-ink">{selCount.toLocaleString()}</span>
            <span className="text-ink-3">{selected.size ? "lassoed" : "matching"} objects</span>

            <input value={tagText} onChange={(e) => setTagText(e.target.value)}
              placeholder="tag, comma separated"
              className="ml-2 bg-bg border border-line px-1.5 py-0.5 text-ink min-w-[180px]" />
            <button onClick={() => applyTags(false)} disabled={!!busy}
              className="px-2 py-0.5 border border-line text-ink-2 hover:border-accent disabled:opacity-40">add tag</button>
            <button onClick={() => applyTags(true)} disabled={!!busy}
              className="px-2 py-0.5 border border-line text-ink-3 hover:border-accent disabled:opacity-40">remove</button>

            <button onClick={saveView} disabled={!!busy}
              className="px-2 py-0.5 border border-line text-ink-2 hover:border-accent disabled:opacity-40">save view</button>
            <button onClick={sendToReview} disabled={!!busy || !selected.size}
              className="px-2 py-0.5 border border-line text-ink-2 hover:border-accent disabled:opacity-40">send to review</button>
            <button onClick={() => router.push("/analytics")}
              className="px-2 py-0.5 border border-line text-ink-3 hover:border-accent">eval drill-down</button>

            {msg && <span className="ml-auto text-accent">{msg}</span>}
          </div>
        </section>

        {/* saved views */}
        <aside className={`w-52 shrink-0 border-l hairline overflow-auto no-scrollbar p-1.5 ${
          focusViews ? "ring-1 ring-accent bg-head/30" : ""}`}>
          <div className={`font-mono text-[10px] uppercase px-1.5 py-1 ${
            focusViews ? "text-accent" : "text-ink-3"}`}>saved views</div>
          {views.length === 0 && (
            <div className="px-1.5 font-mono text-[11px] text-ink-3">
              none yet. filter, then save.
            </div>
          )}
          {views.map((v) => (
            <button key={v.slice_id} onClick={() => { setPredicate(v.predicate ?? {}); setSelected(new Set()); }}
              title={JSON.stringify(v.predicate)}
              className="w-full text-left px-1.5 py-1 font-mono text-[11px] text-ink-3 hover:text-ink">
              {v.name}
            </button>
          ))}
        </aside>
      </div>
    </PageShell>
  );
}
