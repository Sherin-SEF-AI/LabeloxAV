"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useClock } from "@/lib/inspector/clock";
import { api, type InspectorEvent, type InspectorPanel, type InspectorTopic , humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { Spinner } from "@/components/Spinner";
import { ClockProvider } from "@/lib/inspector/clock";
import { McapProvider } from "@/lib/inspector/mcapContext";
import { SessionMcap } from "@/lib/inspector/mcap";
import Timeline from "@/components/inspector/Timeline";
import TopicBrowser from "@/components/inspector/TopicBrowser";
import PanelHost from "@/components/inspector/PanelHost";
import { cycleSpan, fromSaved, movePanel, panelStyle, rowsOf, setRows, toSaved } from "@/lib/inspector/layout";

let PANEL_SEQ = 0;
const mkPanel = (type: string, topic?: string): InspectorPanel => ({ id: `p${++PANEL_SEQ}`, type, topic });

// Bidirectional deep link (annotation workspace -> Inspector): when the URL carries ?ts=<ns>, seek the one
// clock to that moment once the session is open. Runs inside the ClockProvider.
function SeekToParam({ tsNs, startNs }: { tsNs: string | null; startNs: bigint }) {
  const clock = useClock();
  useEffect(() => {
    if (tsNs) clock.seek(Number(BigInt(tsNs) - startNs) / 1e9);
  }, [clock, tsNs, startNs]);
  return null;
}

// Build the config default layout by binding each default panel type to a sensible topic from the index.
function defaultPanels(types: string[], topics: InspectorTopic[]): InspectorPanel[] {
  const find = (pred: (t: InspectorTopic) => boolean) => topics.find(pred)?.name;
  const cam = find((t) => /image/i.test(t.schema) || /cam/i.test(t.name));
  const imu = find((t) => /imu/i.test(t.name));
  const can = find((t) => /can/i.test(t.name));
  const gnss = find((t) => /locationfix/i.test(t.schema) || /gnss|gps/i.test(t.name));
  const raw = topics[0]?.name;
  const bind: Record<string, string | undefined> = { image: cam, imu_plot: imu, can_plot: can, map: gnss, raw };
  return types.map((ty) => mkPanel(ty, bind[ty])).filter((p) => p.topic || p.type === "map");
}

export default function InspectorWorkspace() {
  const router = useRouter();
  const params = useParams();
  const search = useSearchParams();
  const sessionId = String(params.session);
  const tsParam = search.get("ts");
  const [curFrame, setCurFrame] = useState<{ frameId: string | null; tsNs: string } | null>(null);

  const [mcap, setMcap] = useState<SessionMcap | null>(null);
  const [range, setRange] = useState<[bigint, bigint] | null>(null);
  const [topics, setTopics] = useState<InspectorTopic[]>([]);
  const [gaps, setGaps] = useState<Record<string, [number, number][]>>({});
  const [events, setEvents] = useState<InspectorEvent[]>([]);
  const [verdict, setVerdict] = useState<string | null>(null);
  const [panels, setPanels] = useState<InspectorPanel[]>([]);
  const [noRecording, setNoRecording] = useState(false);
  const [dragIdx, setDragIdx] = useState<number | null>(null);
  const [layouts, setLayouts] = useState<{ layout_id: string; name: string; panels: InspectorPanel[]; is_default: boolean }[]>([]);
  const [saveName, setSaveName] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let live = true;
    (async () => {
      setLoading(true);
      try {
        const info = await api.inspectorMcapUrl(sessionId);
        const m = await SessionMcap.open(info.url);
        if (!live) return;
        const idxTopics: InspectorTopic[] = Object.values(info.topics || {});
        // clock bounds from the MCAP statistics (native bigint, no precision loss); fall back to the index.
        const r = m.timeRange() ?? (info.time_range ? [BigInt(info.time_range[0]), BigInt(info.time_range[1])] as [bigint, bigint] : null);
        setMcap(m);
        setRange(r);
        setTopics(idxTopics.length ? idxTopics : m.topics().map((t) => ({ name: t.topic, schema: t.schema, count: 0, rate: 0, first_ts: 0, last_ts: 0 })));
        setGaps(info.gaps || {});
        const [health, lay, ev] = await Promise.all([
          api.inspectorHealth(sessionId).catch(() => ({ verdict: null })),
          api.inspectorLayouts().catch(() => ({ layouts: [], config_default: ["image", "imu_plot", "can_plot", "map", "raw"] })),
          api.inspectorEvents(sessionId).catch(() => ({ events: [] })),
        ]);
        if (!live) return;
        setVerdict(health.verdict);
        setLayouts(lay.layouts);
        setEvents(ev.events || []);
        const def = lay.layouts.find((l) => l.is_default);
        setPanels(def ? fromSaved(def.panels, () => `p${++PANEL_SEQ}`) : defaultPanels(lay.config_default, idxTopics.length ? idxTopics : m.topics().map((t) => ({ name: t.topic, schema: t.schema, count: 0, rate: 0, first_ts: 0, last_ts: 0 }))));
      } catch (e) {
        // No recording is not the same as nothing to inspect. This deployment has one session with an MCAP
        // and 124 with point clouds, so refusing here made the 3D panel unreachable through the product's
        // own navigation. A session without a recording still has its own clock, its extracted frames and
        // its geometry, all addressed by timestamp; what it lacks is topics, and the panels that need one
        // say so themselves.
        if (!live) return;
        // No string sniffing on the error. The previous version tested it for "409" and the humanized
        // message says "session has no MCAP" with no status in it, so the branch never fired. Ask the
        // server what this session has instead: it is the only thing that actually knows, and the answer
        // decides both whether to fall back and what to say when we cannot.
        try {
          const all = await api.inspectorSessions();
          const me = all.find((x) => x.session_id === sessionId);
          if (!me || !me.n_clouds) {
            setErr(me
              ? "This session has no MCAP recording and no point clouds, so there is nothing to inspect."
              : humanizeError(e));
            return;
          }
          setMcap(SessionMcap.empty());
          setRange([BigInt(me.start_ts_ns), BigInt(me.end_ts_ns)]);
          setTopics([]);
          setNoRecording(true);
          const [lay, ev] = await Promise.all([
            api.inspectorLayouts().catch(() => ({ layouts: [], config_default: [] })),
            api.inspectorEvents(sessionId).catch(() => ({ events: [] })),
          ]);
          if (!live) return;
          setLayouts(lay.layouts);
          setEvents(ev.events || []);
          // Only the panels that read this session's own timeline. Restoring a saved layout full of topic
          // panels here would open five that can never draw anything.
          setPanels([mkPanel("scene3d"), mkPanel("image")]);
        } catch (e2) {
          if (live) setErr(humanizeError(e2));
        }
      } finally {
        if (live) setLoading(false);
      }
    })();
    return () => { live = false; };
  }, [sessionId]);

  const addPanel = useCallback((type: string, topic?: string) => setPanels((p) => [...p, mkPanel(type, topic)]), []);
  const removePanel = useCallback((id: string) => setPanels((p) => p.filter((x) => x.id !== id)), []);

  const saveLayout = async () => {
    const name = saveName.trim() || "layout";
    try {
      await api.inspectorSaveLayout(name, panels.map((p) => ({ id: "", ...toSaved(p) })), true);
      setLayouts((await api.inspectorLayouts()).layouts);
      setSaveName("");
    } catch (e) { setErr("save layout failed: " + humanizeError(e)); }
  };
  const loadLayout = (id: string) => {
    const l = layouts.find((x) => x.layout_id === id);
    if (l) setPanels(fromSaved(l.panels, () => `p${++PANEL_SEQ}`));
  };

  const openLichtblick = useCallback(async () => {
    try { const r = await api.inspectorLichtblick(sessionId); window.open(r.url, "_blank", "noopener"); }
    catch (e) { setErr("Open in Lichtblick failed: " + humanizeError(e)); }
  }, [sessionId]);

  const vColor = verdict === "pass" ? "text-pass border-pass" : verdict === "warn" ? "text-warn border-warn" : verdict === "fail" ? "text-block border-block" : "text-ink-3 border-line";

  const right = useMemo(() => (
    <div className="flex items-center gap-2 font-mono text-[11px]">
      {verdict && <span className={`border px-1.5 rounded uppercase ${vColor}`}>{verdict}</span>}
      {curFrame?.frameId && (
        <button onClick={() => router.push(`/frame/${curFrame.frameId}`)} title="open the frame at the current time in the annotation workspace"
          className="border border-accent/50 bg-accent/10 text-accent px-2 py-0.5 rounded hover:bg-accent/20">open frame in workspace</button>
      )}
      {/* Only 1 of 342 sessions in this corpus carries an MCAP, so offering this unconditionally meant the
          button answered 409 for almost every session somebody opened. The precondition is knowable here:
          the page already loaded the recording, or failed to. Say so on the control instead of on a toast
          after the request. */}
      <button onClick={openLichtblick} disabled={!mcap}
        title={mcap ? "open this session's MCAP in Lichtblick"
                    : "this session has no MCAP recording to inspect"}
        className={`border px-2 py-0.5 rounded ${mcap ? "border-line hover:border-accent"
                                                     : "border-line/40 text-ink-3 cursor-not-allowed"}`}>
        open in lichtblick
      </button>
    </div>
  ), [verdict, curFrame, router, mcap, openLichtblick, vColor]);

  return (
    <PageShell active="INSPECT" subtitle="SESSION" right={right}>
      {loading ? (
        <div className="h-full flex items-center justify-center"><Spinner label="opening session mcap" /></div>
      ) : err ? (
        <div className="h-full flex items-center justify-center font-mono text-sm text-block px-6 text-center">{err}</div>
      ) : mcap && range ? (
        <ClockProvider startNs={range[0]} endNs={range[1]}>
          <McapProvider mcap={mcap} sessionId={sessionId}>
            <SeekToParam tsNs={tsParam} startNs={range[0]} />
            <div className="h-full flex flex-col min-h-0">
              <div className="flex-1 flex min-h-0">
                {/* topic browser + layout controls */}
                <div className="w-56 shrink-0 border-r hairline flex flex-col min-h-0">
                  <div className="p-2 border-b hairline space-y-1.5">
                    <div className="flex gap-1">
                      <input value={saveName} onChange={(e) => setSaveName(e.target.value)} placeholder="layout name"
                        className="flex-1 min-w-0 bg-bg-2 border border-line rounded px-1.5 py-1 font-mono text-[10px] text-ink-2 focus:border-accent outline-none" />
                      <button onClick={saveLayout} className="font-mono text-[10px] border border-line px-1.5 rounded hover:border-accent">save</button>
                    </div>
                    {layouts.length > 0 && (
                      <select onChange={(e) => e.target.value && loadLayout(e.target.value)} defaultValue=""
                        className="w-full bg-bg-2 border border-line rounded px-1 py-1 font-mono text-[10px] text-ink-2">
                        <option value="">load layout...</option>
                        {layouts.map((l) => <option key={l.layout_id} value={l.layout_id}>{l.name}{l.is_default ? " (default)" : ""}</option>)}
                      </select>
                    )}
                  </div>
                  {/* Panels that are not driven by an MCAP topic. The 3D view reads the point cloud store
                      keyed on the clock, so there is no topic to pick it from and the topic browser cannot
                      offer it. */}
                  <div className="px-2 py-1.5 border-b hairline">
                    <button onClick={() => addPanel("scene3d")}
                      className="w-full font-mono text-[10px] border border-line rounded px-1.5 py-1 text-ink-2 hover:border-accent hover:text-ink">
                      + 3D point cloud
                    </button>
                  </div>
                  <div className="flex-1 min-h-0">
                    {noRecording ? (
                      // Said plainly, because an empty topic list otherwise reads as a failed load rather
                      // than as a session that was never recorded to an .mcap.
                      <div className="p-2 font-mono text-[10px] text-ink-3 leading-relaxed">
                        no .mcap recording for this session, so there are no topics to browse. The 3D and
                        camera panels read this session&apos;s own timeline and work regardless.
                      </div>
                    ) : (
                      <TopicBrowser topics={topics} onAdd={addPanel} />
                    )}
                  </div>
                </div>
                {/* panel grid */}
                <div className="flex-1 min-h-0 overflow-auto p-2">
                  {panels.length === 0 ? (
                    <div className="h-full flex items-center justify-center font-mono text-xs text-ink-3">add a panel from the topic list</div>
                  ) : (
                    <div className="grid grid-cols-2 gap-2 auto-rows-[minmax(200px,1fr)]">
                      {panels.map((p, i) => (
                        <div key={p.id} style={panelStyle(p)} className="min-h-0">
                          <PanelHost
                            panel={p}
                            dragging={dragIdx === i}
                            onRemove={() => removePanel(p.id)}
                            onFrame={(frameId, tsNs) => setCurFrame({ frameId, tsNs })}
                            onCycleSpan={() => setPanels((ps) => cycleSpan(ps, p.id))}
                            onGrow={() => setPanels((ps) => setRows(ps, p.id, rowsOf(p) + 1))}
                            onShrink={() => setPanels((ps) => setRows(ps, p.id, rowsOf(p) - 1))}
                            onDragStart={() => setDragIdx(i)}
                            // Without preventDefault the browser refuses the drop and the panel snaps back.
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={() => {
                              setPanels((ps) => (dragIdx === null ? ps : movePanel(ps, dragIdx, i)));
                              setDragIdx(null);
                            }}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              <Timeline topics={topics} gaps={gaps} events={events} />
            </div>
          </McapProvider>
        </ClockProvider>
      ) : (
        <div className="h-full flex items-center justify-center font-mono text-sm text-ink-3">no MCAP for this session</div>
      )}
    </PageShell>
  );
}
