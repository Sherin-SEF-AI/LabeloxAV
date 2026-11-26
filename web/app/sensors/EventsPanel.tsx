"use client";

// Event cameras: import a recording, look at it accumulated over a window (event counts, or a time surface
// where what moved last is brightest), and write it out as ordinary frames that open in the normal editor.

import { useEffect, useState } from "react";

import { humanizeError, uploadMultipart } from "@/lib/api";
import EmptyState from "@/components/ui/EmptyState";
import { Badge, Bar, Button, ErrLine, Field, HintBar, NumberInput, Panel, Segmented, Stat, TextInput } from "@/components/ui/kit";
import { compact, fromNs, num } from "@/lib/format";
import { useToolJob } from "@/lib/tools/toolJobs";
import {
  eventsAccumulate, eventsFrames, eventsImport, eventsPreviewUrl, type EventFrames, type EventStreamRow,
} from "@/lib/tools/sensorsApi";

export default function EventsPanel({ streams, sessions, onChanged }: {
  streams: EventStreamRow[]; sessions: { session_id: string; label: string }[]; onChanged: () => void;
}) {
  const [sel, setSel] = useState<string | null>(null);
  const [mode, setMode] = useState<"count" | "time_surface">("count");
  const [windowMs, setWindowMs] = useState(33);
  const [tauMs, setTauMs] = useState(30);
  const [tFrac, setTFrac] = useState(0.1);
  const [frames, setFrames] = useState<EventFrames | null>(null);
  const [maxFrames, setMaxFrames] = useState(600);
  const [jobId, setJobId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // import form
  const [file, setFile] = useState<File | null>(null);
  const [path, setPath] = useState("");
  const [sensor, setSensor] = useState("event_front");
  const [target, setTarget] = useState("");
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [unit, setUnit] = useState<"us" | "ns" | "ms" | "s">("us");
  const [up, setUp] = useState<number | null>(null);

  const stream = streams.find((s) => s.stream_id === sel) ?? null;
  useEffect(() => { if (!sel && streams.length) setSel(streams[0].stream_id); }, [streams, sel]);
  useEffect(() => {
    setFrames(null);
    if (sel) eventsFrames(sel).then(setFrames).catch((e) => setErr(humanizeError(e)));
  }, [sel]);

  const { job } = useToolJob(jobId, {
    onDone: (j) => {
      onChanged();
      if (sel) eventsFrames(sel).then(setFrames).catch((e) => setErr(humanizeError(e)));
      if (j.status === "error") setErr(j.error);
    },
  });

  const doImport = async () => {
    setErr(null);
    try {
      let uri = path.trim();
      if (file) { setUp(0); uri = await uploadMultipart(file, setUp); }
      if (!uri) { setErr("Choose a file or give a server path"); return; }
      const j = await eventsImport({ source_uri: uri, sensor, session_id: target || null, time_unit: unit,
        width: width || null, height: height || null });
      setJobId(j.job_id);
    } catch (e) { setErr(humanizeError(e)); } finally { setUp(null); }
  };

  const accumulate = async () => {
    if (!sel) return;
    setErr(null);
    try { setJobId((await eventsAccumulate(sel, { mode, window_ms: windowMs, tau_ms: tauMs, max_frames: maxFrames, replace: true })).job_id); }
    catch (e) { setErr(humanizeError(e)); }
  };

  const tNs = stream ? Math.round(stream.t0_ns + windowMs * 1e6 + tFrac * Math.max(0, stream.t1_ns - stream.t0_ns - windowMs * 1e6)) : 0;
  const busy = job && job.status !== "done" && job.status !== "error" && job.status !== "canceled";

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[340px_1fr] gap-4">
      <div className="flex flex-col gap-4 min-w-0">
        <Panel title="Import a recording" desc=".npy (x, y, t, p), DSEC .h5, or an MCAP with dvs_msgs / prophesee EventArray messages.">
          <div className="flex flex-col gap-1">
            <input type="file" accept=".npy,.h5,.hdf5,.mcap" className="text-[12px]" onChange={(e) => setFile(e.target.files?.[0] ?? null)} aria-label="Event file" />
            <Field label="or server path" tip="Only inside the directories this server may import from"><TextInput value={path} onChange={setPath} placeholder="/data/events/run1.mcap" className="w-44" /></Field>
            <Field label="Sensor name" tip="The camera id its frames get"><TextInput value={sensor} onChange={setSensor} className="w-44" /></Field>
            <Field label="Session">
              <select className="field w-44" value={target} onChange={(e) => setTarget(e.target.value)}>
                <option value="">A new session</option>
                {sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.label}</option>)}
              </select>
            </Field>
            <Field label="Sensor size" tip="Width x height; read from the file when it says">
              <span className="flex items-center gap-1"><NumberInput value={width} onChange={(n) => setWidth(Math.max(0, Math.round(n || 0)))} w="w-16" />
                <span className="text-ink-3">x</span><NumberInput value={height} onChange={(n) => setHeight(Math.max(0, Math.round(n || 0)))} w="w-16" /></span>
            </Field>
            <Field label="Time unit (.npy)">
              <Segmented value={unit} onChange={setUnit} options={["us", "ns", "ms", "s"].map((u) => ({ value: u as "us", label: u }))} />
            </Field>
            <div className="flex items-center gap-2 mt-2">
              <Button variant="primary" onClick={doImport} disabled={!!busy} tip="Needs the reviewer role">Import</Button>
              {up != null && <span className="flex-1 flex items-center gap-2 text-[12px]"><Bar frac={up} />uploading</span>}
            </div>
          </div>
        </Panel>
        <Panel title="Streams" desc="Every event recording, newest first.">
          {!streams.length ? <p className="text-[12px] text-ink-3">No event recordings yet.</p> : (
            <ul className="flex flex-col gap-1">
              {streams.map((s) => (
                <li key={s.stream_id}>
                  <button type="button" onClick={() => setSel(s.stream_id)}
                    className={`w-full text-left px-2 py-1.5 rounded border text-[12px] ${sel === s.stream_id ? "border-accent bg-accent/10" : "border-line hover:bg-head"}`}>
                    <div className="flex items-center gap-2"><span className="text-ink">{s.sensor}</span><Badge>{s.source}</Badge>
                      <span className="ml-auto text-ink-3">{compact(s.n_events)} events</span></div>
                    <div className="text-ink-3 mt-0.5">{s.session_name ?? "session"} · {s.width}x{s.height} · {s.stats.duration_s?.toFixed(2)} s</div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <div className="flex flex-col gap-4 min-w-0">
        {!stream ? (
          <div className="panel"><EmptyState icon="activity" title="No event recording selected">Import one on the left. There is no event-camera data on this server yet.</EmptyState></div>
        ) : (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Events" value={compact(stream.n_events)} sub={`${num(stream.stats.events_per_s)} per second`} />
              <Stat label="Sensor" value={`${stream.width}x${stream.height}`} sub={stream.sensor} />
              <Stat label="Polarity" value={`${num(stream.stats.positive)} up`} sub={`${num(stream.stats.negative)} down`} />
              <Stat label="Span" value={`${stream.stats.duration_s?.toFixed(2)} s`} sub={fromNs(stream.t0_ns)} />
            </div>
            <Panel title="Accumulate into frames" desc="Each window becomes one image. Counts show where brightness changed; a time surface shows what changed last brightest."
              right={<Segmented value={mode} onChange={setMode} options={[{ value: "count", label: "Event count" }, { value: "time_surface", label: "Time surface" }]} />}>
              <div className="grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-4">
                <div>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={eventsPreviewUrl(stream.stream_id, { t_ns: tNs, mode, window_ms: windowMs, tau_ms: tauMs })}
                    alt="Accumulated events" className="w-full rounded bg-bg-2" style={{ imageRendering: "pixelated", aspectRatio: `${stream.width} / ${stream.height}` }} />
                  <div className="flex items-center gap-3 mt-2 text-[12px]">
                    <input type="range" min={0} max={1} step={0.001} value={tFrac} onChange={(e) => setTFrac(Number(e.target.value))} className="flex-1" aria-label="Time" />
                    <span className="tabular-nums text-ink-2">{((tNs - stream.t0_ns) / 1e6).toFixed(1)} ms</span>
                  </div>
                </div>
                <div className="flex flex-col gap-1">
                  <Field label="Window (ms)"><NumberInput value={windowMs} onChange={(n) => setWindowMs(Math.max(0.1, n || 1))} w="w-20" /></Field>
                  {mode === "time_surface" && <Field label="Decay (ms)" tip="Time constant of the surface"><NumberInput value={tauMs} onChange={(n) => setTauMs(Math.max(0.1, n || 1))} w="w-20" /></Field>}
                  <Field label="At most frames"><NumberInput value={maxFrames} onChange={(n) => setMaxFrames(Math.max(1, Math.min(5000, Math.round(n || 1))))} w="w-20" /></Field>
                  <p className="text-[11px] text-ink-3 mt-1">Writing again replaces this stream&apos;s earlier frames, except any that carry a label.</p>
                  <Button variant="primary" onClick={accumulate} disabled={!!busy} tip="Needs the reviewer role">Write frames</Button>
                  {busy && <span className="flex items-center gap-2 text-[12px] mt-1"><Bar frac={job!.progress} />{job!.stage ?? job!.status}</span>}
                </div>
              </div>
            </Panel>
            <Panel title="Frames to label" desc="Ordinary frames (real, tagged sensor:event). Open one to annotate it in the editor.">
              {!frames?.frames.length ? <HintBar>No frames written from this stream yet.</HintBar> : (
                <>
                  <p className="text-[12px] text-ink-2 mb-2">
                    {frames.accumulation ? `${frames.accumulation.frames} frames, ${frames.accumulation.mode.replace("_", " ")}, ${frames.accumulation.window_ms} ms windows, about ${num(frames.accumulation.events_per_frame)} events each.` : ""}
                  </p>
                  <div className="grid grid-cols-3 md:grid-cols-6 xl:grid-cols-8 gap-2">
                    {frames.frames.slice(0, 48).map((f) => (
                      <a key={f.frame_id} href={`/frame/${f.frame_id}`} className="block rounded border border-line hover:border-accent overflow-hidden" title={fromNs(f.ts_ns)}>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={`/api/frames/${f.frame_id}/image`} alt="Event frame" loading="lazy" className="w-full" style={{ imageRendering: "pixelated" }} />
                        <div className="px-1 py-0.5 text-[11px] text-ink-3 flex justify-between"><span>{((f.ts_ns - stream.t0_ns) / 1e6).toFixed(0)} ms</span>{f.objects > 0 && <span className="text-pass">{f.objects}</span>}</div>
                      </a>
                    ))}
                  </div>
                </>
              )}
            </Panel>
          </>
        )}
        <ErrLine err={err} />
      </div>
    </div>
  );
}
