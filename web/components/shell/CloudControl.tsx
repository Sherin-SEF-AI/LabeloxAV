"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { CloudStatus, CloudOrphan } from "@/lib/types";
import Icon from "@/components/shell/Icon";

// The cloud GPU control: a compact status pill (always showing state and, when connected, live uptime and
// accruing cost) that opens a panel with connect/disconnect, the cost breakdown, and the idle / max-session
// countdowns. Connect requires acknowledging the hourly rate. An orphaned pod surfaces a prominent banner.
// Cost safety is enforced on the backend; this keeps the cost always in view and the rate in front of the
// user before any GPU is fired.

const fmtCost = (usd: number) => `$${usd < 1 ? usd.toFixed(4) : usd.toFixed(2)}`;

function fmtDur(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(sec).padStart(2, "0")}s`;
  return `${sec}s`;
}

export default function CloudControl() {
  const [st, setSt] = useState<CloudStatus | null>(null);
  const [orphans, setOrphans] = useState<CloudOrphan[]>([]);
  const [open, setOpen] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState<"connecting" | "disconnecting" | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const lastPoll = useRef<number>(0);

  const poll = useCallback(async () => {
    try { const s = await api.cloudStatus(); setSt(s); lastPoll.current = Date.now(); } catch { /* keep last */ }
  }, []);
  const pollOrphans = useCallback(async () => {
    try { const r = await api.cloudOrphans(); setOrphans(r.orphans); } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    poll(); pollOrphans();
    const a = setInterval(poll, 3000);
    const b = setInterval(pollOrphans, 20000);
    return () => { clearInterval(a); clearInterval(b); };
  }, [poll, pollOrphans]);
  // a 1s tick interpolates uptime + cost smoothly between polls so the meter never looks frozen
  useEffect(() => { const t = setInterval(() => setTick((x) => x + 1), 1000); return () => clearInterval(t); }, []);

  if (!st) return null;

  const elapsed = st.connected ? Math.max(0, (Date.now() - lastPoll.current) / 1000) : 0;
  const liveCost = st.connected ? st.est_cost + (elapsed * st.hourly_usd) / 3600 : st.est_cost;
  const liveUptime = st.connected ? st.uptime_s + elapsed : st.uptime_s;

  const provisioning = st.state === "provisioning";
  const tearing = st.state === "terminating" || st.state === "pausing";
  const dotClass =
    st.state === "running_job" ? "bg-pass animate-pulse"
      : st.state === "connected" ? "bg-pass"
      : provisioning ? "bg-warn animate-pulse"
      : tearing ? "bg-warn"
      : "bg-ink-3";
  const label =
    st.state === "running_job" ? "running job"
      : st.state === "connected" ? "A100 connected"
      : provisioning ? "provisioning"
      : st.state === "terminating" ? "terminating"
      : st.state === "pausing" ? "pausing"
      : "cloud GPU";

  const doConnect = async () => {
    setConfirm(false); setBusy("connecting"); setErr(null);
    try { setSt(await api.cloudConnect(st.hourly_usd)); } catch (e) { setErr(String(e)); } finally { setBusy(null); poll(); }
  };
  const doDisconnect = async (pause: boolean) => {
    setBusy("disconnecting"); setErr(null);
    try { setSt(await api.cloudDisconnect(pause)); setOpen(false); } catch (e) { setErr(String(e)); } finally { setBusy(null); poll(); }
  };
  const killOrphan = async (podId: string) => {
    try { await api.cloudTerminateOrphan(podId); } finally { pollOrphans(); }
  };

  return (
    <>
      {/* STATUS PILL */}
      <div className="relative">
        <button onClick={() => setOpen((o) => !o)} title="cloud GPU"
          className="flex items-center gap-1.5 h-7 px-2 rounded-md border border-line hover:border-accent">
          <span className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
          <span className="font-mono text-[11px] text-ink-2">{label}</span>
          {st.connected && (
            <span className="font-mono text-[11px] text-ink">{fmtDur(liveUptime)} <span className="text-accent">{fmtCost(liveCost)}</span></span>
          )}
        </button>

        {/* EXPANDED PANEL */}
        {open && (
          <>
            <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
            <div className="absolute right-0 mt-1 z-50 w-[300px] panel p-3 font-mono text-[11px]">
              <div className="flex items-center gap-2 mb-2">
                <span className="flex text-ink-3"><Icon name="cuboid" size={15} /></span>
                <span className="font-display font-semibold text-[12.5px] text-ink">Cloud GPU</span>
                <span className="ml-auto text-ink-3">{fmtCost(st.hourly_usd)}/hr</span>
              </div>

              {!st.configured && (
                <div className="text-warn border border-warn/40 rounded px-2 py-1.5 mb-2">
                  RUNPOD_API_KEY not set on the backend. Export it (and restart) to enable the cloud GPU.
                </div>
              )}

              {st.connected || provisioning || tearing ? (
                <div className="space-y-1.5">
                  <div className="flex items-center"><span className="text-ink-3 w-24">state</span><span className="text-ink-2">{st.state}</span></div>
                  <div className="flex items-center"><span className="text-ink-3 w-24">gpu</span><span className="text-ink-2 truncate">{st.gpu_type ?? "A100"}</span></div>
                  {provisioning && (
                    <div className="flex items-center gap-2 text-warn"><span className="flex animate-pulse"><Icon name="activity" size={13} /></span>cold start, about {st.cold_start_s || 90}s</div>
                  )}
                  {st.connected && (
                    <>
                      <div className="flex items-center"><span className="text-ink-3 w-24">uptime</span><span className="text-ink">{fmtDur(liveUptime)}</span></div>
                      <div className="flex items-center"><span className="text-ink-3 w-24">cost</span><span className="text-accent">{fmtCost(liveCost)}</span><span className="text-ink-3 ml-1">({Math.round(liveUptime)}s x {fmtCost(st.hourly_usd)}/hr)</span></div>
                      <div className="flex items-center"><span className="text-ink-3 w-24">idle stop</span><span className={st.idle_remaining_s != null && st.idle_remaining_s < 120 ? "text-warn" : "text-ink-2"}>{st.idle_remaining_s == null ? "job running" : `in ${fmtDur(st.idle_remaining_s)}`}</span></div>
                      <div className="flex items-center"><span className="text-ink-3 w-24">max session</span><span className="text-ink-2">{st.session_remaining_s == null ? "-" : `in ${fmtDur(st.session_remaining_s)}`}</span></div>
                      {st.last_job_id && <div className="flex items-center"><span className="text-ink-3 w-24">{st.state === "running_job" ? "job" : "last job"}</span><span className="text-info truncate">{st.last_job_id.slice(0, 12)}</span></div>}
                    </>
                  )}
                  <div className="flex gap-2 pt-1.5">
                    <button onClick={() => doDisconnect(false)} disabled={busy != null}
                      className="flex-1 flex items-center justify-center gap-1.5 h-7 rounded-md border border-block text-block hover:bg-block/10 disabled:opacity-40">
                      <Icon name="trash" size={13} />{busy === "disconnecting" ? "terminating..." : "disconnect"}
                    </button>
                    <button onClick={() => doDisconnect(true)} disabled={busy != null} title="keep the pod for a fast reconnect; the network volume still bills"
                      className="h-7 px-2 rounded-md border border-line text-ink-3 hover:border-accent disabled:opacity-40">pause</button>
                  </div>
                  <div className="text-ink-3 text-[10px]">pause keeps the pod for a fast reconnect but still bills volume storage.</div>
                </div>
              ) : (
                <button onClick={() => setConfirm(true)} disabled={!st.configured || busy != null}
                  className="w-full flex items-center justify-center gap-1.5 h-8 rounded-md bg-accent text-bg font-display font-semibold text-[12px] hover:bg-accent/90 disabled:opacity-40">
                  <Icon name="activity" size={14} />{busy === "connecting" ? "provisioning..." : "Connect A100"}
                </button>
              )}
              {err && <div className="text-block mt-2 break-words">{err}</div>}
            </div>
          </>
        )}
      </div>

      {/* CONNECT CONFIRMATION */}
      {confirm && (
        <div className="fixed inset-0 z-[70] bg-bg/70 flex items-center justify-center" onClick={() => setConfirm(false)}>
          <div className="w-[380px] panel" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 px-4 py-3 border-b hairline">
              <span className="flex text-accent"><Icon name="info" size={16} /></span>
              <span className="font-display font-semibold text-[13.5px] text-ink">Fire up the cloud A100</span>
            </div>
            <div className="p-4 flex flex-col gap-3 font-mono text-[11.5px] text-ink-2">
              <div className="flex items-baseline justify-between">
                <span className="text-ink-3">hourly rate</span>
                <span className="text-accent text-[15px]">{fmtCost(st.hourly_usd)}/hr</span>
              </div>
              <div className="flex flex-col gap-1.5 bg-bg-2 border border-line rounded-md p-3 text-[11px]">
                <div className="flex items-center gap-2"><span className="flex text-pass"><Icon name="check" size={13} /></span>Auto-disconnects after {Math.round((st.idle_remaining_s ?? 900) / 60) || 15} min idle.</div>
                <div className="flex items-center gap-2"><span className="flex text-pass"><Icon name="check" size={13} /></span>Hard max-session cap then auto-terminates.</div>
                <div className="flex items-center gap-2"><span className="flex text-pass"><Icon name="check" size={13} /></span>Always terminated on disconnect and on app shutdown.</div>
              </div>
              <div className="text-ink-3 text-[10.5px]">The pod bills until you disconnect or a guard terminates it. Cost stays in view in the top bar.</div>
              <div className="flex gap-2 pt-1">
                <button onClick={() => setConfirm(false)} className="flex-1 h-8 rounded-md border border-line text-ink-2 hover:border-accent">Cancel</button>
                <button onClick={doConnect} className="flex-1 h-8 rounded-md bg-accent text-bg font-display font-semibold hover:bg-accent/90">
                  I understand, connect
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ORPHAN BANNER */}
      {orphans.length > 0 && (
        <div className="fixed top-0 left-0 right-0 z-[80] bg-block/15 border-b border-block flex items-center gap-3 px-4 py-2 font-mono text-[11px]">
          <span className="flex text-block"><Icon name="flag" size={15} /></span>
          <span className="text-block font-semibold">
            {orphans.length} cloud GPU pod{orphans.length > 1 ? "s" : ""} running with no session, billing{" "}
            <span className="text-ink">{fmtCost(orphans.reduce((a, o) => a + o.est_cost, 0))}</span> so far.
          </span>
          <div className="ml-auto flex items-center gap-2">
            <button onClick={doConnect} className="h-6 px-2 rounded border border-line text-ink-2 hover:border-accent">reconnect</button>
            {orphans.map((o) => (
              <button key={o.pod_id} onClick={() => killOrphan(o.pod_id)}
                className="h-6 px-2 rounded border border-block text-block hover:bg-block/10">
                terminate {o.pod_id.slice(0, 8)}
              </button>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
