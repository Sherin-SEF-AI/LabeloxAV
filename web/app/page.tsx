"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { SessionRow, TriageRow } from "@/lib/types";
import { ConfBar, StateBadge } from "@/components/StateBadge";
import TopNav from "@/components/TopNav";
import CorrectionModal from "@/components/CorrectionModal";
import { acceptState, getUser } from "@/lib/user";

const BANDS = [
  { key: "review,annotate", label: "My Queue" },
  { key: "review", label: "Review band" },
  { key: "annotate", label: "Annotate band" },
  { key: "submitted", label: "QA queue" },
];

export default function TriagePage() {
  const router = useRouter();
  const [rows, setRows] = useState<TriageRow[]>([]);
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [session, setSession] = useState<string>("");
  const [states, setStates] = useState<string>("review,annotate");
  const [cursor, setCursor] = useState(0);
  const [loading, setLoading] = useState(false);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [bulkClass, setBulkClass] = useState("");
  const [classes, setClasses] = useState<string[]>([]);
  const [corr, setCorr] = useState<{ objectId: string; old: string; to: string } | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [role, setRole] = useState<string | undefined>(undefined);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params: Record<string, string> = { states, limit: "200" };
      if (session) params.session_id = session;
      setRows(await api.triage(params));
      setCursor(0);
    } finally {
      setLoading(false);
    }
  }, [states, session]);

  useEffect(() => {
    api.sessions().then(setSessions).catch(() => {});
    api.ontology().then((o) => setClasses(o.classes.map((c) => c.name))).catch(() => {});
    setRole(getUser()?.role);
  }, []);
  useEffect(() => {
    load();
    setSel(new Set());
  }, [load]);

  const toggle = (id: string) =>
    setSel((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const allSelected = rows.length > 0 && rows.every((r) => sel.has(r.object_id));
  const selectAll = () => setSel(allSelected ? new Set() : new Set(rows.map((r) => r.object_id)));

  const bulk = useCallback(
    async (action: string, className?: string, state?: string) => {
      if (!sel.size) return;
      setMsg(null);
      try {
        const r = await api.bulkReview([...sel], action, className, state);
        setMsg(`${action}: ${r.updated} objects`);
        setSel(new Set());
        load();
      } catch (e) {
        setMsg(String(e));
      }
    },
    [sel, load],
  );

  const autolabel = useCallback(async (target: "local" | "cloud" = "local") => {
    if (!session) return;
    try {
      const r = await api.startAutolabel(session, undefined, target);
      setMsg(r.status === "queued-cloud"
        ? "queued for the cloud A100 (heavy stack: SAM 3.1 + Qwen3-VL + YOLO26) - see Jobs"
        : "autolabel queued - watch it on Jobs");
    } catch (e) {
      setMsg(String(e).includes("503") ? "GPU busy (training). Try after it finishes." : String(e));
    }
  }, [session]);

  const vlmQa = useCallback(async () => {
    if (!session) return;
    try {
      await api.startVlmQa(session);
      setMsg("VLM auto-QA running - it flags likely-wrong labels into the QA queue + fills attributes");
    } catch (e) {
      setMsg(String(e).includes("503") ? "GPU busy (training). Try after it finishes." : String(e));
    }
  }, [session]);

  const recognizeSigns = useCallback(async () => {
    if (!session) return;
    try {
      const r = await api.recognizeSigns(session);
      setMsg(`typed ${r.recognized} signs (Indian taxonomy), ${r.text_bearing} text-bearing routed to OCR`);
    } catch (e) {
      setMsg(String(e));
    }
  }, [session]);

  const open = useCallback(
    (r?: TriageRow) => {
      const row = r ?? rows[cursor];
      // Open the frame-centric editor focused on the clicked object (the pro annotation surface).
      if (row) router.push(`/frame/${row.frame_id}?focus=${row.object_id}`);
    },
    [rows, cursor, router],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "j") setCursor((c) => Math.min(c + 1, rows.length - 1));
      else if (e.key === "k") setCursor((c) => Math.max(c - 1, 0));
      else if (e.key === "Enter") open();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [rows.length, open]);

  const header = useMemo(
    () => (session ? sessions.find((s) => s.session_id === session) : null),
    [session, sessions],
  );

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav
        active="TRIAGE"
        right={
          <>
            <span className="text-ink-2">
              {header ? `${header.vehicle_id} · ${header.city ?? ""}` : "ALL SESSIONS"}
            </span>
            <span className="border border-line px-2 py-0.5">QUEUE {rows.length}</span>
            <span className={`w-2 h-2 rounded-full ${loading ? "bg-warn" : "bg-pass"}`} />
          </>
        }
      />

      <div className="flex flex-1 min-h-0">
        <aside className="w-56 border-r hairline p-3 space-y-4 text-sm">
          <div>
            {BANDS.map((b) => (
              <button
                key={b.key}
                onClick={() => setStates(b.key)}
                className={`block w-full text-left px-2 py-1 ${
                  states === b.key ? "text-accent" : "text-ink-2 hover:text-ink"
                }`}
              >
                {states === b.key ? "■ " : "  "}
                {b.label}
              </button>
            ))}
          </div>
          <div className="space-y-1">
            <div className="font-mono text-[11px] text-ink-3 uppercase">Filters</div>
            <select
              value={session}
              onChange={(e) => setSession(e.target.value)}
              className="w-full bg-panel border hairline text-ink text-xs px-2 py-1"
            >
              <option value="">all sessions</option>
              {sessions.map((s) => (
                <option key={s.session_id} value={s.session_id}>
                  {s.vehicle_id} · {s.city ?? "?"}
                </option>
              ))}
            </select>
            <button onClick={() => autolabel("local")} disabled={!session}
              title="run autolabel (Path A/B/C + fusion + gate) on this session locally as a background job"
              className="w-full font-mono text-[11px] border border-line text-ink-2 px-2 py-1 hover:border-accent disabled:opacity-40">
              autolabel this session
            </button>
            <button disabled
              title="Cloud A100 dispatch (SAM 3.1 PCS + Qwen3-VL + YOLO26) is not configured in this build. Use 'autolabel this session' to run the local stack now."
              className="w-full font-mono text-[11px] border border-line text-ink-3 px-2 py-1 opacity-40 cursor-not-allowed">
              autolabel on cloud (A100) - not configured
            </button>
            <button onClick={recognizeSigns} disabled={!session}
              title="type traffic signs into the Indian RTO taxonomy (SigLIP2 zero-shot); route text-bearing to OCR"
              className="w-full font-mono text-[11px] border border-line text-ink-2 px-2 py-1 hover:border-accent disabled:opacity-40">
              recognize signs
            </button>
            <button onClick={vlmQa} disabled={!session}
              title="VLM auto-QA: a model-as-reviewer flags likely-wrong labels into the QA queue and pre-fills attributes"
              className="w-full font-mono text-[11px] border border-line text-ink-2 px-2 py-1 hover:border-accent disabled:opacity-40">
              VLM auto-QA this session
            </button>
            {msg && <div className="font-mono text-[10px] text-warn">{msg}</div>}
          </div>
        </aside>

        <main className="flex-1 min-w-0 overflow-auto">
          {sel.size > 0 && (
            <div className="sticky top-0 z-20 flex items-center gap-2 px-3 py-1.5 bg-panel border-b hairline font-mono text-[11px]">
              <span className="text-ink">{sel.size} selected</span>
              <button onClick={() => bulk("accept", undefined, acceptState(role))}
                className="border border-pass text-pass px-2 py-0.5"
                title={role === "annotator" ? "submit for QA" : "approve (accept)"}>
                {role === "annotator" ? "submit" : "approve"}
              </button>
              <button onClick={() => bulk("reject")} className="border border-block text-block px-2 py-0.5">reject</button>
              <input list="cls" value={bulkClass} onChange={(e) => setBulkClass(e.target.value)} placeholder="relabel to class"
                className="bg-bg border border-line px-2 py-0.5 w-40 text-ink" />
              <datalist id="cls">{classes.map((c) => <option key={c} value={c} />)}</datalist>
              <button onClick={() => bulk("reclassify", bulkClass)} disabled={!classes.includes(bulkClass)}
                className="border border-line px-2 py-0.5 hover:border-accent disabled:opacity-40">relabel</button>
              {sel.size === 1 && classes.includes(bulkClass) && (
                <button
                  onClick={() => {
                    const row = rows.find((r) => sel.has(r.object_id));
                    if (row) setCorr({ objectId: row.object_id, old: row.class_name, to: bulkClass });
                  }}
                  title="find visually-similar objects of the old class and bulk-fix them too"
                  className="border border-accent text-accent px-2 py-0.5 hover:bg-accent/10">correct similar →</button>
              )}
              <button onClick={() => setSel(new Set())} className="text-ink-3 hover:text-ink ml-auto">clear</button>
            </div>
          )}
          <table className="w-full text-sm">
            <thead className="text-ink-3 font-mono text-[11px] uppercase border-b hairline sticky top-0 bg-bg">
              <tr>
                <th className="px-3 py-2 w-8"><input type="checkbox" checked={allSelected} onChange={selectAll} /></th>
                <th className="text-left font-normal px-3 py-2 w-10">#</th>
                <th className="text-left font-normal px-3 py-2">object / why</th>
                <th className="text-left font-normal px-3 py-2">class (proposed)</th>
                <th className="text-left font-normal px-3 py-2 w-32">conf</th>
                <th className="text-left font-normal px-3 py-2 w-28">state</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr
                  key={r.object_id}
                  onClick={() => {
                    setCursor(i);
                    open(r);
                  }}
                  onMouseEnter={() => setCursor(i)}
                  className={`border-b hairline cursor-pointer ${
                    sel.has(r.object_id) ? "bg-bg-2" : i === cursor ? "bg-panel" : "hover:bg-bg-2"
                  }`}
                >
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <input type="checkbox" checked={sel.has(r.object_id)} onChange={() => toggle(r.object_id)} />
                  </td>
                  <td className="px-3 py-2 font-mono text-ink-3">{String(i + 1).padStart(2, "0")}</td>
                  <td className="px-3 py-2">
                    <div className="font-mono text-xs">{r.object_id.slice(0, 8)}</div>
                    <div className="text-ink-3 text-xs">{r.why}</div>
                  </td>
                  <td className="px-3 py-2 font-mono">{r.class_name}</td>
                  <td className="px-3 py-2">
                    <ConfBar conf={r.conf} />
                  </td>
                  <td className="px-3 py-2">
                    <StateBadge state={r.state} />
                  </td>
                </tr>
              ))}
              {!rows.length && (
                <tr>
                  <td colSpan={6} className="px-3 py-8 text-center text-ink-3">
                    queue empty for this filter
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </main>
      </div>

      <footer className="h-8 border-t hairline flex items-center px-4 gap-4 font-mono text-[11px] text-ink-3">
        <span>J/K next/prev</span>
        <span>ENTER open</span>
        <span className="ml-auto">ranked by uncertainty x rarity</span>
      </footer>
      {corr && (
        <CorrectionModal
          objectId={corr.objectId}
          kind="class"
          change={{ old: corr.old, new: corr.to }}
          onClose={() => setCorr(null)}
          onApplied={(n) => { setMsg(`corrected ${n} similar objects`); setCorr(null); setSel(new Set()); load(); }}
        />
      )}
    </div>
  );
}
