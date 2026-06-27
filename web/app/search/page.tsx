"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import type { SimilarResponse } from "@/lib/types";
import TopNav from "@/components/TopNav";

// M1.2 similarity search: find visually (DINOv3) or semantically (SigLIP 2) similar frames by uploading
// an image or opening a frame. M1.4 adds natural-language text search to this surface.

export default function SearchPage() {
  const router = useRouter();
  const frameParam = useSearchParams().get("frame");
  const [res, setRes] = useState<SimilarResponse | null>(null);
  const [mode, setMode] = useState<"visual" | "semantic">("visual");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [parsed, setParsed] = useState<{ filters: Record<string, string>; classes: string[] } | null>(null);

  const runText = useCallback(async () => {
    if (!q.trim()) return;
    setBusy(true);
    setNote(null);
    try {
      const r = await api.searchSemantic(q, 24);
      setParsed({ filters: r.filters, classes: r.classes });
      setRes({ kind: "frame", mode: "semantic", results: r.results });
    } catch (e) {
      setNote(String(e));
    } finally {
      setBusy(false);
    }
  }, [q]);

  const runFrame = useCallback(async (fid: string) => {
    setBusy(true);
    setNote(null);
    try {
      setRes(await api.searchSimilar({ frame_id: fid, mode, k: 24 }));
    } catch (e) {
      setNote(String(e));
    } finally {
      setBusy(false);
    }
  }, [mode]);

  useEffect(() => {
    if (frameParam) runFrame(frameParam);
  }, [frameParam, runFrame]);

  const onFile = async (f: File) => {
    setBusy(true);
    setNote(null);
    const b64 = await new Promise<string>((resolve) => {
      const r = new FileReader();
      r.onload = () => resolve(String(r.result).split(",")[1]);
      r.readAsDataURL(f);
    });
    try {
      setRes(await api.searchSimilar({ image_b64: b64, mode, k: 24 }));
    } catch (e) {
      setNote(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="SEARCH" />
      <main className="flex-1 overflow-auto p-4 space-y-4">
        <div className="panel p-3 space-y-2">
          <div className="flex items-center gap-2">
            <input value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && runText()}
              placeholder='natural language, e.g. "night rain autorickshaw" or "dense urban traffic"'
              className="flex-1 bg-bg border border-line px-3 py-1.5 font-mono text-xs text-ink" />
            <button onClick={runText} disabled={busy}
              className="border border-accent text-accent px-3 py-1.5 font-mono text-xs hover:bg-accent/10 disabled:opacity-50">search</button>
          </div>
          {parsed && (parsed.classes.length > 0 || Object.keys(parsed.filters).length > 0) && (
            <div className="font-mono text-[10px] text-ink-3">
              parsed:{" "}
              {Object.entries(parsed.filters).map(([a, v]) => <span key={a} className="text-info mr-2">{a}={v}</span>)}
              {parsed.classes.map((c) => <span key={c} className="text-pass mr-2">class={c}</span>)}
            </div>
          )}
        </div>

        <div className="panel p-3 flex items-center gap-3 flex-wrap font-mono text-[11px]">
          <span className="text-ink-3">find similar frames by</span>
          <label className="border border-line px-2 py-1 hover:border-accent cursor-pointer">
            upload image
            <input type="file" accept="image/*" className="hidden"
              onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
          </label>
          {frameParam && <span className="text-ink-3">frame {frameParam.slice(0, 8)}</span>}
          <span className="ml-2 text-ink-3">mode:</span>
          {(["visual", "semantic"] as const).map((m) => (
            <button key={m} onClick={() => setMode(m)}
              className={`border px-2 py-1 ${mode === m ? "border-accent text-accent" : "border-line text-ink-3 hover:text-ink-2"}`}>{m}</button>
          ))}
          {busy && <span className="text-warn">searching…</span>}
          {note && <span className="text-block">{note}</span>}
        </div>

        {res && res.results.length > 0 ? (
          <div className="panel p-3">
            <div className="font-mono text-[11px] text-ink-3 mb-2">{res.results.length} similar {res.kind}s ({res.mode}, DINOv3/SigLIP2)</div>
            <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-8 gap-2">
              {res.results.map((r, i) => (
                <button key={i} onClick={() => r.frame_id && router.push(`/frame/${r.frame_id}`)}
                  className="border border-line hover:border-accent" title={`sim ${r.score.toFixed(3)}`}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={r.image_url || r.crop_url} alt="" className="w-full h-20 object-cover bg-bg-2" />
                  <div className="font-mono text-[9px] text-ink-3 px-1 py-0.5">{r.score.toFixed(3)}</div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          !busy && (
            <div className="panel px-3 py-10 text-center font-mono text-xs text-ink-3">
              {res ? "no similar frames found." : "upload an image, or open a frame and click “find similar”, to see visual neighbors."}
            </div>
          )
        )}
      </main>
    </div>
  );
}
