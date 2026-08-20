"use client";

import { useRouter } from "next/navigation";

// What an action actually returned, instead of a sentence about it.
//
// Every one of the twenty-one actions on the agent console reported through a single `setMsg` line into a
// transcript at the bottom of the page, well below the button that produced it. Two things went wrong at
// once. The result was out of sight: you pressed a button, it said "mining...", and then nothing visible
// changed anywhere near you. And the line was a summary, so most of the payload was discarded at the point
// of rendering - the coverage report returns class balance, per-axis scene coverage and a geo histogram,
// and the page printed ten strings from one of its five fields.
//
// The worse half is the collections. "mined 47 safety scenarios, see Scenarios" names a count and then
// names a destination in prose, without linking to it. "would auto-accept 12, review 30, annotate 8 across
// 50 top-value frames" is a dry-run preview whose entire value is WHICH fifty frames.
//
// So this renders the whole payload, and does it generically rather than as twenty-one bespoke views: a
// walker over the returned JSON that knows how to draw the shapes these endpoints actually return
// (counters, string lists, nested records) and, crucially, turns anything that looks like an object or
// frame id into something you can click. That is what makes a result reachable rather than merely visible:
// a uuid in a payload is a thing in the corpus, and it should behave like one wherever it appears.

export type ActionResult = {
  /** The handler that produced it, used to pick a headline and any bespoke section. */
  kind: string;
  /** What the button said, so the panel names the thing you pressed. */
  label: string;
  data: unknown;
  at: number;
  /** Where the produced items went, when the action fills a queue on another page. */
  destination?: { href: string; label: string };
};

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// Keys whose values are ids of a known kind. A uuid alone does not say what it identifies, and opening a
// session id in the object inspector would 404 in a way that reads as missing data rather than a wrong link.
const OBJECT_KEYS = new Set(["object_id", "object", "objects"]);
const FRAME_KEYS = new Set(["frame_id", "frame", "frames", "frame_ids"]);

function isNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="border hairline px-2 py-1">
      <div className="font-mono text-[9px] uppercase tracking-wide text-ink-3 truncate">{label}</div>
      <div className="font-mono text-[13px] text-ink tabular-nums">{value}</div>
    </div>
  );
}

/** Recursive renderer. Numbers become stat chips, string lists become lists, records become rows. */
function Value({ name, value, depth, onOpen }: {
  name: string; value: unknown; depth: number;
  onOpen: (kind: "object" | "frame", id: string) => void;
}) {
  if (value === null || value === undefined) {
    // Null is a fact about the result and is drawn as one. Hiding it makes an absent field and an absent
    // measurement look the same.
    return <div className="font-mono text-[11px] text-ink-3">{`${name}: none`}</div>;
  }

  if (typeof value === "boolean") {
    return (
      <div className="font-mono text-[11px]">
        <span className="text-ink-3">{name}: </span>
        <span className={value ? "text-pass" : "text-ink-3"}>{String(value)}</span>
      </div>
    );
  }

  if (isNum(value)) return <Stat label={name} value={value.toLocaleString()} />;

  if (typeof value === "string") {
    const kind = OBJECT_KEYS.has(name) ? "object" : FRAME_KEYS.has(name) ? "frame" : null;
    if (kind && UUID.test(value)) {
      return (
        <button onClick={() => onOpen(kind, value)}
          title={`show this ${kind}`}
          className="font-mono text-[11px] text-accent hover:text-accent-2 underline decoration-dotted">
          {name}: {value.slice(0, 8)}
        </button>
      );
    }
    return (
      <div className="font-mono text-[11px] break-words">
        <span className="text-ink-3">{name}: </span><span className="text-ink-2">{value}</span>
      </div>
    );
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <div className="font-mono text-[11px] text-ink-3">{`${name}: empty`}</div>;
    }
    const ids = FRAME_KEYS.has(name) ? "frame" : OBJECT_KEYS.has(name) ? "object" : null;
    return (
      <div className="space-y-1">
        <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3">{name} ({value.length})</div>
        <div className={ids ? "flex flex-wrap gap-1" : "space-y-1"}>
          {value.slice(0, 200).map((v, i) =>
            ids && typeof v === "string" && UUID.test(v) ? (
              <button key={i} onClick={() => onOpen(ids, v)}
                className="font-mono text-[10px] border border-line px-1.5 py-0.5 text-accent hover:border-accent">
                {v.slice(0, 8)}
              </button>
            ) : typeof v === "string" ? (
              <div key={i} className="font-mono text-[11px] text-ink-2">· {v}</div>
            ) : (
              <div key={i} className="border-l border-line pl-2">
                <Value name={`#${i + 1}`} value={v} depth={depth + 1} onOpen={onOpen} />
              </div>
            ))}
        </div>
        {value.length > 200 && (
          <div className="font-mono text-[10px] text-ink-3">
            {(value.length - 200).toLocaleString()} more not drawn
          </div>
        )}
      </div>
    );
  }

  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    // A flat record of numbers is a histogram, which is worth drawing as bars rather than as a list of
    // key/value pairs: by_kind, decisions, geo and scene_coverage are all this shape.
    const allNums = entries.length > 0 && entries.every(([, v]) => isNum(v));
    if (allNums) {
      const max = Math.max(...entries.map(([, v]) => v as number), 1);
      return (
        <div className="space-y-1">
          <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3">{name}</div>
          {entries.sort((a, b) => (b[1] as number) - (a[1] as number)).map(([k, v]) => (
            <div key={k} className="flex items-center gap-2 font-mono text-[10.5px]">
              <span className="text-ink-2 w-28 truncate shrink-0" title={k}>{k}</span>
              <div className="flex-1 h-1.5 bg-line min-w-[2rem]">
                <div className="h-full bg-accent" style={{ width: `${((v as number) / max) * 100}%` }} />
              </div>
              <span className="text-ink tabular-nums w-14 text-right shrink-0">{(v as number).toLocaleString()}</span>
            </div>
          ))}
        </div>
      );
    }
    return (
      <div className={depth > 0 ? "space-y-1.5 border-l border-line pl-2" : "space-y-1.5"}>
        <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3">{name}</div>
        {entries.map(([k, v]) => (
          <Value key={k} name={k} value={v} depth={depth + 1} onOpen={onOpen} />
        ))}
      </div>
    );
  }

  return null;
}

export default function ActionResultPanel({ result, onOpenObject, onOpenFrame }: {
  result: ActionResult | null;
  onOpenObject: (id: string) => void;
  onOpenFrame: (id: string) => void;
}) {
  const router = useRouter();

  if (!result) {
    return (
      <div className="p-4 font-mono text-[11px] text-ink-3 leading-relaxed">
        Run an action and its full result appears here.
        <div className="mt-2 text-ink-3">
          Every field it returned, not a summary of it, with anything it names that exists in the corpus
          made clickable.
        </div>
      </div>
    );
  }

  const data = result.data as Record<string, unknown> | null;
  const top = data && typeof data === "object" ? Object.entries(data) : [];
  // Scalars first, as chips; the structures below them. Otherwise a five-number headline is buried under a
  // histogram that happened to be declared before it.
  const scalars = top.filter(([, v]) => isNum(v) || typeof v === "boolean");
  const rest = top.filter(([, v]) => !(isNum(v) || typeof v === "boolean"));

  const onOpen = (kind: "object" | "frame", id: string) =>
    kind === "object" ? onOpenObject(id) : onOpenFrame(id);

  return (
    <div className="p-3 space-y-3">
      <div>
        <div className="font-mono text-[11px] text-ink">{result.label}</div>
        <div className="font-mono text-[10px] text-ink-3">
          {new Date(result.at).toLocaleTimeString()}
        </div>
      </div>

      {result.destination && (
        // The message used to name the destination in prose. A count plus the word "Scenarios" is not a
        // way to get to them.
        <button onClick={() => router.push(result.destination!.href)}
          className="w-full border border-accent/50 bg-accent/10 text-accent px-2 py-1 rounded font-mono text-[11px] hover:bg-accent/20">
          {result.destination.label}
        </button>
      )}

      {scalars.length > 0 && (
        <div className="grid grid-cols-2 gap-1">
          {scalars.map(([k, v]) => (
            isNum(v) ? <Stat key={k} label={k} value={v.toLocaleString()} />
              : <Stat key={k} label={k} value={<span className={v ? "text-pass" : "text-ink-3"}>{String(v)}</span>} />
          ))}
        </div>
      )}

      <div className="space-y-2.5">
        {rest.map(([k, v]) => <Value key={k} name={k} value={v} depth={0} onOpen={onOpen} />)}
      </div>

      {top.length === 0 && (
        <div className="font-mono text-[11px] text-ink-3">
          The action returned nothing to show. That is the result, not a failure to render one.
        </div>
      )}
    </div>
  );
}
