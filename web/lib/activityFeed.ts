// The activity feed, read as sentences rather than as the codes it is stored in.
//
// The server records a review as `<action> <class>` ("revert cattle", "adjust_geometry bus", "confirm 7
// objects"), which is exact and is not how a person describes what they did. A review session also records one
// row per object, so the feed for an afternoon was the same line printed four hundred times. This turns each
// row into a sentence and folds runs of the same thing by the same person into one line with a count.
//
// Pure functions, tested without a DOM like the rest of web/lib.

import type { ActivityEvent } from "@/lib/types";
import { humanize, num } from "@/lib/format";

// Review actions in the past tense. The value after it is a class name or "<n> objects".
const PAST: Record<string, string> = {
  confirm: "Confirmed", accept: "Accepted", reject: "Rejected", revert: "Reverted",
  adjust_geometry: "Adjusted the shape of", reclassify: "Reclassified", reclassify_track: "Reclassified a track of",
  create: "Drew", delete: "Deleted", error_fix: "Fixed", split_track: "Split a track of",
  merge_track: "Merged tracks of", edit: "Edited", tag: "Tagged",
};

// Class names that are acronyms read in capitals, and take "an" when the letter is said with a vowel sound.
const ACRONYMS = new Set(["suv", "lcv", "hcv", "mav", "atv", "ev", "ups"]);
const cased = (cls: string) => cls.split(" ").map((w) => (ACRONYMS.has(w) ? w.toUpperCase() : w)).join(" ");
const article = (word: string) => (/^[aeiouAEIOU]|^[FHLMNRSX][A-Z]/.test(word) ? "an" : "a");

export interface ActivityLine {
  key: string;
  text: string;           // the sentence, e.g. "Reverted 14 cattle labels"
  count: number;          // how many raw events this line stands for
  user: string | null;
  first: ActivityEvent;   // the newest event of the run (the feed is newest first)
  last: ActivityEvent;    // the oldest event of the run
  frames: number;         // distinct frames the run touched, when the events link to frames
}

/** One event (or a run of `times` identical events) as a sentence. */
export function sentence(e: Pick<ActivityEvent, "summary" | "label" | "verb">, times = 1): string {
  const raw = (e.summary || e.label || humanize(e.verb) || "").trim();
  const [head, ...rest] = raw.split(/\s+/);
  const verb = PAST[head];
  if (!verb) {
    // Free-text summaries ("took over editing a frame from meera") only need a capital and the count.
    const text = /^[a-z]+(_[a-z]+)+$/.test(head) ? [humanize(head), ...rest].join(" ") : raw;
    const s = text ? text[0].toUpperCase() + text.slice(1) : "";
    return times > 1 ? `${s} (${num(times)} times)` : s;
  }
  // "<n> objects [as <class>]": a batch from the review grid.
  const batch = /^(\d+) objects?(?: as (.+))?$/.exec(rest.join(" "));
  if (batch) {
    const n = Number(batch[1]) * times;
    const as = batch[2] ? ` as ${cased(humanize(batch[2]).toLowerCase())}` : "";
    const runs = times > 1 ? ` in ${num(times)} batches` : "";
    return `${verb} ${num(n)} ${n === 1 ? "object" : "objects"}${as}${runs}`;
  }
  const cls = cased(humanize(rest.join(" ")).toLowerCase());
  if (!cls) return times > 1 ? `${verb} ${num(times)} labels` : `${verb} a label`;
  return times > 1 ? `${verb} ${num(times)} ${cls} labels` : `${verb} ${article(cls)} ${cls} label`;
}

const frameOf = (href: string | null | undefined) => (href && href.startsWith("/frame/") ? href : null);

/** Fold consecutive events with the same person, verb and summary into one line each. */
export function groupActivity(events: ActivityEvent[]): ActivityLine[] {
  const out: { evs: ActivityEvent[] }[] = [];
  for (const e of events) {
    const prev = out[out.length - 1]?.evs[0];
    if (prev && prev.user_name === e.user_name && prev.verb === e.verb && prev.summary === e.summary) {
      out[out.length - 1].evs.push(e);
    } else {
      out.push({ evs: [e] });
    }
  }
  return out.map(({ evs }) => {
    const frames = new Set(evs.map((e) => frameOf(e.href)).filter(Boolean));
    return {
      key: evs[0].event_id,
      text: sentence(evs[0], evs.length),
      count: evs.length,
      user: evs[0].user_name ?? null,
      first: evs[0],
      last: evs[evs.length - 1],
      frames: frames.size,
    };
  });
}

/** A verb's label for the summary chips: "reviewed an object" -> "Reviewed an object", `took_over_frame` -> "Took over frame". */
export function verbLabel(verb: string, label: string | undefined): string {
  const l = label && label !== verb ? label : humanize(verb);
  return l ? l[0].toUpperCase() + l.slice(1) : verb;
}

/** 24 -> "day", 72 -> "3 days", 168 -> "week", 720 -> "month". */
export function windowName(hours: number): string {
  if (hours === 24) return "day";
  if (hours === 168) return "week";
  if (hours === 720) return "month";
  return hours % 24 === 0 ? `${hours / 24} days` : `${hours} hours`;
}
