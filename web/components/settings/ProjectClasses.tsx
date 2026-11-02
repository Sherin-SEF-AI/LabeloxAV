"use client";

import { useEffect, useMemo, useState } from "react";

import { api, apiGet, apiPut, humanizeError } from "@/lib/api";
import { classColor } from "@/lib/colors";
import { toast } from "@/lib/toast";
import type { OntologyClass } from "@/lib/types";

// Which classes a project labels (services/project_classes.py). The ontology is shared and its ids are global;
// a project only narrows what its editor offers and what auto-labelling writes. Ignore regions and the team's
// own custom classes are in every set, so they are shown checked and locked.

export type ClassSet = "india" | "generic" | "custom";
type Template = { name: "india" | "generic"; class_ids: number[]; all: boolean };

const CHOICES: { set: ClassSet; label: string; hint: string }[] = [
  { set: "india", label: "Indian roads", hint: "Every class, including autorickshaws, cycle rickshaws, handcarts and cattle. The default." },
  { set: "generic", label: "Generic driving", hint: "Road users and infrastructure found anywhere, without the Indian long tail." },
  { set: "custom", label: "Custom", hint: "Choose class by class." },
];

const always = (c: OntologyClass) => c.l0 === "ignore" || c.l1 === "custom";

export default function ProjectClasses({ projectId, projectName, classSet, classCount, of, onChanged }: {
  projectId: string; projectName: string; classSet: ClassSet; classCount: number; of: number; onChanged: () => void;
}) {
  const [classes, setClasses] = useState<OntologyClass[] | null>(null);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [choice, setChoice] = useState<ClassSet>(classSet);
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const [o, t, mine] = await Promise.all([
          api.ontology(),
          apiGet<{ templates: Template[] }>("/api/ontology/templates"),
          apiGet<{ active_class_ids: number[] | null }>(`/api/ontology?project_id=${projectId}`),
        ]);
        if (!live) return;
        setClasses(o.classes);
        setTemplates(t.templates);
        setPicked(new Set(mine.active_class_ids ?? o.classes.map((c) => c.id)));
      } catch (e) { toast(humanizeError(e), "error"); }
    })();
    return () => { live = false; };
  }, [projectId]);
  useEffect(() => setChoice(classSet), [classSet]);

  const groups = useMemo(() => {
    if (!classes) return [];
    const q = query.trim().toLowerCase().replace(/[\s-]+/g, "_");
    const by = new Map<string, OntologyClass[]>();
    for (const c of classes) {
      if (q && !c.name.includes(q) && !c.l1.includes(q)) continue;
      const k = `${c.l0} / ${c.l1}`;
      by.set(k, [...(by.get(k) ?? []), c]);
    }
    return [...by.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [classes, query]);

  const save = async (set: ClassSet) => {
    setBusy(true);
    try {
      const body = set === "custom" ? { class_ids: [...picked] } : { template: set };
      const r = await apiPut<{ class_count: number; of: number }>(`/api/projects/${projectId}/classes`, body);
      toast(`${projectName} labels ${r.class_count} of ${r.of} classes`, "success");
      onChanged();
    } catch (e) { toast(humanizeError(e), "error"); }
    finally { setBusy(false); }
  };

  const pick = (set: ClassSet) => {
    setChoice(set);
    if (set !== "custom") {
      const t = templates.find((x) => x.name === set);
      if (t) setPicked(new Set(t.class_ids));
      if (set !== classSet) void save(set);
    }
  };

  const toggle = (c: OntologyClass) => setPicked((cur) => {
    const next = new Set(cur);
    if (next.has(c.id)) next.delete(c.id); else next.add(c.id);
    return next;
  });
  const toggleGroup = (items: OntologyClass[], on: boolean) => setPicked((cur) => {
    const next = new Set(cur);
    for (const c of items) if (!always(c)) { if (on) next.add(c.id); else next.delete(c.id); }
    return next;
  });

  const countOf = (set: ClassSet) => set === "custom" ? picked.size : (templates.find((t) => t.name === set)?.class_ids.length ?? 0);

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-[13px] font-semibold text-ink">
        Classes <span className="font-normal text-ink-3 tabular-nums">({classCount} of {of})</span>
      </h3>
      <p className="text-[12px] text-ink-3">
        What the editor offers and what auto-labelling writes for this project&apos;s sessions. Detections of other classes are left out; labels already saved keep their class.
      </p>
      <div className="grid gap-2 sm:grid-cols-3" role="radiogroup" aria-label="Class set">
        {CHOICES.map((c) => (
          <label key={c.set} className={`flex items-start gap-2.5 rounded border px-3 py-2.5 cursor-pointer ${choice === c.set ? "border-accent/60 bg-accent/5" : "hairline"}`}>
            <input type="radio" name={`classes-${projectId}`} className="mt-0.5" checked={choice === c.set}
                   disabled={busy} onChange={() => pick(c.set)} />
            <span className="flex flex-col min-w-0">
              <span className="text-[13px] text-ink">{c.label} <span className="text-ink-3 tabular-nums">{countOf(c.set) || ""}</span></span>
              <span className="text-[12px] text-ink-3">{c.hint}</span>
            </span>
          </label>
        ))}
      </div>

      {choice === "custom" && classes && (
        <div className="flex flex-col gap-2 border hairline rounded p-3">
          <div className="flex gap-2">
            <input className="field flex-1 min-w-0" value={query} onChange={(e) => setQuery(e.target.value)}
                   placeholder="Filter by class or group, e.g. two_wheeler" aria-label="Filter classes" />
            <button className="btn btn-primary shrink-0" disabled={busy} onClick={() => void save("custom")}>
              Save {picked.size} classes
            </button>
          </div>
          <div className="max-h-80 overflow-y-auto flex flex-col gap-3 pr-1">
            {groups.map(([group, items]) => {
              const on = items.filter((c) => picked.has(c.id)).length;
              return (
                <fieldset key={group} className="flex flex-col gap-1">
                  <legend className="flex w-full items-center gap-2 text-[11px] uppercase tracking-[0.06em] text-ink-3">
                    <span>{group}</span>
                    <span className="tabular-nums normal-case tracking-normal">{on}/{items.length}</span>
                    <button type="button" className="ml-auto normal-case tracking-normal text-accent hover:text-accent-2"
                            onClick={() => toggleGroup(items, on < items.length)}>
                      {on < items.length ? "all" : "none"}
                    </button>
                  </legend>
                  <div className="flex flex-wrap gap-1.5">
                    {items.map((c) => {
                      const locked = always(c);
                      const checked = locked || picked.has(c.id);
                      return (
                        <label key={c.id} title={locked ? "In every project" : undefined}
                               className={`inline-flex items-center gap-1.5 rounded border px-2 py-1 text-[12px] ${checked ? "border-accent/50 text-ink" : "hairline text-ink-3"} ${locked ? "opacity-70" : "cursor-pointer"}`}>
                          <input type="checkbox" className="sr-only" checked={checked} disabled={locked}
                                 onChange={() => toggle(c)} />
                          <span className="inline-block h-2 w-2 rounded-full" style={{ background: classColor(c.id) }} aria-hidden />
                          {c.name}
                        </label>
                      );
                    })}
                  </div>
                </fieldset>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
