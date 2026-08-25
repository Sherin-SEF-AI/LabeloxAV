// Which ontology attributes apply to a class.
//
// The rule is a union of two layers and it has to give the same answer on both sides of the wire: the
// server refuses an attribute the client offered, so a client that computes the scope differently produces
// a control that always 400s. `services/autolabel/ontology.py:attrs_for_class` is the other half, and the
// two are kept deliberately in the same shape.

import type { Ontology } from "@/lib/types";

/** Attribute names applicable to a class, or null when every attribute applies. */
export function attrsForClass(onto: Ontology, classId: number): string[] | null {
  const c = onto.classes.find((k) => k.id === classId);
  if (!c) return null;
  const base = onto.attribute_scope?.[c.l1];
  // An unscoped l1 already means "everything applies", so per-class extras add nothing to it.
  if (!base) return null;
  const extra = onto.attribute_scope_class?.[c.name];
  if (!extra?.length) return base;
  return [...base, ...extra.filter((a) => !base.includes(a))];
}
