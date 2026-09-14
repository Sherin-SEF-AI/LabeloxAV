// Turning an ontology attribute into a keyboard.
//
// The sweep is only faster than the properties panel if the hands never leave the keyboard, and the keys
// have to come from the ontology rather than from a hardcoded table: `load_type` has ten values today and
// will have more, and a table that drifts from the YAML binds a key to the wrong answer with no visible
// symptom. So the bindings are derived, here, as a pure function over the attribute spec.
//
// Not every attribute gets a keyboard, and pretending otherwise is the failure mode to avoid. A float
// tolerance, a per-rider boolean array and an unbounded integer are not answerable with one key, and a
// mode that offers a key for them anyway would record a guess. Those return null and the page says why.
//
// THE BADGES MUST NOT LIE, for the same reason ClassPopover records: a number printed next to a value is
// a promise about which key sets it. The list here is the list that gets numbered, in order, and the page
// renders `keys[i].key` rather than `i + 1`.

/** The largest number of digit keys a hand can reach without looking. 0 is included for ranges. */
const MAX_DIGIT_KEYS = 9;

export type AttrSpec = {
  type: string;
  values?: unknown[] | null;
  range?: number[] | null;
};

export type AttrKey = {
  /** the literal key to press, lowercased */
  key: string;
  /** what to show next to it */
  label: string;
  value: unknown;
};

export type AttrKeymap = {
  keys: AttrKey[];
  /** true when a press toggles rather than commits, so the answer needs an explicit Enter */
  multi: boolean;
};

/**
 * The keys for one attribute, or null when it cannot honestly be answered with a keystroke.
 *
 * `null` is a real answer and callers must render it as one. The alternative, inventing buckets for a
 * float or defaulting a per-rider array to all-false, produces data that looks answered and is not.
 */
export function attrKeymap(spec: AttrSpec): AttrKeymap | null {
  switch (spec.type) {
    case "bool":
      // y/n rather than 1/2: two digits for a binary question reads as an ordering that is not there.
      return { keys: [{ key: "y", label: "yes", value: true }, { key: "n", label: "no", value: false }], multi: false };

    case "enum":
    case "multi_select": {
      const values = spec.values ?? [];
      if (!values.length) return null;
      const keys = values.slice(0, MAX_DIGIT_KEYS).map((v, i) => ({
        key: String(i + 1),
        label: String(v),
        value: v,
      }));
      // Values past the ninth are still answerable by clicking, and deliberately carry no key rather than
      // wrapping onto letters, where the binding would be unguessable.
      return { keys, multi: spec.type === "multi_select" };
    }

    case "int": {
      const range = spec.range;
      if (!range || range.length !== 2) return null;
      const [lo, hi] = range;
      // Only when the whole range fits on the number row, and only when it starts at 0 or 1 so the key
      // IS the answer. `group_size` runs 1..50 and gets no keyboard, which is correct: nobody is counting
      // a herd with one keystroke.
      if (!Number.isInteger(lo) || !Number.isInteger(hi)) return null;
      if (lo < 0 || hi - lo + 1 > MAX_DIGIT_KEYS + 1 || lo > 1) return null;
      const keys: AttrKey[] = [];
      for (let v = lo; v <= hi; v++) keys.push({ key: String(v), label: String(v), value: v });
      return { keys, multi: false };
    }

    // float: a tolerance is a measurement, not a choice, and bucketing it here would invent the buckets.
    // bool_array: one value per rider, so the answer depends on how many riders there are.
    default:
      return null;
  }
}

/** Why an attribute has no keyboard, for the page to show instead of an empty row of keys. */
export function noKeymapReason(spec: AttrSpec): string {
  if (spec.type === "float") return "a measured value, not a choice: set it on the object in the frame editor";
  if (spec.type === "bool_array") return "one value per occupant, so it needs the object's occupant count";
  if (spec.type === "int") return "the range is too wide for the number row";
  return "no keyboard for this attribute type";
}
