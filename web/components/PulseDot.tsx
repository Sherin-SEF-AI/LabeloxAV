// A dot that pulses when, and only when, something is actually happening.
//
// `.running-dot` in globals.css already says "work is running" and eight surfaces use it. It says exactly
// that one thing, in one colour, and two states worth telling apart from it had nowhere to go: something
// that needs attention, and something that has gone quiet when it should not have.
//
// The second is the one that prompted this. `lib/jobStream.ts` tracks whether the event stream is
// connected and says in its own comment that "losing the connection IS the change for anything showing a
// live indicator" - and nothing renders it, so a dropped stream looks exactly like an idle system: same
// grey dot, same "idle" label, no work appearing because none can arrive. The quietest possible failure.
//
// Three rules this component exists to keep.
//
// A dot pulses only while its subject is live. An `idle` dot is deliberately still, because a dot that
// always moves says nothing by moving, and once everything pulses the one that matters is invisible.
//
// Motion is never the only carrier. Colour distinguishes the states without it, the title says the state
// in words, and under `prefers-reduced-motion` the animation is dropped while the dot remains - so the
// indicator degrades to a legible static one rather than disappearing.
//
// And it is labelled. A bare coloured circle is invisible to a screen reader, which is how the eight
// existing `<span className="running-dot" />` usages read today. `label` becomes both the tooltip and the
// accessible name; passing none makes the dot decorative, which is only right beside text that already
// says the same thing.

export type PulseTone = "live" | "good" | "warn" | "bad" | "idle";

/** px. Small enough to sit inside a line of 11px mono without pushing it around. */
const SIZE: Record<NonNullable<PulseDotProps["size"]>, number> = { sm: 5, md: 6, lg: 8 };

export type PulseDotProps = {
  tone: PulseTone;
  size?: "sm" | "md" | "lg";
  /**
   * What this dot is saying, in words.
   *
   * Becomes the tooltip and the accessible name. Omit it only when adjacent text already says the same
   * thing, in which case the dot is marked decorative rather than read out twice.
   */
  label?: string;
  /** A ring for a dot that has to be found rather than read. Off by default; a page of haloes is noise. */
  halo?: boolean;
  className?: string;
};

export default function PulseDot({ tone, size = "md", label, halo, className = "" }: PulseDotProps) {
  const px = SIZE[size];
  const classes = [
    "pulse-dot",
    `pulse-dot--${tone}`,
    halo && tone !== "idle" ? "pulse-dot--halo" : "",
    className,
  ].filter(Boolean).join(" ");

  return (
    <span
      className={classes}
      style={{ width: px, height: px }}
      // A live region would announce every state change as it happened, which for a job stream that
      // reconnects is a stream of interruptions. `img` with a name lets a reader ask what the state is
      // without being told each time it changes.
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      title={label}
    />
  );
}
