/**
 * Small shared pieces: badges, the health bar, cards, progress.
 *
 * Kept together because each is a handful of lines and they are always used in
 * combination. The important invariants live here rather than at each call site:
 * a status colour is always paired with a word, a card always reserves its height,
 * and the health bar never renders a verdict this app computed.
 */

import type { ReactNode } from "react";
import type { Health, HealthVerdict, RunState } from "../api/types";
import { age, EMPTY } from "../lib/format";

/**
 * A status pill: a coloured word on a 16% wash of the same hue.
 *
 * `tone` accepts both health values and lifecycle states, because DESIGN.md wants
 * them visually parallel -- they are genuinely different questions, and a
 * `finished` run is `healthy` and must not look alarming for being silent.
 *
 * The word is always rendered. Colour alone excludes readers with red-green
 * deficiency and is invisible in a screenshot pasted into a chat.
 */
export function Badge(props: { tone: Health | RunState | "unknown"; children?: ReactNode }) {
  return (
    <span className="badge" data-tone={props.tone}>
      {props.children ?? props.tone}
    </span>
  );
}

/**
 * The 4px full-bleed health strip, present on every page.
 *
 * `verdict` comes straight from `GET /api/runs/{id}` → `.health`. Nothing here
 * inspects the snapshot or recomputes anything: the derivation lives in one pure
 * function on the server precisely so a run cannot look healthy in one view and
 * dead in another.
 *
 * The reason text appears whenever the verdict is not `healthy` -- the bar alone
 * says "look", the text says "at what". `unknown` gets it too: "we could not tell"
 * is real information and must not be laundered into "fine", and it is not a
 * loading state.
 */
export function HealthBar(props: { verdict: HealthVerdict | null }) {
  const verdict = props.verdict;
  const health: Health = verdict?.health ?? "unknown";
  const showReason = health !== "healthy" && verdict !== null;

  return (
    <>
      <div
        className="healthbar"
        data-health={health}
        role="status"
        aria-label={`Run health: ${health}`}
      />
      {showReason && (
        <div className="healthbar-reason" data-health={health}>
          <span className="healthbar-reason-label">{health}</span>
          {/* Verbatim from the server. This string is the entire explanation of a
              non-green bar, and paraphrasing it would drop the numbers it carries. */}
          <span>{verdict.reason}</span>
        </div>
      )}
    </>
  );
}

/** A card: a faint uppercase label, a monospaced value, an optional hint. */
export function Card(props: {
  label: string;
  /** Right-aligned adornment in the label row, e.g. a badge. */
  adornment?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="card">
      <div className="card-label">
        <span>{props.label}</span>
        {props.adornment}
      </div>
      {props.children}
    </section>
  );
}

/**
 * A card's hero number.
 *
 * `empty` styles a missing value faint rather than rendering a zero: a server that
 * reported no ETA and a server that reported an ETA of zero are different facts.
 */
export function CardValue(props: { children: ReactNode; small?: boolean; empty?: boolean }) {
  return (
    <div
      className={props.small ? "card-value card-value-sm" : "card-value"}
      data-empty={props.empty ? "true" : undefined}
    >
      {props.children}
    </div>
  );
}

export function CardHint(props: { children: ReactNode; strong?: boolean }) {
  return (
    <div className={props.strong ? "card-hint card-hint-strong" : "card-hint"}>{props.children}</div>
  );
}

/**
 * A progress track.
 *
 * With no horizon the track renders indeterminate -- a static stripe, not a fake
 * percentage. Runners derive the effective `max_steps` from `max_steps` and
 * `max_epochs` together, and a run that reports none genuinely does not know how
 * far it has to go; inventing a denominator would be the wrong kind of certainty.
 */
export function Progress(props: { step: number; maxSteps: number | null; semantics: string }) {
  const { step, maxSteps, semantics } = props;
  const fraction = maxSteps && maxSteps > 0 ? Math.min(1, step / maxSteps) : null;

  return (
    <div className="progress">
      <div className="progress-track" data-indeterminate={fraction === null ? "true" : undefined}>
        <div
          className="progress-fill"
          style={fraction === null ? undefined : { width: `${(fraction * 100).toFixed(2)}%` }}
        />
      </div>
      <div className="progress-legend">
        <span>{semantics}</span>
        <span>{fraction === null ? "no horizon" : `${Math.round(fraction * 100)}%`}</span>
      </div>
    </div>
  );
}

/** A label/value row inside a card. Values are monospaced and nowrap. */
export function Row(props: { label: string; children: ReactNode }) {
  return (
    <div className="card-row">
      <span>{props.label}</span>
      <span className="card-row-value">{props.children}</span>
    </div>
  );
}

/** An inline code span: run ids, metric keys, paths, resume commands. */
export function Code(props: { children: ReactNode; title?: string }) {
  return (
    <code className="code" title={props.title}>
      {props.children}
    </code>
  );
}

/**
 * "As of" readout for a live view.
 *
 * Shows the age of the data, not a spinner. A long-session console needs to say
 * how stale what you are looking at is; a spinner says only that something is
 * happening somewhere.
 */
export function AsOf(props: { updatedAt: number | null; now: number }) {
  if (props.updatedAt === null) return <span className="faint">{EMPTY}</span>;
  return <span className="faint num">{age((props.now - props.updatedAt) / 1000)}</span>;
}

export function Note(props: { tone?: "error" | "warn"; title?: string; children: ReactNode }) {
  return (
    <div className="note" data-tone={props.tone}>
      {props.title && <div className="note-title">{props.title}</div>}
      {props.children}
    </div>
  );
}
