/**
 * Client-side metric signals, kept strictly separate from the server's verdict.
 *
 * The server owns liveness (`healthy` / `degraded` / `unreachable` / `unknown`)
 * and that verdict is rendered verbatim, never recomputed here. What the server
 * deliberately does not compute is anything that requires reading the *series*:
 * `derive_health` is a pure function of a snapshot and a clock, by design, so it
 * can be unit-tested without a filesystem and cannot disagree with itself across
 * views.
 *
 * That leaves three warning signals only a series reader can produce:
 *
 * * step time degraded past 2x its own baseline -> amber;
 * * eval with no improvement for K consecutive rounds -> amber;
 * * a non-finite value anywhere in a series -> red.
 *
 * These are labelled as metric signals everywhere they surface, and never merged
 * into the health badge or the health bar's colour. A reader must be able to tell
 * "the server says this run is hung" from "the client noticed the loss went NaN",
 * because the two have completely different remedies.
 */

import type { NorthStar, RunTemplate, Series, SeriesPoint } from "../api/types";

/** Signal severity. Only two levels: amber warns, red says something is wrong. */
export type SignalLevel = "amber" | "red";

export interface MetricSignal {
  /** Stable id, so a list of signals does not reorder between SSE updates. */
  id: string;
  level: SignalLevel;
  /** Short phrase. Paired with the colour, never carried by colour alone. */
  title: string;
  /** The evidence, with the numbers it was derived from. */
  detail: string;
  /** Metric keys involved, for a link into the deep dive. */
  keys: string[];
}

/** Consecutive eval rounds without improvement before the plateau signal fires. */
export const EVAL_PLATEAU_K = 5;

/** Step time ratio, recent versus baseline, above which the signal fires. */
export const STEP_TIME_RATIO = 2.0;

/**
 * Watch-set cap for the overview.
 *
 * The overview must be legible in five seconds, and requesting all 56 series of a
 * long run is megabytes of JSON. The set is the first key of each chart in
 * template order -- which is authored priority, task performance before
 * throughput -- capped here. The deep-dive view checks every series it renders,
 * so nothing is permanently invisible; the overview says how many it checked.
 */
export const WATCH_SET_CAP = 12;

function finiteValues(points: SeriesPoint[]): number[] {
  const out: number[] = [];
  for (const point of points) {
    if (point.value !== null && Number.isFinite(point.value)) out.push(point.value);
  }
  return out;
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid] as number;
  return ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
}

/**
 * A non-finite value anywhere in a series.
 *
 * `value: null` on the wire means non-finite, not missing: `SeriesPoint.value` is
 * a required float on the server, and pydantic serialises `float('nan')` and
 * `float('inf')` to JSON `null`. Verified against a real TensorBoard event file
 * containing both. So this is the NaN/Inf detector the criterion asks for, and a
 * run whose loss went NaN at iteration 31 shows a red signal naming that step.
 */
export function nonFiniteSignal(series: Record<string, Series>): MetricSignal | null {
  const hits: { key: string; step: number; count: number }[] = [];

  for (const [key, entry] of Object.entries(series)) {
    let first: number | null = null;
    let count = 0;
    for (const point of entry.points) {
      if (point.value === null || !Number.isFinite(point.value)) {
        if (first === null) first = point.step;
        count += 1;
      }
    }
    if (first !== null) hits.push({ key, step: first, count });
  }
  if (hits.length === 0) return null;

  hits.sort((a, b) => a.step - b.step || a.key.localeCompare(b.key));
  const worst = hits[0] as { key: string; step: number; count: number };
  const others = hits.length - 1;

  return {
    id: "non-finite",
    level: "red",
    title: "Non-finite metric value",
    detail:
      `${worst.key} first went NaN or Inf at step ${worst.step} ` +
      `(${worst.count} of ${series[worst.key]?.points.length ?? 0} points)` +
      (others > 0 ? `, and in ${others} other ${others === 1 ? "series" : "series"}.` : "."),
    keys: hits.map((hit) => hit.key),
  };
}

/**
 * Step time degraded past 2x its own baseline.
 *
 * The baseline is the run's own early steps rather than a fixed threshold, for the
 * same reason the server's timeouts are relative: a reasoning step is seconds and
 * an embodied Pi0.5 step on 4xH20 measured 428s, so no absolute number means
 * anything across RLinf.
 *
 * The first two steps are excluded from the baseline. Step 0 carries CUDA graph
 * capture, JIT warmup and the first simulator reset, and is routinely several
 * times a steady-state step -- including it would make every run look like it got
 * *faster* and mask a real regression.
 */
export function stepTimeSignal(series: Series | null | undefined): MetricSignal | null {
  if (!series) return null;
  const values = finiteValues(series.points);
  // Below this there is no baseline to compare against; a 4-step run that got
  // slower has not demonstrated a trend.
  if (values.length < 8) return null;

  const warmup = 2;
  const window = Math.max(3, Math.floor((values.length - warmup) / 5));
  const baseline = median(values.slice(warmup, warmup + window));
  const recent = median(values.slice(-window));
  if (baseline === null || recent === null || baseline <= 0) return null;

  const ratio = recent / baseline;
  if (ratio < STEP_TIME_RATIO) return null;

  return {
    id: "step-time",
    level: "amber",
    title: "Step time degraded",
    detail:
      `${series.key} is ${ratio.toFixed(1)}x its early baseline ` +
      `(${recent.toFixed(1)}s now versus ${baseline.toFixed(1)}s over steps ` +
      `${warmup}-${warmup + window - 1}).`,
    keys: [series.key],
  };
}

/**
 * Eval with no improvement for K consecutive rounds.
 *
 * "No improvement" allows a 0.5% relative tolerance rather than requiring a
 * strictly lower value. On a noisy eval curve a 1e-4 wobble is not progress, and
 * a strict comparison would let any noise reset the counter and the signal would
 * never fire on exactly the runs it is for.
 *
 * Direction comes from the template's `north_star.goal`, so a template whose
 * headline metric is a loss gets the comparison the right way round without this
 * function knowing anything about metric names.
 */
export function evalPlateauSignal(
  series: Series | null | undefined,
  goal: "maximize" | "minimize",
  k = EVAL_PLATEAU_K,
): MetricSignal | null {
  if (!series) return null;
  const values = finiteValues(series.points);
  // Need a history to plateau against, not just K flat points from the start.
  if (values.length < k + 2) return null;

  const head = values.slice(0, values.length - k);
  const tail = values.slice(-k);
  const best = goal === "maximize" ? Math.max(...head) : Math.min(...head);
  const tolerance = Math.max(Math.abs(best) * 0.005, 1e-9);

  const improved = tail.some((value) =>
    goal === "maximize" ? value > best + tolerance : value < best - tolerance,
  );
  if (improved) return null;

  const bestTail = goal === "maximize" ? Math.max(...tail) : Math.min(...tail);
  return {
    id: "eval-plateau",
    level: "amber",
    title: `No eval improvement in ${k} rounds`,
    detail:
      `${series.key} has not beaten ${best.toPrecision(3)} in its last ${k} ` +
      `evaluations (best recent ${bestTail.toPrecision(3)}).`,
    keys: [series.key],
  };
}

/**
 * The eval counterpart of a metric key: `env/success_once` -> `eval/success_once`.
 *
 * Derived rather than hardcoded so the plateau check follows whatever the template
 * declared as its north star, in any task type.
 */
export function evalCounterpart(key: string | null | undefined): string | null {
  if (!key) return null;
  if (key.startsWith("eval/")) return key;
  const slash = key.indexOf("/");
  if (slash < 0) return `eval/${key}`;
  return `eval/${key.slice(slash + 1)}`;
}

/** Goal direction for the plateau check; templates default to maximize. */
export function northStarGoal(northStar: NorthStar | null | undefined): "maximize" | "minimize" {
  return northStar?.goal === "minimize" ? "minimize" : "maximize";
}

/**
 * The keys the overview should fetch to run these checks.
 *
 * Assembled from the template rather than a fixed list, because the point of the
 * template system is that this bundle never learns which metrics a task type
 * emits.
 */
export function watchSetKeys(template: RunTemplate | null, available: Set<string>): string[] {
  const keys: string[] = [];
  const add = (key: string | null | undefined) => {
    if (key && available.has(key) && !keys.includes(key)) keys.push(key);
  };

  add(template?.north_star?.resolved ? template.north_star.key : null);
  add(evalCounterpart(template?.north_star?.key));
  // The step time check needs the series the throughput group charts, which is
  // `time/step` in every template that has one.
  add("time/step");

  for (const group of template?.groups ?? []) {
    for (const chart of group.charts) {
      if (keys.length >= WATCH_SET_CAP) return keys;
      add(chart.keys[0]);
    }
  }
  return keys;
}

/** Every eval key the template charts, for the plateau fallback. */
function evalRateKeys(template: RunTemplate | null): string[] {
  const out: string[] = [];
  for (const group of template?.groups ?? []) {
    for (const chart of group.charts) {
      // `format: percent` is the template author declaring the metric a rate, and
      // a rate is maximized. Using the declaration avoids guessing from names.
      if (chart.format !== "percent") continue;
      for (const key of chart.keys) {
        if (key.startsWith("eval/") && !out.includes(key)) out.push(key);
      }
    }
  }
  return out;
}

/**
 * Run every metric-side check over the series that were fetched.
 *
 * Returns them ordered red before amber, then by id, so the list does not shuffle
 * when an SSE update lands -- a list that reorders while someone is reading it is
 * called out in DESIGN.md as a page-geometry violation.
 */
export function collectSignals(
  template: RunTemplate | null,
  series: Record<string, Series>,
): MetricSignal[] {
  const signals: MetricSignal[] = [];

  const nonFinite = nonFiniteSignal(series);
  if (nonFinite) signals.push(nonFinite);

  const stepTime = stepTimeSignal(series["time/step"]);
  if (stepTime) signals.push(stepTime);

  const goal = northStarGoal(template?.north_star);
  const evalKey = evalCounterpart(template?.north_star?.key);
  const plateauKey =
    evalKey && series[evalKey]
      ? evalKey
      : evalRateKeys(template).find((key) => series[key]) ?? null;
  const plateau = plateauKey ? evalPlateauSignal(series[plateauKey], goal) : null;
  if (plateau) signals.push(plateau);

  const rank = (signal: MetricSignal) => (signal.level === "red" ? 0 : 1);
  return signals.sort((a, b) => rank(a) - rank(b) || a.id.localeCompare(b.id));
}

/** The worst level present, for a summary chip. `null` when nothing fired. */
export function worstLevel(signals: MetricSignal[]): SignalLevel | null {
  if (signals.some((signal) => signal.level === "red")) return "red";
  if (signals.length > 0) return "amber";
  return null;
}
