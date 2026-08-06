/**
 * Turning API series into uPlot's column-major data, plus smoothing.
 *
 * uPlot wants `[xs, ys1, ys2, ...]` with every y array the same length as x. API
 * series do not share a step axis: eval metrics are logged every fifth iteration
 * while train metrics are logged every one, so they must be aligned onto a union
 * axis with holes rather than zip-indexed. Zipping them would silently plot an
 * eval value at a train step, which is a chart that lies.
 */

import type { Series, SeriesPoint } from "../api/types";

/** uPlot's data shape: x first, then one array per series. `null` is a gap. */
export type PlotData = [number[], ...(number | null)[][]];

/** The eight-slot series ramp, assigned by position. Order matches DESIGN.md. */
export const SERIES_COLORS = [
  "var(--color-series-1)",
  "var(--color-series-2)",
  "var(--color-series-3)",
  "var(--color-series-4)",
  "var(--color-series-5)",
  "var(--color-series-6)",
    "var(--color-series-7)",
  "var(--color-series-8)",
] as const;

export function seriesColor(index: number): string {
  return SERIES_COLORS[index % SERIES_COLORS.length] as string;
}

/**
 * Server key for one rank's copy of a metric, matching `?expand=ranks`.
 *
 * The suffix is `@<group>/rank_<n>`, which is also the on-disk directory the
 * numbers came from -- a legend entry doubles as the path to go and grep.
 */
export function rankKey(key: string, worker: string): string {
  return `${key}@${worker}`;
}

/**
 * How a chart can draw its ranks, given how many lines that comes to.
 *
 * Three modes because three different questions are being asked, and which one is
 * answerable depends only on the line count:
 *
 * * `distinct` -- few enough lines that every rank gets its own colour and legend
 *   entry. "Compare the ranks."
 * * `outliers` -- too many to colour, but one metric, so the lowest, highest and
 *   median can be named against it. "Is any rank an outlier?"
 * * `bundle` -- too many, and several metrics. Nothing is named: the notable
 *   palette would have to repeat across metrics, and two legend entries of the
 *   same colour meaning different metrics is the ambiguity the cap exists to
 *   avoid. The bundle's width is still the answer to "how tight is the spread".
 *
 * Decided once per chart rather than per metric, so two metrics in one panel are
 * never drawn under two different conventions.
 */
export type RankMode = "distinct" | "outliers" | "bundle";

/**
 * @param keyCount How many aggregate series the chart draws.
 * @param rankLineCount How many rank lines will be drawn across all of them.
 *   A count rather than a rank count, because a worker that never logged one of
 *   the chart's keys contributes no line for it.
 */
export function rankMode(keyCount: number, rankLineCount: number): RankMode {
  // Aggregates and rank lines share the eight-slot ramp. Past it `seriesColor`
  // cycles, and two ranks drawn in the same colour make a legend that cannot
  // answer "which one is rank 9" -- worse than no colours, which is what the other
  // two modes do instead.
  if (keyCount + rankLineCount <= SERIES_COLORS.length) return "distinct";
  return keyCount === 1 ? "outliers" : "bundle";
}

/**
 * Which of `workers` actually logged `metric`.
 *
 * Not every worker logs every metric: `env/*` comes from the env group alone, so
 * an actor rank has no series for it and the server omits it rather than sending
 * an empty one. Narrowing here keeps a chart from spending a colour and a legend
 * entry on a line that would be drawn flat at the axis for a metric that group
 * does not measure.
 */
export function workersWith(
  metric: string,
  workers: string[],
  series: Record<string, Series>,
): string[] {
  return workers.filter((worker) => series[rankKey(metric, worker)] !== undefined);
}

/** One line in an expanded chart. */
export interface RankLine {
  /** Server key: `rankKey(metric, worker)`. */
  key: string;
  /** The metric this rank is a breakdown of. */
  metric: string;
  /** Worker label, e.g. `EnvGroup/rank_3`. */
  worker: string;
  color: string;
  /**
   * Whether this line is a subject of the chart or context for it.
   *
   * `false` puts it in the bundle: drawn, but with no legend entry and no
   * tooltip row. See `bundleRanks` for why a wide bundle is drawn that way.
   */
  legend: boolean;
  /** Why it was singled out, for the legend's title text. */
  note?: string;
}

/**
 * Palette for the named lines in `outliers` mode.
 *
 * Fixed rather than taken from the ramp by position: these three are picked by
 * *role*, and the reader has to be able to learn "amber is the slowest rank" once
 * and have it hold across every chart on the page.
 */
const NOTABLE_COLORS = [
  "var(--color-series-5)",
  "var(--color-series-3)",
  "var(--color-series-2)",
] as const;

/**
 * Decide how to draw N ranks of one metric.
 *
 * In `distinct` mode every rank gets its own colour and its own legend entry. In
 * the other two the bundle is drawn faint -- the rest are still drawn, because the
 * shape of the bundle is how you see whether it is one straggler or a wide spread
 * -- and `outliers` additionally names the lowest, highest and median.
 *
 * `colorOffset` is the first ramp slot this metric's ranks may use, so a chart can
 * lay several metrics' ranks side by side without two lines sharing a colour.
 *
 * `workers` must already be narrowed to those that logged `metric` -- see
 * `workersWith`. A worker without the key would otherwise take a colour, a legend
 * entry and a flat line at the axis for a metric it does not measure.
 */
export function bundleRanks(
  metric: string,
  workers: string[],
  series: Record<string, Series>,
  options: { colorOffset?: number; mode?: RankMode } = {},
): RankLine[] {
  const colorOffset = options.colorOffset ?? 1;
  const mode = options.mode ?? rankMode(1, workers.length);

  if (mode === "distinct") {
    return workers.map((worker, index) => ({
      key: rankKey(metric, worker),
      metric,
      worker,
      color: seriesColor(colorOffset + index),
      legend: true,
    }));
  }

  const notable = new Map<string, string>();
  if (mode === "outliers") {
    const ranked = workers
      .map((worker) => ({ worker, value: lastValue(series[rankKey(metric, worker)]) }))
      .filter((entry): entry is { worker: string; value: number } => entry.value !== null)
      .sort((a, b) => a.value - b.value);
    if (ranked.length > 0) {
      notable.set((ranked[0] as { worker: string }).worker, "lowest");
      notable.set((ranked[ranked.length - 1] as { worker: string }).worker, "highest");
      // Set after the extremes, so a median that collides with one of them keeps
      // the more informative label rather than overwriting it.
      const middle = (ranked[Math.floor(ranked.length / 2)] as { worker: string }).worker;
      if (!notable.has(middle)) notable.set(middle, "median");
    }
  }

  let taken = 0;
  return workers.map((worker) => {
    const note = notable.get(worker);
    if (note === undefined) {
      return {
        key: rankKey(metric, worker),
        metric,
        worker,
        color: "var(--color-text-faint)",
        legend: false,
      };
    }
    const color = NOTABLE_COLORS[taken % NOTABLE_COLORS.length] as string;
    taken += 1;
    return { key: rankKey(metric, worker), metric, worker, color, legend: true, note };
  });
}

/**
 * Resolve a `var(--...)` token to a literal colour.
 *
 * uPlot draws to a canvas, and canvas `strokeStyle` does not understand CSS custom
 * properties. Reading the computed value keeps the palette in tokens.css as the
 * single source rather than duplicating hex literals in chart code.
 */
export function resolveColor(token: string, element: Element): string {
  const match = /^var\((--[^)]+)\)$/.exec(token.trim());
  if (!match) return token;
  const value = getComputedStyle(element).getPropertyValue(match[1] as string).trim();
  return value || token;
}

/**
 * Align several series onto a union step axis.
 *
 * A step present in one series and absent in another gets `null` there, which
 * uPlot renders as a gap. That is the honest rendering: the metric has no value at
 * that step, and interpolating one would invent data.
 */
export function alignSeries(list: (Series | undefined)[]): PlotData {
  const steps = new Set<number>();
  for (const entry of list) {
    for (const point of entry?.points ?? []) steps.add(point.step);
  }
  const xs = [...steps].sort((a, b) => a - b);
  const index = new Map<number, number>();
  xs.forEach((step, i) => index.set(step, i));

  const columns: (number | null)[][] = list.map((entry) => {
    const column: (number | null)[] = new Array(xs.length).fill(null);
    for (const point of entry?.points ?? []) {
      const at = index.get(point.step);
      if (at === undefined) continue;
      // A non-finite value arrives as `null` and stays `null`: uPlot cannot draw
      // NaN, and the fact that it happened is reported as a signal instead.
      column[at] =
        point.value !== null && Number.isFinite(point.value) ? point.value : null;
    }
    return column;
  });

  return [xs, ...columns];
}

/**
 * The x extent to pin a chart to, or `null` when there is nothing to pin.
 *
 * A one-step run has no extent, and uPlot will happily accept `min === max`: its
 * own zero-width rejection sits behind `dataLen > 1`, which a single point does
 * not satisfy. The axis then picks an increment of `1e-16` and walks
 * `for (val = min; val <= max; val += incr)` — but `1 + 1e-16` *is* `1` in
 * float64, so the loop never advances and pushes splits until the array hits its
 * length limit. That is a dead tab, not a mis-drawn chart: Chrome kills the
 * renderer ("Error code: 5") and Safari stops responding. A one-iteration run is
 * a real case, so the degenerate extent has to be padded here.
 *
 * One step of headroom either side, rather than uPlot's own proportional padding
 * (`rangeNum` stretches step 7 to 0–14): the axis labels integers only, so ±1
 * keeps them whole and leaves the single point mid-panel.
 */
export function xExtent(xs: number[]): { min: number; max: number } | null {
  if (xs.length === 0) return null;
  const first = xs[0] as number;
  const last = xs[xs.length - 1] as number;
  if (first !== last) return { min: first, max: last };
  // 1 is exact for any step a run can reach. The relative floor is for a step so
  // large that 1 falls below its float spacing — a corrupt event file rather than
  // a real run, but it reaches this line from disk and must not hang the tab.
  // 2^-40 puts the pad ~4096 steps of spacing clear of it.
  const pad = Math.max(1, Math.abs(first) * 2 ** -40);
  return { min: first - pad, max: last + pad };
}

/**
 * A widened y range when new data falls outside the current one, else `null`.
 *
 * Live updates go through `setData(data, false)`, which deliberately does not
 * recompute scales — that is what stops the gridlines walking every two seconds
 * and the page reflowing under the reader. The cost is that the y axis keeps the
 * extent it was built with, so a value arriving *later* that exceeds it is drawn
 * outside the plot area: clipped, invisible, and indistinguishable from the run
 * being flat. A loss explosion or a reward collapse is exactly the value that
 * arrives late and exceeds the range, which made this the one push the chart
 * could not show.
 *
 * Growth only, never shrink. A range that tracked the data in both directions
 * would rescale on almost every push and bring back the walking gridlines; one
 * that only grows changes rarely, and only when something happened that the
 * reader needs to see. The axis therefore records the run's worst excursion
 * rather than its current window, which is the more useful of the two.
 *
 * @param columns Series columns (`data[1..]`), y values with `null` gaps. For a
 *   stacked chart these are already cumulative, so the maximum is the top of the
 *   stack — which is the bound that has to be admitted.
 * @param current The scale's present bounds, from `plot.scales.y`.
 * @param opts `positiveOnly` for log scales, which cannot render `<= 0` and
 *   would be handed a range uPlot refuses to draw; `stacked` to keep zero on the
 *   axis, since a stack whose baseline is not zero misstates every band's share.
 */
export function yGrowth(
  columns: (number | null)[][],
  current: { min: number; max: number } | null | undefined,
  opts: { positiveOnly?: boolean; stacked?: boolean } = {},
): { min: number; max: number } | null {
  const { positiveOnly = false, stacked = false } = opts;
  if (!current || !Number.isFinite(current.min) || !Number.isFinite(current.max)) return null;

  let low = Infinity;
  let high = -Infinity;
  for (const column of columns) {
    for (const value of column) {
      // Non-finite values are the divergence signal, but they have no position
      // on an axis; `nonFiniteSignal` reports them, the scale ignores them.
      if (value === null || !Number.isFinite(value)) continue;
      if (positiveOnly && value <= 0) continue;
      if (value < low) low = value;
      if (value > high) high = value;
    }
  }
  if (low === Infinity) return null;

  const belowBy = current.min - low;
  const aboveBy = high - current.max;
  if (belowBy <= 0 && aboveBy <= 0) return null;

  // The same padding `Chart.tsx`'s initial `range` callback applies, so that a
  // chart which grew into a range and one freshly built from the same data land
  // on identical bounds. Without that, reloading the page would visibly shift
  // every axis that had grown — the reader would learn to distrust whichever one
  // they were not looking at. `check:scales` asserts the two agree.
  const pad = low === high ? (Math.abs(low) > 0 ? Math.abs(low) * 0.15 : 1) : (high - low) * 0.08;
  const lowBound = stacked ? Math.min(0, low) : low - pad;
  return {
    min: belowBy > 0 ? lowBound : current.min,
    max: aboveBy > 0 ? high + pad : current.max,
  };
}

/**
 * Exponential moving average with a window expressed in points.
 *
 * EMA rather than a boxcar mean because it needs no lookahead: a live run's most
 * recent point is the one being watched, and a centred window cannot smooth it
 * until half a window later. `weight` is derived so that a window of 1 is the
 * identity, which is what makes the smoothing control's zero position honest.
 *
 * Gaps (`null`) are preserved rather than bridged, and the accumulator carries
 * across them: smoothing an eval curve that is logged every fifth step must not
 * fill in the four missing steps.
 */
export function smooth(column: (number | null)[], window: number): (number | null)[] {
  if (window <= 1) return column;
  const weight = 2 / (window + 1);
  let accumulator: number | null = null;
  return column.map((value) => {
    if (value === null) return null;
    accumulator = accumulator === null ? value : accumulator + weight * (value - accumulator);
    return accumulator;
  });
}

/**
 * Convert a column into a cumulative stack against a running total.
 *
 * Stacking is done here rather than by uPlot's band support because the template
 * declares it per chart, and the totals must be computed on aligned columns where
 * a `null` contributes nothing but does not break the stack below it.
 */
export function stackColumns(columns: (number | null)[][]): (number | null)[][] {
  const length = columns[0]?.length ?? 0;
  const totals = new Array<number>(length).fill(0);
  return columns.map((column) => {
    const out: (number | null)[] = new Array(length).fill(null);
    for (let i = 0; i < length; i += 1) {
      const value = column[i];
      if (value === null || value === undefined) {
        // A missing part of a stack is zero contribution, but the band still has
        // to be drawn at the running total or the layers above it detach.
        out[i] = totals[i] as number;
        continue;
      }
      totals[i] = (totals[i] as number) + value;
      out[i] = totals[i] as number;
    }
    return out;
  });
}

/** Last finite value of a series, for a legend readout and a hero number. */
export function lastValue(series: Series | undefined): number | null {
  if (!series) return null;
  for (let i = series.points.length - 1; i >= 0; i -= 1) {
    const point = series.points[i] as SeriesPoint;
    if (point.value !== null && Number.isFinite(point.value)) return point.value;
  }
  return null;
}

/** Last step at which the series has a finite value. */
export function lastStep(series: Series | undefined): number | null {
  if (!series) return null;
  for (let i = series.points.length - 1; i >= 0; i -= 1) {
    const point = series.points[i] as SeriesPoint;
    if (point.value !== null && Number.isFinite(point.value)) return point.step;
  }
  return null;
}

/** How many points across a set of series, for a "sampled" note. */
export function totalPoints(list: (Series | undefined)[]): number {
  return list.reduce((sum, entry) => sum + (entry?.total_points ?? 0), 0);
}

/** True when any series in the set was decimated server-side. */
export function anyDecimated(list: (Series | undefined)[]): boolean {
  return list.some((entry) => entry?.decimated === true);
}

/**
 * Shorten a metric key for a legend.
 *
 * Keeps the last two path segments: within one chart every key shares a prefix
 * (`train/actor/`), so the prefix is noise and the leaf is the distinguishing
 * part. The full key stays in the `title` attribute.
 */
export function shortKey(key: string): string {
  const parts = key.split("/");
  if (parts.length <= 2) return key;
  return parts.slice(-2).join("/");
}
