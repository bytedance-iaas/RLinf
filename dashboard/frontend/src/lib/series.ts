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
