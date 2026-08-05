/**
 * The single uPlot wrapper. Every chart in the app is this component.
 *
 * uPlot rather than a React charting library because an RL run puts tens of
 * thousands of points in one series and the server decimates only above 4000 per
 * key -- a chart per metric, several metrics per chart, on one page. uPlot draws to
 * a canvas in one pass; a virtual-DOM charting library allocates a node per point.
 *
 * Two behaviours here exist specifically to satisfy DESIGN.md rather than because
 * uPlot wants them that way:
 *
 * * **The plot never re-fits its axes on a push.** uPlot's default is to rescale
 *   to the incoming data on every `setData`, so a live chart's y axis creeps every
 *   two seconds and the gridlines walk. Scales are recomputed only when the data's
 *   own range changes enough to matter, and a user zoom pins them entirely.
 * * **The plot area keeps its height with no data.** A chart that collapses while
 *   loading makes the page jump when it arrives, so axes are drawn against an
 *   empty plot instead.
 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { PlotData } from "../lib/series";
import { resolveColor } from "../lib/series";
import { metric as formatMetric } from "../lib/format";

export interface ChartSeriesSpec {
  /** Legend label. The full metric key goes in `title`. */
  label: string;
  /** Full key or run id, shown on hover. */
  title?: string;
  /** A `var(--color-series-N)` token, resolved to a literal for the canvas. */
  color: string;
  /** Hidden by a legend toggle. Kept in the data so indices stay stable. */
  hidden?: boolean;
  /** Draw as a filled band up to its value; set for `stacked` charts. */
  fill?: boolean;
  /** Dashed stroke, for a run that has different step semantics than the axis. */
  dashed?: boolean;
  /**
   * Context rather than subject: drawn, but kept out of the tooltip.
   *
   * For the bundle of per-rank lines behind a drill-down. The bundle's *shape* is
   * the content -- whether one line strays from it -- and thirty-two rows of
   * numbers on hover would bury the handful that were singled out by name.
   */
  muted?: boolean;
}

export interface ChartProps {
  data: PlotData;
  series: ChartSeriesSpec[];
  /** X axis label, from the template's `step_axis_label` or step semantics. */
  xLabel: string;
  /** Y axis suffix from the template's `unit`. */
  unit?: string;
  /** Template `format: percent`: display as 0-100% and pin the axis to [0,1]. */
  percent?: boolean;
  /** Template `scale: log`. Falls back to linear when the data spans zero. */
  logScale?: boolean;
  /** Template `stacked: true`. Series are pre-stacked by the caller. */
  stacked?: boolean;
  /** Taller variant for the one chart a page emphasises. */
  tall?: boolean;
  /**
   * Sparkline variant: the 44px strip inside an overview card.
   *
   * The height comes from the stylesheet, not from the caller. An inline height
   * on a wrapping element does not constrain a child that sets its own, which is
   * how the default 180px once escaped the bottom of a card.
   */
  spark?: boolean;
  /** Crosshair group id: charts sharing one read out together on hover. */
  cursorGroup?: string;
}

/**
 * Whether a log scale is usable for this data.
 *
 * A log axis cannot show zero or negative values, and uPlot silently drops the
 * points rather than warning. A template author writing `scale: log` on grad norm
 * is right in general and wrong for the run whose grad norm hit exactly 0, so the
 * decision is made per render against the actual values.
 */
function logIsUsable(data: PlotData): boolean {
  for (let i = 1; i < data.length; i += 1) {
    for (const value of data[i] as (number | null)[]) {
      if (value !== null && value <= 0) return false;
    }
  }
  return true;
}

/** True when at least one series has two or more finite points to join. */
function hasSpan(data: PlotData): boolean {
  for (let i = 1; i < data.length; i += 1) {
    let seen = 0;
    for (const value of data[i] as (number | null)[]) {
      if (value !== null) seen += 1;
      if (seen >= 2) return true;
    }
  }
  return false;
}

function countPoints(data: PlotData): number {
  let total = 0;
  for (let i = 1; i < data.length; i += 1) {
    for (const value of data[i] as (number | null)[]) if (value !== null) total += 1;
  }
  return total;
}

export function Chart(props: ChartProps) {
  const { data, series, xLabel, unit, percent, logScale, stacked, tall, spark, cursorGroup } = props;
  const hostRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [hover, setHover] = useState<{ left: number; idx: number } | null>(null);

  const pointCount = countPoints(data);
  const useLog = logScale === true && logIsUsable(data);
  // A line through one point draws nothing. A one-iteration run is a real case --
  // the verification tree is exactly that -- so points are drawn whenever there is
  // no span to draw, and at low densities where they are legible anyway.
  const showPoints = !hasSpan(data) || pointCount <= 60;

  // The host's box drives the canvas size. uPlot needs explicit pixels, and
  // measuring here (rather than on window resize) keeps the canvas correct inside
  // a grid that reflows at a breakpoint without a window resize event.
  useLayoutEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const observer = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (!box) return;
      setSize((previous) =>
        // Sub-pixel jitter from a scrollbar appearing must not trigger a redraw
        // loop, so only whole-pixel changes count.
        Math.round(box.width) === Math.round(previous.width) &&
        Math.round(box.height) === Math.round(previous.height)
          ? previous
          : { width: box.width, height: box.height },
      );
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const options = useMemo<uPlot.Options>(() => {
    const host = hostRef.current ?? document.body;
    const grid = resolveColor("var(--color-border)", host);
    const axisText = resolveColor("var(--color-text-faint)", host);
    const strokeWidth = 1.5;

    const uSeries: uPlot.Series[] = [
      { label: xLabel },
      ...series.map((spec) => {
        const stroke = resolveColor(spec.color, host);
        const entry: uPlot.Series = {
          label: spec.label,
          stroke,
          // A muted line is drawn thinner as well as fainter. At 1.5px a bundle of
          // thirty ranks is a solid block; at 1px the individual paths stay
          // separable, which is the only reason to draw them at all.
          width: spec.muted ? 1 : strokeWidth,
          show: spec.hidden !== true,
          // Square-cornered data marks: DESIGN.md forbids rounding a mark, since
          // it misrepresents the value at the pixel level.
          points: { show: showPoints, size: 5, fill: stroke, stroke },
          value: (_self, value) => (value === null ? "—" : formatMetric(value, { percent })),
        };
        if (spec.fill) {
          // 22% keeps five stacked bands distinguishable where a heavier fill
          // turns the whole panel into one mass.
          entry.fill = `color-mix(in srgb, ${stroke} 22%, transparent)`;
        }
        if (spec.dashed) entry.dash = [4, 3];
        return entry;
      }),
    ];

    return {
      width: Math.max(1, Math.floor(size.width)),
      height: Math.max(1, Math.floor(size.height)),
      // Nothing animates: DESIGN.md rules out repeating motion, and a chart that
      // tweens on every SSE push cannot be read mid-tween.
      ms: 1,
      // A spark has no axes, so it uses its whole box; the others leave room for
      // the topmost gridline label not to be clipped.
      padding: spark ? [2, 1, 0, 1] : [8, 8, 0, 0],
      legend: { show: false },
      cursor: spark
        ? // Nothing to interact with in a 44px strip: it carries no axes to read a
          // value against, and its tooltip would be several times its own height.
          // The full chart is one click away, which is what the card links to.
          { show: false }
        : {
            y: false,
            // Zoom by drag on x only. A y-drag zoom on a live chart is a trap: the
            // next push extends x and the view silently stops tracking.
            drag: { x: true, y: false },
            sync: cursorGroup ? { key: cursorGroup, setSeries: false } : undefined,
            points: { size: 6 },
          },
      scales: {
        x: { time: false },
        y: percent
          ? // A rate is read against 0 and 1, not against its own range: a
            // success rate wobbling between 0.61 and 0.63 must not look like it
            // swept the full axis.
            { range: [0, 1] }
          : useLog
            ? { distr: 3 }
            : {
                range: (_self, min, max) => {
                  if (min === max) {
                    // A flat or single-point series has no range. Pad it so the
                    // line sits mid-panel rather than on an axis.
                    const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.15 : 1;
                    return [min - pad, max + pad];
                  }
                  const pad = (max - min) * 0.08;
                  // Stacked parts are durations: an axis that does not include
                  // zero misstates every band's share of the total.
                  return [stacked ? Math.min(0, min) : min - pad, max + pad];
                },
              },
      },
      // A spark draws no axes, gridlines or ticks. At 44px a y axis would take
      // more than half the width and the labels would not be legible anyway; the
      // shape of the line is the entire content, and the card states the current
      // value in figures right above it.
      axes: spark
        ? [{ show: false }, { show: false }]
        : [
            {
              stroke: axisText,
              grid: { stroke: grid, width: 1 },
              ticks: { stroke: grid, width: 1, size: 4 },
              font: `12px ${getComputedStyle(host).getPropertyValue("--font-jetbrains-mono") || "monospace"}`,
              // Integers only: a step axis has no half-steps, and uPlot's default
              // splits produce "12.5" on a short run.
              values: (_self, splits) =>
                splits.map((value) => (Number.isInteger(value) ? String(value) : "")),
            },
            {
              stroke: axisText,
              grid: { stroke: grid, width: 1 },
              ticks: { show: false },
              size: 52,
              font: `12px ${getComputedStyle(host).getPropertyValue("--font-jetbrains-mono") || "monospace"}`,
              values: (_self, splits) =>
                splits.map((value) => {
                  if (percent) return `${Math.round(value * 100)}%`;
                  const text = formatMetric(value, { digits: undefined });
                  return unit ? `${text}${unit}` : text;
                }),
            },
          ],
      series: uSeries,
      hooks: {
        setCursor: [
          (self) => {
            const idx = self.cursor.idx;
            const left = self.cursor.left ?? -1;
            setHover(idx === null || idx === undefined || left < 0 ? null : { left, idx });
          },
        ],
      },
    };
    // `series` identity changes on every toggle; the effect below applies show/hide
    // without rebuilding, so it is intentionally not a dependency of the rebuild.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    size.width,
    size.height,
    xLabel,
    unit,
    percent,
    useLog,
    stacked,
    spark,
    cursorGroup,
    showPoints,
    series.length,
    series
      .map((spec) => `${spec.label}|${spec.color}|${spec.fill}|${spec.dashed}|${spec.muted}`)
      .join(","),
  ]);

  // Create and destroy. Recreated only when the option identity above changes,
  // which excludes data and visibility.
  useEffect(() => {
    const host = hostRef.current;
    if (!host || size.width <= 0 || size.height <= 0) return;
    const plot = new uPlot(options, data as unknown as uPlot.AlignedData, host);
    plotRef.current = plot;
    return () => {
      plot.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options]);

  // Data updates go through `setData` so the canvas is redrawn without a new DOM
  // node. `resetScales: false` is the mechanism behind "the page does not reflow
  // when data arrives": the axes keep the extents they already had, so gridlines
  // and tick labels stay put across a push.
  useEffect(() => {
    const plot = plotRef.current;
    if (!plot) return;
    plot.setData(data as unknown as uPlot.AlignedData, false);
    // The x extent does have to follow a growing run, or new points land off the
    // right edge and the chart appears frozen. Only x, and only when not zoomed.
    const xs = data[0];
    const zoomed = plot.select.width > 0;
    if (!zoomed && xs.length > 0) {
      plot.setScale("x", { min: xs[0] as number, max: xs[xs.length - 1] as number });
    }
  }, [data]);

  // Visibility toggles, applied in place for the same reason.
  useEffect(() => {
    const plot = plotRef.current;
    if (!plot) return;
    series.forEach((spec, index) => {
      const wanted = spec.hidden !== true;
      if (plot.series[index + 1]?.show !== wanted) plot.setSeries(index + 1, { show: wanted });
    });
  }, [series]);

  // `cursor: { show: false }` means a spark never sets `hover`, so this is already
  // null there; the explicit guard is so the invariant survives a future edit to
  // the cursor options rather than silently reintroducing an overflowing tooltip.
  const tooltip =
    hover === null || spark ? null : buildTooltip(data, series, hover.idx, xLabel, percent);
  // Flip past the midpoint so the tooltip never leaves the panel and never covers
  // the point being read. Computed here rather than inline so the null check is one
  // branch instead of three.
  const tooltipStyle =
    hover === null
      ? undefined
      : hover.left > size.width / 2
        ? { right: size.width - hover.left + 12, top: 8 }
        : { left: hover.left + 12, top: 8 };

  return (
    <div
      className="chart-plot"
      data-tall={tall ? "true" : undefined}
      data-variant={spark ? "spark" : undefined}
      ref={hostRef}
    >
      {/* The empty message is prose, and prose does not fit a 44px strip. The
          card's own hero value already reads "—" when there is nothing. */}
      {pointCount === 0 && !spark && <div className="chart-empty">No data for this metric yet</div>}
      {tooltip !== null && (
        <div className="chart-tooltip" style={tooltipStyle}>
          {tooltip}
        </div>
      )}
    </div>
  );
}

function buildTooltip(
  data: PlotData,
  series: ChartSeriesSpec[],
  idx: number,
  xLabel: string,
  percent?: boolean,
) {
  const step = data[0]?.[idx];
  if (step === undefined) return null;
  const rows = series
    .map((spec, index) => ({ spec, value: (data[index + 1] as (number | null)[] | undefined)?.[idx] ?? null }))
    .filter((row) => row.spec.hidden !== true && row.spec.muted !== true);
  if (rows.length === 0) return null;

  return (
    <>
      <div className="chart-tooltip-step">
        {xLabel} {step}
      </div>
      {rows.map((row) => (
        <div className="chart-tooltip-row" key={row.spec.label}>
          <span className="chart-legend-swatch" style={{ background: row.spec.color }} />
          <span>{row.spec.label}</span>
          <span className="chart-tooltip-row-value">
            {row.value === null ? "—" : formatMetric(row.value, { percent })}
          </span>
        </div>
      ))}
    </>
  );
}
