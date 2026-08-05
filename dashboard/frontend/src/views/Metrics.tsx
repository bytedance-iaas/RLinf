/**
 * The one generic, template-driven chart renderer.
 *
 * There is no per-task-type code anywhere in this file. It renders whatever
 * `GET /api/runs/{id}/template` returns, honouring every option the template
 * schema can express: `format: percent`, `unit`, `scale: log`, `stacked`, group
 * `collapsed`, `north_star` (including `resolved: false`), `step_axis_label`, and
 * `unmatched`. Adding an `sft` or `reasoning` template is a YAML change on the
 * server and nothing here changes -- which is the whole point, since the four
 * shipped templates differ only in data.
 *
 * `unmatched` is rendered, in a collapsed group. A metric the server found but no
 * chart claimed would otherwise vanish, and a dropped metric looks like a missing
 * feature rather than like an unclaimed key.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { useFetch } from "../api/useLive";
import type { RunStatus, RunTemplate, Series, TemplateChart, TemplateGroup } from "../api/types";
import { Chart, type ChartSeriesSpec } from "../components/Chart";
import { Code, Note } from "../components/primitives";
import { metric as formatMetric, semanticsLabel } from "../lib/format";
import {
  alignSeries,
  anyDecimated,
  lastValue,
  seriesColor,
  shortKey,
  smooth,
  stackColumns,
  totalPoints,
  type PlotData,
} from "../lib/series";
import { nonFiniteSignal } from "../lib/signals";
import { SmoothingControl } from "../components/SmoothingControl";

export interface MetricsProps {
  status: RunStatus;
  template: RunTemplate | null;
  templateError: string | null;
  /** Bumped when the run's step advances, to refetch series without a timer. */
  dataVersion: number;
}

export function Metrics(props: MetricsProps) {
  const { status, template, templateError, dataVersion } = props;
  const runId = status.run_id;

  const axisLabel =
    template?.step_axis_label ??
    semanticsLabel(status.snapshot?.progress.step_semantics ?? status.manifest?.step_semantics);

  // Every key the template asks for, deduplicated. One request for the whole page
  // rather than one per chart: 25 charts is 25 round trips, and the server reads
  // one event-file accumulator for all of them anyway.
  const allKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const group of template?.groups ?? []) {
      for (const chart of group.charts) for (const key of chart.keys) keys.add(key);
    }
    for (const key of template?.unmatched ?? []) keys.add(key);
    return [...keys];
  }, [template]);

  const seriesQuery = useFetch<Record<string, Series>>(
    useCallback(
      (signal) => api.series(runId, allKeys, signal),
      [runId, allKeys],
    ),
    [runId, allKeys.join("\0"), dataVersion],
  );
  const series = seriesQuery.data ?? {};

  const [smoothing, setSmoothing] = useState(0);
  // Group folding is per group title, initialised from the template's `collapsed`.
  const [folded, setFolded] = useState<Record<string, boolean> | null>(null);
  useEffect(() => {
    if (!template) return;
    const initial: Record<string, boolean> = {};
    for (const group of template.groups) {
      initial[groupKey(group)] = group.collapsed === true;
    }
    initial[UNMATCHED_KEY] = true;
    setFolded(initial);
  }, [template]);

  const toggle = useCallback((key: string) => {
    setFolded((current) => ({ ...(current ?? {}), [key]: !(current?.[key] ?? false) }));
  }, []);

  if (templateError) {
    return (
      <Note tone="error" title="No template">
        {templateError}
      </Note>
    );
  }
  if (!template) {
    return <Note>Loading the run's chart layout…</Note>;
  }

  const northStarKey = template.north_star?.resolved ? template.north_star.key : null;
  const unmatched = template.unmatched ?? [];

  return (
    <div className="stack">
      <div className="controls">
        <div className="control">
          <span>Template</span>
          <Code>{template.name}</Code>
        </div>
        <div className="control">
          <span>{axisLabel} axis</span>
          {/* The axis label is the run's own step semantics. It is shown as a
              control-bar fact rather than only on each axis, because the whole
              page shares one x meaning and a reader has to know it once. */}
        </div>
        <SmoothingControl value={smoothing} onChange={setSmoothing} />
        {anyDecimated(Object.values(series)) && (
          <span className="chip" title="The server strided-sampled these series to stay under its point cap">
            sampled
          </span>
        )}
        <span className="control-value faint">
          {Object.keys(series).length}/{allKeys.length} keys
        </span>
      </div>

      {template.caveats?.map((caveat) => (
        <Note tone="warn" key={caveat}>
          {caveat}
        </Note>
      ))}

      {seriesQuery.error && (
        <Note tone="error" title="Series request failed">
          {seriesQuery.error}
        </Note>
      )}

      {/* The north-star chart is promoted above its group and rendered tall, which
          is the one place `chart-height-tall` is used. */}
      {northStarKey && (
        <ChartPanel
          chart={{
            keys: [northStarKey],
            title: template.north_star?.label ?? northStarKey,
            format: template.north_star?.format,
          }}
          series={series}
          axisLabel={axisLabel}
          smoothing={smoothing}
          tall
          cursorGroup={`run-${runId}`}
        />
      )}
      {template.north_star && template.north_star.resolved === false && template.north_star.key && (
        <Note tone="warn" title="North-star metric not logged">
          The <Code>{template.name}</Code> template expects{" "}
          <Code>{template.north_star.key}</Code>, which this run does not log. Every other chart
          below is unaffected.
        </Note>
      )}

      {template.groups.map((group) => (
        <Group
          key={groupKey(group)}
          group={group}
          series={series}
          axisLabel={axisLabel}
          smoothing={smoothing}
          folded={folded?.[groupKey(group)] ?? group.collapsed === true}
          onToggle={() => toggle(groupKey(group))}
          cursorGroup={`run-${runId}`}
        />
      ))}

      {unmatched.length > 0 && (
        <Group
          group={{
            title: "Unmatched keys",
            description:
              "Logged by this run but claimed by no chart in the template. Shown so a metric cannot silently disappear.",
            charts: unmatched.map((key) => ({ keys: [key], title: key })),
          }}
          series={series}
          axisLabel={axisLabel}
          smoothing={smoothing}
          folded={folded?.[UNMATCHED_KEY] ?? true}
          onToggle={() => toggle(UNMATCHED_KEY)}
          cursorGroup={`run-${runId}`}
        />
      )}
    </div>
  );
}

const UNMATCHED_KEY = "\0unmatched";

function groupKey(group: TemplateGroup): string {
  return group.title ?? "(untitled)";
}

function Group(props: {
  group: TemplateGroup;
  series: Record<string, Series>;
  axisLabel: string;
  smoothing: number;
  folded: boolean;
  onToggle: () => void;
  cursorGroup: string;
}) {
  const { group, folded } = props;
  return (
    <section className="group">
      <button className="group-head" onClick={props.onToggle} type="button" aria-expanded={!folded}>
        <span className="group-caret">{folded ? "▸" : "▾"}</span>
        <span className="group-title">{group.title ?? "Metrics"}</span>
        {group.description && <span className="group-desc">{group.description}</span>}
        <span className="group-count">{group.charts.length}</span>
      </button>
      {/* Folded groups are unmounted, not hidden. A collapsed group holding twenty
          canvases would keep paying for their redraws on every push. */}
      {!folded && (
        <div className="chart-grid">
          {group.charts.map((chart) => (
            <ChartPanel
              key={chart.keys.join("|")}
              chart={chart}
              // A group-level `unit` applies to every chart in it (the fallback
              // template's `time/` bucket uses this); a chart-level one wins.
              groupUnit={group.unit}
              series={props.series}
              axisLabel={props.axisLabel}
              smoothing={props.smoothing}
              cursorGroup={props.cursorGroup}
            />
          ))}
        </div>
      )}
    </section>
  );
}

export function ChartPanel(props: {
  chart: TemplateChart;
  groupUnit?: string;
  series: Record<string, Series>;
  axisLabel: string;
  smoothing: number;
  tall?: boolean;
  cursorGroup?: string;
}) {
  const { chart, series, smoothing } = props;
  const unit = chart.unit ?? props.groupUnit;
  const percent = chart.format === "percent";
  const logScale = chart.scale === "log";
  const stacked = chart.stacked === true;

  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const entries = chart.keys.map((key) => series[key]);

  const data = useMemo<PlotData>(() => {
    const aligned = alignSeries(entries);
    const [xs, ...columns] = aligned;
    // Smoothing is applied before stacking, so the bands still sum to the smoothed
    // total rather than to a mix of raw and smoothed parts.
    const smoothed = columns.map((column) => smooth(column, smoothing));
    return [xs, ...(stacked ? stackColumns(smoothed) : smoothed)];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chart.keys.join("|"), entries.map((entry) => entry?.total_points ?? 0).join(","), smoothing, stacked, series]);

  const specs: ChartSeriesSpec[] = chart.keys.map((key, index) => ({
    label: shortKey(key),
    title: key,
    color: seriesColor(index),
    hidden: hidden[key] === true,
    fill: stacked,
  }));

  // A non-finite value in a charted series is called out on the panel, not just in
  // the overview's anomalies card: the chart shows a gap where the point should be,
  // and a gap with no explanation reads as missing data rather than as a NaN.
  const nonFinite = useMemo(() => {
    const subset: Record<string, Series> = {};
    for (const key of chart.keys) {
      const entry = series[key];
      if (entry) subset[key] = entry;
    }
    return nonFiniteSignal(subset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chart.keys.join("|"), series]);

  const logDropped = logScale && data.slice(1).some((column) =>
    (column as (number | null)[]).some((value) => value !== null && value <= 0),
  );

  return (
    <div className="chart-panel" data-tall={props.tall ? "true" : undefined}>
      <div className="chart-head">
        <span className="chart-title">{chart.title ?? chart.keys[0]}</span>
        {unit && <span className="chart-unit">({unit})</span>}
        <span className="chart-flags">
          {stacked && <span className="chart-flag" title="Series are stacked">stacked</span>}
          {logScale && (
            <span
              className="chart-flag"
              title={
                logDropped
                  ? "Template asks for a log scale, but this run has zero or negative values; linear is used instead"
                  : "Log scale"
              }
            >
              {logDropped ? "log n/a" : "log"}
            </span>
          )}
          {percent && <span className="chart-flag" title="Displayed as a percentage">%</span>}
        </span>
      </div>

      {/* Legend above the plot, so plot width does not depend on key length.
          Each entry is a toggle and shows the series' latest value. */}
      <div className="chart-legend">
        {chart.keys.map((key, index) => {
          const off = hidden[key] === true;
          const value = lastValue(series[key]);
          return (
            <button
              className="chart-legend-item"
              key={key}
              data-off={off ? "true" : undefined}
              title={key}
              type="button"
              onClick={() => setHidden((current) => ({ ...current, [key]: !off }))}
            >
              <span className="chart-legend-swatch" style={{ background: seriesColor(index) }} />
              <span>{shortKey(key)}</span>
              <span className="chart-legend-value">{formatMetric(value, { percent })}</span>
            </button>
          );
        })}
      </div>

      <Chart
        data={data}
        series={specs}
        xLabel={props.axisLabel}
        unit={unit}
        percent={percent}
        logScale={logScale && !logDropped}
        stacked={stacked}
        tall={props.tall}
        cursorGroup={props.cursorGroup}
      />

      {nonFinite && (
        <div className="chart-note" data-level="red">
          {nonFinite.detail}
        </div>
      )}
      {!nonFinite && totalPoints(entries) === 1 && (
        <div className="chart-note">single point — plotted as a marker</div>
      )}
    </div>
  );
}
