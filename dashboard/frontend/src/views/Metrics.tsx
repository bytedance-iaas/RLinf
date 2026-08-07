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
 * `unmatched` is rendered, in a collapsed group titled "Other keys". A metric the
 * server found but no chart claimed would otherwise vanish, and a dropped metric
 * looks like a missing feature rather than like an unclaimed key. The title says
 * what the group holds rather than why it exists: "unmatched" is a fact about the
 * template, and nobody reading a run wants to be told about the template.
 *
 * Groups render a title and a chart count, and no prose. Standing subtitles like
 * "Is the update stable?" are read once and then sit there costing vertical space
 * on every visit, pushing the charts -- the thing being looked at -- down the page.
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
  bundleRanks,
  lastValue,
  rankMode,
  seriesColor,
  shortKey,
  smooth,
  stackColumns,
  totalPoints,
  workersWith,
  type PlotData,
  type RankLine,
} from "../lib/series";
import { nonFiniteSignal } from "../lib/signals";
import { SmoothingControl } from "../components/SmoothingControl";

export interface MetricsProps {
  status: RunStatus;
  template: RunTemplate | null;
  templateError: string | null;
  /** Bumped when the run's step advances, to refetch series without a timer. */
  dataVersion: number;
  /**
   * `(group, rank)` labels this run wrote per-worker metrics for, from `/keys`.
   *
   * Empty for every run that did not set `runner.per_worker_log: true`, which is
   * the default -- so the drill-down control is absent rather than present and
   * empty.
   */
  workers?: string[];
}

export function Metrics(props: MetricsProps) {
  const { status, template, templateError, dataVersion } = props;
  const runId = status.run_id;
  const workers = props.workers ?? [];

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

  /**
   * Whether to break every chart out per `(worker group, rank)`.
   *
   * Off by default even for a run that has the data: the aggregate is what the
   * page is for, and expanding multiplies both the response and the number of
   * lines by the rank count. It is asked for when the aggregate has already
   * raised the question ("iteration time doubled") and the next question is
   * whether one card is responsible.
   */
  const [expanded, setExpanded] = useState(false);
  const expandRanks = expanded && workers.length > 0;
  // Expansion has to be requested from the server, not derived: the per-rank
  // series live in different event directories than the aggregate.
  const chartWorkers = expandRanks ? workers : [];

  const seriesQuery = useFetch<Record<string, Series>>(
    useCallback(
      (signal) => api.series(runId, allKeys, signal, expandRanks),
      [runId, allKeys, expandRanks],
    ),
    [runId, allKeys.join("\0"), expandRanks, dataVersion],
  );
  const series = seriesQuery.data ?? {};
  // Counted against the template's keys rather than over the response, which
  // carries one entry per rank as well once expanded.
  const resolvedKeys = allKeys.filter((key) => series[key] !== undefined).length;

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
        {/* Absent, not disabled, for a run without per-worker logging: a control
            that can never do anything is a question about a feature rather than an
            answer about this run. */}
        {workers.length > 0 && (
          <label className="control">
            <input
              type="checkbox"
              checked={expanded}
              onChange={(event) => setExpanded(event.target.checked)}
            />
            <span>Expand to ranks</span>
            <span className="control-value" title={workers.join("\n")}>
              {workers.length}
            </span>
          </label>
        )}
        {anyDecimated(Object.values(series)) && (
          <span className="chip" title="The server strided-sampled these series to stay under its point cap">
            sampled
          </span>
        )}
        <span className="control-value faint">
          {resolvedKeys}/{allKeys.length} keys
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
          workers={chartWorkers}
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
          workers={chartWorkers}
          folded={folded?.[groupKey(group)] ?? group.collapsed === true}
          onToggle={() => toggle(groupKey(group))}
          cursorGroup={`run-${runId}`}
        />
      ))}

      {unmatched.length > 0 && (
        <Group
          group={{
            title: "Other keys",
            charts: unmatched.map((key) => ({ keys: [key], title: key })),
          }}
          series={series}
          axisLabel={axisLabel}
          smoothing={smoothing}
          workers={chartWorkers}
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
  /** Worker labels to break each chart out by; empty when not expanded. */
  workers: string[];
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
              workers={props.workers}
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
  /**
   * Worker labels to break this chart's keys out by. Empty means aggregate only,
   * which is every chart until the drill-down is switched on.
   */
  workers?: string[];
  tall?: boolean;
  cursorGroup?: string;
}) {
  const { chart, series, smoothing } = props;
  const unit = chart.unit ?? props.groupUnit;
  const percent = chart.format === "percent";
  const logScale = chart.scale === "log";
  // Stacking and a rank breakdown are mutually exclusive. A stack asserts that its
  // bands sum to a meaningful total; adding four ranks of each band stacks the same
  // seconds five times over and the top of the stack becomes a number that
  // describes nothing. So the stack keeps its aggregates and the expansion is
  // dropped for this chart, which the panel says out loud rather than silently.
  const rankWorkers = chart.stacked === true ? [] : (props.workers ?? []);
  const stacked = chart.stacked === true;
  const stackBlockedRanks = stacked && (props.workers?.length ?? 0) > 0;

  const [hidden, setHidden] = useState<Record<string, boolean>>({});

  /**
   * The per-rank lines, one run of them per metric this chart draws.
   *
   * `mode` is decided across the whole chart so two metrics in one panel are never
   * drawn under two different conventions, and each metric's ranks continue from
   * the ramp slot the previous one stopped at, so no two lines in the panel share a
   * colour in `distinct` mode.
   */
  const rankLines = useMemo<RankLine[]>(() => {
    if (rankWorkers.length === 0) return [];
    // Narrowed per key before counting: `env/*` exists only on the env group's
    // ranks, so the line count -- and therefore the mode -- depends on which
    // workers logged what, not on how many workers the run has.
    const perKey = chart.keys.map((key) => workersWith(key, rankWorkers, series));
    const mode = rankMode(
      chart.keys.length,
      perKey.reduce((total, list) => total + list.length, 0),
    );

    const lines: RankLine[] = [];
    chart.keys.forEach((key, index) => {
      lines.push(
        ...bundleRanks(key, perKey[index] as string[], series, {
          // Aggregates hold slots 0..keys.length-1; ranks continue from there.
          colorOffset: chart.keys.length + lines.length,
          mode,
        }),
      );
    });
    return lines;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chart.keys.join("|"), rankWorkers.join("|"), series]);

  // Aggregates first, then the ranks. Order is the draw order, so a faint bundle
  // never paints over the aggregate line it is context for.
  const plotKeys = [...chart.keys, ...rankLines.map((line) => line.key)];
  const entries = plotKeys.map((key) => series[key]);

  const data = useMemo<PlotData>(() => {
    const aligned = alignSeries(entries);
    const [xs, ...columns] = aligned;
    // Smoothing is applied before stacking, so the bands still sum to the smoothed
    // total rather than to a mix of raw and smoothed parts.
    const smoothed = columns.map((column) => smooth(column, smoothing));
    return [xs, ...(stacked ? stackColumns(smoothed) : smoothed)];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plotKeys.join("|"), entries.map((entry) => entry?.total_points ?? 0).join(","), smoothing, stacked, series]);

  const specs: ChartSeriesSpec[] = [
    ...chart.keys.map((key, index) => ({
      label: shortKey(key),
      title: key,
      color: seriesColor(index),
      hidden: hidden[key] === true,
      fill: stacked,
    })),
    ...rankLines.map((line) => ({
      // The worker label alone: within one chart the metric prefix is shared, and
      // `Chart.buildTooltip` keys its rows by label, so what has to be unique is
      // the part that differs. For a multi-metric chart the metric leaf is kept.
      label: chart.keys.length > 1 ? `${shortKey(line.metric)} ${line.worker}` : line.worker,
      title: line.note ? `${line.key} (${line.note})` : line.key,
      color: line.color,
      hidden: hidden[line.key] === true,
      muted: !line.legend,
    })),
  ];

  // A non-finite value in a charted series is called out on the panel, not just in
  // the overview's anomalies card: the chart shows a gap where the point should be,
  // and a gap with no explanation reads as missing data rather than as a NaN.
  //
  // Over the rank lines too, once they are drawn. The aggregate is a mean, so one
  // rank's NaN does poison it -- but the mean can only say "somewhere", and naming
  // the rank is the entire reason to be looking at this chart expanded.
  const nonFinite = useMemo(() => {
    const subset: Record<string, Series> = {};
    for (const key of plotKeys) {
      const entry = series[key];
      if (entry) subset[key] = entry;
    }
    return nonFiniteSignal(subset);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plotKeys.join("|"), series]);

  const logDropped = logScale && data.slice(1).some((column) =>
    (column as (number | null)[]).some((value) => value !== null && value <= 0),
  );

  const bundled = rankLines.filter((line) => !line.legend).length;

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
          {/* Per panel, not just on the control at the top of the page. The
              control scrolls out of view, the setting survives navigating into a
              run and back, and a smoothed curve is a different shape from the
              data — so a panel that does not say so is showing something the
              reader did not ask for and cannot see they are getting. The compare
              view has always carried this flag; the metrics view had not. */}
          {smoothing > 0 && (
            <span
              className="chart-flag"
              title={
                `Exponential moving average over ${smoothing} points, applied for display ` +
                `only — the underlying values are unchanged. Smoothing flattens and delays ` +
                `spikes, so set it to off before reading this chart for anomalies.`
              }
            >
              smoothed {smoothing}pt
            </span>
          )}
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
              <span>
                {shortKey(key)}
                {/* Named as the mean once ranks are beside it: the aggregate is an
                    arithmetic mean across ranks, so "which line is the whole run"
                    has to be answerable from the legend. */}
                {rankLines.length > 0 && <span className="faint"> mean</span>}
              </span>
              <span className="chart-legend-value">{formatMetric(value, { percent })}</span>
            </button>
          );
        })}
        {/* Only the singled-out ranks get an entry. In `distinct` mode that is all
            of them; in the wide-bundle modes a legend listing thirty-two ranks
            would be taller than the plot and could not be read anyway. */}
        {rankLines
          .filter((line) => line.legend)
          .map((line) => {
            const off = hidden[line.key] === true;
            return (
              <button
                className="chart-legend-item"
                key={line.key}
                data-off={off ? "true" : undefined}
                title={line.note ? `${line.key} — ${line.note}` : line.key}
                type="button"
                onClick={() => setHidden((current) => ({ ...current, [line.key]: !off }))}
              >
                <span className="chart-legend-swatch" style={{ background: line.color }} />
                <span>
                  {chart.keys.length > 1 && `${shortKey(line.metric)} `}
                  {line.worker}
                  {line.note && <span className="faint"> {line.note}</span>}
                </span>
                <span className="chart-legend-value">
                  {formatMetric(lastValue(series[line.key]), { percent })}
                </span>
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
      {/* Said once per panel, because a stacked chart silently refusing to expand
          while its neighbours expanded would read as missing per-rank data. */}
      {stackBlockedRanks && (
        <div className="chart-note">
          stacked — shown as the aggregate; stacking N ranks of each band would sum
          the same time N times
        </div>
      )}
      {/* How many lines the legend is not naming, so a faint band is understood as
          the other ranks rather than as a rendering artefact. */}
      {bundled > 0 && (
        <div className="chart-note">
          {bundled} more {bundled === 1 ? "rank" : "ranks"} drawn unlabelled
          {rankLines.some((line) => line.legend)
            ? " — named lines are the extremes and the median"
            : ""}
        </div>
      )}
      {!nonFinite && totalPoints(entries) === 1 && (
        <div className="chart-note">single point — plotted as a marker</div>
      )}
    </div>
  );
}
