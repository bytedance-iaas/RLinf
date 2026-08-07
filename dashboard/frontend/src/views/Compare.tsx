/**
 * Multi-run compare: the same metric key overlaid, per-run toggles, and a
 * smoothing control.
 *
 * The one thing this view has to get right beyond overlaying lines is the x axis.
 * Two runs with different `step_semantics` on one axis is a meaningless comparison
 * that will be believed anyway, so a mixed selection is called out in the UI and
 * the minority runs are drawn dashed. The overlay is still allowed -- refusing to
 * draw it would just push the operator to two browser tabs and a guess -- but it
 * cannot be silent.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { useFetch } from "../api/useLive";
import type { RunSummary, Series } from "../api/types";
import { Chart, type ChartSeriesSpec } from "../components/Chart";
import { Code, Note } from "../components/primitives";
import { SmoothingControl } from "../components/SmoothingControl";
import { duration, metric as formatMetric, semanticsLabel } from "../lib/format";
import {
  alignSeries,
  lastValue,
  seriesColor,
  smooth,
  type PlotData,
} from "../lib/series";

export interface CompareProps {
  runs: RunSummary[];
  selected: string[];
  metricKey: string | null;
  onChange: (runIds: string[], key: string | null) => void;
}

export function Compare(props: CompareProps) {
  const { runs, selected, metricKey } = props;

  // Only runs the user picked contribute keys. Listing the union across every
  // discovered run would offer keys that no selected run has, and the resulting
  // empty chart looks like a broken request.
  const keyQueries = useFetch<Record<string, string[]>>(
    useCallback(
      async (signal) => {
        const out: Record<string, string[]> = {};
        await Promise.all(
          selected.map(async (runId) => {
            try {
              out[runId] = (await api.keys(runId, signal)).keys;
            } catch {
              // A run that has gone away must not blank the whole picker.
              out[runId] = [];
            }
          }),
        );
        return out;
      },
      [selected],
    ),
    [selected.join(" ")],
  );

  /** Keys present in every selected run, which is what is actually comparable. */
  const sharedKeys = useMemo(() => {
    const lists = Object.values(keyQueries.data ?? {});
    if (lists.length === 0) return [];
    const [first, ...rest] = lists as string[][];
    return (first ?? []).filter((key) => rest.every((list) => list.includes(key))).sort();
  }, [keyQueries.data]);

  // Keys some but not all selected runs log. Offered separately, because a metric
  // only one arm logs is often exactly what you want to look at -- but the chart
  // will show one line, and the reason should not be a mystery.
  const partialKeys = useMemo(() => {
    const lists = Object.entries(keyQueries.data ?? {});
    const counts = new Map<string, number>();
    for (const [, list] of lists) {
      for (const key of list) counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts.entries()]
      .filter(([, count]) => count > 0 && count < lists.length)
      .map(([key]) => key)
      .sort();
  }, [keyQueries.data]);

  useEffect(() => {
    // Default to the first shared key rather than leaving the chart empty. No
    // preference for a particular metric name: this view is task-type agnostic.
    if (metricKey === null && sharedKeys.length > 0) {
      props.onChange(selected, sharedKeys[0] as string);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [metricKey, sharedKeys]);

  const seriesQuery = useFetch<Record<string, Series | undefined>>(
    useCallback(
      async (signal) => {
        if (!metricKey) return {};
        const out: Record<string, Series | undefined> = {};
        await Promise.all(
          selected.map(async (runId) => {
            try {
              out[runId] = (await api.series(runId, [metricKey], signal))[metricKey];
            } catch {
              out[runId] = undefined;
            }
          }),
        );
        return out;
      },
      [selected, metricKey],
    ),
    [selected.join(" "), metricKey],
  );

  const [smoothing, setSmoothing] = useState(0);
  const [off, setOff] = useState<Record<string, boolean>>({});

  const byId = useMemo(() => new Map(runs.map((run) => [run.run_id, run])), [runs]);

  // Step semantics across the selection. The most common one owns the axis; the
  // rest are drawn dashed and named in a warning above the chart.
  const semanticsCounts = new Map<string, number>();
  for (const runId of selected) {
    const value = byId.get(runId)?.step_semantics ?? "unknown";
    semanticsCounts.set(value, (semanticsCounts.get(value) ?? 0) + 1);
  }
  const dominant =
    [...semanticsCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "unknown";
  const mixed = semanticsCounts.size > 1;

  const seriesList = selected.map((runId) => seriesQuery.data?.[runId]);

  const data = useMemo<PlotData>(() => {
    const [xs, ...columns] = alignSeries(seriesList);
    return [xs, ...columns.map((column) => smooth(column, smoothing))];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seriesQuery.data, selected.join(" "), smoothing]);

  const specs: ChartSeriesSpec[] = selected.map((runId, index) => {
    const run = byId.get(runId);
    return {
      label: run?.experiment_name ?? runId,
      title: runId,
      color: seriesColor(index),
      hidden: off[runId] === true,
      // A dashed line for a run whose steps do not mean what the axis says.
      dashed: mixed && (run?.step_semantics ?? "unknown") !== dominant,
    };
  });

  const percent = metricKey !== null && /success|accuracy|pass@|_frac|fraction|_ratio/.test(metricKey);

  return (
    <div className="stack">
      <div className="controls">
        <label className="control">
          <span>Metric</span>
          <select
            value={metricKey ?? ""}
            onChange={(event) => props.onChange(selected, event.target.value || null)}
          >
            <option value="">— pick a metric —</option>
            {sharedKeys.length > 0 && (
              <optgroup label={`In all ${selected.length} runs`}>
                {sharedKeys.map((key) => (
                  <option value={key} key={key}>
                    {key}
                  </option>
                ))}
              </optgroup>
            )}
            {partialKeys.length > 0 && (
              <optgroup label="In some runs only">
                {partialKeys.map((key) => (
                  <option value={key} key={key}>
                    {key}
                  </option>
                ))}
              </optgroup>
            )}
          </select>
        </label>
        <SmoothingControl value={smoothing} onChange={setSmoothing} />
        <span className="control-value faint">
          {selected.length} run{selected.length === 1 ? "" : "s"}
        </span>
      </div>

      {/* Do not put two runs with different step semantics on one axis without
          saying so -- the comparison is meaningless and the chart will be believed
          anyway. Named here, and dashed on the plot. */}
      {mixed && (
        <Note tone="warn" title="Mixed step semantics on one axis">
          The selected runs do not agree on what a step is (
          {[...semanticsCounts.entries()]
            .map(([value, count]) => `${count}x ${semanticsLabel(value)}`)
            .join(", ")}
          ). The axis is labelled <Code>{semanticsLabel(dominant)}</Code>; runs using a different
          unit are drawn dashed. Their x values are not comparable to the others.
        </Note>
      )}

      <div className="row">
        {/* Per-run toggles. Colour is never the only cue: the label is always
            present and struck through when the run is hidden. */}
        {selected.map((runId, index) => {
          const run = byId.get(runId);
          const hidden = off[runId] === true;
          return (
            <button
              className="toggle"
              key={runId}
              type="button"
              data-off={hidden ? "true" : undefined}
              onClick={() => setOff((current) => ({ ...current, [runId]: !hidden }))}
              title={runId}
            >
              <span className="toggle-swatch" style={{ background: seriesColor(index) }} />
              <span className="toggle-label">{run?.experiment_name ?? runId}</span>
              <span className="toggle-meta">
                {formatMetric(lastValue(seriesQuery.data?.[runId]), { percent })}
              </span>
            </button>
          );
        })}
      </div>

      {selected.length === 0 ? (
        <Note title="Nothing selected">
          Pick two or more runs on the run list, then choose <em>Compare</em>.
        </Note>
      ) : (
        <div className="chart-panel" data-tall="true">
          <div className="chart-head">
            <span className="chart-title">{metricKey ?? "No metric selected"}</span>
            <span className="chart-flags">
              {percent && <span className="chart-flag">%</span>}
              {smoothing > 0 && <span className="chart-flag">smoothed {smoothing}pt</span>}
            </span>
          </div>
          <Chart
            data={data}
            series={specs}
            xLabel={semanticsLabel(dominant)}
            percent={percent}
            tall
            cursorGroup="compare"
          />
        </div>
      )}

      {selected.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>Run</th>
              <th>State</th>
              <th className="col-num">Step</th>
              <th className="col-num">Latest {metricKey ? "value" : ""}</th>
              <th className="col-num">Elapsed</th>
              <th>Step semantics</th>
            </tr>
          </thead>
          <tbody>
            {selected.map((runId) => {
              const run = byId.get(runId);
              return (
                <tr key={runId}>
                  <td title={runId}>{run?.experiment_name ?? runId}</td>
                  <td>{run?.state ?? "unknown"}</td>
                  <td className="col-num">{run?.step ?? "—"}</td>
                  <td className="col-num">
                    {formatMetric(lastValue(seriesQuery.data?.[runId]), { percent })}
                  </td>
                  <td className="col-num">{duration(run?.elapsed_s)}</td>
                  <td>{semanticsLabel(run?.step_semantics)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
