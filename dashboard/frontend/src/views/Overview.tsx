/**
 * The Overview page: exactly eight cards.
 *
 * The count is eight because DESIGN.md's grid is four columns and the goal is
 * "job status legible within five seconds": two rows of four is one fixation per
 * row, and the card order is the reading order of the question being
 * asked. State first (is it alive at all), then what it is doing, then how far
 * along, then how long, then what it has saved, then the verdict, then the number
 * the run exists to move, then anything that wants attention.
 *
 * Nothing on this page recomputes health. The health card and the bar above it
 * render `status.health` verbatim; the anomalies card holds the metric-side signals
 * the server does not compute, and says so.
 */

import { useMemo } from "react";
import type { RunStatus, RunTemplate, Series } from "../api/types";
import type { MetricSignal } from "../lib/signals";
import {
  Badge,
  Card,
  CardHint,
  CardValue,
  Code,
  Note,
  Progress,
  Row,
} from "../components/primitives";
import { Chart } from "../components/Chart";
import {
  age,
  ageSince,
  basename,
  bytes,
  clockTime,
  duration,
  EMPTY,
  integer,
  metric as formatMetric,
  semanticsLabel,
  timestamp,
} from "../lib/format";
import { alignSeries, lastStep, lastValue, seriesColor } from "../lib/series";

export interface OverviewProps {
  status: RunStatus;
  template: RunTemplate | null;
  /** The watch set: series fetched for the north star and the signals. */
  series: Record<string, Series>;
  signals: MetricSignal[];
  /** How many series the signal checks looked at, for the anomalies card. */
  watchedCount: number;
  now: number;
  onOpenMetrics: () => void;
}

export function Overview(props: OverviewProps) {
  const { status, template, series, signals, watchedCount, now } = props;
  const snapshot = status.snapshot;
  const progress = snapshot?.progress;
  const timing = snapshot?.timing;
  const semantics = semanticsLabel(progress?.step_semantics ?? status.manifest?.step_semantics);

  const northStarKey = template?.north_star?.resolved ? template.north_star.key : null;
  const northStar = northStarKey ? series[northStarKey] : undefined;
  const northStarValue = lastValue(northStar);
  const northStarPercent = template?.north_star?.format === "percent";

  const components = Object.entries(snapshot?.components ?? {});
  const checkpoint = snapshot?.latest_checkpoint ?? null;

  // The north-star sparkline. `useMemo` keyed on the series identity keeps the
  // aligned arrays stable across re-renders so the chart is not fed a new array
  // (and forced to redraw) every time an unrelated card updates.
  const northStarData = useMemo(() => alignSeries([northStar]), [northStar]);

  // Startup and damage look identical on disk -- no snapshot either way -- and
  // reporting both as damage made every launch look like an incident. The server
  // tells them apart; the page only has to stop conflating them.
  const initializing = status.initializing === true;
  // A run in a terminal state has no future to forecast. Several cards read
  // differently once that is true, so the fact is named once.
  const terminal =
    snapshot?.state === "finished" ||
    snapshot?.state === "failed" ||
    snapshot?.state === "stopped";

  return (
    <>
      {initializing && (
        <Note title="Starting up">
          The run has registered but has not published its first snapshot yet.
          Cluster boot, worker allocation and model load all happen in this
          window
          {status.startup_elapsed_s != null
            ? `, and it has been ${duration(status.startup_elapsed_s)} so far.`
            : "."}
        </Note>
      )}
      {status.error && (
        <Note tone="error" title="Snapshot unreadable">
          {status.error}
        </Note>
      )}

      <div className="cards">
        {/* 1. State -- the lifecycle fact the training process recorded. */}
        <Card
          label="State"
          adornment={<Badge tone={snapshot?.state ?? (initializing ? "pending" : "unknown")} />}
        >
          <CardValue small empty={!snapshot?.state && !initializing}>
            {snapshot?.state ?? (initializing ? "initializing" : "unknown")}
          </CardValue>
          <CardHint>
            {snapshot?.exit
              ? snapshot.exit.reason
              : status.manifest?.experiment_name
                ? `${status.manifest.experiment_name} · ${status.manifest.task_type}`
                : (status.manifest?.task_type ?? "no manifest")}
          </CardHint>
          <div className="card-foot">
            <CardHint>
              Started {timestamp(snapshot?.timing.started_at ?? status.manifest?.started_at)}
            </CardHint>
          </div>
        </Card>

        {/* 2. Phase -- or a component strip, when the runner is async.
             An async run has env, rollout and actor all live for the whole loop
             (they are started before the `while`, not inside it), so a single
             scalar phase is a semantic error for it. Both shapes render. */}
        <Card
          label={components.length > 0 ? "Components" : "Phase"}
          adornment={
            components.length > 0 ? <span className="chip">async</span> : undefined
          }
        >
          {components.length > 0 ? (
            <div className="components">
              {components.map(([name, state]) => (
                <div className="component" key={name} data-active={state.active ? "true" : "false"}>
                  <span className="component-dot" />
                  <span className="component-name">{name}</span>
                  <span className="sr-only">{state.active ? "active" : "idle"}</span>
                  {/* `since` is when the component entered its current state, so
                      the same number means "active for" or "idle for" depending
                      on which state that is. Printed bare it read as an update
                      time, which is a different fact entirely. */}
                  <span
                    className="component-since"
                    title={`${state.active ? "Active" : "Idle"} since ${state.since ?? "unknown"}`}
                  >
                    {state.since
                      ? `${state.active ? "active" : "idle"} for ${age(ageSince(state.since, now))}`
                      : EMPTY}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <>
              <CardValue small empty={!snapshot?.phase}>
                {snapshot?.phase ?? (snapshot?.state === "running" ? "—" : "not running")}
              </CardValue>
              <CardHint>
                {snapshot?.phase_since
                  ? `in phase for ${duration(ageSince(snapshot.phase_since, now))}`
                  : "no phase recorded"}
              </CardHint>
            </>
          )}
          <div className="card-foot">
            <CardHint>
              {status.manifest?.cluster?.num_nodes
                ? `${status.manifest.cluster.num_nodes} node${status.manifest.cluster.num_nodes === 1 ? "" : "s"}`
                : "placement unknown"}
              {status.manifest?.algorithm?.loss_type
                ? ` · ${status.manifest.algorithm.loss_type}`
                : ""}
            </CardHint>
          </div>
        </Card>

        {/* 3. Progress -- with the step semantics beside it, always. The same
             "step 400" is an RL iteration here and a minibatch elsewhere. */}
        <Card label="Progress">
          <CardValue>
            {integer(progress?.step ?? 0)}
            <span className="faint"> / {progress?.max_steps ? integer(progress.max_steps) : "?"}</span>
          </CardValue>
          <Progress
            step={progress?.step ?? 0}
            maxSteps={progress?.max_steps ?? null}
            semantics={semantics}
          />
          <div className="card-foot">
            <CardHint>
              {progress?.epoch !== null && progress?.epoch !== undefined
                ? `epoch ${progress.epoch}`
                : "no epoch reported"}
            </CardHint>
          </div>
        </Card>

        {/* 4. Timing -- elapsed as the hero, with the ETA and its confidence.
             The confidence is shown because a `low`-confidence ETA on a two-step
             run is a projection from one sample and should not be trusted.
             A run that has stopped has no time remaining to estimate, so the row
             states the outcome instead: `ETA 0.0s (medium)` on a finished run is
             a forecast of the past. */}
        <Card label="Timing">
          <CardValue>{duration(timing?.elapsed_s)}</CardValue>
          <div className="card-rows">
            <Row label={terminal ? "finished" : "ETA"}>
              {terminal
                ? `${snapshot?.state ?? "ended"} after ${duration(timing?.elapsed_s)}`
                : timing?.eta_s === null || timing?.eta_s === undefined
                  ? EMPTY
                  : `${duration(timing.eta_s)}${timing.eta_confidence ? ` (${timing.eta_confidence})` : ""}`}
            </Row>
            <Row label={`per ${semantics.toLowerCase()}`}>{duration(timing?.step_time_p50)}</Row>
          </div>
        </Card>

        {/* 5. Latest checkpoint -- with the resume command assembled from the
             fields, not from a pre-baked string. A stored command goes stale the
             moment anything about the launch changes. */}
        <Card
          label="Latest checkpoint"
          adornment={checkpoint?.is_best ? <span className="chip">best</span> : undefined}
        >
          <CardValue small empty={!checkpoint}>
            {checkpoint ? basename(checkpoint.path) : "none yet"}
          </CardValue>
          {checkpoint ? (
            <div className="card-rows">
              <Row label="saved">{clockTime(checkpoint.saved_at)}</Row>
              <Row label="size">{bytes(checkpoint.size_bytes)}</Row>
              <Row label="took">{duration(checkpoint.duration_s)}</Row>
            </div>
          ) : (
            <CardHint title="The index is appended only after a save finishes, so a half-written checkpoint is never listed.">
              No checkpoints saved yet.
            </CardHint>
          )}
        </Card>

        {/* 6. Health -- the server's verdict and its reason, verbatim. */}
        <Card label="Health" adornment={<Badge tone={status.health.health} />}>
          <CardValue small>{status.health.health}</CardValue>
          <CardHint>{status.health.reason}</CardHint>
          <div className="card-foot">
            <div className="card-rows">
              {status.health.heartbeat_age_s !== null && (
                <Row label="heartbeat">{age(status.health.heartbeat_age_s)}</Row>
              )}
              {status.health.progress_age_s !== null && (
                <Row label="last step">{age(status.health.progress_age_s)}</Row>
              )}
              {status.health.budget_s !== null && (
                <Row label="budget">{duration(status.health.budget_s)}</Row>
              )}
            </div>
          </div>
        </Card>

        {/* 7. North star -- the one metric the template says this run exists to
             move. `resolved: false` says the run does not log it, rather than
             showing an empty hero number that reads as a broken run. */}
        <Card
          label={template?.north_star?.label ?? "North-star metric"}
          adornment={
            northStarKey ? (
              <button className="chart-flag" onClick={props.onOpenMetrics} type="button">
                open metric
              </button>
            ) : undefined
          }
        >
          {northStarKey ? (
            <>
              <CardValue empty={northStarValue === null}>
                {formatMetric(northStarValue, { percent: northStarPercent })}
              </CardValue>
              <CardHint>
                <Code title={northStarKey}>{northStarKey}</Code>
                {lastStep(northStar) !== null && (
                  <span className="faint"> at {semantics.toLowerCase()} {lastStep(northStar)}</span>
                )}
              </CardHint>
              <div className="card-foot">
                {/* A 44px strip, not a chart: the shape is the information here,
                    and the full chart is one click away on the metrics tab. The
                    height is `spark`, not an inline style on a wrapper -- a
                    wrapper does not constrain a child that sets its own height,
                    which is how this drew through the bottom of the card. */}
                <Chart
                  data={northStarData}
                  series={[
                    {
                      label: northStarKey,
                      color: seriesColor(0),
                      fill: true,
                    },
                  ]}
                  xLabel={semantics}
                  percent={northStarPercent}
                  spark
                />
              </div>
            </>
          ) : (
            <>
              <CardValue small empty>
                not logged
              </CardValue>
              <CardHint>
                {template?.north_star?.key
                  ? `This run logs no ${template.north_star.key}. The ${template?.name ?? "default"} template expects it.`
                  : `The ${template?.name ?? "default"} template declares no north-star metric for this task type.`}
              </CardHint>
            </>
          )}
        </Card>

        {/* Metric-side anomalies stay separate from the server health verdict;
            they require series data that snapshot-only health does not read. */}
        <Card
          label="Anomalies"
          adornment={
            <span className="chip" title="Computed in the browser from metric series">
              derived from metrics
            </span>
          }
        >
          {signals.length === 0 ? (
            <>
              <CardValue small empty>
                none
              </CardValue>
              <CardHint>
                No step-time regression, eval plateau or non-finite value in{" "}
                {watchedCount} watched series.
              </CardHint>
            </>
          ) : (
            <div className="card-rows">
              {signals.map((signal) => (
                <div className="signal" data-level={signal.level} key={signal.id}>
                  <span className="signal-mark" />
                  <span className="signal-body">
                    <span className="signal-title">{signal.title}</span>{" "}
                    <span className="sr-only">
                      ({signal.level === "red" ? "critical" : "warning"})
                    </span>
                    {signal.detail}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
