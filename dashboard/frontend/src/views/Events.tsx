/**
 * The event log: `events.jsonl`, newest first.
 *
 * This is the view an operator lands on after the health bar has gone amber, so it
 * is a log, not a dashboard: dense rows, a fixed timestamp column, and the payload
 * shown as-is. The payload is arbitrary JSON written by the training side, so it is
 * rendered as compact key=value pairs rather than being interpreted -- guessing at
 * a shape the runner is free to change would make a wrong reading look authoritative.
 */

import { useCallback, useMemo, useState } from "react";
import { api } from "../api/client";
import { useFetch } from "../api/useLive";
import type { CheckpointEntry, EventKind, RunEvent, RunStatus } from "../api/types";
import { Code, Note } from "../components/primitives";
import { bytes, clockTime, duration, integer, semanticsShort, timestamp } from "../lib/format";

export interface EventsProps {
  status: RunStatus;
  /** Bumped when the run advances, so the log refetches without a timer. */
  dataVersion: number;
}

/** Kinds the "problems only" filter keeps. */
const PROBLEM_KINDS: ReadonlySet<string> = new Set<EventKind>(["warn", "error"]);

export function Events(props: EventsProps) {
  const runId = props.status.run_id;
  const stepUnit = semanticsShort(
    props.status.snapshot?.progress.step_semantics ?? props.status.manifest?.step_semantics,
  );

  const [problemsOnly, setProblemsOnly] = useState(false);

  const eventsQuery = useFetch<RunEvent[]>(
    useCallback((signal) => api.events(runId, 500, signal), [runId]),
    [runId, props.dataVersion],
  );
  const checkpointsQuery = useFetch<CheckpointEntry[]>(
    useCallback((signal) => api.checkpoints(runId, signal), [runId]),
    [runId, props.dataVersion],
  );

  // Newest first. The server returns the tail of the file in file order, which is
  // append order; an operator reading a log after an alert wants the last thing
  // that happened at the top.
  const rows = useMemo(() => {
    const all = [...(eventsQuery.data ?? [])].reverse();
    return problemsOnly ? all.filter((event) => PROBLEM_KINDS.has(event.kind)) : all;
  }, [eventsQuery.data, problemsOnly]);

  const problemCount = (eventsQuery.data ?? []).filter((event) =>
    PROBLEM_KINDS.has(event.kind),
  ).length;

  const exit = props.status.snapshot?.exit ?? null;
  const checkpoints = checkpointsQuery.data ?? [];

  return (
    <div className="stack">
      <div className="controls">
        <div className="control">
          <span>Filter</span>
          <div className="control-group">
            <button
              className="btn"
              type="button"
              data-active={!problemsOnly ? "true" : undefined}
              onClick={() => setProblemsOnly(false)}
            >
              all
            </button>
            <button
              className="btn"
              type="button"
              data-active={problemsOnly ? "true" : undefined}
              onClick={() => setProblemsOnly(true)}
            >
              warn + error
            </button>
          </div>
        </div>
        <span className="control-value faint">
          {rows.length} of {(eventsQuery.data ?? []).length}
        </span>
        {problemCount > 0 && (
          <span className="chip">
            {problemCount} warn/error
          </span>
        )}
      </div>

      {/* The exit reason and its traceback tail, when the run ended badly. This is
          the single most useful thing on the page for a failed run, so it sits above
          the log rather than being the last row of it. */}
      {exit && (
        <Note tone={props.status.snapshot?.state === "failed" ? "error" : undefined} title="Exit">
          <div>{exit.reason}</div>
          {exit.traceback_tail && <code className="code-block">{exit.traceback_tail}</code>}
        </Note>
      )}

      {eventsQuery.error && (
        <Note tone="error" title="Event log unreadable">
          {eventsQuery.error}
        </Note>
      )}

      {!eventsQuery.loading && rows.length === 0 && (
        <Note title={problemsOnly ? "No warnings or errors" : "No events"}>
          {problemsOnly
            ? "Nothing in this run's log is a warning or an error."
            : "This run wrote no events.jsonl entries. The file is appended by the runner at phase boundaries, checkpoint saves and evals, so an empty log usually means the run has not reached one yet."}
        </Note>
      )}

      <div className="events">
        {rows.map((event, index) => (
          <div className="event" data-kind={event.kind} key={`${event.ts}-${index}`}>
            <span className="event-time" title={timestamp(event.ts)}>
              {clockTime(event.ts)}
            </span>
            <span className="event-kind">{event.kind}</span>
            <span className="event-step">
              {event.step === null ? "" : `${stepUnit} ${event.step}`}
            </span>
            <span className="event-payload">{payloadText(event.payload)}</span>
          </div>
        ))}
      </div>

      {/* The checkpoint index, on the same page: "when did it last save" and "what
          happened" are the same question during an incident, and the resume path is
          what the operator needs next. */}
      {checkpoints.length > 0 && (
        <section className="section">
          <div className="section-head">
            <span className="section-title">Checkpoints</span>
            <span className="section-desc">
              Appended only after a save completes, so a half-written checkpoint is never
              listed.
            </span>
            <span className="section-meta">{checkpoints.length}</span>
          </div>
          <table className="table">
            <thead>
              <tr>
                <th className="col-num">{stepUnit}</th>
                <th>Saved</th>
                <th className="col-num">Size</th>
                <th className="col-num">Took</th>
                <th>Path</th>
              </tr>
            </thead>
            <tbody>
              {[...checkpoints].reverse().map((entry) => (
                <tr key={`${entry.step}-${entry.path}`}>
                  <td className="col-num">
                    {integer(entry.step)}
                    {entry.is_best && <span className="chip">best</span>}
                  </td>
                  <td>{clockTime(entry.saved_at)}</td>
                  <td className="col-num">{bytes(entry.size_bytes)}</td>
                  <td className="col-num">{duration(entry.duration_s)}</td>
                  <td>
                    <Code title={entry.path}>{entry.path}</Code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {/* Assembled from the recorded fields rather than from a stored command
              string: a baked command goes stale the moment the launch changes. */}
          {resumeHint(checkpoints)}
        </section>
      )}
    </div>
  );
}

/** The payload as compact `key=value` pairs, in the order the writer emitted them. */
function payloadText(payload: Record<string, unknown>): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(payload)) {
    if (value === null || value === undefined) continue;
    parts.push(`${key}=${typeof value === "object" ? JSON.stringify(value) : String(value)}`);
  }
  return parts.join("  ");
}

function resumeHint(checkpoints: CheckpointEntry[]) {
  const latest = [...checkpoints].reverse().find((entry) => entry.resume_dir !== null);
  if (!latest?.resume_dir) return null;
  const script = latest.entry_script ?? "python examples/<task>/train_*.py";
  const config = latest.config_name ? ` --config-name ${latest.config_name}` : "";
  return (
    <Note title="Resume from the latest checkpoint">
      <code className="code-block">
        {script}
        {config} runner.resume_dir={latest.resume_dir}
      </code>
    </Note>
  );
}
