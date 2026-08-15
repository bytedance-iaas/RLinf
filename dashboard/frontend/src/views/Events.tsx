/**
 * The event log: `events.jsonl`, newest first.
 *
 * This is the view an operator lands on after the health bar has gone amber, so it
 * is a log, not a dashboard: dense rows, a fixed timestamp column, and the payload
 * shown as-is. The payload is arbitrary JSON written by the training side, so it is
 * rendered as compact key=value pairs rather than being interpreted -- guessing at
 * a shape the runner is free to change would make a wrong reading look authoritative.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
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

/** Rows per page. */
const PAGE_SIZE = 50;

/**
 * How many events are fetched.
 *
 * The endpoint returns the *tail* of `events.jsonl`, so a run with more than
 * this has already lost its oldest entries before the page sees them. That is
 * worth stating rather than hiding behind pagination: paging to the last page
 * of a truncated log looks exactly like reaching the start of the run.
 */
const FETCH_LIMIT = 500;

export function Events(props: EventsProps) {
  const runId = props.status.run_id;
  const stepUnit = semanticsShort(
    props.status.snapshot?.progress.step_semantics ?? props.status.manifest?.step_semantics,
  );

  const [problemsOnly, setProblemsOnly] = useState(false);

  const eventsQuery = useFetch<RunEvent[]>(
    useCallback((signal) => api.events(runId, FETCH_LIMIT, signal), [runId]),
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

  const [page, setPage] = useState(0);
  // Back to the first page whenever the set of rows changes meaning. Staying on
  // page 4 of a filter that now has one page reads as an empty log.
  useEffect(() => setPage(0), [runId, problemsOnly]);

  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  // Clamped rather than stored back: a live run can shrink the filtered set
  // between renders, and writing the correction into state would re-render to
  // fix a number the user never saw.
  const current = Math.min(page, pageCount - 1);
  const start = current * PAGE_SIZE;
  const pageRows = rows.slice(start, start + PAGE_SIZE);

  // The tail was full, so `events.jsonl` had at least this many lines and the
  // oldest are not on this page or any other.
  const truncated = (eventsQuery.data ?? []).length >= FETCH_LIMIT;

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
          {rows.length === 0
            ? `0 of ${(eventsQuery.data ?? []).length}`
            : `${start + 1}–${start + pageRows.length} of ${rows.length}`}
          {problemsOnly && ` (filtered from ${(eventsQuery.data ?? []).length})`}
        </span>
        {problemCount > 0 && (
          <span className="chip">
            {problemCount} warn/error
          </span>
        )}
        {pageCount > 1 && (
          <div className="control">
            <span>Page</span>
            <div className="control-group">
              <button
                className="btn"
                type="button"
                onClick={() => setPage(0)}
                disabled={current === 0}
                aria-label="First page"
              >
                ‹‹
              </button>
              <button
                className="btn"
                type="button"
                onClick={() => setPage(current - 1)}
                disabled={current === 0}
                aria-label="Previous page"
              >
                ‹
              </button>
              <span className="control-value" aria-live="polite">
                {current + 1} / {pageCount}
              </span>
              <button
                className="btn"
                type="button"
                onClick={() => setPage(current + 1)}
                disabled={current >= pageCount - 1}
                aria-label="Next page"
              >
                ›
              </button>
              <button
                className="btn"
                type="button"
                onClick={() => setPage(pageCount - 1)}
                disabled={current >= pageCount - 1}
                aria-label="Last page"
              >
                ››
              </button>
            </div>
          </div>
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
        {pageRows.map((event, index) => (
          <div className="event" data-kind={event.kind} key={`${event.ts}-${start + index}`}>
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

      {/* Said on the last page only, where "no more rows" would otherwise read as
          "this is where the run began". The server returns the tail of the file,
          so what is missing is the oldest entries -- the launch and early phases,
          which is exactly what someone paging backwards is looking for. */}
      {truncated && current === pageCount - 1 && (
        <Note title="Older events not shown">
          This log is the most recent {FETCH_LIMIT} entries. The run wrote more
          before them, including its start, and they are in{" "}
          <Code>events.jsonl</Code> under the run root.
        </Note>
      )}

      {/* The checkpoint index, on the same page: "when did it last save" and "what
          happened" are the same question during an incident, and the resume path is
          what the operator needs next. */}
      {checkpoints.length > 0 && (
        <section className="section">
          <div className="section-head">
            <span className="section-title">Checkpoints</span>
            <span className="section-meta">{checkpoints.length}</span>
          </div>
          <div className="table-scroll">
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
          </div>
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
