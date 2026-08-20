/**
 * The run list: every discovered run, live over SSE.
 *
 * Sort order is fixed and does not follow the data. A list that reorders itself
 * while someone is reading it is called out in DESIGN.md, and it is worse here than
 * elsewhere: the rows are click targets, and a row that moves under the pointer
 * opens the wrong run. So rows are ordered by start time descending -- an immutable
 * property -- and the "needs attention" grouping is expressed by a leading marker,
 * not by moving the row.
 */

import { useMemo, useState } from "react";
import type { Health, RunState, RunSummary } from "../api/types";
import { Badge, Code, Note } from "../components/primitives";
import { age, ageSince, duration, integer, semanticsLabel } from "../lib/format";
import { statusLabel, t, tNode } from "../lib/i18n";

export interface RunListProps {
  runs: RunSummary[];
  /**
   * The first discovery has not returned yet.
   *
   * Separate from `runs.length === 0` because the two demand opposite copy: one
   * says "wait", the other says "something is misconfigured". Collapsing them
   * meant the very first render accused the user's scan root of not existing,
   * seconds before listing the runs it found under it.
   */
  discovering: boolean;
  /** From `GET /api/health`, so an empty result can name the root it searched. */
  scanRoot?: { path: string; exists: boolean; run_count: number };
  selected: string[];
  now: number;
  onOpen: (runId: string) => void;
  onToggleSelect: (runId: string) => void;
  onCompare: () => void;
}

/**
 * Attention order, for the "needs attention" card only -- never for row order.
 *
 * `unknown` sorts above `healthy`: "we cannot tell" is not "fine".
 */
const HEALTH_RANK: Record<Health, number> = {
  unreachable: 0,
  degraded: 1,
  unknown: 2,
  healthy: 3,
};

/**
 * React key for a row.
 *
 * `run_id` alone is not unique. One scan root can hold two copies of the same
 * tree -- a run copied off a cluster beside the original is the ordinary case --
 * and the server lists both, since deduplicating them would hide the fact that
 * there are two. `run_root` is what actually distinguishes them, so it is
 * appended: a key collision here makes React drop or duplicate rows silently,
 * which on a click target is a wrong-run-opened bug rather than a cosmetic one.
 */
export function rowKey(run: RunSummary): string {
  return `${run.run_id}\0${run.run_root}`;
}

export function RunList(props: RunListProps) {
  const { runs, selected, now } = props;
  const [stateFilter, setStateFilter] = useState<RunState | "all">("all");
  const [query, setQuery] = useState("");
  // Scan-root diagnostics come from the server's configured discovery root.
  const root = props.scanRoot;

  const rows = useMemo(() => {
    const filtered = runs.filter((run) => {
      if (stateFilter !== "all" && run.state !== stateFilter) return false;
      if (query.trim() === "") return true;
      const needle = query.trim().toLowerCase();
      return (
        run.run_id.toLowerCase().includes(needle) ||
        (run.experiment_name ?? "").toLowerCase().includes(needle) ||
        (run.task_type ?? "").toLowerCase().includes(needle)
      );
    });
    // Started-at descending, with the run id and then the run root as tiebreaks so
    // the order is total: two runs that started in the same second never swap
    // places between pushes, and neither do two copies of the same run found under
    // different paths under the scan root.
    return [...filtered].sort((a, b) => {
      const at = a.started_at ? Date.parse(a.started_at) : 0;
      const bt = b.started_at ? Date.parse(b.started_at) : 0;
      return (
        bt - at || a.run_id.localeCompare(b.run_id) || a.run_root.localeCompare(b.run_root)
      );
    });
  }, [runs, stateFilter, query]);

  /**
   * Every run that is not healthy, `unknown` included.
   *
   * `unknown` belongs here for the reason `HEALTH_RANK` puts it above `healthy`:
   * "we cannot tell" is not "fine". It is what a run with no heartbeat file at
   * all looks like, which is usually a reporter that never started.
   *
   * Worst first, unlike the table below. Reordering is only a hazard where rows
   * are click targets; these are labels, and the run that most needs opening
   * should be the one read first.
   */
  const attention = useMemo(
    () =>
      runs
        // A run inside its startup window has nothing to report yet, so its
        // `unknown` health is the absence of an answer rather than a bad one.
        // Listing it here made every launch open with a warning banner.
        .filter((run) => run.health !== "healthy" && run.initializing !== true)
        .sort(
          (a, b) =>
            HEALTH_RANK[a.health] - HEALTH_RANK[b.health] ||
            a.run_id.localeCompare(b.run_id),
        ),
    [runs],
  );

  // Run ids that appear more than once, i.e. the same tree found at two paths.
  // Those rows are otherwise identical, so they get their run root shown --
  // without it the list reads as a duplicated row, which looks like a bug in the
  // dashboard rather than two copies of a tree on disk.
  const duplicated = useMemo(() => {
    const counts = new Map<string, number>();
    for (const run of runs) counts.set(run.run_id, (counts.get(run.run_id) ?? 0) + 1);
    return new Set([...counts.entries()].filter(([, n]) => n > 1).map(([id]) => id));
  }, [runs]);

  /**
   * Ids shared by runs with *different* experiment names.
   *
   * The benign duplicate above is one tree reachable by two paths: the
   * rows describe the same run, so resolving the id to either is correct and
   * showing `run_root` is enough to explain the repetition.
   *
   * This is the other kind, and it is not cosmetic. These are different runs --
   * different names, different log paths, different numbers -- that collided on
   * an id, which can happen with explicit or legacy run IDs. Every part of the
   * product addresses a run by id, so opening any of
   * them shows whichever the server resolves first, and the page gives no sign
   * it is not the one that was clicked.
   */
  const collided = useMemo(() => {
    const names = new Map<string, Set<string>>();
    for (const run of runs) {
      const set = names.get(run.run_id) ?? new Set<string>();
      set.add(run.experiment_name ?? run.run_id);
      names.set(run.run_id, set);
    }
    return new Set(
      [...names.entries()].filter(([, seen]) => seen.size > 1).map(([id]) => id),
    );
  }, [runs]);

  return (
    <div className="stack">
      <div className="controls">
        <label className="control">
          <span>{t("runlist.search")}</span>
          <input
            type="search"
            value={query}
            placeholder={t("runlist.searchPlaceholder")}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <div className="control">
          <span>{t("runlist.state")}</span>
          <div className="control-group">
            {(["all", "running", "finished", "failed", "stopped", "pending"] as const).map((value) => (
              <button
                className="btn"
                key={value}
                type="button"
                data-active={stateFilter === value ? "true" : undefined}
                onClick={() => setStateFilter(value)}
              >
                {statusLabel(value)}
              </button>
            ))}
          </div>
        </div>
        <button
          className="btn btn-primary"
          type="button"
          disabled={selected.length < 2}
          onClick={props.onCompare}
          title={selected.length < 2 ? t("runlist.compareHint") : undefined}
        >
          {selected.length > 0
            ? t("runlist.compareN", { count: selected.length })
            : t("runlist.compare")}
        </button>
      </div>

      {collided.size > 0 && (
        <Note
          tone="error"
          title={t(collided.size === 1 ? "runlist.collided.one" : "runlist.collided.other", {
            count: collided.size,
          })}
        >
          <div>
            {tNode("runlist.collidedBody", {
              emphasis: <strong>{t("runlist.collidedEmphasis")}</strong>,
            })}
          </div>
          {[...collided].map((id) => (
            <div key={id} className="faint" style={{ marginTop: "var(--space-xs)" }}>
              <Code>{id}</Code>{" "}
              {runs
                .filter((run) => run.run_id === id)
                .map((run) => run.experiment_name ?? run.run_id)
                .join(", ")}
            </div>
          ))}
          <div style={{ marginTop: "var(--space-xs)" }}>
            {tNode("runlist.collidedHint", { code: <Code>runner.run_id</Code> })}
          </div>
        </Note>
      )}

      {attention.length > 0 && (
        <Note
          tone="warn"
          title={t(
            attention.length === 1 ? "runlist.attention.one" : "runlist.attention.other",
            { count: attention.length },
          )}
        >
          {attention.map((run) => (
            <div key={rowKey(run)}>
              <Badge tone={run.health} /> {run.experiment_name ?? run.run_id}
            </div>
          ))}
        </Note>
      )}

      <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th style={{ width: 32 }} aria-label={t("runlist.selectForCompare")} />
              <th>{t("runlist.col.run")}</th>
              <th>{t("runlist.col.state")}</th>
              <th>{t("runlist.col.health")}</th>
              <th>{t("runlist.col.phase")}</th>
              <th className="col-num">{t("runlist.col.step")}</th>
              <th className="col-num">{t("runlist.col.elapsed")}</th>
              <th className="col-num">{t("runlist.col.eta")}</th>
              <th className="col-num">{t("runlist.col.ckpt")}</th>
              <th className="col-num">{t("runlist.col.heartbeat")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((run) => (
              <tr
                key={rowKey(run)}
                data-selected={selected.includes(rowKey(run)) ? "true" : undefined}
              >
                <td>
                  {/* Selection uses row identity because run IDs may collide. */}
                  <input
                    type="checkbox"
                    checked={selected.includes(rowKey(run))}
                    onChange={() => props.onToggleSelect(rowKey(run))}
                    aria-label={t("runlist.selectRunForCompare", {
                      name: run.experiment_name ?? run.run_id,
                    })}
                  />
                </td>
                <td>
                  <a
                    className="table-link"
                    href={`#/runs/${encodeURIComponent(run.run_id)}`}
                    title={run.run_id}
                  >
                    {run.experiment_name ?? run.run_id}
                  </a>
                  <div className="faint" style={{ fontSize: "var(--type-label-sm-size)" }}>
                    {run.task_type} · {semanticsLabel(run.step_semantics)}
                    {duplicated.has(run.run_id) && <> · {run.run_root}</>}
                  </div>
                </td>
                <td>
                  {run.state ? (
                    <Badge tone={run.state} />
                  ) : run.initializing ? (
                    <Badge tone="pending">{t("status.initializing")}</Badge>
                  ) : (
                    <Badge tone="unknown" />
                  )}
                </td>
                <td>
                  <Badge tone={run.health} />
                </td>
                <td className="muted">{run.phase ?? "—"}</td>
                <td className="col-num">
                  {integer(run.step)}
                  {run.max_steps ? <span className="faint">/{integer(run.max_steps)}</span> : null}
                </td>
                <td className="col-num">{duration(run.elapsed_s)}</td>
                <td className="col-num">{run.eta_s === null ? "—" : duration(run.eta_s)}</td>
                <td className="col-num">
                  {run.latest_checkpoint_step === null ? "—" : integer(run.latest_checkpoint_step)}
                </td>
                <td className="col-num">{age(ageSince(run.heartbeat_at, now))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {rows.length === 0 &&
        (props.discovering ? (
          <Note title={t("runlist.discoveringTitle")}>{t("runlist.discoveringBody")}</Note>
        ) : runs.length === 0 ? (
          <Note title={t("runlist.noneTitle")}>
            {root === undefined
              ? t("runlist.noneNoRoot")
              : !root.exists
                ? t("runlist.noneMissingRoot", { path: root.path })
                : t("runlist.noneEmptyRoot", { path: root.path })}
          </Note>
        ) : (
          <Note title={t("runlist.noMatchTitle")}>{t("runlist.noMatchBody")}</Note>
        ))}
    </div>
  );
}
