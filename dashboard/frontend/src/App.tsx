/**
 * The app shell: header, health bar, tabs, and the route switch.
 *
 * All of the live wiring lives here rather than in the views, for one reason: a run
 * page shows the same `RunStatus` in four places (the bar, the header, the overview
 * cards, the media view's step semantics), and four independent subscriptions to the
 * same SSE stream would let those four disagree by up to a push interval. One
 * subscription, passed down.
 *
 * The `now` ticker is here for the same reason. Every "3s ago" on the page is
 * derived from a single clock read, so two ages rendered side by side can never be
 * one second apart from each other.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { useFetch, useRun, useRuns, type LiveState } from "./api/useLive";
import type { Health, RunStatus, RunSummary, RunTemplate, Series, ServerHealth } from "./api/types";
import { AsOf, Badge, Code, HealthBar, Note } from "./components/primitives";
import { href, routeRunId, useRoute, type Route } from "./lib/router";
import { collectSignals, watchSetKeys } from "./lib/signals";
import { Compare } from "./views/Compare";
import { Events } from "./views/Events";
import { Media } from "./views/Media";
import { Metrics } from "./views/Metrics";
import { Overview } from "./views/Overview";
import { RunList } from "./views/RunList";

/** How often derived ages are recomputed. One second, because ages are shown to it. */
const TICK_MS = 1000;

/**
 * The tabs every run has. Media is not among them.
 *
 * Only embodied runs record simulator video. An SFT or reasoning run shown a
 * Media tab gets a tab that always leads to an empty page -- and a navigation
 * item that never has content teaches people to distrust the rest of the nav. So
 * the tab is derived rather than fixed; see `mediaTabState`.
 */
const BASE_TABS = [
  { name: "overview", label: "Overview" },
  { name: "metrics", label: "Metrics" },
] as const;

const EVENTS_TAB = { name: "events", label: "Events" } as const;

/**
 * Whether to offer a Media tab, from the two independent facts the server sends.
 *
 * * `template.has_media_view` -- can this *kind* of run have video? A property of
 *   the task type, declared in the template YAML, so a new task type answers it
 *   in data rather than here.
 * * `status.has_media` -- did *this* run record any? `enable_dump_video: false`
 *   is the common case even for embodied runs.
 *
 * `"pending"` while the template is still in flight: the tab strip must not
 * flicker a Media tab in and then remove it, because a tab that appears and
 * vanishes under the pointer is a misclick.
 */
function mediaTabState(
  template: RunTemplate | null,
  status: RunStatus | null,
): "show" | "hide" | "pending" {
  if (template === null || status === null) return "pending";
  if (template.has_media_view === false) return "hide";
  return status.has_media ? "show" : "hide";
}

export function App() {
  const [route, navigate] = useRoute();
  const runId = routeRunId(route);

  // One clock for the whole page.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), TICK_MS);
    return () => window.clearInterval(timer);
  }, []);

  const runs = useRuns();
  const run = useRun(runId);
  const status = run.data;

  // Compare selection. Held here rather than in the list so it survives navigating
  // into a run and back, which is how a comparison actually gets assembled.
  const [selected, setSelected] = useState<string[]>([]);
  useEffect(() => {
    // A compare deep link is authoritative: someone pasted it, and the selection
    // it encodes has to win over whatever this tab had selected.
    if (route.name === "compare" && route.runIds.length > 0) setSelected(route.runIds);
  }, [route]);

  /**
   * Bumped whenever the run advances, and by the refresh button.
   *
   * Series, templates, events and media are fetched once rather than pushed, so
   * they need an invalidation trigger. Tying it to the step rather than to a timer
   * means a finished run stops refetching on its own, and a fast run refetches as
   * fast as it produces data.
   */
  const step = status?.snapshot?.progress.step ?? null;
  const [refreshNonce, setRefreshNonce] = useState(0);
  const [dataVersion, setDataVersion] = useState(0);
  useEffect(() => {
    setDataVersion((value) => value + 1);
  }, [runId, step, refreshNonce]);

  // The template is a property of the run's task type, so it is fetched once per
  // run and not on every advance.
  const templateQuery = useFetch<RunTemplate | null>(
    useCallback(
      (signal) => (runId ? api.template(runId, signal) : Promise.resolve(null)),
      [runId],
    ),
    [runId],
  );
  const template = templateQuery.data;

  // Keys and workers come from one response: both describe what this run logged,
  // and the drill-down toggle must not appear a request later than the charts.
  const keysQuery = useFetch<{ keys: string[]; workers: string[] }>(
    useCallback(
      async (signal) =>
        runId ? await api.keys(runId, signal) : { keys: [], workers: [] },
      [runId],
    ),
    [runId, dataVersion],
  );
  const workers = keysQuery.data?.workers ?? [];

  // The overview's watch set: enough series to run the metric-side checks without
  // pulling every key of a long run into the page that has to be legible in five
  // seconds. The deep-dive view checks everything it renders.
  const watchKeys = useMemo(
    () => watchSetKeys(template, new Set(keysQuery.data?.keys ?? [])),
    [template, keysQuery.data],
  );
  const watchQuery = useFetch<Record<string, Series>>(
    useCallback(
      (signal) =>
        runId && watchKeys.length > 0 ? api.series(runId, watchKeys, signal) : Promise.resolve({}),
      [runId, watchKeys],
    ),
    [runId, watchKeys.join(" "), dataVersion],
  );
  const watchSeries = watchQuery.data ?? {};

  const signals = useMemo(
    () => collectSignals(template, watchSeries),
    [template, watchSeries],
  );

  const serverQuery = useFetch<ServerHealth>(
    useCallback((signal) => api.serverHealth(signal), []),
    [],
  );

  const mediaTab = mediaTabState(template, status);
  const tabs = useMemo(
    () => (mediaTab === "show"
      ? [...BASE_TABS, { name: "media", label: "Media" } as const, EVENTS_TAB]
      : [...BASE_TABS, EVENTS_TAB]),
    [mediaTab],
  );

  // A deep link to a media view this run does not have -- a bookmark from another
  // run, or a link shared before `enable_dump_video` was turned off -- lands on
  // the overview rather than on a page whose only content is an explanation of
  // why it is empty. Replaced rather than pushed so Back does not bounce.
  useEffect(() => {
    if (route.name === "media" && mediaTab === "hide") {
      window.location.replace(href({ name: "overview", runId: route.runId }));
    }
  }, [route, mediaTab]);

  const liveState: LiveState = runId ? run.liveState : runs.liveState;
  const updatedAt = runId ? run.updatedAt : runs.updatedAt;

  const toggleSelect = useCallback((id: string) => {
    setSelected((current) =>
      current.includes(id) ? current.filter((value) => value !== id) : [...current, id],
    );
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <a className="header-brand" href={href({ name: "runs" })}>
            <span className="header-brand-mark">RLinf</span>
            <span className="faint" style={{ fontSize: "var(--type-body-sm-size)" }}>
              dashboard
            </span>
          </a>

          <nav className="header-crumbs" aria-label="Breadcrumb">
            <a href={href({ name: "runs" })}>Runs</a>
            {route.name === "compare" && (
              <>
                <span className="header-crumb-sep">/</span>
                <span>Compare ({selected.length})</span>
              </>
            )}
            {runId && (
              <>
                <span className="header-crumb-sep">/</span>
                <Code title={runId}>{status?.manifest?.experiment_name ?? runId}</Code>
                {status?.snapshot?.state && <Badge tone={status.snapshot.state} />}
              </>
            )}
          </nav>

          <div className="header-right">
            {/* Refresh invalidates the fetched-once resources. The live document
                does not need it -- SSE owns that -- so this is deliberately not a
                "reload the page" button. */}
            {runId && (
              <button className="btn" type="button" onClick={() => setRefreshNonce((n) => n + 1)}>
                Refresh
              </button>
            )}
            <span className="header-live" data-state={liveState} title={`SSE: ${liveState}`}>
              <span className="header-live-dot" />
              {liveState}
            </span>
            <AsOf updatedAt={updatedAt} now={now} />
          </div>
        </div>
      </header>

      {/* The 4px strip, on every page. On a run page it carries that run's
          verdict and the server's reason for it. On the list there is no single
          run to explain, so the strip is the worst verdict as a colour only --
          the naming of which runs is the list's own "needs attention" card, and
          saying it in both places was the same sentence twice. */}
      {runId ? (
        <HealthBar verdict={status?.health ?? null} />
      ) : (
        <RollupBar runs={runs.data ?? []} />
      )}

      <main className="app-main">
        {/* A transport error is shown without discarding the last good document: a
            dropped stream and a dead run are different facts. */}
        {(runId ? run.error : runs.error) && (
          <Note tone="error" title="Live stream reported an error">
            {runId ? run.error : runs.error}
          </Note>
        )}

        {runId && (
          <nav className="tabs" aria-label="Run views">
            {tabs.map((tab) => (
              <a
                className="tab"
                key={tab.name}
                href={href({ name: tab.name, runId })}
                data-active={route.name === tab.name ? "true" : undefined}
              >
                {tab.label}
              </a>
            ))}
          </nav>
        )}

        <div className="stack" style={{ marginTop: "var(--space-lg)" }}>
          {renderRoute({
            route,
            runs: runs.data ?? [],
            status,
            template,
            templateError: templateQuery.error,
            watchSeries,
            watchKeys,
            workers,
            signals,
            dataVersion,
            now,
            selected,
            navigate,
            onToggleSelect: toggleSelect,
          })}

          {route.name === "runs" && serverQuery.data && (
            <Note title="Server">
              <div className="kv">
                <dt>version</dt>
                <dd>
                  <Code>{serverQuery.data.version}</Code>
                </dd>
                <dt>runs</dt>
                <dd>{serverQuery.data.run_count}</dd>
                <dt>scan roots</dt>
                <dd>
                  {serverQuery.data.scan_roots.map((root) => (
                    <div key={root.path}>
                      <Code>{root.path}</Code>{" "}
                      {/* A scan root that does not exist is the single most common
                          reason for an empty dashboard, so it is stated here rather
                          than left to be inferred from a blank table. */}
                      {root.exists ? (
                        <span className="faint">exists</span>
                      ) : (
                        <Badge tone="unreachable">missing</Badge>
                      )}
                    </div>
                  ))}
                </dd>
              </div>
            </Note>
          )}
        </div>
      </main>
    </div>
  );
}

interface RenderArgs {
  route: Route;
  runs: RunSummary[];
  status: RunStatus | null;
  template: RunTemplate | null;
  templateError: string | null;
  watchSeries: Record<string, Series>;
  watchKeys: string[];
  /** `(group, rank)` labels with per-worker metrics; empty for most runs. */
  workers: string[];
  signals: ReturnType<typeof collectSignals>;
  dataVersion: number;
  now: number;
  selected: string[];
  navigate: (route: Route) => void;
  onToggleSelect: (runId: string) => void;
}

function renderRoute(args: RenderArgs) {
  const { route, status } = args;

  if (route.name === "runs") {
    return (
      <RunList
        runs={args.runs}
        selected={args.selected}
        now={args.now}
        onOpen={(runId) => args.navigate({ name: "overview", runId })}
        onToggleSelect={args.onToggleSelect}
        onCompare={() =>
          args.navigate({ name: "compare", runIds: args.selected, key: null })
        }
      />
    );
  }

  if (route.name === "compare") {
    return (
      <Compare
        runs={args.runs}
        selected={args.selected}
        metricKey={route.key}
        onChange={(runIds, key) => args.navigate({ name: "compare", runIds, key })}
      />
    );
  }

  // Every remaining route is about one run, and none of them can render without
  // the status document. Held as one branch so a slow first fetch shows one
  // placeholder rather than four half-empty panels.
  if (!status) {
    return <Note>Loading run…</Note>;
  }

  switch (route.name) {
    case "overview":
      return (
        <Overview
          status={status}
          template={args.template}
          series={args.watchSeries}
          signals={args.signals}
          watchedCount={args.watchKeys.length}
          now={args.now}
          onOpenMetrics={() => args.navigate({ name: "metrics", runId: status.run_id })}
        />
      );
    case "metrics":
      return (
        <Metrics
          status={status}
          template={args.template}
          templateError={args.templateError}
          dataVersion={args.dataVersion}
          workers={args.workers}
        />
      );
    case "media":
      return <Media status={status} />;
    case "events":
      return <Events status={status} dataVersion={args.dataVersion} />;
  }
}

/** Attention order. `unknown` sorts above `healthy`: "we cannot tell" is not "fine". */
const HEALTH_RANK: Record<Health, number> = {
  unreachable: 0,
  degraded: 1,
  unknown: 2,
  healthy: 3,
};

/**
 * The health strip for pages that are not about one run.
 *
 * Colour only, and deliberately so. It used to carry a sentence naming every
 * unhealthy run, which was the same information the list's "N runs need
 * attention" card states directly below it -- twice on one screen, the second
 * time as a comma-joined paragraph rather than a scannable list. The strip's job
 * here is peripheral: is anything wrong at all. Which runs, and why, belong to
 * the card and to each run's own page.
 *
 * It still does not synthesise a `reason`. The list endpoint returns each run's
 * `health` but not the sentence behind it, and writing a plausible one here is the
 * "recomputing health in the browser" failure the design forbids.
 */
function RollupBar(props: { runs: RunSummary[] }) {
  const runs = props.runs;
  const worst = runs.reduce<Health>(
    (acc, run) => (HEALTH_RANK[run.health] < HEALTH_RANK[acc] ? run.health : acc),
    "healthy",
  );
  const health: Health = runs.length === 0 ? "unknown" : worst;
  const notHealthy = runs.filter((run) => run.health !== "healthy").length;

  return (
    <div
      className="healthbar"
      data-health={health}
      role="status"
      // The strip is silent, so the count lives in the accessible name: a screen
      // reader gets what the colour conveys, which is the whole point of the
      // `role="status"` here.
      aria-label={
        runs.length === 0
          ? "No runs discovered yet"
          : `Worst health across ${runs.length} runs: ${health}. ${notHealthy} not healthy.`
      }
    />
  );
}
