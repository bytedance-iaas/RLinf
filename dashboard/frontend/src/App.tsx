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
import rlinfLogo from "./assets/rlinf-logo.svg";
import { api } from "./api/client";
import { useFetch, useRun, useRuns, type LiveState } from "./api/useLive";
import type { Health, RunStatus, RunSummary, RunTemplate, Series, ServerHealth } from "./api/types";
import { AsOf, Badge, Code, HealthBar, Note } from "./components/primitives";
import { href, routeRunId, useRoute, type Route } from "./lib/router";
import { collectSignals, watchSetKeys } from "./lib/signals";
import { setLang, statusLabel, t, useLang, useT } from "./lib/i18n";
import { setTheme, useTheme } from "./lib/theme";
import { Compare } from "./views/Compare";
import { Events } from "./views/Events";
import { Media } from "./views/Media";
import { Metrics } from "./views/Metrics";
import { Overview } from "./views/Overview";
import { RunList, rowKey } from "./views/RunList";

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
const BASE_TABS = [{ name: "overview" }, { name: "metrics" }] as const;

const EVENTS_TAB = { name: "events" } as const;

/**
 * Whether to offer a Media tab.
 *
 * Decided by `template.has_media_view` alone -- can this *kind* of run have
 * video? A property of the task type, declared in the template YAML, so a new
 * task type answers it in data rather than here.
 *
 * `status.has_media` is deliberately not consulted. It is a fact about what has
 * been written *so far*, and on a live run that starts false and turns true at
 * the first video dump, so gating on it made the tab strip change shape partway
 * through a run. For a task type that records video the tab is part of the run's
 * navigation whether or not a clip has landed; the empty view says so, which is
 * a stable answer rather than a missing one.
 *
 * `"pending"` while the template is still in flight: the tab strip must not
 * flicker a Media tab in and then remove it, because a tab that appears and
 * vanishes under the pointer is a misclick.
 */
function mediaTabState(template: RunTemplate | null): "show" | "hide" | "pending" {
  if (template === null) return "pending";
  return template.has_media_view === false ? "hide" : "show";
}

export function App() {
  const [route, navigate] = useRoute();
  const runId = routeRunId(route);
  const theme = useTheme();
  // The one language subscription in the app. Everything below re-renders with
  // the shell, so views and helpers call the plain `t` -- see `lib/i18n.ts`.
  const lang = useLang();
  useT();

  // The tab title is chrome too, and it is what the window list shows for a
  // console left open for days.
  useEffect(() => {
    document.title = t("app.title");
  }, [lang]);

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
  //
  // Holds *row* keys (`run_id\0run_root`), not run ids, because a run id can be
  // shared by more than one row and a selection keyed by id cannot say which of
  // them was ticked. The compare route still speaks run ids -- that is all the
  // API can address -- so the two are converted at this boundary.
  const [selected, setSelected] = useState<string[]>([]);
  const runRows = runs.data ?? [];
  useEffect(() => {
    // A compare deep link is authoritative: someone pasted it, and the selection
    // it encodes has to win over whatever this tab had selected. The link carries
    // ids, so every row claiming one of them is ticked; with a duplicated id that
    // is the honest answer, since the link cannot say which was meant.
    if (route.name !== "compare" || route.runIds.length === 0) return;
    const wanted = new Set(route.runIds);
    setSelected(runRows.filter((r) => wanted.has(r.run_id)).map(rowKey));
    // `runRows` is a dependency because the runs list usually arrives after the
    // route does, and a link opened in a cold tab would otherwise select nothing.
  }, [route, runRows]);

  /** Run ids for the selected rows, deduplicated, in selection order. */
  const selectedRunIds = useMemo(() => {
    const byKey = new Map(runRows.map((r) => [rowKey(r), r.run_id]));
    const ids: string[] = [];
    for (const key of selected) {
      const id = byKey.get(key);
      if (id !== undefined && !ids.includes(id)) ids.push(id);
    }
    return ids;
  }, [selected, runRows]);

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

  const mediaTab = mediaTabState(template);
  const tabs = useMemo(
    () => (mediaTab === "show"
      ? [...BASE_TABS, { name: "media" } as const, EVENTS_TAB]
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
          <a
            className="header-brand"
            href={href({ name: "runs" })}
          >
            <img className="header-brand-logo" src={rlinfLogo} alt={t("app.brandAlt")} />
          </a>

          <nav
            className="header-crumbs"
            aria-label={t("app.breadcrumb")}
            data-run={runId ? "true" : undefined}
          >
            <a href={href({ name: "runs" })}>{t("runlist.runs")}</a>
            {route.name === "compare" && (
              <>
                <span className="header-crumb-sep">/</span>
                <span>{t("compare.title", { count: selectedRunIds.length })}</span>
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
            <button
              className="theme-toggle"
              type="button"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label={t(theme === "dark" ? "app.themeToLight" : "app.themeToDark")}
              title={t(theme === "dark" ? "app.themeToLight" : "app.themeToDark")}
            >
              {theme === "dark" ? (
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M20.5 14.3A8.5 8.5 0 0 1 9.7 3.5a8.5 8.5 0 1 0 10.8 10.8Z" />
                </svg>
              )}
              <span className="theme-toggle-label">
                {t(theme === "dark" ? "app.themeLight" : "app.themeDark")}
              </span>
            </button>
            {/* Two languages, so a button rather than a picker: a select for a
                binary choice costs a click and a menu to say what a label
                already says. The label is the language it switches *to*, written
                in that language -- someone who cannot read the current UI has to
                be able to find the way out of it. */}
            <button
              className="theme-toggle lang-toggle"
              type="button"
              onClick={() => setLang(lang === "zh" ? "en" : "zh")}
              aria-label={t("app.langToggleTitle")}
              title={t("app.langToggleTitle")}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="12" cy="12" r="9" />
                <path d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18" />
              </svg>
              <span className="theme-toggle-label">{t("app.langToggle")}</span>
            </button>
            {/* Refresh invalidates the fetched-once resources. The live document
                does not need it -- SSE owns that -- so this is deliberately not a
                "reload the page" button. */}
            {runId && (
              <button className="btn" type="button" onClick={() => setRefreshNonce((n) => n + 1)}>
                {t("app.refresh")}
              </button>
            )}
            <span
              className="header-live"
              data-state={liveState}
              title={t("app.liveTitle", { state: liveState })}
            >
              <span className="header-live-dot" />
              {t(`live.${liveState}`)}
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
          <Note tone="error" title={t("app.streamError")}>
            {runId ? run.error : runs.error}
          </Note>
        )}

        {runId && (
          <nav className="tabs" aria-label={t("app.runViews")}>
            {tabs.map((tab) => (
              <a
                className="tab"
                key={tab.name}
                href={href({ name: tab.name, runId })}
                data-active={route.name === tab.name ? "true" : undefined}
              >
                {t(`tab.${tab.name}`)}
              </a>
            ))}
          </nav>
        )}

        <div className="stack" style={{ marginTop: "var(--space-lg)" }}>
          {renderRoute({
            route,
            runs: runs.data ?? [],
            // `data === null` with no error is the only shape that means "the
            // first fetch has not landed". Passed through rather than inferred
            // from an empty array, which cannot tell "none" from "not yet".
            discovering: runs.data === null && runs.error === null,
            scanRoot: serverQuery.data?.scan_root,
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
            selectedRunIds,
            navigate,
            onToggleSelect: toggleSelect,
          })}

          {route.name === "runs" && serverQuery.data && (
            <Note title={t("server.title")}>
              <div className="kv">
                <dt>{t("server.version")}</dt>
                <dd>
                  <Code>{serverQuery.data.version}</Code>
                </dd>
                <dt>{t("server.runs")}</dt>
                {/* Counted from the same live list the table renders, not from
                    the one-shot health fetch. That fetch happens once at page
                    load and never again, so its count froze at whatever was
                    there when the tab opened while the table went on updating --
                    two numbers on one screen disagreeing about the same fact. */}
                <dd>{runRows.length}</dd>
                <dt>{t("server.scanRoot")}</dt>
                <dd>
                  <Code>{serverQuery.data.scan_root.path}</Code>{" "}
                  {/* A scan root that does not exist is the commonest reason for
                      an empty dashboard, and a root one level off the runs is the
                      next -- which "exists" alone cannot distinguish, since both
                      report true. Those two get a badge each.
                      A root that exists and holds runs gets nothing: the count is
                      already the row above, and printing it twice invited the
                      reader to compare two numbers that are allowed to differ --
                      this one froze at page load, that one follows the list. */}
                  {!serverQuery.data.scan_root.exists ? (
                    <Badge tone="unreachable">{t("server.missing")}</Badge>
                  ) : serverQuery.data.scan_root.run_count === 0 ? (
                    <Badge tone="unknown">{t("server.noRunsFound")}</Badge>
                  ) : null}
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
  discovering: boolean;
  scanRoot?: { path: string; exists: boolean; run_count: number };
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
  selectedRunIds: string[];
  navigate: (route: Route) => void;
  onToggleSelect: (runId: string) => void;
}

function renderRoute(args: RenderArgs) {
  const { route, status } = args;

  if (route.name === "runs") {
    return (
      <RunList
        runs={args.runs}
        discovering={args.discovering}
        scanRoot={args.scanRoot}
        selected={args.selected}
        now={args.now}
        onOpen={(runId) => args.navigate({ name: "overview", runId })}
        onToggleSelect={args.onToggleSelect}
        onCompare={() =>
          args.navigate({ name: "compare", runIds: args.selectedRunIds, key: null })
        }
      />
    );
  }

  if (route.name === "compare") {
    return (
      <Compare
        runs={args.runs}
        selected={args.selectedRunIds}
        metricKey={route.key}
        onChange={(runIds, key) => args.navigate({ name: "compare", runIds, key })}
      />
    );
  }

  // Every remaining route is about one run, and none of them can render without
  // the status document. Held as one branch so a slow first fetch shows one
  // placeholder rather than four half-empty panels.
  if (!status) {
    return <Note>{t("app.loadingRun")}</Note>;
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
 * The strip communicates only whether anything needs attention. Run identities
 * and explanations belong to the list and run detail pages.
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
          ? t("rollup.none")
          : t("rollup.summary", {
              total: runs.length,
              health: statusLabel(health),
              bad: notHealthy,
            })
      }
    />
  );
}
