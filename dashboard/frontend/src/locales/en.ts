/**
 * English copy. The source of truth for both catalogues.
 *
 * `zh.ts` is typed as `Record<keyof typeof en, string>`, so a key added here and
 * not there fails `tsc`, and a key that exists only there fails too. Empty
 * strings, drifted `{placeholders}` and keys no view references are caught by
 * `npm run check:i18n`.
 *
 * Conventions:
 *
 * * Keys are `area.thing`, and `.one` / `.other` where English needs a plural.
 *   Chinese has one form and maps both to the same string; the call site keeps
 *   the count test, because that is where the number is.
 * * `{placeholders}` are named, never positional -- word order differs between
 *   the two languages and a positional `%s` cannot survive that.
 * * A message that embeds an element (a `<Code>`, an `<em>`) is still one
 *   message, with the element as a placeholder. See `tNode`.
 * * Anything the server wrote -- metric keys, paths, health sentences, event
 *   payloads -- is not in this file. It is rendered verbatim in both languages.
 */

export const en = {
  // -- Shell ---------------------------------------------------------------
  "app.title": "RLinf Control Plane",
  "app.brandAlt": "RLinf",
  "app.breadcrumb": "Breadcrumb",
  "app.runViews": "Run views",
  "app.streamError": "Live stream reported an error",
  "app.loadingRun": "Loading run…",
  "app.refresh": "Refresh",
  "app.themeToLight": "Switch to light theme",
  "app.themeToDark": "Switch to dark theme",
  "app.themeLight": "Light",
  "app.themeDark": "Dark",
  // The label names the language being switched *to*, written in that language:
  // someone who cannot read the current one still has to be able to find it.
  "app.langToggle": "中文",
  "app.langToggleTitle": "切换到中文 (Switch to Chinese)",
  "app.liveTitle": "SSE: {state}",
  "app.updated": "updated {age}",

  "tab.overview": "Overview",
  "tab.metrics": "Metrics",
  "tab.media": "Media",
  "tab.events": "Events",

  // The visible word is binary -- the reader is deciding whether to trust the
  // numbers on screen. The four states below stay, for the tooltip.
  "live.connected": "connected",
  "live.disconnected": "disconnected",
  "live.connecting": "connecting",
  "live.live": "live",
  "live.reconnecting": "reconnecting",
  "live.error": "error",

  // -- Shared vocabulary ---------------------------------------------------
  "status.healthy": "healthy",
  "status.degraded": "degraded",
  "status.unreachable": "unreachable",
  "status.unknown": "unknown",
  "status.running": "running",
  "status.finished": "finished",
  "status.failed": "failed",
  "status.stopped": "stopped",
  "status.pending": "pending",
  "status.initializing": "initializing",
  "status.all": "all",

  "confidence.low": "low",
  "confidence.medium": "medium",
  "confidence.high": "high",

  "semantics.rl_iteration": "RL iteration",
  "semantics.minibatch": "Minibatch",
  "semantics.optimizer_step": "Optimizer step",
  "semantics.step": "Step",
  "semantics.short.rl_iteration": "iter",
  "semantics.short.minibatch": "mb",
  "semantics.short.optimizer_step": "step",
  "semantics.short.step": "step",

  "format.justNow": "just now",
  "format.secondsAgo": "{n}s ago",
  "format.minutesAgo": "{n}m ago",
  "format.hoursAgo": "{n}h ago",
  "format.daysAgo": "{n}d ago",

  "healthbar.aria": "Run health: {health}",

  "pager.label": "Pagination",
  "pager.page": "Page {current} of {total}",
  "pager.range": "{from}–{to} of {total}",
  "pager.first": "First page",
  "pager.prev": "Previous page",
  "pager.next": "Next page",
  "pager.last": "Last page",
  "progress.noHorizon": "no horizon",

  // -- Server card (run list) ----------------------------------------------
  "server.title": "Server",
  "server.version": "version",
  "server.runs": "runs",
  "server.scanRoot": "scan root",
  "server.missing": "missing",
  "server.noRunsFound": "no runs found",
  "server.scanRootChange": "change",
  "server.scanRootSave": "Save",
  "server.scanRootCancel": "Cancel",
  "server.scanRootReset": "reset",
  "server.scanRootResetTitle": "Back to the root this server was started with: {path}",
  "server.scanRootLabel": "Scan root path",
  "server.scanRootPlaceholder": "a directory on the server",

  "rollup.none": "No runs discovered yet",
  "rollup.summary": "Worst health across {total} runs: {health}. {bad} not healthy.",

  // -- Run list ------------------------------------------------------------
  "runlist.runs": "Runs",
  "runlist.search": "Search",
  "runlist.searchPlaceholder": "run id, experiment, task type",
  "runlist.state": "State",
  "runlist.compare": "Compare",
  "runlist.compareN": "Compare ({count})",
  "runlist.compareHint": "Select two or more runs to compare",
  "runlist.selectForCompare": "Select for compare",
  "runlist.selectRunForCompare": "Select {name} for compare",

  "runlist.collided.one": "{count} run id is shared by different runs",
  "runlist.collided.other": "{count} run ids are shared by different runs",
  "runlist.collidedBody":
    "These runs have different names and different log paths but the same run id, " +
    "and every URL and API call in this dashboard addresses a run by id. Opening any " +
    "of them shows whichever the server finds first — {emphasis} — and the same " +
    "applies to comparing them.",
  "runlist.collidedEmphasis": "not necessarily the one you clicked",
  "runlist.collidedHint":
    "The default id is a second-resolution timestamp plus the experiment name, so this " +
    "also happens when a copied config pins {code}. Give each run its own id to tell " +
    "them apart.",

  "common.close": "Close",

  "runlist.attention.one": "1 run needs attention",
  "runlist.attention.other": "{count} runs need attention",
  "runlist.attentionMore": "… and {count} more",

  "runlist.col.run": "Run",
  "runlist.col.state": "State",
  "runlist.col.health": "Health",
  "runlist.col.phase": "Phase",
  "runlist.col.step": "Step",
  "runlist.col.elapsed": "Elapsed",
  "runlist.col.eta": "ETA",
  "runlist.col.ckpt": "Ckpt",
  "runlist.col.heartbeat": "Heartbeat",

  "runlist.discoveringTitle": "Discovering runs",
  "runlist.discoveringBody":
    "Scanning for runs. Nothing is claimed about what is there until this finishes.",
  "runlist.noneTitle": "No runs discovered",
  "runlist.noneNoRoot": "The server has not reported its scan root yet.",
  "runlist.noneMissingRoot": "The scan root {path} does not exist.",
  "runlist.noneEmptyRoot":
    "Searched {path}, which exists but holds no run yet. A run is a directory holding " +
    "_rlinf/runs/<id>/manifest.json, up to six levels below the root.",
  "runlist.noMatchTitle": "No runs match",
  "runlist.noMatchBody":
    "Every discovered run is filtered out by the current search or state filter.",

  // -- Overview ------------------------------------------------------------
  "overview.startingTitle": "Starting up",
  "overview.startingBody":
    "The run has registered but has not published its first snapshot yet. Cluster boot, " +
    "worker allocation and model load all happen in this window.",
  "overview.startingBodyElapsed":
    "The run has registered but has not published its first snapshot yet. Cluster boot, " +
    "worker allocation and model load all happen in this window, and it has been " +
    "{elapsed} so far.",
  "overview.snapshotUnreadable": "Snapshot unreadable",

  "overview.state": "State",
  "overview.components": "Components",
  "overview.phase": "Phase",
  "overview.progress": "Progress",
  "overview.timing": "Timing",
  "overview.checkpoint": "Latest checkpoint",
  "overview.health": "Health",
  "overview.northStar": "North-star metric",
  "overview.anomalies": "Anomalies",

  "overview.noManifest": "no manifest",
  "overview.started": "Started {time}",
  "overview.async": "async",
  "overview.active": "active",
  "overview.idle": "idle",
  "overview.activeFor": "active for {age}",
  "overview.idleFor": "idle for {age}",
  "overview.activeSince": "Active since {time}",
  "overview.idleSince": "Idle since {time}",
  "overview.notRunning": "not running",
  "overview.inPhaseFor": "in phase for {age}",
  "overview.noPhase": "no phase recorded",
  "overview.nodes.one": "{count} node",
  "overview.nodes.other": "{count} nodes",
  "overview.placementUnknown": "placement unknown",
  "overview.epoch": "epoch {epoch}",
  "overview.noEpoch": "no epoch reported",

  "overview.finished": "finished",
  "overview.eta": "ETA",
  "overview.endedAfter": "{state} after {elapsed}",
  "overview.ended": "ended",
  "overview.etaWithConfidence": "{eta} ({confidence})",
  "overview.perStep": "per {unit}",

  "overview.best": "best",
  "overview.noCheckpoint": "none yet",
  "overview.saved": "saved",
  "overview.size": "size",
  "overview.took": "took",
  "overview.noCheckpointHint": "No checkpoints saved yet.",
  "overview.noCheckpointTitle":
    "The index is appended only after a save finishes, so a half-written checkpoint is " +
    "never listed.",

  "overview.heartbeat": "heartbeat",
  "overview.lastStep": "last step",
  "overview.budget": "budget",

  "overview.openMetric": "open metric",
  "overview.notLogged": "not logged",
  "overview.atStep": "at {unit} {step}",
  "overview.northStarMissing":
    "This run logs no {key}. The {template} template expects it.",
  "overview.northStarUndeclared":
    "The {template} template declares no north-star metric for this task type.",
  "overview.templateDefault": "default",

  "overview.derivedFromMetrics": "derived from metrics",
  "overview.derivedTitle": "Computed in the browser from metric series",
  "overview.anomaliesNone": "none",
  "overview.anomaliesNoneHint":
    "No step-time regression, eval plateau or non-finite value in {count} watched series.",
  "overview.critical": "critical",
  "overview.warning": "warning",

  // -- Metric signals (computed in the browser) ----------------------------
  "signal.nonFiniteTitle": "Non-finite metric value",
  "signal.nonFiniteDetail":
    "{key} first went NaN or Inf at step {step} ({count} of {total} points).",
  "signal.nonFiniteDetailMore":
    "{key} first went NaN or Inf at step {step} ({count} of {total} points), and in " +
    "{others} other series.",
  "signal.stepTimeTitle": "Step time degraded",
  "signal.stepTimeDetail":
    "{key} is {ratio}x its early baseline ({recent}s now versus {baseline}s over steps " +
    "{from}-{to}).",
  "signal.plateauTitle": "No eval improvement in {k} rounds",
  "signal.plateauDetail":
    "{key} has not beaten {best} in its last {k} evaluations (best recent {recent}).",

  // -- Metrics -------------------------------------------------------------
  "metrics.noTemplate": "No template",
  "metrics.loadingLayout": "Loading the run's chart layout…",
  "metrics.template": "Template",
  "metrics.axis": "{label} axis",
  "metrics.expandRanks": "Expand to ranks",
  "metrics.sampled": "sampled",
  "metrics.sampledTitle":
    "The server strided-sampled these series to stay under its point cap",
  "metrics.keyCount": "{resolved}/{total} keys",
  "metrics.seriesFailed": "Series request failed",
  "metrics.northStarMissingTitle": "North-star metric not logged",
  "metrics.northStarMissingBody":
    "The {template} template expects {key}, which this run does not log. Every other " +
    "chart below is unaffected.",
  "metrics.otherKeys": "Other keys",
  "metrics.groupFallback": "Metrics",
  "metrics.stackBlockedNote":
    "stacked — shown as the aggregate; stacking N ranks of each band would sum the same " +
    "time N times",
  "metrics.bundled.one": "{count} more rank drawn unlabelled",
  "metrics.bundled.other": "{count} more ranks drawn unlabelled",
  "metrics.bundledNamed": " — named lines are the extremes and the median",
  "metrics.singlePoint": "single point — plotted as a marker",

  // -- Charts --------------------------------------------------------------
  "chart.empty": "No data for this metric yet",
  "chart.total": "Total",
  "chart.zoomReset": "Zoomed · reset",
  "chart.mean": "mean",
  "chart.stacked": "stacked",
  "chart.stackedTitle": "Series are stacked",
  "chart.log": "log",
  "chart.logNa": "log n/a",
  "chart.logTitle": "Log scale",
  "chart.logDroppedTitle":
    "Template asks for a log scale, but this run has zero or negative values; linear is " +
    "used instead",
  "chart.percentTitle": "Displayed as a percentage",
  "chart.smoothed": "smoothed {n}pt",
  "chart.smoothedTitle":
    "Exponential moving average over {n} points, applied for display only — the " +
    "underlying values are unchanged. Smoothing flattens and delays spikes, so set it " +
    "to off before reading this chart for anomalies.",
  "chart.smoothing": "Smoothing",
  "chart.smoothingAria": "Smoothing window in points",
  "chart.smoothingOff": "off",
  "chart.smoothingPoints": "{n}pt",

  // -- Media ---------------------------------------------------------------
  "media.split": "Split",
  "split.all": "all",
  "split.train": "train",
  "split.eval": "eval",
  "media.allSteps": "all",
  "media.clips.one": "{count} clip",
  "media.clips.other": "{count} clips",
  "media.tally": "{success}/{envs} envs succeeded",
  "media.tallyTitle": "Summed over the clips currently shown",
  "media.unrecorded": "{count} not recorded",
  "media.unrecordedTitle":
    "These clips have no recorded outcome. They are excluded from the tally rather than " +
    "counted as failures.",
  "media.requestFailed": "Media request failed",
  "media.emptyTitle": "No video for this run",
  "media.emptyBody":
    "Videos are written by env workers into a sharded index. A run with {code}, or one " +
    "whose recording step has not come round yet, has none.",
  "media.succeeded": "succeeded",
  "media.succeededTitle": "Single-env clip: the episode reached the goal.",
  "media.notSucceeded": "did not succeed",
  "media.notSucceededTitle": "Single-env clip: the episode did not reach the goal.",
  "media.outcomeUnrecorded": "outcome not recorded",
  "media.outcomeUnrecordedTitle":
    "This clip has no recorded outcome. That is not a failure: the environment may track " +
    "no success notion, or the clip predates the field.",
  "media.successCount": "{success}/{envs} succeeded",
  "media.successCountTitle":
    "{success} of the {envs} environments tiled in this clip reached the goal.",
  "media.playAria": "Play clip at {unit} {step}",
  "media.decodeFailed": "This clip could not be decoded by the browser.",
  "media.noUrl": "The server returned no URL for this clip.",
  "media.seed": "seed {seed}",
  "media.shard": "shard {shard}",
  "media.path": "path",

  // -- Events --------------------------------------------------------------
  "events.filter": "Filter",
  "events.all": "all",
  "events.warnError": "warn + error",
  "events.rangeEmpty": "0 of {total}",
  "events.filteredFrom": " (filtered from {total})",
  "events.problemCount": "{count} warn/error",
  "events.exit": "Exit",
  "events.logUnreadable": "Event log unreadable",
  "events.noneWarnTitle": "No warnings or errors",
  "events.noneWarnBody": "Nothing in this run's log is a warning or an error.",
  "events.noneTitle": "No events",
  "events.noneBody":
    "This run wrote no events.jsonl entries. The file is appended by the runner at phase " +
    "boundaries, checkpoint saves and evals, so an empty log usually means the run has " +
    "not reached one yet.",
  "events.truncatedTitle": "Older events not shown",
  "events.truncatedBody":
    "This log is the most recent {limit} entries. The run wrote more before them, " +
    "including its start, and they are in {code} under the run root.",
  "events.checkpoints": "Checkpoints",
  "events.best": "best",
  "events.col.saved": "Saved",
  "events.col.size": "Size",
  "events.col.took": "Took",
  "events.col.path": "Path",

  // -- Compare -------------------------------------------------------------
  "compare.title": "Compare ({count})",
  "compare.metric": "Metric",
  "compare.pickMetric": "— pick a metric —",
  "compare.inAllRuns": "In all {count} runs",
  "compare.inSomeRuns": "In some runs only",
  "compare.runs.one": "{count} run",
  "compare.runs.other": "{count} runs",
  "compare.mixedTitle": "Mixed step semantics on one axis",
  "compare.mixedBody":
    "The selected runs do not agree on what a step is ({counts}). The axis is labelled " +
    "{code}; runs using a different unit are drawn dashed. Their x values are not " +
    "comparable to the others.",
  "compare.mixedCount": "{count}x {label}",
  "compare.mixedSeparator": ", ",
  "compare.nothingTitle": "Nothing selected",
  "compare.nothingBody":
    "Pick two or more runs on the run list, then choose {compare}.",
  "compare.noMetric": "No metric selected",
  "compare.col.run": "Run",
  "compare.col.state": "State",
  "compare.col.step": "Step",
  "compare.col.latest": "Latest",
  "compare.col.latestValue": "Latest value",
  "compare.col.elapsed": "Elapsed",
  "compare.col.semantics": "Step semantics",
} as const;
