/**
 * TypeScript mirror of `rlinf_dashboard/models.py`.
 *
 * Hand-written rather than generated from the OpenAPI schema, and kept in one
 * file, because the whole cross-process contract is HTTP + the filesystem: there
 * is no import path from the Python side into this bundle, so the only thing
 * keeping the two in step is that they are short enough to diff by eye.
 *
 * The two closed enums below are closed on the server too. `Health` has exactly
 * four values and `RunState` exactly five; widening either here would let the UI
 * render a state the server cannot produce, which is how "unknown" ends up
 * styled as a loading spinner.
 */

/** Read-side liveness verdict. Exactly four values, per `models.Health`. */
export type Health = "healthy" | "degraded" | "unreachable" | "unknown";

/** Write-side lifecycle fact. Exactly five values, per `models.RunState`. */
export type RunState = "pending" | "running" | "finished" | "failed" | "stopped";

/**
 * What one step means. The same "step 400" is an RL iteration in an embodied run
 * and a minibatch in a reasoning run, so every axis and every compare view has to
 * carry this.
 */
export type StepSemantics = "rl_iteration" | "minibatch" | "optimizer_step";

export type EtaConfidence = "low" | "medium" | "high";

/** Server-side event kinds from `events.jsonl`. */
export type EventKind =
  | "run_start"
  | "phase_enter"
  | "phase_exit"
  | "ckpt_saved"
  | "eval_done"
  | "warn"
  | "error"
  | "run_end";

export interface ComponentState {
  active: boolean;
  since: string | null;
}

export interface Progress {
  step: number;
  max_steps: number | null;
  epoch: number | null;
  step_semantics: StepSemantics | null;
}

export interface Timing {
  started_at: string | null;
  elapsed_s: number;
  step_time_p50: number | null;
  step_time_recent: number[];
  eta_s: number | null;
  eta_confidence: EtaConfidence | null;
}

export interface CheckpointEntry {
  step: number;
  path: string;
  saved_at: string | null;
  size_bytes: number | null;
  duration_s: number | null;
  is_best: boolean;
  /** Values arrive as strings from the training side as often as numbers. */
  metrics: Record<string, unknown>;
  resume_dir: string | null;
  entry_script: string | null;
  config_name: string | null;
}

export interface AlgorithmInfo {
  loss_type: string | null;
  adv_type: string | null;
}

export interface ClusterInfo {
  num_nodes: number | null;
  component_placement: Record<string, unknown> | null;
}

export interface ExitInfo {
  reason: string;
  traceback_tail: string | null;
}

export interface RunSnapshot {
  schema_version: number;
  run_id: string;
  task_type: string;
  algorithm: AlgorithmInfo | null;
  state: RunState;
  /**
   * Scalar phase, `null` on a finished run and on async runners. Async runners
   * report `components` instead: env, rollout and actor all run for the whole
   * loop, so no single phase is true.
   */
  phase: string | null;
  phase_since: string | null;
  components: Record<string, ComponentState>;
  heartbeat_at: string | null;
  heartbeat_seq: number;
  last_progress_at: string | null;
  last_metric_at: string | null;
  progress: Progress;
  timing: Timing;
  latest_checkpoint: CheckpointEntry | null;
  paths: Record<string, string | null>;
  cluster: ClusterInfo | null;
  exit: ExitInfo | null;
}

export interface RunManifest {
  schema_version: number;
  run_id: string;
  task_type: string;
  experiment_name: string | null;
  project_name: string | null;
  step_semantics: StepSemantics | null;
  algorithm: AlgorithmInfo | null;
  cluster: ClusterInfo | null;
  git_sha: string | null;
  hostname: string | null;
  pid: number | null;
  started_at: string | null;
  resumed_from: string | null;
  paths: Record<string, string | null>;
  metric_aliases: Record<string, string>;
}

/**
 * The server's liveness call plus the ages it was computed from.
 *
 * `reason` is rendered verbatim. The derivation lives in one pure function on the
 * server precisely so a run cannot look healthy in one view and dead in another;
 * recomputing any part of it here reintroduces exactly that.
 */
export interface HealthVerdict {
  health: Health;
  reason: string;
  heartbeat_age_s: number | null;
  progress_age_s: number | null;
  metric_age_s: number | null;
  budget_s: number | null;
}

export interface RunStatus {
  run_id: string;
  manifest: RunManifest | null;
  snapshot: RunSnapshot | null;
  health: HealthVerdict;
  run_root: string;
  /** Set when run.json is missing or unparseable but the directory exists. */
  error: string | null;
  /** Present when launch-time absolute paths were translated for this machine. */
  relocation?: Record<string, string> | null;
  /**
   * Whether this run recorded any video at all.
   *
   * The claim about *this* run, as opposed to the template's `has_media_view`,
   * which is about its task type. Both are needed: a reasoning run has no video
   * view even in the impossible case that a clip appeared, and an embodied run
   * with `enable_dump_video: false` has nothing to show.
   */
  has_media?: boolean;
  /**
   * Startup, as distinct from failure. The training process cannot report "I
   * have not started reporting yet", so this is derived on the read side from a
   * manifest with no snapshot beside it and a deadline that has not passed.
   */
  initializing?: boolean;
  /** Seconds spent in startup so far, reported past the deadline too. */
  startup_elapsed_s?: number | null;
}

export interface RunSummary {
  run_id: string;
  task_type: string | null;
  experiment_name: string | null;
  state: RunState | null;
  health: Health;
  phase: string | null;
  step: number;
  max_steps: number | null;
  step_semantics: StepSemantics | null;
  started_at: string | null;
  heartbeat_at: string | null;
  elapsed_s: number;
  eta_s: number | null;
  latest_checkpoint_step: number | null;
  run_root: string;
  relocation?: Record<string, string> | null;
  /**
   * The manifest exists, the snapshot does not, and the startup deadline has not
   * passed. A read-side derivation like `health`, not a `state` the training
   * process could have written -- see `RunStatus.initializing`.
   */
  initializing?: boolean;
  /** Seconds spent in startup so far, reported past the deadline too. */
  startup_elapsed_s?: number | null;
}

export interface SeriesPoint {
  step: number;
  /**
   * `null` where the training side logged a non-finite value.
   *
   * This is not a nullable metric: pydantic serialises `float('nan')` and
   * `float('inf')` to JSON `null`, verified against a real event file. So a
   * `null` here means "the run logged NaN or Inf at this step", which is a red
   * signal, not missing data. See `nonFiniteRun` in `signals.ts`.
   */
  value: number | null;
  wall_time: number | null;
}

export interface Series {
  key: string;
  points: SeriesPoint[];
  /** `tensorboard`, or `none` when no source had the key. */
  source: string;
  decimated: boolean;
  total_points: number;
  /**
   * Worker group, on a series from `?expand=ranks`. `null` on the aggregate.
   *
   * The aggregate is a mean across ranks, so these two fields are what let the
   * UI say *which* rank, rather than only that the mean moved.
   */
  group: string | null;
  /** Rank within `group`. `null` on the aggregate. */
  rank: number | null;
}

export interface MediaEntry {
  path: string;
  step: number | null;
  split: string;
  seed: number | null;
  num_frames: number | null;
  fps: number | null;
  shard: number;
  /**
   * How many of the clip's envs had succeeded. One mp4 is a tiled grid of every
   * env in a worker, so this is a count. `null` means the outcome was not
   * recorded -- which is not the same as zero and must never render as a failure.
   */
  num_success: number | null;
  num_envs: number | null;
  /** Scalar outcome, set only for a single-env clip. `null` for a grid. */
  success: boolean | null;
  url: string | null;
}

export interface RunEvent {
  ts: string;
  kind: EventKind;
  step: number | null;
  payload: Record<string, unknown>;
}

// ---------------------------------------------------------------- templates

/**
 * One chart, as the server's YAML template declares it.
 *
 * Every field here is honoured by the single generic renderer. That is the whole
 * point of driving charts from templates: adding one is a YAML change on the
 * server, never a change in this bundle, so a field the renderer silently ignores
 * would be a template author writing into a void.
 */
export interface TemplateChart {
  /** Already filtered by the server to keys this run actually logged. */
  keys: string[];
  title?: string;
  /** `percent` scales a 0..1 fraction for display and pins the axis to 0..100. */
  format?: "percent" | string;
  /** Axis suffix, e.g. `s` or `tokens`. */
  unit?: string;
  /** `log` for quantities that span orders of magnitude (grad norm, lr). */
  scale?: "log" | string;
  /** Stack the series as filled bands; only meaningful for additive parts. */
  stacked?: boolean;
}

export interface TemplateGroup {
  title?: string;
  /** Start folded. Used for eval and for the auto-binned "Other" bucket. */
  collapsed?: boolean;
  unit?: string;
  charts: TemplateChart[];
}

export interface NorthStar {
  key: string | null;
  /**
   * These three always describe the key in `key`, never the metric the template
   * would have preferred. The server resolves the candidate list and sends the
   * winner's own semantics, so a template whose accuracy metric is missing and
   * whose loss fallback won arrives labelled as a loss, unformatted, and with
   * `goal: minimize` -- rather than as an accuracy percentage to be maximized,
   * which would invert every trend verdict computed from it.
   */
  label?: string;
  format?: "percent" | string;
  goal?: "maximize" | "minimize" | string;
  /**
   * False when no candidate key had data. A headline metric with no data reads as
   * a broken run rather than as a metric this run does not log, so the UI says
   * which it is instead of showing an empty hero number.
   */
  resolved?: boolean;
}

export interface RunTemplate {
  name: string;
  task_types?: string[];
  /** Human label for the x axis, e.g. "RL iteration" or "Minibatch". */
  step_axis_label?: string;
  north_star?: NorthStar | null;
  groups: TemplateGroup[];
  /**
   * Keys no chart claimed. Rendered in a collapsed group: a metric that vanishes
   * from the page looks like a missing feature, not like an unclaimed key.
   */
  unmatched?: string[];
  auto_group?: boolean;
  caveats?: string[];
  /**
   * Whether this *kind* of run can have simulator video.
   *
   * A claim about the task type, not about this run: `embodied` is true even when
   * `env.enable_dump_video` was off. Pair it with `RunStatus.has_media`, which is
   * the claim about this run, before offering a Media view.
   */
  has_media_view?: boolean;
}

export interface ServerHealth {
  status: string;
  version: string;
  scan_roots: { path: string; exists: boolean }[];
  run_count: number;
}
