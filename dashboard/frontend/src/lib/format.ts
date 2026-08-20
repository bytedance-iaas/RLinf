/**
 * Value formatting for a console read at a glance.
 *
 * Every function here targets a fixed character width where it can. Tabular
 * figures keep digits in fixed columns, but only if the number of digits is
 * stable too -- `1.2345` becoming `12.345` still shifts a right-aligned column,
 * so durations and byte counts pick a unit and a precision rather than printing
 * whatever the float happens to be.
 *
 * The functions that produce *words* -- ages, step semantics -- translate; the
 * ones that produce numbers do not. A duration is `1:23` and a byte count is
 * `4.2 GiB` in every language, and localising the digit grouping of a step count
 * would break the column alignment the whole file exists to protect.
 */

import { activeLang, localeTag, t } from "./i18n";

/** Rendered in place of a number the server did not report. */
export const EMPTY = "—";

/**
 * Duration as a fixed-shape clock string.
 *
 * `h:mm:ss` above an hour, `m:ss` below, and sub-minute values keep one decimal
 * because an RL step time of 3.4s versus 3.9s is a real difference. A run left
 * open for days needs the day component, hence the `d` prefix.
 */
export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return EMPTY;
  if (seconds < 0) return EMPTY;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;

  const total = Math.floor(seconds);
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (value: number) => String(value).padStart(2, "0");

  if (days > 0) return `${days}d ${pad(hours)}:${pad(minutes)}:${pad(secs)}`;
  if (hours > 0) return `${hours}:${pad(minutes)}:${pad(secs)}`;
  return `${minutes}:${pad(secs)}`;
}

/** Compact age, for "as of" readouts where the exact second does not matter. */
export function age(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return EMPTY;
  if (seconds < 1) return t("format.justNow");
  if (seconds < 60) return t("format.secondsAgo", { n: Math.round(seconds) });
  if (seconds < 3600) return t("format.minutesAgo", { n: Math.round(seconds / 60) });
  if (seconds < 86400) return t("format.hoursAgo", { n: Math.round(seconds / 3600) });
  return t("format.daysAgo", { n: Math.round(seconds / 86400) });
}

/** Seconds between an ISO timestamp and now, or null if it was never recorded. */
export function ageSince(iso: string | null | undefined, now = Date.now()): number | null {
  if (!iso) return null;
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return null;
  return (now - then) / 1000;
}

/**
 * A metric value at a readable precision.
 *
 * Switches to exponential outside roughly 1e-4..1e6 rather than printing sixteen
 * digits or a row of zeros: learning rates and grad norms both live off the end
 * of the fixed-point range, and they are read as magnitudes, not as exact values.
 */
export function metric(
  value: number | null | undefined,
  options: { percent?: boolean; digits?: number } = {},
): string {
  if (value === null || value === undefined) return EMPTY;
  if (!Number.isFinite(value)) return value > 0 ? "+Inf" : Number.isNaN(value) ? "NaN" : "-Inf";

  if (options.percent) return `${(value * 100).toFixed(1)}%`;

  const magnitude = Math.abs(value);
  if (value === 0) return "0";
  if (magnitude >= 1e6 || magnitude < 1e-4) return value.toExponential(2);

  const digits = options.digits ?? (magnitude >= 100 ? 1 : magnitude >= 1 ? 3 : 4);
  return value.toFixed(digits);
}

/** Integer with thin-space grouping, so a step count of 128000 is scannable. */
export function integer(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  return Math.round(value).toLocaleString("en-US");
}

/** Byte count in binary units. Checkpoints are tens of gigabytes. */
export function bytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return EMPTY;
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let scaled = value;
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${scaled.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

/** Local wall-clock time, for timestamps a person cross-references with logs. */
export function timestamp(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return EMPTY;
  return parsed.toLocaleString(localeTag(), {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/** Time only, for dense lists where the date is the same for every row. */
export function clockTime(iso: string | null | undefined): string {
  if (!iso) return EMPTY;
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return EMPTY;
  return parsed.toLocaleTimeString(localeTag(), {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/**
 * `step_semantics` as a phrase, and as a unit.
 *
 * Written out rather than shown raw because `rl_iteration` on an axis reads as a
 * variable name, and the whole reason to surface the field is that a reader must
 * not silently assume an embodied "step 400" and a reasoning "step 400" are the
 * same thing.
 */
const SEMANTICS_VALUES = ["rl_iteration", "minibatch", "optimizer_step"] as const;

function known(value: string | null | undefined): (typeof SEMANTICS_VALUES)[number] | null {
  return SEMANTICS_VALUES.find((known) => known === value) ?? null;
}

export function semanticsLabel(value: string | null | undefined): string {
  if (!value) return t("semantics.step");
  const match = known(value);
  // A semantics string this bundle has never heard of is shown raw. It came from
  // a runner newer than the dashboard, and the raw name is at least accurate.
  return match ? t(`semantics.${match}`) : value;
}

export function semanticsShort(value: string | null | undefined): string {
  if (!value) return t("semantics.short.step");
  const match = known(value);
  return match ? t(`semantics.short.${match}`) : value;
}

/**
 * The same label as it reads *inside* a sentence -- "per rl iteration", "at rl
 * iteration 95".
 *
 * English lowercases it, because a capitalised noun mid-sentence reads as a
 * proper name. Chinese has no case at all, and lowercasing there does real
 * damage: "RL 迭代" becomes "rl 迭代", which turns an acronym into a typo. So
 * the rule branches on the language rather than on the message -- casing is a
 * property of the script, and no catalogue entry can express it.
 */
export function semanticsInline(value: string | null | undefined): string {
  const label = semanticsLabel(value);
  return activeLang() === "en" ? label.toLowerCase() : label;
}

/** Trailing path component, for a checkpoint dir shown next to its full path. */
export function basename(path: string | null | undefined): string {
  if (!path) return EMPTY;
  const parts = path.replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || path;
}
