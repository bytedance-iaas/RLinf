/**
 * The media view: recorded simulator video for an embodied run.
 *
 * One rule dominates this file. A media row carries `num_success` / `num_envs` and
 * a `success` that is usually `null`, because one mp4 is a tiled grid of every env
 * in a worker -- a scalar verdict over eight tiles would be a lie. So the outcome
 * renders as a count ("3/8 succeeded"), and a `null` is rendered as "not recorded",
 * never as a failure. Unrecorded is not failed: the env may track no success notion
 * at all, or the clip may predate the field.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { useFetch } from "../api/useLive";
import type { MediaEntry, RunStatus } from "../api/types";
import { Code, Note } from "../components/primitives";
import { integer, semanticsLabel } from "../lib/format";

export interface MediaProps {
  status: RunStatus;
}

/** Outcome of one clip, as a phrase plus the styling flags behind it. */
function outcome(entry: MediaEntry): {
  text: string;
  all?: boolean;
  none?: boolean;
  unrecorded?: boolean;
  title: string;
} {
  // A single-env clip is the only case where a scalar is honest, and the server
  // sets `success` exactly then.
  if (entry.success !== null && entry.num_envs === 1) {
    return entry.success
      ? { text: "succeeded", all: true, title: "Single-env clip: the episode reached the goal." }
      : {
          text: "did not succeed",
          none: true,
          title: "Single-env clip: the episode did not reach the goal.",
        };
  }

  if (entry.num_success === null || entry.num_envs === null) {
    return {
      text: "outcome not recorded",
      unrecorded: true,
      // Said explicitly, because the alternative reading -- that nothing succeeded
      // -- is the one a reader will otherwise reach for.
      title:
        "This clip has no recorded outcome. That is not a failure: the environment " +
        "may track no success notion, or the clip predates the field.",
    };
  }

  const { num_success: success, num_envs: envs } = entry;
  return {
    text: `${success}/${envs} succeeded`,
    all: envs > 0 && success === envs,
    none: success === 0,
    title: `${success} of the ${envs} environments tiled in this clip reached the goal.`,
  };
}

export function Media(props: MediaProps) {
  const runId = props.status.run_id;
  const semantics = semanticsLabel(
    props.status.snapshot?.progress.step_semantics ?? props.status.manifest?.step_semantics,
  );

  const [split, setSplit] = useState<"all" | "train" | "eval">("all");
  const [step, setStep] = useState<number | "all">("all");

  const stepsQuery = useFetch<number[]>(
    useCallback((signal) => api.mediaSteps(runId, signal), [runId]),
    [runId],
  );
  const mediaQuery = useFetch<MediaEntry[]>(
    useCallback(
      (signal) =>
        api.media(
          runId,
          {
            ...(split === "all" ? {} : { split }),
            ...(step === "all" ? {} : { step }),
          },
          signal,
        ),
      [runId, split, step],
    ),
    [runId, split, step],
  );

  const entries = mediaQuery.data ?? [];
  const steps = stepsQuery.data ?? [];

  // A per-step tally over what is currently shown. The point of a video view for an
  // RL run is "did the policy actually solve it", and eight separate "0/8" labels
  // answer that far more slowly than one total.
  const tally = useMemo(() => {
    let success = 0;
    let envs = 0;
    let unrecorded = 0;
    for (const entry of entries) {
      if (entry.num_success === null || entry.num_envs === null) {
        unrecorded += 1;
        continue;
      }
      success += entry.num_success;
      envs += entry.num_envs;
    }
    return { success, envs, unrecorded };
  }, [entries]);

  return (
    <div className="stack">
      <div className="controls">
        <div className="control">
          <span>Split</span>
          <div className="control-group">
            {(["all", "train", "eval"] as const).map((value) => (
              <button
                className="btn"
                key={value}
                type="button"
                data-active={split === value ? "true" : undefined}
                onClick={() => setSplit(value)}
              >
                {value}
              </button>
            ))}
          </div>
        </div>
        <label className="control">
          <span>{semantics}</span>
          <select
            value={step === "all" ? "" : String(step)}
            onChange={(event) => setStep(event.target.value === "" ? "all" : Number(event.target.value))}
          >
            <option value="">all</option>
            {steps.map((value) => (
              <option value={String(value)} key={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <span className="control-value faint">
          {entries.length} clip{entries.length === 1 ? "" : "s"}
        </span>
        {tally.envs > 0 && (
          <span className="chip" title="Summed over the clips currently shown">
            {tally.success}/{tally.envs} envs succeeded
          </span>
        )}
        {tally.unrecorded > 0 && (
          <span
            className="chip"
            title="These clips have no recorded outcome. They are excluded from the tally rather than counted as failures."
          >
            {tally.unrecorded} not recorded
          </span>
        )}
      </div>

      {mediaQuery.error && (
        <Note tone="error" title="Media request failed">
          {mediaQuery.error}
        </Note>
      )}

      {!mediaQuery.loading && entries.length === 0 && (
        <Note title="No video for this run">
          Videos are written by env workers into a sharded index. A run with{" "}
          <Code>env.&lt;split&gt;.video_cfg.save_video: false</Code>, or one whose recording step has not come round
          yet, has none.
        </Note>
      )}

      <div className="media-grid">
        {entries.map((entry) => (
          <Clip entry={entry} key={`${entry.path}-${entry.split}-${entry.shard}`} semantics={semantics} />
        ))}
      </div>
    </div>
  );
}

function Clip(props: { entry: MediaEntry; semantics: string }) {
  const { entry } = props;
  const url = api.mediaUrl(entry);
  const [failed, setFailed] = useState(false);
  const [near, setNear] = useState(false);
  const frameRef = useRef<HTMLDivElement>(null);
  const result = outcome(entry);

  /**
   * Mount the player only once its card is near the viewport.
   *
   * Loading every clip can saturate the browser's per-origin connections and
   * delay unrelated API calls. `rootMargin` starts the next card slightly before
   * it enters the viewport.
   */
  useEffect(() => {
    const node = frameRef.current;
    if (!node || near) return;
    if (typeof IntersectionObserver === "undefined") {
      setNear(true);
      return;
    }
    const observer = new IntersectionObserver(
      (records) => {
        if (records.some((record) => record.isIntersecting)) {
          setNear(true);
          observer.disconnect();
        }
      },
      { rootMargin: "300px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [near]);

  return (
    <figure className="media-card" style={{ margin: 0 }}>
      <div className="media-frame" ref={frameRef}>
        {url && !failed ? (
          near ? (
            // `controls` and no autoplay: this is evidence being examined, not
            // ambient motion, and a grid of autoplaying clips is unreadable.
            //
            // `preload="none"` because the grid is a contact sheet, not a
            // player: the browser fetches nothing until someone presses play on
            // a specific clip, and the server's FileResponse handles Range, so
            // playback and seeking still work without downloading the whole
            // file. Metadata preloading would put the cost back, one HTTP
            // request per visible card, for a poster frame nobody asked for.
            <video src={url} controls preload="none" onError={() => setFailed(true)} />
          ) : (
            // Same box, no element: reserving the space keeps the grid from
            // reflowing as cards mount, which is the rule the rest of the page
            // follows.
            <div className="media-frame-idle" aria-hidden="true" />
          )
        ) : (
          <div className="media-frame-error">
            {url
              ? "This clip could not be decoded by the browser."
              : "The server returned no URL for this clip."}
          </div>
        )}
      </div>
      <figcaption className="media-meta">
        <span className="chip">{entry.split}</span>
        <span>
          {props.semantics.toLowerCase()} {entry.step ?? "—"}
        </span>
        {entry.seed !== null && <span>seed {entry.seed}</span>}
        <span
          className="media-count"
          data-all={result.all ? "true" : undefined}
          data-none={result.none ? "true" : undefined}
          data-unrecorded={result.unrecorded ? "true" : undefined}
          title={result.title}
        >
          {result.text}
        </span>
      </figcaption>
      <div className="media-meta">
        {entry.num_frames !== null && <span>{integer(entry.num_frames)}f</span>}
        {entry.fps !== null && <span>{entry.fps}fps</span>}
        <span className="faint">shard {entry.shard}</span>
      </div>
      <div className="media-path" title={entry.path}>
        {entry.path}
      </div>
    </figure>
  );
}
