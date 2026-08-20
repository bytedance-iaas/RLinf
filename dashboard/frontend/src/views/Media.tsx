/**
 * The media view: recorded simulator video for an embodied run.
 *
 * Loading is staged, because the clips are not cheap and there are a lot of them.
 * A card near the viewport fetches a ~15KB server-rendered poster; the ~1.1MB
 * clip behind it is fetched only when someone clicks that specific card. The
 * grid is a contact sheet first and a player second.
 *
 * The reason the poster is rendered server-side rather than by seeking a hidden
 * `<video>`: clips are written by `imageio.get_writer` without `+faststart`, so
 * the `moov` atom sits at the end of the file (measured at 98% in). A browser
 * wanting one frame therefore needs several Range round-trips per clip, and at
 * forty cards against a six-connection origin limit that is precisely the queue
 * that leaves unrelated API calls hanging.
 *
 * Every fetch this view starts is also one it can abandon -- see the cleanups in
 * `Clip`. Navigating away must free the connections immediately, not when the
 * video finishes.
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
import { EMPTY, integer, semanticsInline } from "../lib/format";
import { t, tNode } from "../lib/i18n";

export interface MediaProps {
  status: RunStatus;
}

/**
 * A split as a word.
 *
 * `split` is a free string on the wire, and only the two the index actually
 * writes have a translation. Anything else is shown as the server wrote it,
 * rather than as a blank chip.
 */
const SPLIT_KEYS = { all: "split.all", train: "split.train", eval: "split.eval" } as const;

function splitLabel(value: string): string {
  const key = SPLIT_KEYS[value as keyof typeof SPLIT_KEYS];
  return key ? t(key) : value;
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
      ? { text: t("media.succeeded"), all: true, title: t("media.succeededTitle") }
      : {
          text: t("media.notSucceeded"),
          none: true,
          title: t("media.notSucceededTitle"),
        };
  }

  if (entry.num_success === null || entry.num_envs === null) {
    return {
      text: t("media.outcomeUnrecorded"),
      unrecorded: true,
      // Said explicitly, because the alternative reading -- that nothing succeeded
      // -- is the one a reader will otherwise reach for.
      title: t("media.outcomeUnrecordedTitle"),
    };
  }

  const { num_success: success, num_envs: envs } = entry;
  return {
    text: t("media.successCount", { success, envs }),
    all: envs > 0 && success === envs,
    none: success === 0,
    title: t("media.successCountTitle", { success, envs }),
  };
}

export function Media(props: MediaProps) {
  const runId = props.status.run_id;
  // The media view only ever uses the label mid-phrase -- as a filter caption and
  // beside a step number -- so it takes the inline form once, here.
  const semantics = semanticsInline(
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
          <span>{t("media.split")}</span>
          <div className="control-group">
            {(["all", "train", "eval"] as const).map((value) => (
              <button
                className="btn"
                key={value}
                type="button"
                data-active={split === value ? "true" : undefined}
                onClick={() => setSplit(value)}
              >
                {splitLabel(value)}
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
            <option value="">{t("media.allSteps")}</option>
            {steps.map((value) => (
              <option value={String(value)} key={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <span className="control-value faint">
          {t(entries.length === 1 ? "media.clips.one" : "media.clips.other", {
            count: entries.length,
          })}
        </span>
        {tally.envs > 0 && (
          <span className="chip" title={t("media.tallyTitle")}>
            {t("media.tally", { success: tally.success, envs: tally.envs })}
          </span>
        )}
        {tally.unrecorded > 0 && (
          <span className="chip" title={t("media.unrecordedTitle")}>
            {t("media.unrecorded", { count: tally.unrecorded })}
          </span>
        )}
      </div>

      {mediaQuery.error && (
        <Note tone="error" title={t("media.requestFailed")}>
          {mediaQuery.error}
        </Note>
      )}

      {!mediaQuery.loading && entries.length === 0 && (
        <Note title={t("media.emptyTitle")}>
          {tNode("media.emptyBody", {
            code: <Code>env.&lt;split&gt;.video_cfg.save_video: false</Code>,
          })}
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
  const posterUrl = api.posterUrl(entry);
  const [failed, setFailed] = useState(false);
  const [near, setNear] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [posterFailed, setPosterFailed] = useState(false);
  const frameRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const posterRef = useRef<HTMLImageElement>(null);
  const result = outcome(entry);

  /**
   * Load the poster only once its card is near the viewport.
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

  /**
   * Drop an in-flight poster fetch when this card goes away.
   *
   * Leaving the tab unmounts every card at once. React removes the `<img>`, but
   * removal alone does not oblige the browser to abandon a request already on
   * the wire -- clearing `src` does, which is what frees the connection for the
   * view being navigated *to*. Six connections per origin is the whole budget,
   * and forty cards can hold all of it.
   *
   * The node is captured here rather than read from the ref inside the cleanup,
   * because by then React may already have detached it.
   */
  useEffect(() => {
    const node = posterRef.current;
    if (!node) return;
    return () => node.removeAttribute("src");
  }, [posterUrl, near]);

  /**
   * Same contract for the player, which is the expensive one.
   *
   * A clip is ~1.1MB streamed over Range requests; abandoning that on navigation
   * is the difference between the next tab rendering now and rendering when the
   * video finishes. `load()` after clearing `src` is what actually resets the
   * media element and cancels the pending fetches -- removing the attribute on
   * its own leaves the current request running.
   */
  useEffect(() => {
    const node = videoRef.current;
    if (!node) return;
    return () => {
      node.pause();
      node.removeAttribute("src");
      node.load();
    };
  }, [playing, url]);

  // A poster is worth showing only until someone asks for the real thing.
  const showPoster = Boolean(posterUrl) && !posterFailed && near && !playing;
  // Clicking must work even with no poster: a deployment without ffmpeg still
  // has clips, and gating playback on a thumbnail would make a cosmetic
  // dependency into a functional one.
  const canPlay = Boolean(url) && !failed && !playing;

  return (
    <figure className="media-card" style={{ margin: 0 }}>
      <div className="media-frame" ref={frameRef}>
        {url && !failed ? (
          playing ? (
            // `controls` and `autoPlay`: autoplay only ever follows an explicit
            // click on this card, so it is the continuation of a gesture rather
            // than ambient motion. A grid that started playing on its own would
            // be unreadable.
            //
            // `preload="auto"` is right *here* and wrong on the grid: by this
            // point someone has asked for this specific clip.
            <video
              ref={videoRef}
              src={url}
              controls
              autoPlay
              preload="auto"
              onError={() => setFailed(true)}
            />
          ) : (
            <button
              type="button"
              className="media-play"
              onClick={() => setPlaying(true)}
              disabled={!canPlay}
              aria-label={t("media.playAria", {
                unit: props.semantics,
                step: entry.step ?? t("status.unknown"),
              })}
            >
              {showPoster && (
                <img
                  ref={posterRef}
                  src={posterUrl as string}
                  alt=""
                  loading="lazy"
                  decoding="async"
                  onError={() => setPosterFailed(true)}
                />
              )}
              {/* Reserving the box keeps the grid from reflowing as posters
                  arrive, which is the rule the rest of the page follows. */}
              {!showPoster && <div className="media-frame-idle" aria-hidden="true" />}
              <span className="media-play-badge" aria-hidden="true">
                ▶
              </span>
            </button>
          )
        ) : (
          <div className="media-frame-error">
            {url ? t("media.decodeFailed") : t("media.noUrl")}
          </div>
        )}
      </div>
      <figcaption className="media-meta">
        <span className="chip">{splitLabel(entry.split)}</span>
        <span>
          {props.semantics} {entry.step ?? EMPTY}
        </span>
        {entry.seed !== null && <span>{t("media.seed", { seed: entry.seed })}</span>}
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
        <span className="faint">{t("media.shard", { shard: entry.shard })}</span>
      </div>
      {/* The path is diagnosis material, not caption material: it is the widest
          thing on the card and the least often needed, and forty of them made
          the grid read as a file listing with videos attached. Kept one click
          away rather than removed -- when the question *is* "which file is
          this", nothing else answers it. */}
      <details className="media-path">
        <summary>{t("media.path")}</summary>
        <div className="media-path-value">{entry.path}</div>
      </details>
    </figure>
  );
}
