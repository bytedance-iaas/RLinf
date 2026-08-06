#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a multi-run scan root so the frontend can be checked against curves.

The real verification tree (``/tmp/rlinf-reloc2``) is a genuine one-iteration run
and is the right thing to test the single-point rendering case against. It cannot
exercise the cases that only appear with history:

* a curve, and therefore smoothing, log scales and stacked areas;
* multi-run compare, which needs a second run to overlay;
* the client-side metric signals -- a step time that doubles, an eval that
  plateaus, a NaN in a series -- none of which a one-point run can contain;
* the per-worker drill-down, which needs a run written with
  ``runner.per_worker_log: true`` and at least one rank behaving differently from
  the others -- a tree where every rank agrees cannot show whether the expansion
  works, because the aggregate already answers the question;
* the four lifecycle states other than ``finished``, and the health verdicts that
  only a *running* snapshot can produce.

So this writes synthetic trees in the same on-disk layout the training side
writes, using only ``tensorboard``'s event writer and ``json``. It deliberately
does not import ``rlinf``: the dashboard's whole contract is the filesystem, and
a fixture generator that reached into the training package would be testing
something other than that contract.

Usage:
    python scripts/make_demo_runs.py [--root /tmp/rlinf-demo] [--clean]
    python scripts/make_demo_runs.py --root /tmp/rlinf-demo --touch
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timedelta, timezone

from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


#: A real mp4 to copy into every clip slot, set by ``--sample-clip``. ``None``
#: writes 64-byte stubs instead, which the browser cannot decode -- fine for
#: checking layout and counts, useless for checking playback.
SAMPLE_CLIP: str | None = None


def _write_clip(path: str) -> None:
    if SAMPLE_CLIP is not None:
        shutil.copyfile(SAMPLE_CLIP, path)
        return
    with open(path, "wb") as handle:
        handle.write(b"\x00" * 64)


def touch_runs(root: str) -> int:
    """Re-stamp every run's liveness timestamps to now, keeping relative offsets.

    Health is a function of *age*, so a generated tree decays: the budget is 5x the
    run's own step time, around 95s here, which means the healthy runs read
    `unreachable` about eight minutes after generation. That is correct behaviour --
    a heartbeat that stopped is a heartbeat that stopped -- but it makes the fixture
    useless for checking the healthy and `degraded` renderings, since a full rebuild
    takes two minutes and has itself gone stale by the time anyone has clicked
    through four views.

    So this shifts the whole tree forward instead, by the interval between the clock
    this run was generated against and now. Every timestamp moves by that one delta,
    which is what preserves the deliberately-constructed verdicts: the wedged run
    stays 6000s stale, the frozen-snapshot run keeps its fresh heartbeat *file* over
    a stale snapshot, and `unknown` keeps having no heartbeat at all.

    The generation clock is recovered as ``timing.started_at + timing.elapsed_s``,
    which is the one anchor the generator leaves that does not itself encode a
    staleness offset. Using a liveness field as the reference instead -- as an
    earlier version of this did -- normalises the stale runs to fresh and silently
    turns the fixture into eleven healthy runs.

    Only liveness fields move. Series data, event log and media stay put, because
    they are the run's history and rewriting them would make the metrics disagree
    with the steps the snapshot claims.

    Args:
        root: A scan root previously written by this script.

    Returns:
        How many runs were re-stamped.
    """
    now = datetime.now(timezone.utc)
    touched = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if "run.json" not in filenames:
            continue
        dirnames[:] = []
        snapshot_path = os.path.join(dirpath, "run.json")
        with open(snapshot_path, encoding="utf-8") as handle:
            snapshot = json.load(handle)

        timing = snapshot.get("timing") or {}
        started = _parse(timing.get("started_at"))
        elapsed = timing.get("elapsed_s")
        if started is None or not isinstance(elapsed, (int, float)):
            continue
        generated_at = started + timedelta(seconds=elapsed)
        shift = now - generated_at

        # Both ends move by the same delta: `started_at` too, or `elapsed_s` stops
        # agreeing with it and the Timing card contradicts the State card.
        for field in (
            "heartbeat_at",
            "last_progress_at",
            "last_metric_at",
            "phase_since",
        ):
            moment = _parse(snapshot.get(field))
            if moment is not None:
                snapshot[field] = _iso(moment + shift)
        timing["started_at"] = _iso(started + shift)
        snapshot["timing"] = timing
        # Component `since` values are rendered as durations, so they shift too or
        # the components strip claims the run has been up for three hours.
        for state in (snapshot.get("components") or {}).values():
            moment = _parse(state.get("since"))
            if moment is not None:
                state["since"] = _iso(moment + shift)
        checkpoint = snapshot.get("latest_checkpoint")
        if isinstance(checkpoint, dict):
            moment = _parse(checkpoint.get("saved_at"))
            if moment is not None:
                checkpoint["saved_at"] = _iso(moment + shift)

        with open(snapshot_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)

        # The manifest carries the same `started_at`, and discovery sorts the run
        # list by it, so leaving it behind would reorder the list against the
        # snapshots.
        manifest_path = os.path.join(dirpath, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            moment = _parse(manifest.get("started_at"))
            if moment is not None:
                manifest["started_at"] = _iso(moment + shift)
                with open(manifest_path, "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, indent=2)

        # The heartbeat *file*'s mtime is a separate signal -- it is what splits
        # "the process is gone" from "the process is alive and its snapshot writes
        # are failing" -- so it is shifted by the same delta rather than set to now.
        heartbeat_path = os.path.join(dirpath, "heartbeat")
        if os.path.exists(heartbeat_path):
            stamp = os.path.getmtime(heartbeat_path) + shift.total_seconds()
            os.utime(heartbeat_path, (stamp, stamp))
        touched += 1
    return touched


def _parse(raw: str | None) -> datetime | None:
    """Parse an ISO stamp written by :func:`_iso`, or return ``None``."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _write_scalars(
    log_dir: str, series: dict[str, list[float]], t0: float, step_offset: int = 0
) -> None:
    """Write one event file containing every series, one point per step.

    ``step_offset`` shifts the step numbers the events carry. It exists because
    step *0* is not a representative first step: a one-point run at 0 cannot
    reproduce the axis-split hang that killed the Metrics tab, since the hang
    needs ``min + incr == min`` and ``0 + 1e-16`` is not ``0``. Real runners log
    their first point at 1, so a fixture that wants to stand in for one has to
    say so.
    """
    os.makedirs(log_dir, exist_ok=True)
    writer = EventFileWriter(log_dir)
    steps = max(len(values) for values in series.values())
    for step in range(steps):
        for tag, values in series.items():
            if step >= len(values):
                continue
            event = event_pb2.Event(
                wall_time=t0 + step * 30.0,
                step=step + step_offset,
                summary=summary_pb2.Summary(
                    value=[
                        summary_pb2.Summary.Value(tag=tag, simple_value=values[step])
                    ]
                ),
            )
            writer.add_event(event)
    writer.close()


def _write_worker_scalars(
    worker_root: str,
    series: dict[str, list[float]],
    t0: float,
    ranks: int,
    straggler: int,
) -> None:
    """Break a few series out per ``(group, rank)``, as ``per_worker_log`` does.

    Mirrors the layout `MetricLogger._get_scoped_logger` writes --
    ``<root>/<group>/rank_<n>/tensorboard`` -- because that path *is* the index the
    server globs for; there is no manifest listing of ranks to fake instead.

    Only the groups that really log each namespace get it: `env/*` comes from the
    env workers alone, so an actor rank has no series for it. Leaving those absent
    rather than writing zeros is the fixture's job, since "absent" is exactly the
    case the frontend has to be seen not drawing a line for.

    One rank is made slow on purpose. A fixture where every rank agrees cannot show
    whether the drill-down works: the aggregate mean already answers it. The whole
    claim is that one card lagging is visible per-rank and washed out in the mean.

    The per-rank factors are normalised to average to 1, so the ranks written here
    really do average to the aggregate curve already on disk -- which is what
    `_aggregate_numeric_metrics` produces on the training side. Skipping that step
    leaves the aggregate as the *unskewed* curve, and then the fixture shows the
    straggler at 4.4x the aggregate instead of 2.3x: it would make the drill-down
    look less necessary than it is, by making the mean look like it caught the
    problem.
    """
    scoped = {
        "EnvGroup": [k for k in series if k.startswith(("env/", "time/env/"))],
        "RolloutGroup": [k for k in series if k.startswith("time/rollout/")],
        "ActorGroup": [k for k in series if k.startswith("time/actor/")],
    }
    # A spread wide enough to be visible but not so wide it looks like a different
    # metric, plus a 4x tail on the straggler's timings -- then divided by their own
    # mean so the ranks average back to the aggregate.
    spread = [1.0 + (rank - (ranks - 1) / 2) * 0.06 for rank in range(ranks)]
    heavy = [4.0 if rank == straggler else 1.0 for rank in range(ranks)]
    timing = [s * h for s, h in zip(spread, heavy)]
    timing = [f / (sum(timing) / ranks) for f in timing]

    for group, keys in scoped.items():
        if not keys:
            continue
        for rank in range(ranks):
            per_rank = {
                key: [
                    value * (timing[rank] if key.startswith("time/") else spread[rank])
                    for value in series[key]
                ]
                for key in keys
            }
            _write_scalars(
                os.path.join(worker_root, group, f"rank_{rank}", "tensorboard"),
                per_rank,
                t0,
            )


def _curves(
    steps: int,
    seed: int,
    *,
    plateau_from: int | None,
    nan_at: int | None,
    step_time_doubles: bool,
    drop_critic: bool = False,
) -> dict[str, list[float]]:
    """Synthesise the 56 keys of a real embodied PPO run, with plantable faults.

    Shapes follow what the real LIBERO run looks like -- success climbing from
    zero, entropy decaying, grad norm spanning orders of magnitude -- because a
    chart that only ever sees smooth noise never shows whether a log scale or a
    stacked area is wired up correctly.
    """
    rng = random.Random(seed)
    out: dict[str, list[float]] = {}

    def ramp(lo: float, hi: float, noise: float, *, curve: float = 1.0) -> list[float]:
        vals = []
        for i in range(steps):
            frac = (i / max(1, steps - 1)) ** curve
            vals.append(lo + (hi - lo) * frac + rng.gauss(0, noise))
        return vals

    def decay(hi: float, lo: float, noise: float) -> list[float]:
        return [
            lo
            + (hi - lo) * math.exp(-3.0 * i / max(1, steps - 1))
            + rng.gauss(0, noise)
            for i in range(steps)
        ]

    out["env/success_once"] = [
        max(0.0, min(1.0, v)) for v in ramp(0.0, 0.62, 0.04, curve=1.4)
    ]
    out["env/return"] = ramp(0.0, 4.2, 0.3)
    out["env/reward"] = ramp(0.0, 0.07, 0.006)
    out["env/episode_len"] = decay(60.0, 34.0, 1.5)
    out["env/num_trajectories"] = [8.0] * steps

    out["train/actor/policy_loss"] = decay(0.42, 0.03, 0.02)
    out["train/actor/policy_loss_abs"] = [
        abs(v) for v in out["train/actor/policy_loss"]
    ]
    out["train/actor/total_loss"] = [v + 0.1 for v in out["train/actor/policy_loss"]]
    out["train/actor/entropy_loss"] = decay(2.9, 0.6, 0.05)
    out["train/actor/approx_kl"] = [abs(v) for v in ramp(0.002, 0.031, 0.004)]
    out["train/actor/clip_fraction"] = ramp(0.02, 0.19, 0.02)
    out["train/actor/clipped_ratio"] = ramp(0.01, 0.12, 0.015)
    out["train/actor/dual_cliped_ratio"] = ramp(0.0, 0.03, 0.005)
    out["train/actor/ratio"] = [1.0 + rng.gauss(0, 0.03) for _ in range(steps)]
    out["train/actor/ratio_abs"] = [abs(v) for v in out["train/actor/ratio"]]
    # Three orders of magnitude, which is the case `scale: log` exists for.
    out["train/actor/grad_norm"] = [
        max(
            1e-3,
            30.0 * math.exp(-4.0 * i / max(1, steps - 1)) * (1 + rng.gauss(0, 0.2)),
        )
        for i in range(steps)
    ]
    out["train/actor/lr"] = [
        1e-5 * (0.5 ** (i / max(1, steps / 3))) for i in range(steps)
    ]

    out["train/critic/value_loss"] = decay(1.8, 0.11, 0.05)
    out["train/critic/explained_variance"] = ramp(-0.4, 0.71, 0.06)
    out["train/critic/value_clip_ratio"] = ramp(0.3, 0.04, 0.02)
    out["train/critic/lr"] = list(out["train/actor/lr"])

    out["rollout/rewards"] = list(out["env/reward"])
    out["rollout/returns_mean"] = list(out["env/return"])
    out["rollout/returns_min"] = [v - 1.4 for v in out["env/return"]]
    out["rollout/returns_max"] = [v + 1.9 for v in out["env/return"]]
    out["rollout/advantages_mean"] = [rng.gauss(0, 0.02) for _ in range(steps)]
    out["rollout/advantages_min"] = [-1.7 + rng.gauss(0, 0.1) for _ in range(steps)]
    out["rollout/advantages_max"] = [1.9 + rng.gauss(0, 0.1) for _ in range(steps)]

    # The stacked chart's five keys must actually sum to something like time/step,
    # or "stacked: true" renders a shape nobody can sanity-check.
    rollout_t = [58.0 + rng.gauss(0, 3) for _ in range(steps)]
    train_t = [31.0 + rng.gauss(0, 2) for _ in range(steps)]
    adv_t = [2.1 + rng.gauss(0, 0.2) for _ in range(steps)]
    sync_t = [3.4 + rng.gauss(0, 0.3) for _ in range(steps)]
    eval_t = [9.0 if i % 5 == 0 else 0.0 for i in range(steps)]
    if step_time_doubles:
        # A simulator that degrades halfway through: exactly the >2x step-time
        # regression the client is meant to flag, since the server's health
        # derivation is about silence, not slowdown.
        #
        # The factors are on the *phases*, but the signal reads `time/step`, which
        # is their sum -- so slowing rollout alone by 2.6x only moves the total to
        # 1.98x and the signal (correctly) does not fire. Slowing the two dominant
        # phases puts the total unambiguously past the threshold.
        for i in range(steps // 2, steps):
            rollout_t[i] *= 3.4
            train_t[i] *= 1.4
    out["time/generate_rollouts"] = rollout_t
    out["time/actor_training"] = train_t
    out["time/cal_adv_and_returns"] = adv_t
    out["time/sync_weights"] = sync_t
    out["time/eval"] = eval_t
    out["time/step"] = [
        rollout_t[i] + train_t[i] + adv_t[i] + sync_t[i] + eval_t[i]
        for i in range(steps)
    ]
    out["time/env/interact"] = [v * 0.82 for v in rollout_t]
    out["time/env/env_interact_step"] = [v * 0.55 for v in rollout_t]
    out["time/env/run_interact_once"] = [v * 0.21 for v in rollout_t]
    out["time/env/compute_bootstrap_rewards"] = [
        0.9 + rng.gauss(0, 0.1) for _ in range(steps)
    ]
    out["time/env/env/bootstrap_step"] = [0.4] * steps
    out["time/env/env/send_rollout_trajectories"] = [1.2] * steps
    out["time/env/evaluate"] = list(eval_t)
    out["time/rollout/predict"] = [v * 0.61 for v in rollout_t]
    out["time/rollout/generate_one_epoch"] = [v * 0.9 for v in rollout_t]
    out["time/rollout/sync_model_from_actor"] = list(sync_t)
    out["time/rollout/evaluate"] = list(eval_t)
    out["time/rollout/rollout/generate"] = [v * 0.58 for v in rollout_t]
    out["time/actor/run_training"] = [v * 0.95 for v in train_t]
    out["time/actor/actor/compute_adv"] = list(adv_t)
    out["time/actor/actor/recv_traj"] = [2.8] * steps
    out["time/actor/actor/sync_model_to_rollout"] = [v * 0.7 for v in sync_t]

    # Eval runs every fifth iteration, so its series is sparser than the rest --
    # which is itself a rendering case (a chart whose x values are not 0..N).
    eval_steps = [i for i in range(steps) if i % 5 == 0]
    eval_success = []
    for n, i in enumerate(eval_steps):
        if plateau_from is not None and i >= plateau_from:
            # Flat for K consecutive evals: the "no improvement" amber signal.
            eval_success.append(0.375)
        else:
            # A saturating curve rather than a hard cap. A `min(0.75, ...)` would
            # make every long run flat at the ceiling and fire the plateau signal
            # on runs that are meant to be clean, so the healthy branch of the
            # signal check would never be exercised.
            eval_success.append(round(0.86 * (1.0 - math.exp(-0.09 * n)), 4))
    for tag, values in (
        ("eval/success_once", eval_success),
        ("eval/success_at_end", [max(0.0, v - 0.12) for v in eval_success]),
        ("eval/return", [v * 6.0 for v in eval_success]),
        ("eval/reward", [v * 0.1 for v in eval_success]),
        ("eval/episode_len", [60.0 - 20 * v for v in eval_success]),
        ("eval/num_trajectories", [8.0] * len(eval_steps)),
    ):
        # Densify onto the full step axis by repeating the last known value is
        # wrong -- it would invent eval results. The event file simply carries
        # fewer points for these tags, which is what the real run does too.
        out[tag] = values

    if nan_at is not None and nan_at < steps:
        # A single NaN is the realistic failure: one bad batch poisons the loss and
        # every later value is NaN too.
        for i in range(nan_at, steps):
            out["train/actor/policy_loss"][i] = float("nan")
        out["train/actor/grad_norm"][nan_at] = float("inf")

    if drop_critic:
        # A GRPO-style run has no value head, so it logs no `train/critic/*` at all.
        # Real, and the reason the compare view separates keys that only some of the
        # selected runs have: picking a critic metric across a PPO and a GRPO arm
        # must explain the single line rather than look like a failed request.
        for key in [k for k in out if k.startswith("train/critic/")]:
            del out[key]

    return out


def _write_eval_sparse(
    log_dir: str, out: dict[str, list[float]], steps: int, t0: float
) -> None:
    """Write the eval tags on their own (sparse) step axis."""
    eval_steps = [i for i in range(steps) if i % 5 == 0]
    eval_tags = {k: v for k, v in out.items() if k.startswith("eval/")}
    writer = EventFileWriter(log_dir, filename_suffix=".eval")
    for n, step in enumerate(eval_steps):
        for tag, values in eval_tags.items():
            if n >= len(values):
                continue
            writer.add_event(
                event_pb2.Event(
                    wall_time=t0 + step * 30.0,
                    step=step,
                    summary=summary_pb2.Summary(
                        value=[
                            summary_pb2.Summary.Value(tag=tag, simple_value=values[n])
                        ]
                    ),
                )
            )
    writer.close()


def make_run(
    root: str,
    run_id: str,
    experiment: str,
    *,
    steps: int,
    seed: int,
    state: str,
    max_steps: int | None,
    components: bool,
    stale_heartbeat_s: float | None = None,
    stale_progress_s: float | None = None,
    heartbeat_file_age_s: float | None = None,
    drop_heartbeat_at: bool = False,
    plateau_from: int | None = None,
    nan_at: int | None = None,
    step_time_doubles: bool = False,
    exit_reason: str | None = None,
    with_media: bool = True,
    semantics: str = "rl_iteration",
    drop_critic: bool = False,
    per_worker_ranks: int = 0,
    step_offset: int = 0,
) -> None:
    """Write one run tree: manifest, snapshot, events, checkpoints, media, scalars."""
    log_path = os.path.join(root, experiment)
    run_root = os.path.join(log_path, "_rlinf", "runs", run_id)
    # With `per_worker_log` on, the real writer puts the aggregate bundle under an
    # `all/` subdirectory (`MetricLogger` passes `log_path_suffix="all"`), so the
    # fixture has to as well -- otherwise it would not reproduce the layout the
    # path-recovery fix in 9ad4dd81 exists for.
    tb_dir = os.path.join(log_path, "tensorboard")
    if per_worker_ranks:
        tb_dir = os.path.join(tb_dir, "all")
    worker_root = os.path.join(log_path, "worker_logs") if per_worker_ranks else None
    video_root = os.path.join(log_path, "video")
    ckpt_root = os.path.join(log_path, "checkpoints")
    for path in (run_root, tb_dir, ckpt_root):
        os.makedirs(path, exist_ok=True)

    now = datetime.now(timezone.utc)
    started = now - timedelta(seconds=steps * 105 + 40)
    t0 = time.time() - steps * 105 - 40

    series = _curves(
        steps,
        seed,
        plateau_from=plateau_from,
        nan_at=nan_at,
        step_time_doubles=step_time_doubles,
        drop_critic=drop_critic,
    )
    _write_scalars(
        tb_dir,
        {k: v for k, v in series.items() if not k.startswith("eval/")},
        t0,
        step_offset,
    )
    _write_eval_sparse(tb_dir, series, steps, t0)
    if worker_root is not None:
        _write_worker_scalars(
            worker_root,
            {k: v for k, v in series.items() if not k.startswith("eval/")},
            t0,
            per_worker_ranks,
            straggler=per_worker_ranks - 1,
        )

    step_times = series["time/step"]
    recent = step_times[-8:]
    p50 = sorted(step_times)[len(step_times) // 2]

    heartbeat = now - timedelta(seconds=stale_heartbeat_s or 1.0)
    progress = now - timedelta(seconds=stale_progress_s or 4.0)

    # No value head means no critic loss, so the algorithm the run reports has to
    # match the keys it emits or the template's critic group would look broken
    # rather than absent.
    algorithm = (
        {"loss_type": "grpo", "adv_type": "grpo"}
        if drop_critic
        else {"loss_type": "actor_critic", "adv_type": "gae"}
    )

    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "task_type": "embodied",
        "experiment_name": experiment,
        "project_name": "rlinf",
        "step_semantics": semantics,
        "algorithm": algorithm,
        "cluster": {
            "num_nodes": 1,
            "component_placement": {"actor,env,rollout": "all"},
        },
        "git_sha": "937d85c51b99c1816fbe30e7298d5e514af22655",
        "hostname": "rlinf-demo",
        "pid": 4242 + seed,
        "started_at": _iso(started),
        "resumed_from": None,
        "paths": {
            "log_path": log_path,
            "tensorboard": tb_dir,
            "video_root": video_root,
            "checkpoint_root": ckpt_root,
            "run_root": run_root,
            # Recorded only when per-worker logging is on, matching
            # `RunStateReporter._worker_log_root`. A reader must not guess it from
            # `log_path`: every embodied example config shares `../results`, so a
            # guessed path makes one run advertise another's ranks.
            "worker_logs": worker_root,
        },
        "metric_aliases": {
            "actor/training/": "train/actor/",
            "critic/training/": "train/critic/",
        },
    }

    checkpoint = {
        "step": steps - 1,
        "path": os.path.join(ckpt_root, f"global_step_{steps - 1}"),
        "saved_at": _iso(now - timedelta(seconds=90)),
        "size_bytes": 20350835238,
        "duration_s": 77.5,
        "is_best": True,
        "metrics": {
            "eval/success_once": f"{series['eval/success_once'][-1]:.4f}",
            "eval/return": f"{series['eval/return'][-1]:.4f}",
            "eval/num_trajectories": 8,
        },
        "resume_dir": os.path.join(ckpt_root, f"global_step_{steps - 1}"),
        "entry_script": "examples/embodiment/train_embodied_agent.py",
        "config_name": experiment,
    }

    snapshot = {
        "schema_version": 2,
        "run_id": run_id,
        "task_type": "embodied",
        "algorithm": algorithm,
        "state": state,
        # An async runner reports concurrent components and no scalar phase; a
        # synchronous one reports the reverse. Both shapes have to render.
        "phase": None if components else "rollout",
        "phase_since": None if components else _iso(now - timedelta(seconds=22)),
        "components": (
            {
                "env": {"active": True, "since": _iso(started)},
                "rollout": {"active": True, "since": _iso(started)},
                "actor": {"active": True, "since": _iso(started)},
            }
            if components
            else {}
        ),
        # A running snapshot with no heartbeat at all is the server's `unknown`
        # verdict -- the writer started but never ticked. `unknown` is not a loading
        # state, so the UI has to be seen rendering it as a real verdict.
        "heartbeat_at": None if drop_heartbeat_at else _iso(heartbeat),
        "heartbeat_seq": steps * 7,
        "last_progress_at": _iso(progress),
        "last_metric_at": _iso(progress),
        "progress": {
            # Offset with the events, or the header's step and the curve's last
            # point disagree and the page looks like it lost an iteration.
            "step": steps + step_offset,
            "max_steps": max_steps,
            "epoch": 1,
            "step_semantics": semantics,
        },
        "timing": {
            "started_at": _iso(started),
            "elapsed_s": (now - started).total_seconds(),
            "step_time_p50": p50,
            "step_time_recent": recent,
            "eta_s": (max_steps - steps) * p50 if max_steps else None,
            "eta_confidence": "high" if steps > 8 else "low",
        },
        "latest_checkpoint": checkpoint,
        "paths": manifest["paths"],
        "cluster": manifest["cluster"],
        "exit": {"reason": exit_reason, "traceback_tail": None}
        if exit_reason
        else None,
    }

    with open(os.path.join(run_root, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(os.path.join(run_root, "run.json"), "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
    with open(
        os.path.join(run_root, "checkpoints.jsonl"), "w", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(checkpoint) + "\n")
    heartbeat_path = os.path.join(run_root, "heartbeat")
    with open(heartbeat_path, "w", encoding="utf-8") as handle:
        handle.write(_iso(heartbeat))
    # The server reads this file's *mtime*, not its contents, to tell "the process
    # is gone" from "the process is alive but its snapshot writes are failing"
    # (`_refine_with_heartbeat_file`). Writing it now leaves the mtime fresh, which
    # downgrades an `unreachable` verdict to `degraded` -- the stale-snapshot
    # signature. Backdating it past the budget produces `unreachable` instead, so
    # both branches are reachable from this fixture.
    file_age = heartbeat_file_age_s if heartbeat_file_age_s is not None else 1.0
    stamp = (now - timedelta(seconds=file_age)).timestamp()
    os.utime(heartbeat_path, (stamp, stamp))

    events = [
        {
            "ts": _iso(started),
            "kind": "run_start",
            "step": 0,
            "payload": {"run_id": run_id, "task_type": "embodied"},
        },
    ]
    for i in range(0, steps, max(1, steps // 6)):
        events.append(
            {
                "ts": _iso(started + timedelta(seconds=i * 105)),
                "kind": "phase_enter",
                "step": i,
                "payload": {"scope": "generate_rollouts", "phase": "rollout"},
            }
        )
    for i in range(0, steps, 5):
        events.append(
            {
                "ts": _iso(started + timedelta(seconds=i * 105 + 80)),
                "kind": "eval_done",
                "step": i,
                "payload": {"eval/success_once": series["eval/success_once"][i // 5]},
            }
        )
    events.append(
        {
            "ts": _iso(now - timedelta(seconds=90)),
            "kind": "ckpt_saved",
            "step": steps - 1,
            "payload": {"path": checkpoint["path"], "is_best": True},
        }
    )
    if nan_at is not None:
        events.append(
            {
                "ts": _iso(started + timedelta(seconds=nan_at * 105)),
                "kind": "warn",
                "step": nan_at,
                "payload": {
                    "message": "non-finite loss observed; skipping optimizer step"
                },
            }
        )
    if exit_reason:
        events.append(
            {
                "ts": _iso(now - timedelta(seconds=5)),
                "kind": "error",
                "step": steps,
                "payload": {"reason": exit_reason},
            }
        )
        events.append(
            {
                "ts": _iso(now - timedelta(seconds=4)),
                "kind": "run_end",
                "step": steps,
                "payload": {"state": state},
            }
        )
    with open(os.path.join(run_root, "events.jsonl"), "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")

    if with_media:
        # Sharded per rank, as env workers write it. By default the mp4s are stubs
        # so no binary has to ship with the repo: a browser cannot decode them, and
        # the media view's decode-failure path is itself worth seeing. Pass
        # `--sample-clip <file.mp4>` to copy a real clip into every slot instead,
        # which is what actually exercises playback.
        for rank in range(4):
            rows = []
            for split in ("train", "eval"):
                for step in range(0, steps, max(1, steps // 3)):
                    clip_dir = os.path.join(video_root, split, f"seed_{rank}")
                    os.makedirs(clip_dir, exist_ok=True)
                    path = os.path.join(clip_dir, f"{step}.mp4")
                    if not os.path.exists(path):
                        _write_clip(path)
                    num_envs = 8
                    frac = series["env/success_once"][min(step, steps - 1)]
                    rows.append(
                        {
                            "path": path,
                            "step": step,
                            "split": split,
                            "seed": rank,
                            "num_frames": 61,
                            "fps": 30,
                            "shard": rank,
                            # A tiled grid of 8 envs has a count, not a flag; `success`
                            # stays null so the UI is forced to render the count.
                            "num_success": int(round(frac * num_envs)),
                            "num_envs": num_envs,
                            "success": None,
                        }
                    )
            # One single-env clip per run, which is the only case where a scalar
            # `success` is honest -- so both branches of the media UI get covered.
            solo_dir = os.path.join(video_root, "eval", f"solo_{rank}")
            os.makedirs(solo_dir, exist_ok=True)
            solo = os.path.join(solo_dir, "0.mp4")
            _write_clip(solo)
            rows.append(
                {
                    "path": solo,
                    "step": steps - 1,
                    "split": "eval",
                    "seed": rank,
                    "num_frames": 61,
                    "fps": 30,
                    "shard": rank,
                    "num_success": 1 if rank % 2 == 0 else 0,
                    "num_envs": 1,
                    "success": rank % 2 == 0,
                }
            )
            # And one row with no recorded outcome at all: `num_success` null must
            # read as "not recorded", never as a failure.
            rows.append(
                {
                    "path": solo,
                    "step": steps - 1,
                    "split": "train",
                    "seed": rank,
                    "num_frames": 61,
                    "fps": 30,
                    "shard": rank,
                    "num_success": None,
                    "num_envs": None,
                    "success": None,
                }
            )
            with open(
                os.path.join(run_root, f"media.rank{rank}.jsonl"), "w", encoding="utf-8"
            ) as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")


def main() -> int:
    """Write the demo scan root and report where it landed."""
    global SAMPLE_CLIP

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/tmp/rlinf-demo")
    parser.add_argument("--clean", action="store_true", help="Remove the root first")
    parser.add_argument(
        "--sample-clip",
        default=None,
        help=(
            "Path to a real .mp4 to copy into every clip slot. Without it the "
            "clips are undecodable stubs, which exercises the media view's error "
            "path but not playback."
        ),
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=4,
        help=(
            "How many ranks per worker group to write for the one run that has "
            "per-worker logging on. 4 matches a single H20 node and leaves every "
            "single-metric chart inside the eight-slot colour ramp; 8 or more "
            "pushes past it, which is how the drill-down's outlier mode -- only "
            "the extremes and the median get names -- becomes reachable. Set 0 to "
            "write no per-worker tree at all."
        ),
    )
    parser.add_argument(
        "--touch",
        action="store_true",
        help=(
            "Re-stamp an existing tree's liveness timestamps to now and exit, "
            "instead of regenerating. Takes under a second where a rebuild takes "
            "two minutes, and preserves every constructed health verdict."
        ),
    )
    args = parser.parse_args()

    if args.touch:
        if not os.path.isdir(args.root):
            parser.error(f"--touch needs an existing tree: {args.root}")
        count = touch_runs(args.root)
        print(f"re-stamped {count} run(s) under {args.root}")
        return 0

    if args.sample_clip:
        if not os.path.isfile(args.sample_clip):
            parser.error(f"--sample-clip does not exist: {args.sample_clip}")
        SAMPLE_CLIP = os.path.abspath(args.sample_clip)

    if args.clean and os.path.isdir(args.root):
        shutil.rmtree(args.root)

    # Baseline and variant: same metric keys, different trajectories, so the
    # multi-run overlay has two curves that are actually comparable.
    make_run(
        args.root,
        "20260801-101500-libero_10_ppo_baseline",
        "libero_10_ppo_baseline",
        steps=120,
        seed=1,
        state="running",
        max_steps=200,
        components=False,
    )
    # Also the per-worker run. Its step time degrades, which is precisely the
    # situation where "is the whole job slow, or is one card holding it up?" is the
    # question -- and the aggregate is an arithmetic mean across ranks, so one card
    # at 4x shows up as only ~1.75x there. Four ranks over three groups gives the
    # drill-down twelve extra event directories to find, and `env/*` on the env
    # group alone gives it the absent-not-empty case.
    make_run(
        args.root,
        "20260801-142200-libero_10_ppo_lr3e6",
        "libero_10_ppo_lr3e6",
        steps=96,
        seed=7,
        state="running",
        max_steps=200,
        components=True,
        plateau_from=45,
        step_time_doubles=True,
        per_worker_ranks=args.ranks,
    )
    # No horizon: the progress card must render indeterminate rather than a
    # fabricated percentage.
    make_run(
        args.root,
        "20260802-090000-libero_10_ppo_openended",
        "libero_10_ppo_openended",
        steps=40,
        seed=13,
        state="running",
        max_steps=None,
        components=True,
    )
    # Heartbeat far past its budget in the snapshot *and* on the heartbeat file:
    # `unreachable`, the driver process is gone.
    make_run(
        args.root,
        "20260802-233000-libero_10_ppo_wedged",
        "libero_10_ppo_wedged",
        steps=61,
        seed=21,
        state="running",
        max_steps=200,
        components=False,
        stale_heartbeat_s=6000,
        stale_progress_s=6200,
        heartbeat_file_age_s=6000,
    )
    # Snapshot stale but the heartbeat *file* still fresh: the server downgrades
    # `unreachable` to `degraded` with the "snapshot writes are failing" reason. A
    # different remedy than the run above, so both must be seen rendered verbatim.
    make_run(
        args.root,
        "20260802-201000-libero_10_ppo_frozen_snapshot",
        "libero_10_ppo_frozen_snapshot",
        steps=57,
        seed=67,
        state="running",
        max_steps=200,
        components=False,
        stale_heartbeat_s=6000,
        stale_progress_s=6200,
        heartbeat_file_age_s=20,
    )
    # No heartbeat at all in a running snapshot: `unknown`.
    make_run(
        args.root,
        "20260804-060000-libero_10_ppo_noheartbeat",
        "libero_10_ppo_noheartbeat",
        steps=17,
        seed=71,
        state="running",
        max_steps=200,
        components=False,
        drop_heartbeat_at=True,
    )
    # Heartbeat fresh, progress stale: the hung-training-thread signature.
    make_run(
        args.root,
        "20260803-041200-libero_10_ppo_hung",
        "libero_10_ppo_hung",
        steps=33,
        seed=29,
        state="running",
        max_steps=200,
        components=True,
        stale_heartbeat_s=3,
        stale_progress_s=4200,
    )
    # Failed, with a NaN in the loss that only the client can see.
    make_run(
        args.root,
        "20260803-180000-libero_10_ppo_nan",
        "libero_10_ppo_nan",
        steps=52,
        seed=37,
        state="failed",
        max_steps=200,
        components=False,
        nan_at=31,
        exit_reason="loss became non-finite at iteration 31",
    )
    make_run(
        args.root,
        "20260804-070000-libero_10_ppo_queued",
        "libero_10_ppo_queued",
        steps=2,
        seed=43,
        state="pending",
        max_steps=200,
        components=False,
        with_media=False,
    )
    # The shape that killed the browser, kept as a fixture because it took a real
    # run to find it.
    #
    # One logged point pins the x scale to a zero-width range; uPlot only rejects
    # those when `dataLen > 1`, so a single point walks straight past the guard
    # into an axis-split loop that never terminates. Chrome killed the renderer
    # ("Error code: 5") and Safari stopped responding.
    #
    # `step_offset=1` is the whole point. Before this, the shortest fixture was
    # two steps starting at 0, and even a one-step version at step 0 would not
    # have reproduced it: the hang needs `min + incr == min`, and `0 + 1e-16` is
    # not `0` while `1 + 1e-16` is. The fixture that looked like the degenerate
    # case was the one value that could not fail.
    make_run(
        args.root,
        "20260804-081500-libero_10_ppo_onestep",
        "libero_10_ppo_onestep",
        steps=1,
        step_offset=1,
        seed=47,
        state="finished",
        max_steps=1,
        components=False,
        with_media=False,
    )
    # Same keys, but a step is a minibatch rather than an RL iteration. Selecting
    # this run alongside any other in the compare view is the only way to reach the
    # mixed-step-semantics warning, which is what keeps a meaningless overlay
    # from being read as a meaningful one.
    make_run(
        args.root,
        "20260803-063000-libero_10_ppo_minibatch",
        "libero_10_ppo_minibatch",
        steps=88,
        seed=51,
        state="running",
        max_steps=200,
        components=False,
        semantics="minibatch",
    )
    # A GRPO arm: same env and actor keys, no critic. Compared against a PPO arm it
    # puts real entries in the compare picker's "in some runs only" group.
    make_run(
        args.root,
        "20260802-151500-libero_10_grpo",
        "libero_10_grpo",
        steps=74,
        seed=59,
        state="running",
        max_steps=200,
        components=True,
        drop_critic=True,
    )

    print(f"wrote demo runs under {args.root}")
    print("point the server at it with:")
    print(f"  RLINF_DASHBOARD_SCAN_ROOTS={args.root} python -m rlinf_dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
