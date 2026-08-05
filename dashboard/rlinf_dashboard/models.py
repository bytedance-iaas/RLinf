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

"""Read-side models for the run control plane.

These mirror ``docs/schemas/run.v2.schema.json``, which is the single source of
truth shared with the training side. The two sides keep separate models on
purpose -- one dataclass, one pydantic -- because the dashboard cannot import
``rlinf``. ``tests/test_contract.py`` validates the same fixtures against both
the schema and these models, which is what keeps them from drifting.

Every field is optional except the ones the schema marks required. A reader must
survive a snapshot written by an older or newer training side: refusing to parse
is strictly worse than showing a run with blanks.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RunState(str, Enum):
    """Write-side lifecycle fact, as recorded by the training process.

    Deliberately has no ``stalled``: a process that was ``kill -9``'d cannot
    record its own death. Liveness is a read-side derivation -- see
    :class:`Health`.
    """

    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    STOPPED = "stopped"


class Health(str, Enum):
    """Read-side liveness verdict. Never persisted by the training side."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


TaskType = Literal[
    "embodied",
    "embodied_eval",
    "reasoning",
    "reasoning_eval",
    "coding_online_rl",
    "sft",
    "offline",
]

StepSemantics = Literal["rl_iteration", "minibatch", "optimizer_step"]


class _Lenient(BaseModel):
    """Base that tolerates unknown fields.

    A newer training side may add a field before this package is upgraded.
    Dropping the unknown key and rendering the rest beats failing the request.
    """

    model_config = ConfigDict(extra="ignore")


class ComponentState(_Lenient):
    """One concurrently running component of an async runner.

    A single scalar ``phase`` is a semantic error for async runners, where env,
    rollout and actor all run for the entire loop (they are started before the
    ``while``, not inside it).
    """

    active: bool
    since: datetime | None = None


class Progress(_Lenient):
    """How far along, and what one step means here.

    ``max_steps`` is the *effective* horizon the runner computed, not the config
    cap: runners derive it from ``max_steps`` and ``max_epochs`` together, so the
    cap can exceed what the run will ever reach.
    """

    step: int = 0
    max_steps: int | None = None
    epoch: int | None = None
    step_semantics: StepSemantics | None = None

    @property
    def fraction(self) -> float | None:
        """Completed fraction, or ``None`` when the horizon is unknown."""
        if not self.max_steps:
            return None
        return min(1.0, self.step / self.max_steps)


class Timing(_Lenient):
    """Elapsed time and the ETA derived from observed step times."""

    started_at: datetime | None = None
    elapsed_s: float = 0.0
    step_time_p50: float | None = None
    step_time_recent: list[float] = Field(default_factory=list)
    eta_s: float | None = None
    eta_confidence: Literal["low", "medium", "high"] | None = None


class CheckpointEntry(_Lenient):
    """One completed checkpoint.

    Only appended after the save finishes, which is the entire mechanism behind
    checkpoint visibility: a reader of the index can never observe a
    half-written checkpoint, so no ``WRITING``/``READY`` protocol exists.

    Resume information is stored as separate fields rather than a pre-baked
    shell command -- a command string goes stale the moment anything about the
    launch changes. The frontend assembles a display command from these.
    """

    step: int
    path: str
    saved_at: datetime | None = None
    size_bytes: int | None = None
    duration_s: float | None = None
    is_best: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)
    resume_dir: str | None = None
    entry_script: str | None = None
    config_name: str | None = None


class AlgorithmInfo(_Lenient):
    """Optional secondary template key, when ``task_type`` is too coarse."""

    loss_type: str | None = None
    adv_type: str | None = None


class ClusterInfo(_Lenient):
    """Where the run was placed, copied from ``cluster.*`` at launch."""

    num_nodes: int | None = None
    component_placement: dict[str, Any] | None = None


class ExitInfo(_Lenient):
    """Why a run ended. Non-null only for ``failed`` / ``stopped``."""

    reason: str
    traceback_tail: str | None = None


class RunSnapshot(_Lenient):
    """A parsed ``run.json``.

    Field names and types follow the frozen v2 schema exactly. ``health`` is
    absent here because it is not part of the snapshot -- see :class:`RunStatus`.
    """

    schema_version: int = 2
    run_id: str
    task_type: TaskType | str
    algorithm: AlgorithmInfo | None = None
    state: RunState
    phase: str | None = None
    phase_since: datetime | None = None
    components: dict[str, ComponentState] = Field(default_factory=dict)
    heartbeat_at: datetime | None = None
    heartbeat_seq: int = 0
    last_progress_at: datetime | None = None
    last_metric_at: datetime | None = None
    progress: Progress = Field(default_factory=Progress)
    timing: Timing = Field(default_factory=Timing)
    latest_checkpoint: CheckpointEntry | None = None
    paths: dict[str, str | None] = Field(default_factory=dict)
    cluster: ClusterInfo | None = None
    exit: ExitInfo | None = None


class RunManifest(_Lenient):
    """A parsed ``manifest.json`` -- the invariants of one launch.

    ``metric_aliases`` is embedded here by the training side precisely so a
    reader can resolve legacy metric keys without importing ``rlinf``.
    """

    schema_version: int = 2
    run_id: str
    task_type: TaskType | str
    experiment_name: str | None = None
    project_name: str | None = None
    step_semantics: StepSemantics | None = None
    algorithm: AlgorithmInfo | None = None
    cluster: ClusterInfo | None = None
    git_sha: str | None = None
    hostname: str | None = None
    pid: int | None = None
    started_at: datetime | None = None
    resumed_from: str | None = None
    paths: dict[str, str | None] = Field(default_factory=dict)
    metric_aliases: dict[str, str] = Field(default_factory=dict)


class HealthVerdict(_Lenient):
    """The read-side liveness call, with the evidence behind it.

    ``reason`` exists so the UI can explain a yellow badge instead of just
    showing one. An unexplained warning gets ignored.
    """

    health: Health
    reason: str
    heartbeat_age_s: float | None = None
    progress_age_s: float | None = None
    metric_age_s: float | None = None
    budget_s: float | None = None


class RunStatus(_Lenient):
    """What ``/runs/{id}`` returns: the snapshot plus read-side derivations."""

    run_id: str
    manifest: RunManifest | None = None
    snapshot: RunSnapshot | None = None
    health: HealthVerdict
    run_root: str
    #: Set when ``run.json`` is missing or unparseable but the run directory
    #: exists -- a run that crashed between ``makedirs`` and its first flush.
    error: str | None = None
    #: Set when the run's launch-time absolute paths were translated into this
    #: machine's namespace (see :mod:`.relocate`). Reported rather than applied
    #: silently, so a path in the UI that differs from what the training side
    #: logged can be traced to the translation instead of looking like a bug.
    relocation: dict[str, str] | None = None


class RunSummary(_Lenient):
    """One row of ``/runs``. Flat on purpose: the list view sorts on these."""

    run_id: str
    task_type: str | None = None
    experiment_name: str | None = None
    state: RunState | None = None
    health: Health = Health.UNKNOWN
    phase: str | None = None
    step: int = 0
    max_steps: int | None = None
    step_semantics: StepSemantics | None = None
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    elapsed_s: float = 0.0
    eta_s: float | None = None
    latest_checkpoint_step: int | None = None
    run_root: str


class SeriesPoint(_Lenient):
    """One sample of one metric."""

    step: int
    value: float
    #: Wall-clock, when the source has it. TensorBoard event files do; optional
    #: so a source without it need not invent one.
    wall_time: float | None = None


class Series(_Lenient):
    """One metric's history, plus where it came from and what was dropped."""

    key: str
    points: list[SeriesPoint] = Field(default_factory=list)
    #: Which source answered: ``tensorboard``, or ``none`` when no source had
    #: data. Left as a free string rather than a closed enum because
    #: ``MetricSource.name`` is open -- pinning the values here would mean a new
    #: source crashed inside ``make_series`` instead of just working.
    source: str = "none"
    #: True when decimation dropped points, so the UI can say "sampled".
    decimated: bool = False
    total_points: int = 0


class MediaEntry(_Lenient):
    """One recorded video, from the sharded media index."""

    path: str
    step: int | None = None
    split: str = "train"
    seed: int | None = None
    num_frames: int | None = None
    fps: int | None = None
    shard: int = 0
    #: How many of the clip's envs had succeeded when it was written. One MP4 is
    #: a tiled grid of every env in a worker, so success is a count rather than a
    #: flag; ``None`` means the env tracks no success notion (or is pre-#1425).
    num_success: int | None = None
    num_envs: int | None = None
    #: Scalar success, set only when the clip shows a single env. ``None`` for a
    #: grid -- deliberately not a collapsed "any env succeeded", which would read
    #: as a claim about the whole clip.
    success: bool | None = None
    #: Server-relative URL for streaming, filled in by the API layer.
    url: str | None = None
