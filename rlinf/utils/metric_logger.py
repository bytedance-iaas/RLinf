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

"""Metric logging: the single write entry point for the data plane.

Every runner logs through :class:`MetricLogger`, and nothing writes metrics
around it. That is what makes it possible to add a backend, normalize key
names, or notify the control plane in one place and have all runners benefit.

Backends implement :class:`MetricBackend`. wandb and swanlab are wrapped in
thin adapters rather than used as bare modules so that ``flush`` and
``log_media`` mean the same thing everywhere and a missing capability degrades
to a warning instead of an exception.
"""

from __future__ import annotations

import os
import time
from typing import Protocol, runtime_checkable

from omegaconf import DictConfig, OmegaConf

from rlinf.utils.logging import get_logger
from rlinf.utils.metric_naming import dual_write

#: Default seconds between forced backend flushes. TensorBoard's own buffering
#: can otherwise leave a live run looking stalled in the UI for minutes, which
#: defeats the point of watching a run in progress.
DEFAULT_FLUSH_INTERVAL_S = 10.0


@runtime_checkable
class MetricBackend(Protocol):
    """What a metric backend must provide.

    ``log`` is the only required capability. The rest have no-op or
    warn-and-skip defaults in the adapters, so an unsupported operation
    degrades observability without failing a training run.
    """

    def log(self, data: dict, step: int) -> None: ...

    def log_table(self, df_data, name: str, step: int) -> None: ...

    def log_media(self, path: str, name: str, step: int, **kwargs) -> None: ...

    def flush(self) -> None: ...

    def finish(self) -> None: ...


class _BaseBackend:
    """Shared warn-and-skip defaults for optional capabilities."""

    name = "base"

    def log(self, data: dict, step: int) -> None:
        raise NotImplementedError

    def log_table(self, df_data, name: str, step: int) -> None:
        get_logger().warning(
            f"Backend '{self.name}' does not support tables; skipping '{name}'."
        )

    def log_media(self, path: str, name: str, step: int, **kwargs) -> None:
        get_logger().warning(
            f"Backend '{self.name}' does not support media; skipping '{name}'."
        )

    def flush(self) -> None:
        pass

    def finish(self) -> None:
        pass


class _TensorboardLogger(_BaseBackend):
    name = "tensorboard"

    def __init__(self, log_path):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_path)
        self._closed = False

    def log(self, data: dict[str, float], step: int) -> None:
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def flush(self) -> None:
        if not self._closed:
            self.writer.flush()

    def finish(self):
        # Idempotent: `finish()` is called explicitly by runner teardown and
        # again from `MetricLogger.__del__`, and SummaryWriter.close() on an
        # already-closed writer is not safe to assume.
        if not self._closed:
            self._closed = True
            self.writer.close()


class _WandbBackend(_BaseBackend):
    """One wandb run addressed through its explicit run handle.

    Module-level logging follows the most recently initialized run, which is
    unsafe when aggregate and per-worker bundles coexist. The module remains
    available for its ``Table`` and ``Video`` constructors.
    """

    name = "wandb"

    def __init__(self, wandb_module, run=None):
        self._wandb = wandb_module
        if run is None:
            candidate = getattr(wandb_module, "run", None)
            if callable(getattr(candidate, "log", None)) and callable(
                getattr(candidate, "finish", None)
            ):
                run = candidate
        self._run = run
        self._finished = False

    @property
    def _target(self):
        """The run to write to, falling back to the module if there is none."""
        return self._run if self._run is not None else self._wandb

    def log(self, data: dict, step: int) -> None:
        self._target.log(data=data, step=step)

    def log_table(self, df_data, name: str, step: int) -> None:
        table = self._wandb.Table(dataframe=df_data)
        self._target.log({name: table}, step=step)

    def log_media(self, path: str, name: str, step: int, **kwargs) -> None:
        self._target.log({name: self._wandb.Video(path, **kwargs)}, step=step)

    def finish(self) -> None:
        if not self._finished:
            self._finished = True
            self._target.finish()


class _SwanlabBackend(_BaseBackend):
    """One swanlab run. Same handle-versus-module hazard as :class:`_WandbBackend`."""

    name = "swanlab"

    def __init__(self, swanlab_module, run=None):
        self._swanlab = swanlab_module
        self._run = run
        self._finished = False

    @property
    def _target(self):
        return self._run if self._run is not None else self._swanlab

    def log(self, data: dict, step: int) -> None:
        self._target.log(data=data, step=step)

    def finish(self) -> None:
        if not self._finished:
            self._finished = True
            self._target.finish()


class MetricLogger:
    supported_logger = ["wandb", "swanlab", "tensorboard"]

    #: Class-level defaults so ``__del__`` is safe on a partially constructed
    #: logger. ``__init__`` can raise before these are set per instance (an
    #: unsupported backend name asserts), and the resulting ``AttributeError``
    #: inside ``__del__`` would surface as an unraisable exception that masks
    #: the real config error. Empty tuple rather than list: this default is
    #: shared across instances and must not be appendable.
    _finished = False
    _all_loggers = ()

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        logger_cfg = cfg.runner.logger

        self.log_path = logger_cfg.get("log_path", "logs")
        self.project_name = logger_cfg.get("project_name", "rlinf")
        self.experiment_name = logger_cfg.get("experiment_name", "default")
        self.per_worker_log = bool(cfg.runner.get("per_worker_log", False))
        self.per_worker_log_root = cfg.runner.get(
            "per_worker_log_path", os.path.join(self.log_path, "worker_logs")
        )

        logger_backends = logger_cfg.get("logger_backends", ["tensorboard"])
        if isinstance(logger_backends, str):
            self.logger_backends = [logger_backends]
        elif logger_backends is None:
            self.logger_backends = []
        else:
            self.logger_backends = logger_backends

        self.wandb_proxy = logger_cfg.get("wandb_proxy", None)
        self.wandb_entity = logger_cfg.get("wandb_entity", None)
        self.swanlab_mode = logger_cfg.get("swanlab_mode", "cloud")
        if len(self.logger_backends) > 0:
            assert all(
                backend in self.supported_logger for backend in self.logger_backends
            ), f"Unsupported logger backend: {self.logger_backends}"

        self.flush_interval_s = float(
            logger_cfg.get("flush_interval_s", DEFAULT_FLUSH_INTERVAL_S)
        )
        self._last_flush = time.monotonic()
        self._finished = False
        # Set by the runner so the control plane learns when metrics last
        # reached a backend. Optional and one-directional: MetricLogger never
        # imports run_state.
        self._on_log = None

        self.config = OmegaConf.to_container(cfg, resolve=True)
        self._all_loggers = []
        self._worker_loggers: dict[tuple[str, int], dict] = {}
        self.logger = self._create_logger_bundle(
            log_path=self.log_path,
            experiment_name=self.experiment_name,
            log_path_suffix="all" if self.per_worker_log else "",
        )

    def set_log_callback(self, callback) -> None:
        """Register a callable invoked after each successful log.

        Used by the run-state reporter to maintain ``last_metric_at``, which
        detects a broken metric path while training itself still advances.
        """
        self._on_log = callback

    def _create_logger_bundle(
        self, log_path: str, experiment_name: str, log_path_suffix: str = ""
    ) -> dict:
        logger = {}
        if "wandb" in self.logger_backends:
            import wandb

            wandb_log_path = os.path.join(log_path, "wandb", log_path_suffix)
            os.makedirs(wandb_log_path, exist_ok=True)

            settings = None
            if self.wandb_proxy:
                settings = wandb.Settings(https_proxy=self.wandb_proxy)
            # Each bundle keeps its own handle so per-worker runs stay isolated.
            run = wandb.init(
                entity=self.wandb_entity,
                project=self.project_name,
                name=experiment_name,
                config=self.config,
                settings=settings,
                dir=wandb_log_path,
                reinit=True,
            )
            logger["wandb"] = _WandbBackend(wandb, run)

        if "swanlab" in self.logger_backends:
            import swanlab

            swanlab_log_path = os.path.join(log_path, "swanlab", log_path_suffix)
            os.makedirs(swanlab_log_path, exist_ok=True)

            run = swanlab.init(
                project=self.project_name,
                experiment_name=experiment_name,
                config=self.config,
                logdir=swanlab_log_path,
                mode=self.swanlab_mode,
            )
            logger["swanlab"] = _SwanlabBackend(swanlab, run)

        if "tensorboard" in self.logger_backends:
            tensorboard_log_path = os.path.join(
                log_path, "tensorboard", log_path_suffix
            )
            os.makedirs(tensorboard_log_path, exist_ok=True)

            config_yaml_path = os.path.join(tensorboard_log_path, "config.yaml")
            OmegaConf.save(self.cfg, config_yaml_path, resolve=True)

            logger["tensorboard"] = _TensorboardLogger(tensorboard_log_path)

        self._all_loggers.append(logger)
        return logger

    def _get_scoped_logger(self, worker_group_name: str, rank: int) -> dict:
        key = (worker_group_name, int(rank))
        if key in self._worker_loggers:
            return self._worker_loggers[key]

        scoped_log_path = os.path.join(
            self.per_worker_log_root,
            worker_group_name,
            f"rank_{int(rank)}",
        )
        scoped_experiment_name = (
            f"{self.experiment_name}-{worker_group_name}-rank_{int(rank)}"
        )
        scoped_logger = self._create_logger_bundle(
            log_path=scoped_log_path,
            experiment_name=scoped_experiment_name,
        )
        self._worker_loggers[key] = scoped_logger
        return scoped_logger

    def log(
        self,
        data,
        step,
        backend=None,
        worker_group_name: str | None = None,
        rank: int | None = None,
    ):
        target_logger = self.logger
        if self.per_worker_log and worker_group_name is not None and rank is not None:
            target_logger = self._get_scoped_logger(
                worker_group_name=worker_group_name,
                rank=rank,
            )
        data = dual_write(data)
        for default_backend, logger_instance in target_logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

        if self._on_log is not None:
            try:
                self._on_log()
            except Exception as exc:  # noqa: BLE001 - observability is not critical
                get_logger().debug(f"Metric log callback failed: {exc}")
        self._maybe_flush()

    def _maybe_flush(self) -> None:
        """Flush backends periodically so a live run is visible while running."""
        now = time.monotonic()
        if now - self._last_flush < self.flush_interval_s:
            return
        self._last_flush = now
        self.flush()

    def flush(self) -> None:
        for logger in self._all_loggers:
            for logger_instance in logger.values():
                try:
                    logger_instance.flush()
                except Exception as exc:  # noqa: BLE001
                    get_logger().debug(f"Backend flush failed: {exc}")

    def log_table(self, df_data, name, step):
        # Dispatch through the backends so an unsupported one warns and skips.
        for logger_instance in self.logger.values():
            logger_instance.log_table(df_data=df_data, name=name, step=step)

    def log_media(self, path, name, step, **kwargs):
        for logger_instance in self.logger.values():
            logger_instance.log_media(path=path, name=name, step=step, **kwargs)

    def __del__(self):
        self.finish()

    def finish(self):
        # Idempotent: runner teardown calls this explicitly and `__del__` calls
        # it again at interpreter shutdown.
        if self._finished:
            return
        self._finished = True
        for logger in self._all_loggers:
            for logger_instance in logger.values():
                logger_instance.finish()
