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

"""Suppression of third-party start-up noise.

A single embodied run spawns dozens of processes -- one Ray worker per rank plus
one simulator subprocess per environment -- and every one of them replays the
same third-party banners, so a few chatty imports turn into hundreds of log
lines that bury the actual training output. This module collects the
suppression mechanisms, grouped by the layer each one has to act on:

* :func:`apply_noisy_dependency_env_vars` -- banners printed from native code
  (TensorFlow) before Python can install any filter, so they can only be
  disabled through the environment.
* :func:`quiet_noisy_third_party_loggers` -- chatty ``logging`` loggers,
  silenced by raising their level.
* :func:`suppress_gym_notice` -- a notice printed with a bare ``print`` to
  stderr, which neither of the above can reach.
* :func:`suppress_fsdp_no_shard_warnings` -- ``warnings`` warnings, filtered by
  message and category.

:func:`suppress_start_up_noise` applies the first three and is called from
``rlinf/__init__.py``, so it runs in every process that touches RLinf --
including the ``spawn``-ed simulator subprocesses, which re-import from scratch.
The FSDP filter is applied from ``fsdp_model_manager`` instead, since only
training processes import it.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

TF_ENV_VARS: dict[str, str] = {
    # "3" rather than "2": a 4-GPU run whose config already set "2" through
    # `hydra.job.env_set` still logged all 26 `port.cc:153` / `cudart_stub.cc:31`
    # banners, so "2" is demonstrably not enough for these. The cost is that
    # TensorFlow's C++ ERROR logs are hidden too; that is acceptable here because
    # TensorFlow is only reached from `rlinf/envs/utils.py::crop_and_resize`, and
    # a genuine import or call failure still raises a Python exception.
    "TF_CPP_MIN_LOG_LEVEL": "3",
}
"""Environment variables that mute TensorFlow's start-up banners.

TensorFlow is an indirect dependency of the embodied stack and prints these
banners from C++ during import, i.e. before any Python-level filter can be
installed, so the environment is the only lever. It also has to be set before
the process starts: ``hydra.job.env_set`` applies inside the job body, by which
point the driver has already imported its dependencies.

Deliberately *not* set here: ``TF_ENABLE_ONEDNN_OPTS=0``. It would remove the
`oneDNN custom operations are on ...` notice, but it does so by turning the
optimisation off, which changes how TensorFlow computes rather than only what it
logs. Export it manually if you want that trade.
"""

NOISY_THIRD_PARTY_LOGGERS: tuple[str, ...] = (
    "absl",
    "datasets",
    "filelock",
    "h5py",
    "matplotlib",
    "OpenGL",
    "PIL",
    "robosuite_logs",
    "urllib3",
)
"""Loggers whose INFO output is noise for RLinf runs."""

_GYM_NOTICE_MODULES: tuple[str, ...] = ("gym_notices", "gym_notices.notices")


def resolve_noisy_dependency_env_vars(
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return the effective value of every noise-suppressing variable.

    An explicit setting always wins, so someone debugging a TensorFlow issue can
    export ``TF_CPP_MIN_LOG_LEVEL=0`` and keep it. Use this when handing the
    variables to another process: the result is complete, so the receiving
    process gets the intended value whether or not it inherits this environment.

    Args:
        environ: Environment to read overrides from. Defaults to ``os.environ``.

    Returns:
        Mapping of variable name to its effective value.
    """
    if environ is None:
        environ = os.environ
    return {name: environ.get(name, default) for name, default in TF_ENV_VARS.items()}


def apply_noisy_dependency_env_vars(
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """Set the noise-suppressing variables that are not already set.

    Existing values are left untouched, so this never overrides an explicit
    setting.

    Args:
        environ: Environment to update. Defaults to ``os.environ``.

    Returns:
        The variables that were actually set.
    """
    if environ is None:
        environ = os.environ
    applied = {
        name: value for name, value in TF_ENV_VARS.items() if name not in environ
    }
    environ.update(applied)
    return applied


def quiet_noisy_third_party_loggers(
    level: int = logging.WARNING,
    logger_names: tuple[str, ...] = NOISY_THIRD_PARTY_LOGGERS,
) -> None:
    """Raise the level of known-chatty third-party loggers.

    RLinf's own loggers are untouched: workers log through a dedicated
    ``logging.Logger`` with ``propagate = False``, so restricting these
    namespaces cannot hide RLinf output.

    Args:
        level: Level to apply to each logger.
        logger_names: Logger namespaces to quiet down.
    """
    for name in logger_names:
        logging.getLogger(name).setLevel(level)


def suppress_gym_notice(modules: dict | None = None) -> tuple[str, ...]:
    """Stop legacy ``gym`` from printing its unmaintained-package notice.

    ``gym/__init__.py`` ends with::

        try:
            import gym_notices.notices as notices

            notice = notices.notices.get(__version__)
            if notice:
                print(notice, file=sys.stderr)
        except Exception:
            pass

    Because that is a bare ``print`` to stderr rather than a ``warnings``
    warning, neither ``PYTHONWARNINGS`` nor ``warnings.filterwarnings`` can
    reach it. Blocking ``gym_notices`` makes the ``import`` raise, which gym
    already swallows, so the notice is skipped and nothing else changes --
    ``gym_notices`` carries no functionality beyond these notices.

    Uninstalling ``gym_notices`` would have the same effect, but it is a declared
    dependency of ``gym``, so removing it leaves the environment failing a
    dependency check and any later ``uv sync`` puts it back. Blocking the import
    is contained to the process and works in environments that already exist.
    Modules that are already imported are left alone so we never yank a module
    out from under code that is using it.

    Args:
        modules: Module table to update. Defaults to ``sys.modules``.

    Returns:
        Names that were newly blocked.
    """
    if modules is None:
        modules = sys.modules
    blocked = []
    for name in _GYM_NOTICE_MODULES:
        if name not in modules:
            # A ``None`` entry makes ``import name`` raise ImportError.
            modules[name] = None
            blocked.append(name)
    return tuple(blocked)


def suppress_fsdp_no_shard_warnings() -> None:
    """Filter out torch's ``NO_SHARD`` deprecation and state-dict warnings.

    ``no_shard`` is a deliberate choice for the embodied configs -- every rank
    holding a full replica is what keeps per-step time flat -- so migrating to
    DDP or FSDP2 is a behaviour change, not a logging fix. torch warns once per
    FSDP wrap, i.e. twice per rank, which is 16 lines on a 4-GPU run.

    Both patterns are matched narrowly enough that unrelated ``UserWarning`` and
    ``FutureWarning`` output stays visible.
    """
    warnings.filterwarnings(
        "ignore",
        message=".*NO_SHARD.*full_state_dict.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*NO_SHARD.*sharding strategy is deprecated.*",
        category=FutureWarning,
    )


def suppress_start_up_noise() -> None:
    """Apply every start-up noise suppression mechanism.

    Safe to call repeatedly and from any process. Must run before the noisy
    dependencies are imported, hence the call from ``rlinf/__init__.py``.
    """
    apply_noisy_dependency_env_vars()
    quiet_noisy_third_party_loggers()
    suppress_gym_notice()
