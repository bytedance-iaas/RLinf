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

import logging
import warnings

from rlinf.utils.log_noise import (
    TF_ENV_VARS,
    apply_noisy_dependency_env_vars,
    quiet_noisy_third_party_loggers,
    resolve_noisy_dependency_env_vars,
    suppress_fsdp_no_shard_warnings,
    suppress_gym_notice,
)

# Verbatim from a 4-GPU run's log, so the filters are matched against the text
# torch actually emits rather than a paraphrase of it.
_NO_SHARD_DEPRECATION = (
    "The `NO_SHARD` sharding strategy is deprecated. If having issues, "
    "please use `DistributedDataParallel` instead."
)


def test_resolve_reports_every_variable_when_environment_is_empty():
    # The worker-side injection relies on this being complete rather than a diff:
    # a Ray worker on a node started outside the launch script inherits nothing.
    assert resolve_noisy_dependency_env_vars({}) == TF_ENV_VARS


def test_resolve_keeps_an_explicitly_set_value():
    resolved = resolve_noisy_dependency_env_vars({"TF_CPP_MIN_LOG_LEVEL": "0"})
    assert resolved["TF_CPP_MIN_LOG_LEVEL"] == "0"


def test_resolve_still_reports_variables_the_user_did_not_set():
    # A user override of one variable must not drop the others from the mapping.
    resolved = resolve_noisy_dependency_env_vars({"TF_CPP_MIN_LOG_LEVEL": "0"})
    assert set(resolved) == set(TF_ENV_VARS)


def test_apply_sets_missing_variables():
    environ = {}
    applied = apply_noisy_dependency_env_vars(environ)
    assert applied == TF_ENV_VARS
    assert environ == TF_ENV_VARS


def test_apply_does_not_override_an_explicit_value():
    environ = {"TF_CPP_MIN_LOG_LEVEL": "0"}
    applied = apply_noisy_dependency_env_vars(environ)
    assert "TF_CPP_MIN_LOG_LEVEL" not in applied
    assert environ["TF_CPP_MIN_LOG_LEVEL"] == "0"


def test_tf_log_level_is_high_enough_for_the_native_banners():
    # A run whose config set "2" still logged every port.cc / cudart_stub banner.
    assert TF_ENV_VARS["TF_CPP_MIN_LOG_LEVEL"] == "3"


def test_quieting_blocks_third_party_info():
    name = "test_log_noise_fake_lib"
    logging.getLogger(name).setLevel(logging.NOTSET)
    quiet_noisy_third_party_loggers(logger_names=(name,))
    assert not logging.getLogger(name).isEnabledFor(logging.INFO)


def test_quieting_keeps_third_party_warnings_visible():
    # Noise reduction must not turn into blanket silence: a real problem reported
    # at WARNING still has to reach the log.
    name = "test_log_noise_fake_lib_warn"
    quiet_noisy_third_party_loggers(logger_names=(name,))
    assert logging.getLogger(name).isEnabledFor(logging.WARNING)


def test_quieting_a_namespace_covers_its_child_loggers():
    # The log shows `INFO:OpenGL.acceleratesupport`, not `INFO:OpenGL`, so
    # suppression has to reach child loggers to be worth anything.
    logging.getLogger().setLevel(logging.INFO)
    quiet_noisy_third_party_loggers()
    assert not logging.getLogger("OpenGL.acceleratesupport").isEnabledFor(logging.INFO)
    assert not logging.getLogger("datasets.builder").isEnabledFor(logging.INFO)


def test_quieting_leaves_the_root_logger_alone():
    # ~100 places in this repo log through the root logger; that output must
    # survive, which is why suppression is per-namespace rather than global.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    quiet_noisy_third_party_loggers()
    assert root.isEnabledFor(logging.INFO)


def test_gym_notice_import_is_blocked():
    modules = {}
    blocked = suppress_gym_notice(modules)
    assert "gym_notices" in blocked
    # A ``None`` entry is what makes the ``import`` inside gym raise, which gym
    # already swallows.
    assert modules["gym_notices"] is None


def test_gym_notice_leaves_an_already_imported_module_alone():
    sentinel = object()
    modules = {"gym_notices": sentinel}
    blocked = suppress_gym_notice(modules)
    assert "gym_notices" not in blocked
    assert modules["gym_notices"] is sentinel


def test_gym_notice_suppression_is_idempotent():
    modules = {}
    suppress_gym_notice(modules)
    assert suppress_gym_notice(modules) == ()


def test_no_shard_deprecation_is_filtered():
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        suppress_fsdp_no_shard_warnings()
        warnings.warn(_NO_SHARD_DEPRECATION, FutureWarning)
    assert caught == []


def test_no_shard_full_state_dict_warning_is_filtered():
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        suppress_fsdp_no_shard_warnings()
        warnings.warn("NO_SHARD is not supported with full_state_dict", UserWarning)
    assert caught == []


def test_unrelated_future_warnings_still_surface():
    # The filters have to be narrow: a genuine deprecation elsewhere in the
    # stack must not be swallowed along with the NO_SHARD noise.
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        suppress_fsdp_no_shard_warnings()
        warnings.warn("some other API is deprecated", FutureWarning)
    assert len(caught) == 1


def test_unrelated_user_warnings_still_surface():
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        suppress_fsdp_no_shard_warnings()
        warnings.warn("an unrelated full_state_dict problem", UserWarning)
    assert len(caught) == 1
