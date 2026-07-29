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

from omegaconf import OmegaConf

from rlinf.utils.logging import print_config_if_enabled


def test_config_dump_is_off_by_default(capsys):
    cfg = OmegaConf.create({"runner": {"task_type": "embodied"}})
    assert not print_config_if_enabled(cfg)
    assert capsys.readouterr().out == ""


def test_config_dump_runs_when_explicitly_enabled(capsys):
    cfg = OmegaConf.create({"runner": {"print_config": True}, "actor": {"lr": 1e-5}})
    assert print_config_if_enabled(cfg)
    assert "actor" in capsys.readouterr().out


def test_config_dump_resolves_interpolations(capsys):
    # The point of the dump is to show effective values, so it must resolve.
    cfg = OmegaConf.create(
        {"runner": {"print_config": True}, "a": 7, "b": "${a}"},
    )
    print_config_if_enabled(cfg)
    assert '"b": 7' in capsys.readouterr().out


def test_config_dump_is_off_when_struct_mode_hides_the_key(capsys):
    # `validate_cfg` turns on struct mode; reading a missing key must not raise.
    cfg = OmegaConf.create({"runner": {"task_type": "embodied"}})
    OmegaConf.set_struct(cfg, True)
    assert not print_config_if_enabled(cfg)
    assert capsys.readouterr().out == ""


def test_config_dump_is_off_when_there_is_no_runner_section(capsys):
    cfg = OmegaConf.create({"actor": {"lr": 1e-5}})
    assert not print_config_if_enabled(cfg)
    assert capsys.readouterr().out == ""
