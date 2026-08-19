# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import OmegaConf

from rlinf.workers.rollout.hf.huggingface_worker import build_rollout_model_config


def _cfg(rollout_model: dict) -> tuple:
    cfg = OmegaConf.create(
        {
            "rollout": {"model": rollout_model},
            "actor": {
                "model": {
                    "model_path": "/actor/path",
                    "precision": "bfloat16",
                    "openpi": {"enable_fused_prefix": True},
                }
            },
        }
    )
    return cfg, cfg.actor.model


def test_rollout_inherits_actor_fused_prefix_when_unset() -> None:
    cfg, model_cfg = _cfg({"model_path": "/rollout/path", "precision": "float32"})

    resolved = build_rollout_model_config(cfg, model_cfg)

    assert resolved.openpi.enable_fused_prefix is True
    assert resolved.model_path == "/rollout/path"
    assert resolved.precision == "float32"


def test_rollout_can_disable_fused_prefix_independently() -> None:
    cfg, model_cfg = _cfg(
        {
            "model_path": "/rollout/path",
            "precision": "float32",
            "openpi": {"enable_fused_prefix": False},
        }
    )

    resolved = build_rollout_model_config(cfg, model_cfg)

    assert resolved.openpi.enable_fused_prefix is False
    # the actor's own config must not be mutated by the rollout override
    assert cfg.actor.model.openpi.enable_fused_prefix is True


def test_rollout_can_enable_fused_prefix_independently() -> None:
    cfg, model_cfg = _cfg({"model_path": "/rollout/path", "precision": "float32"})
    cfg.actor.model.openpi.enable_fused_prefix = False
    OmegaConf.update(
        cfg, "rollout.model.openpi.enable_fused_prefix", True, force_add=True
    )

    resolved = build_rollout_model_config(cfg, cfg.actor.model)

    assert resolved.openpi.enable_fused_prefix is True
    assert cfg.actor.model.openpi.enable_fused_prefix is False
