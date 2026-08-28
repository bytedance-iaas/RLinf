# Copyright 2026 The RLinf Authors.
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
"""Unit tests for the LeRobot -> OpenPI PyTorch checkpoint convertor.

Runs on synthetic checkpoints, so it needs neither a real pi0.5 file nor a GPU.
"""

import json

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file as save_np
from safetensors.torch import save_file as save_pt

from rlinf.utils.ckpt_convertor.openpi import lerobot_to_openpi_pytorch as conv

STATS = ("mean", "std", "q01", "q99")


def _write_lerobot_weights(path, prefix="model."):
    sd = {
        f"{prefix}action_in_proj.weight": torch.randn(4, 3, dtype=torch.float32),
        f"{prefix}paligemma_with_expert.layer.weight": torch.randn(
            2, 2, dtype=torch.bfloat16
        ),
        f"{prefix}norm.bias": torch.randn(2, dtype=torch.float32),
        "already_bare.weight": torch.randn(2, dtype=torch.bfloat16),
    }
    save_pt(sd, str(path), metadata={"format": "pt"})
    return sd


def _write_lerobot_norm(path, dim=6, features=("observation.state", "action")):
    payload = {}
    for feat in features:
        payload[f"{feat}.mean"] = np.arange(dim, dtype=np.float32)
        payload[f"{feat}.std"] = np.ones(dim, dtype=np.float32)
        payload[f"{feat}.q01"] = -np.ones(dim, dtype=np.float32) * 50
        payload[f"{feat}.q99"] = np.ones(dim, dtype=np.float32) * 50
    save_np(payload, str(path))


# --- weights ---------------------------------------------------------------


def test_strips_the_lerobot_prefix(tmp_path):
    src = tmp_path / "model.safetensors"
    _write_lerobot_weights(src)
    out = tmp_path / "out"
    report = conv.convert_weights(src, out)

    from safetensors.torch import load_file

    got = load_file(str(out / "model.safetensors"))
    assert report["tensors_in"] == 4 and report["tensors_out"] == 4
    assert "action_in_proj.weight" in got
    assert not any(k.startswith("model.") for k in got)
    assert "already_bare.weight" in got, "a bare key must survive untouched"


def test_preserves_mixed_dtypes(tmp_path):
    """The whole point of not reusing the bf16-casting default in _core.

    A LeRobot pi0.5 checkpoint mixes fp32 and bf16; casting the fp32 tensors down
    would quietly lose precision the training run kept.
    """
    src = tmp_path / "model.safetensors"
    _write_lerobot_weights(src)
    out = tmp_path / "out"
    conv.convert_weights(src, out)

    from safetensors.torch import load_file

    got = load_file(str(out / "model.safetensors"))
    assert got["action_in_proj.weight"].dtype == torch.float32
    assert got["paligemma_with_expert.layer.weight"].dtype == torch.bfloat16


def test_values_are_bit_identical(tmp_path):
    src = tmp_path / "model.safetensors"
    original = _write_lerobot_weights(src)
    out = tmp_path / "out"
    conv.convert_weights(src, out)

    from safetensors.torch import load_file

    got = load_file(str(out / "model.safetensors"))
    for k, v in original.items():
        bare = k[len("model.") :] if k.startswith("model.") else k
        assert torch.equal(got[bare].float(), v.float()), f"{bare} changed"


def test_colliding_keys_raise_instead_of_dropping(tmp_path):
    """Two source keys collapsing to one bare key must fail loudly."""
    src = tmp_path / "model.safetensors"
    save_pt(
        {
            "model.weight": torch.zeros(2),
            "weight": torch.ones(2),  # collides after the strip
        },
        str(src),
        metadata={"format": "pt"},
    )
    with pytest.raises(ValueError, match="duplicate bare key"):
        conv.convert_weights(src, tmp_path / "out")


# --- norm stats ------------------------------------------------------------


def test_norm_stats_shape_and_keys(tmp_path):
    src = tmp_path / "norm.safetensors"
    _write_lerobot_norm(src, dim=6)
    out = tmp_path / "asset" / "norm_stats.json"
    dims = conv.convert_norm_stats(src, out)

    assert dims == {"state": 6, "actions": 6}
    payload = json.loads(out.read_text())
    assert set(payload) == {"norm_stats"}
    assert set(payload["norm_stats"]) == {"state", "actions"}
    for sec in ("state", "actions"):
        assert set(payload["norm_stats"][sec]) == set(STATS)
        assert len(payload["norm_stats"][sec]["mean"]) == 6


def test_norm_stats_keep_native_width(tmp_path):
    """Stats must not be padded to max_action_dim; openpi slices before padding."""
    src = tmp_path / "norm.safetensors"
    _write_lerobot_norm(src, dim=6)
    out = tmp_path / "a" / "norm_stats.json"
    conv.convert_norm_stats(src, out)
    mean = json.loads(out.read_text())["norm_stats"]["state"]["mean"]
    assert len(mean) == 6, "padding to 32 would bury the real action width"


def test_missing_statistic_raises(tmp_path):
    src = tmp_path / "norm.safetensors"
    save_np(
        {
            "observation.state.mean": np.zeros(6, dtype=np.float32),
            "observation.state.std": np.ones(6, dtype=np.float32),
            # q01/q99 absent
        },
        str(src),
    )
    with pytest.raises(ValueError, match="missing statistics"):
        conv.convert_norm_stats(
            src, tmp_path / "n.json", {"observation.state": "state"}
        )


def test_degenerate_quantile_span_raises(tmp_path):
    """q99 == q01 would make quantile normalization divide by ~zero."""
    src = tmp_path / "norm.safetensors"
    payload = {}
    for stat, val in (("mean", 0.0), ("std", 1.0), ("q01", 3.0), ("q99", 3.0)):
        payload[f"observation.state.{stat}"] = np.full(6, val, dtype=np.float32)
    save_np(payload, str(src))
    with pytest.raises(ValueError, match="degenerate"):
        conv.convert_norm_stats(
            src, tmp_path / "n.json", {"observation.state": "state"}
        )


def test_feature_map_is_overridable(tmp_path):
    src = tmp_path / "norm.safetensors"
    _write_lerobot_norm(src, dim=7, features=("obs.custom", "act.custom"))
    out = tmp_path / "a" / "norm_stats.json"
    dims = conv.convert_norm_stats(
        src, out, {"obs.custom": "state", "act.custom": "actions"}
    )
    assert dims == {"state": 7, "actions": 7}


# --- wiring ----------------------------------------------------------------


def test_mode_is_registered_in_the_unified_entry_point():
    from rlinf.utils.ckpt_convertor.openpi import convert as convert_cli

    parser = convert_cli.build_parser()
    action = next(
        a
        for a in parser._actions
        if getattr(a, "choices", None) and "sft2deploy" in a.choices
    )
    assert "lerobot_to_openpi_pytorch" in action.choices


def test_end_to_end_writes_both_artifacts(tmp_path):
    weights = tmp_path / "src" / "model.safetensors"
    weights.parent.mkdir()
    _write_lerobot_weights(weights)
    norm = tmp_path / "norm.safetensors"
    _write_lerobot_norm(norm)
    out = tmp_path / "out"

    conv.convert(
        input_model=weights.parent,  # a directory, not the file
        input_norm_stats=norm,
        output_model=out,
        asset_id="ds/subset",
    )
    assert (out / "model.safetensors").exists()
    assert (out / "ds" / "subset" / "norm_stats.json").exists()
