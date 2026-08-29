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
"""Unit tests for the RLinf SFT -> LeRobot export mode.

Synthetic checkpoints throughout: no real pi0.5 file and no GPU needed.
"""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from rlinf.utils.ckpt_convertor.openpi import sft_to_lerobot as conv

POLICY_KEYS = ("action_in_proj.weight", "paligemma_with_expert.layer.weight")


def _make_sft_ckpt(root, extra=None, wrappers=""):
    """An RLinf checkpoint: actor/model_state_dict/full_weights.pt."""
    d = root / "actor" / "model_state_dict"
    d.mkdir(parents=True)
    sd = {f"{wrappers}{k}": torch.randn(2, 2) for k in POLICY_KEYS}
    sd.update(extra or {})
    torch.save(sd, d / "full_weights.pt")
    return root


def _make_template(root, prefix="", tied=False):
    """A LeRobot checkpoint supplying the target key set, config and processors."""
    root.mkdir(parents=True, exist_ok=True)
    keys = {f"{prefix}{k}": torch.zeros(2, 2) for k in POLICY_KEYS}
    if tied:
        keys[f"{prefix}{conv.TIED_KEY}"] = torch.zeros(2, 2)
    save_file(keys, str(root / "model.safetensors"), metadata={"format": "pt"})
    (root / "config.json").write_text(
        json.dumps(
            {
                "chunk_size": 50,
                "n_action_steps": 50,
                "num_inference_steps": 10,
                "normalization_mapping": {"STATE": "QUANTILES", "ACTION": "QUANTILES"},
            }
        )
    )
    save_file(
        {
            f"{feat}.{stat}": torch.zeros(6)
            for feat in ("observation.state", "action")
            for stat in ("q01", "q99", "mean", "std")
        },
        str(root / "policy_preprocessor_step_3_normalizer_processor.safetensors"),
        metadata={"format": "pt"},
    )
    return root


def _stat_values(stat, dim):
    """Distinct per-stat values, with q01 < q99 so the span is not degenerate."""
    base = [float(i) for i in range(dim)]
    return {
        "mean": base,
        "std": [v + 1.0 for v in base],
        "q01": [v - 50.0 for v in base],
        "q99": [v + 50.0 for v in base],
    }[stat]


def _norm_stats(path, dim=6):
    payload = {
        "norm_stats": {
            sec: {s: _stat_values(s, dim) for s in ("mean", "std", "q01", "q99")}
            for sec in ("state", "actions")
        }
    }
    path.write_text(json.dumps(payload))
    return path


# --- weights ---------------------------------------------------------------


def test_drops_rl_only_heads(tmp_path):
    """value_head / noise_head are RL machinery with no LeRobot slot."""
    ckpt = _make_sft_ckpt(
        tmp_path / "ckpt",
        extra={
            "value_head.weight": torch.randn(2, 2),
            "noise_head.mlp.weight": torch.randn(2, 2),
        },
    )
    tpl = _make_template(tmp_path / "tpl")
    report = conv.convert_weights(ckpt, tpl, tmp_path / "out")

    assert report["dropped_rl_heads"] == 2
    got = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert not any(k.startswith(("value_head.", "noise_head.")) for k in got)


def test_applies_the_template_prefix(tmp_path):
    """LeRobot wraps the flow model; the prefix is read off the template."""
    ckpt = _make_sft_ckpt(tmp_path / "ckpt")
    tpl = _make_template(tmp_path / "tpl", prefix="model.")
    conv.convert_weights(ckpt, tpl, tmp_path / "out")

    got = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert all(k.startswith("model.") for k in got)


def test_follows_the_template_on_the_tied_embedding(tmp_path):
    """Some LeRobot checkpoints store embed_tokens, some dedup it away."""
    ckpt = _make_sft_ckpt(tmp_path / "ckpt", extra={conv.TIED_KEY: torch.randn(2, 2)})
    # template without it -> must be dropped
    tpl = _make_template(tmp_path / "tpl", tied=False)
    report = conv.convert_weights(ckpt, tpl, tmp_path / "out")
    assert report["dropped_tied"] == 1

    # template with it -> must be kept
    tpl2 = _make_template(tmp_path / "tpl2", tied=True)
    report2 = conv.convert_weights(ckpt, tpl2, tmp_path / "out2")
    assert report2["dropped_tied"] == 0
    assert conv.TIED_KEY in load_file(str(tmp_path / "out2" / "model.safetensors"))


def test_strips_training_wrappers_but_not_lerobots_own(tmp_path):
    ckpt = _make_sft_ckpt(tmp_path / "ckpt", wrappers="_fsdp_wrapped_module.")
    tpl = _make_template(tmp_path / "tpl")
    conv.convert_weights(ckpt, tpl, tmp_path / "out")
    got = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert set(got) == set(POLICY_KEYS)


def test_key_mismatch_refuses_to_write(tmp_path):
    """A partial export must never reach a robot."""
    ckpt = _make_sft_ckpt(tmp_path / "ckpt")
    tpl = _make_template(tmp_path / "tpl")
    # template expects a key the checkpoint does not have
    extra = load_file(str(tpl / "model.safetensors"))
    extra["an_extra_key.weight"] = torch.zeros(2, 2)
    save_file(extra, str(tpl / "model.safetensors"), metadata={"format": "pt"})

    with pytest.raises(ValueError, match="key sets do not match"):
        conv.convert_weights(ckpt, tpl, tmp_path / "out")


def test_shape_mismatch_refuses_to_write(tmp_path):
    ckpt = _make_sft_ckpt(tmp_path / "ckpt")
    tpl = _make_template(tmp_path / "tpl")
    wrong = {k: torch.zeros(3, 3) for k in load_file(str(tpl / "model.safetensors"))}
    save_file(wrong, str(tpl / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="key sets do not match"):
        conv.convert_weights(ckpt, tpl, tmp_path / "out")


# --- config ----------------------------------------------------------------


def test_overrides_the_two_behavior_changing_fields(tmp_path):
    """num_inference_steps is the subtle one: 10 vs 4 is a different policy."""
    tpl = _make_template(tmp_path / "tpl")
    cfg = conv.convert_config(tpl, tmp_path / "out", chunk_size=10, num_steps=4)

    assert cfg["chunk_size"] == 10 and cfg["n_action_steps"] == 10
    assert cfg["num_inference_steps"] == 4
    written = json.loads((tmp_path / "out" / "config.json").read_text())
    assert written["num_inference_steps"] == 4, "template default 10 must not survive"


# --- norm stats ------------------------------------------------------------


def test_rewrites_processor_stats_from_our_lineage(tmp_path):
    """The template's stats come from ITS dataset and must not be kept."""
    tpl = _make_template(tmp_path / "tpl")
    stats = _norm_stats(tmp_path / "norm_stats.json")
    written = conv.convert_norm_stats(tpl, tmp_path / "out", stats)

    assert written == ["policy_preprocessor_step_3_normalizer_processor.safetensors"]
    got = load_file(str(tmp_path / "out" / written[0]))
    for feat in ("observation.state", "action"):
        for stat in ("q01", "q99", "mean", "std"):
            expected = torch.tensor(_stat_values(stat, 6))
            assert torch.equal(got[f"{feat}.{stat}"], expected), f"{feat}.{stat} kept"


def test_norm_stat_width_mismatch_raises(tmp_path):
    tpl = _make_template(tmp_path / "tpl")
    stats = _norm_stats(tmp_path / "norm_stats.json", dim=7)  # template holds 6
    with pytest.raises(ValueError, match="shape"):
        conv.convert_norm_stats(tpl, tmp_path / "out", stats)


# --- wiring ----------------------------------------------------------------


def test_mode_is_registered():
    from rlinf.utils.ckpt_convertor.openpi import convert as convert_cli

    parser = convert_cli.build_parser()
    action = next(
        a
        for a in parser._actions
        if getattr(a, "choices", None) and "sft2deploy" in a.choices
    )
    assert "sft_to_lerobot" in action.choices


def test_round_trip_direction_pair_is_present():
    """Both directions exist: import for training, export for deployment."""
    from rlinf.utils.ckpt_convertor.openpi import convert as convert_cli

    assert {"lerobot_to_openpi_pytorch", "sft_to_lerobot"} <= set(convert_cli._MODES)


# --- round trip ------------------------------------------------------------


def test_weights_round_trip_losslessly(tmp_path):
    """RLinf -> LeRobot -> RLinf must return the exact tensors.

    The weight half of the conversion is pure key renaming, so anything other
    than a bit-identical return means a tensor was dropped, reshaped or recast.
    Verified on real 9.3 GB checkpoints in both directions; this keeps it honest
    on every commit.
    """
    from rlinf.utils.ckpt_convertor.openpi import lerobot_to_openpi_pytorch as fwd

    ckpt = _make_sft_ckpt(tmp_path / "ckpt")
    tpl = _make_template(tmp_path / "tpl", prefix="model.")
    start = torch.load(
        ckpt / "actor" / "model_state_dict" / "full_weights.pt", weights_only=True
    )

    conv.convert_weights(ckpt, tpl, tmp_path / "mid")
    fwd.convert_weights(tmp_path / "mid", tmp_path / "end")

    end = load_file(str(tmp_path / "end" / "model.safetensors"))
    assert set(end) == set(start)
    for k in start:
        assert torch.equal(end[k], start[k]), f"{k} changed across the round trip"


def test_norm_stats_round_trip_within_fp32(tmp_path):
    """Stats survive a round trip to within fp32, which is how LeRobot stores them.

    Not bit-identical on purpose: the processor safetensors are fp32, so a
    float64 JSON value cannot come back unchanged. Measured on real stats the
    relative error is ~5e-8, far below anything that moves a policy.
    """
    from rlinf.utils.ckpt_convertor.openpi import lerobot_to_openpi_pytorch as fwd

    tpl = _make_template(tmp_path / "tpl")
    stats_path = _norm_stats(tmp_path / "norm_stats.json")
    start = json.loads(stats_path.read_text())["norm_stats"]

    conv.convert_norm_stats(tpl, tmp_path / "mid", stats_path)
    fwd.convert_norm_stats(
        tmp_path
        / "mid"
        / "policy_preprocessor_step_3_normalizer_processor.safetensors",
        tmp_path / "end" / "norm_stats.json",
    )

    end = json.loads((tmp_path / "end" / "norm_stats.json").read_text())["norm_stats"]
    assert set(end) == set(start)
    for section in start:
        for stat in ("mean", "std", "q01", "q99"):
            for a, b in zip(start[section][stat], end[section][stat]):
                assert abs(a - b) <= 1e-6 + 1e-6 * abs(a), f"{section}.{stat} drifted"


def test_config_deliberately_does_not_round_trip(tmp_path):
    """The export overrides chunk_size and num_inference_steps by design.

    Carrying the template's values through would be the bug this guards against:
    a 10-step-horizon policy exported with the template's 50, or run at LeRobot's
    default 10 denoising steps instead of RLinf's 4, is a different policy.
    """
    tpl = _make_template(tmp_path / "tpl")
    before = json.loads((tpl / "config.json").read_text())
    assert (before["chunk_size"], before["num_inference_steps"]) == (50, 10)

    after = conv.convert_config(tpl, tmp_path / "out", chunk_size=10, num_steps=4)
    assert (after["chunk_size"], after["num_inference_steps"]) == (10, 4)
