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
"""Unit tests for the SO101 ManiSkill adaptation.

Deliberately importable without a simulator: ``so101_calib`` depends only on
numpy (torch lazily), so the conversion contract -- the part that silently
destroys a policy when it is wrong -- is testable in plain CI.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _load_calib():
    """Import so101_calib directly, without executing ``rlinf.envs.__init__``."""
    path = _REPO / "rlinf" / "envs" / "maniskill" / "so101_calib.py"
    spec = importlib.util.spec_from_file_location("so101_calib_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


calib = _load_calib()


def _sample_norm(n=64, seed=0):
    """Random LeRobot-normalized poses: arm in [-80,80], gripper in [5,95]."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.uniform(-80, 80, (n, 5)), rng.uniform(5, 95, (n, 1))], axis=1
    )


# --- shape / layout contract ---------------------------------------------


def test_action_dim_is_six():
    assert calib.SO101_ACTION_DIM == 6
    assert len(calib.SO101_JOINT_NAMES) == 6
    assert len(calib.SO101_CALIB) == 6


def test_calib_table_order_matches_joint_names():
    assert tuple(j["name"] for j in calib.SO101_CALIB) == calib.SO101_JOINT_NAMES


def test_gripper_index_points_at_the_gripper():
    assert calib.SO101_JOINT_NAMES[calib.GRIPPER_IDX] == "gripper"
    assert calib.SO101_CALIB[calib.GRIPPER_IDX]["mode"] == "r0_100"


def test_action_dim_literal_matches_policy_module():
    """The policy module duplicates the literal to stay import-light."""
    path = _REPO / "rlinf/models/embodiment/openpi/policies/so101_policy.py"
    text = path.read_text()
    assert f"SO101_ACTION_DIM = {calib.SO101_ACTION_DIM}" in text


@pytest.mark.parametrize("bad_width", [5, 7, 32])
def test_conversion_rejects_wrong_width(bad_width):
    """A 7-wide action must fail loudly, not broadcast into garbage targets."""
    with pytest.raises(ValueError):
        calib.norm_to_rad(np.zeros((4, bad_width)))
    with pytest.raises(ValueError):
        calib.rad_to_norm(np.zeros((4, bad_width)))


@pytest.mark.parametrize("shape", [(6,), (4, 6), (2, 5, 6)])
def test_conversion_preserves_leading_shape(shape):
    assert calib.norm_to_rad(np.zeros(shape)).shape == shape
    assert calib.rad_to_norm(np.zeros(shape)).shape == shape


# --- numerical contract ---------------------------------------------------


def test_unit_round_trip():
    """norm -> rad -> norm must be identity inside the joint limits."""
    norm = _sample_norm()
    back = calib.rad_to_norm(calib.norm_to_rad(norm))
    np.testing.assert_allclose(back, norm, atol=1e-3)


def test_round_trip_survives_the_clip():
    """Values outside the URDF limits clip, and clipping must be monotone."""
    extreme = np.full((1, 6), 100.0)
    rad = calib.norm_to_rad(extreme)
    assert np.all(rad >= calib.JOINT_LIMITS_LOW - 1e-6)
    assert np.all(rad <= calib.JOINT_LIMITS_HIGH + 1e-6)


def test_gripper_uses_its_own_map_not_the_arm_conversion():
    """The regression that made grasping impossible.

    Applying the arm tick->rad conversion to the gripper leaves the jaws at an
    ~8 cm gap while the target cube is 2.9 cm, so the policy can never close on
    it. The gripper must instead land on its own measured closed/open angles.
    """
    closed = calib.norm_to_rad(np.zeros((1, 6)))[0, calib.GRIPPER_IDX]
    opened = np.zeros((1, 6))
    opened[0, calib.GRIPPER_IDX] = 100.0
    opened = calib.norm_to_rad(opened)[0, calib.GRIPPER_IDX]

    assert closed == pytest.approx(calib.GRIPPER_RAD_CLOSED, abs=1e-5)
    assert opened == pytest.approx(calib.GRIPPER_RAD_OPEN, abs=1e-5)
    assert closed < opened, "norm 0 must be the CLOSED end"

    # And it must NOT agree with what the arm-style conversion would produce.
    tick = 0.0 * calib._RAW_SCALE[calib.GRIPPER_IDX] + calib._RAW_OFFSET[
        calib.GRIPPER_IDX
    ]
    arm_style = (tick - calib.CENTER_TICK) * calib.RAD_PER_TICK
    assert abs(arm_style - closed) > 0.1, (
        "gripper conversion collapsed back onto the arm formula"
    )


def test_wrist_flex_offset_is_applied():
    """wrist_flex carries a +0.6 rad zero offset (gripper points down-forward)."""
    idx = calib.SO101_JOINT_NAMES.index("wrist_flex")
    assert calib.OFFSET[idx] == pytest.approx(0.6)
    centered = np.zeros((1, 6))
    centered[0, idx] = 0.0
    # A normalized 0 on wrist_flex maps to the calibration midpoint plus OFFSET.
    rad = calib.norm_to_rad(centered)[0, idx]
    no_offset = calib.norm_to_rad(centered)[0, idx] - calib.OFFSET[idx]
    assert rad - no_offset == pytest.approx(0.6)


# --- torch parity ---------------------------------------------------------


def test_torch_rad_to_norm_matches_numpy():
    """The observation path uses the torch version; it must not drift."""
    torch = pytest.importorskip("torch")
    rad = calib.norm_to_rad(_sample_norm())
    expected = calib.rad_to_norm(rad)
    got = calib.rad_to_norm_torch(torch.from_numpy(rad)).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-3)


def test_torch_rad_to_norm_does_not_mutate_input():
    """qpos belongs to ManiSkill; converting must not write through it."""
    torch = pytest.importorskip("torch")
    rad = torch.from_numpy(calib.norm_to_rad(_sample_norm()))
    before = rad.clone()
    calib.rad_to_norm_torch(rad)
    assert torch.equal(rad, before)


def test_torch_rad_to_norm_rejects_wrong_width():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        calib.rad_to_norm_torch(torch.zeros(4, 7))


# --- framework wiring -----------------------------------------------------


def test_control_mode_maps_so101_to_joint_position():
    """SO101 must select pd_joint_pos, not a Franka/WidowX EE controller."""
    text = (_REPO / "rlinf/config.py").read_text()
    assert '"so100" in robot or "so101" in robot' in text
    assert 'return "pd_joint_pos"' in text


def test_action_utils_converts_before_the_generic_seven_dof_path():
    """The so101 branch must precede the 7-DoF reshape, which would corrupt it."""
    text = (_REPO / "rlinf/envs/action_utils.py").read_text()
    so101_at = text.index('"so100" in policy or "so101" in policy')
    reshape_at = text.index("reshaped_actions = raw_chunk_actions.reshape")
    assert so101_at < reshape_at


def test_wrist_camera_is_optional_in_the_default_wrap():
    """Tasks without a wrist_camera must still get None, not a KeyError."""
    text = (_REPO / "rlinf/envs/maniskill/maniskill_env.py").read_text()
    assert '"wrist_camera" in sensor_data' in text
    assert '"wrist_images": wrist_image' in text


# --- openpi transforms (skipped where openpi is absent) -------------------


def _load_policy_module():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies import so101_policy

    return so101_policy


def test_outputs_slice_to_six_without_binarizing_the_gripper():
    so101_policy = _load_policy_module()
    actions = np.tile(np.arange(32, dtype=np.float32), (4, 1))
    actions[:, 5] = 0.37  # a graded aperture, not a flag
    out = so101_policy.SO101Outputs()({"actions": actions})["actions"]
    assert out.shape == (4, 6)
    assert out[:, 5] == pytest.approx(0.37), "gripper was binarized or dropped"


def test_outputs_reject_a_too_narrow_action_head():
    so101_policy = _load_policy_module()
    with pytest.raises(ValueError, match="action dims"):
        so101_policy.SO101Outputs()({"actions": np.zeros((4, 4))})


def test_outputs_apply_an_optional_unit_converter():
    """The hook the IsaacLab line needs; unused (None) on the ManiSkill path."""
    so101_policy = _load_policy_module()
    out = so101_policy.SO101Outputs(unit_converter=lambda a: a * 2.0)(
        {"actions": np.ones((2, 6))}
    )["actions"]
    assert out == pytest.approx(2.0)


def test_inputs_mask_out_an_absent_wrist_view():
    """A missing wrist view must be masked, not shown as a black frame."""
    so101_policy = _load_policy_module()
    from openpi.models import model as _model

    data = {
        "observation/state": np.zeros(6),
        "observation/image": np.zeros((480, 640, 3), dtype=np.uint8),
    }
    got = so101_policy.SO101Inputs(model_type=_model.ModelType.PI0, has_wrist_image=False)(data)
    assert bool(got["image_mask"]["left_wrist_0_rgb"]) is False
    assert bool(got["image_mask"]["base_0_rgb"]) is True
