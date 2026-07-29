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

import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "requirements"
    / "embodied"
    / "setup_robosuite_macros.py"
)
_SPEC = importlib.util.spec_from_file_location("setup_robosuite_macros", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
setup_robosuite_macros = _MODULE.setup_robosuite_macros


def test_setup_robosuite_macros_creates_empty_override(tmp_path):
    macros_path = tmp_path / "macros.py"
    macros_path.write_text("IMAGE_CONVENTION = 'opengl'\n")

    private_path = setup_robosuite_macros(tmp_path)

    assert private_path == tmp_path / "macros_private.py"
    assert private_path.read_text() == ""


def test_setup_robosuite_macros_preserves_existing_overrides(tmp_path):
    (tmp_path / "macros.py").write_text("IMAGE_CONVENTION = 'opengl'\n")
    private_path = tmp_path / "macros_private.py"
    private_path.write_text("IMAGE_CONVENTION = 'opencv'\n")

    assert setup_robosuite_macros(tmp_path) == private_path
    assert private_path.read_text() == "IMAGE_CONVENTION = 'opencv'\n"
