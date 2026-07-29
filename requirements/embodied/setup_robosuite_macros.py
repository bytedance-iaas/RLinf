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

"""Create robosuite's optional private macro file without importing robosuite."""

from __future__ import annotations

from importlib.metadata import distribution
from pathlib import Path


def setup_robosuite_macros(package_dir: Path | None = None) -> Path:
    """Create an empty ``macros_private.py`` when it is absent.

    An empty override file keeps the installed robosuite defaults active while
    preventing its repeated missing-file warnings. Existing user overrides are
    never changed.

    Args:
        package_dir: Robosuite package directory. The installed package is
            located automatically when omitted.

    Returns:
        Path to the existing or newly created private macro file.

    Raises:
        FileNotFoundError: If robosuite's default macro file cannot be found.
    """
    if package_dir is None:
        package_dir = Path(distribution("robosuite").locate_file("robosuite"))

    macros_path = package_dir / "macros.py"
    macros_private_path = package_dir / "macros_private.py"
    if not macros_path.is_file():
        raise FileNotFoundError(f"Robosuite macro defaults not found: {macros_path}")

    macros_private_path.touch(exist_ok=True)
    return macros_private_path


if __name__ == "__main__":
    print(setup_robosuite_macros())
