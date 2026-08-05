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

"""Control-plane dashboard for RLinf training runs.

This package must never import ``rlinf``. RLinf training environments are many
and heavy (isaac-sim / omnigibson / sglang, each in its own venv), so the only
cross-process contract is the filesystem layout under
``<log_path>/_rlinf/runs/<run_id>/`` -- frozen in
``docs/schemas/run.v2.schema.json`` -- plus HTTP for time series.
"""

__version__ = "0.1.0"
