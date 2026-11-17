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

"""Run a simulator on another machine, through HeteroPulse.

``env_client`` and ``env_server`` serve a whole simulator from a GPU host as a
HeteroPulse session. Remote rendering, which keeps physics next to the policy
and draws on a GPU host, needs no RLinf code on that host: LIBERO installs
:mod:`heteropulse.render.robosuite` in its simulator processes, and the host
runs ``heteropulse-render``.
"""
