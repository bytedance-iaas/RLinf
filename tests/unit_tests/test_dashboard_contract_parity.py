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

"""Keep the standalone dashboard's frozen contract synchronized with RLinf."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER_SCHEMA = REPO_ROOT / "docs" / "schemas" / "run.v2.schema.json"
DASHBOARD_SCHEMA = (
    REPO_ROOT
    / "dashboard"
    / "rlinf_dashboard"
    / "schemas"
    / "run.v2.schema.json"
)
PRODUCER_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "run_state"
DASHBOARD_FIXTURES = REPO_ROOT / "dashboard" / "tests" / "fixtures" / "run_state"


def test_vendored_schema_matches_the_producer_contract():
    assert DASHBOARD_SCHEMA.read_bytes() == PRODUCER_SCHEMA.read_bytes()


def test_vendored_fixtures_match_the_producer_contract():
    producer_names = {path.name for path in PRODUCER_FIXTURES.glob("*.json")}
    dashboard_names = {path.name for path in DASHBOARD_FIXTURES.glob("*.json")}
    assert dashboard_names == producer_names

    for name in sorted(producer_names):
        assert (DASHBOARD_FIXTURES / name).read_bytes() == (
            PRODUCER_FIXTURES / name
        ).read_bytes(), name


def test_dashboard_carries_the_repository_license():
    assert (REPO_ROOT / "dashboard" / "LICENSE").read_bytes() == (
        REPO_ROOT / "LICENSE"
    ).read_bytes()
