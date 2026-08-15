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

"""The command line, for the ways it can be invoked wrongly.

The server takes one scan root. It used to take several, so the invocations that
have to fail are the ones that used to work -- and they have to fail *loudly*.
An operator whose second directory is silently dropped sees a dashboard missing
half its runs and goes looking for a discovery bug.
"""

from __future__ import annotations

import pytest

from rlinf_dashboard.__main__ import main


@pytest.fixture(autouse=True)
def _no_inherited_config(monkeypatch, tmp_path):
    monkeypatch.delenv("RLINF_DASHBOARD_SCAN_ROOT", raising=False)
    monkeypatch.delenv("RLINF_DASHBOARD_SCAN_ROOTS", raising=False)
    monkeypatch.delenv("RLINF_DASHBOARD_AUTH_MODE", raising=False)
    monkeypatch.delenv("RLINF_DASHBOARD_AUTH_USERNAME", raising=False)
    monkeypatch.delenv("RLINF_DASHBOARD_AUTH_PASSWORD", raising=False)
    monkeypatch.chdir(tmp_path)


def test_two_paths_are_refused_with_an_instruction(capsys):
    code = main(["/mnt/team-a/runs", "/mnt/team-b/runs"])

    assert code == 2
    message = capsys.readouterr().err
    assert "single directory" in message
    # Naming both paths matters: the fix is to find their ancestor, and the
    # reader should not have to scroll back to their own command to see them.
    assert "/mnt/team-a/runs" in message
    assert "/mnt/team-b/runs" in message


def test_the_old_plural_environment_variable_is_refused(monkeypatch, capsys):
    """Being ignored is the dangerous outcome, so it is an error instead.

    ``extra="ignore"`` means the old name would otherwise be dropped without a
    word, leaving a server on ./logs and an operator reading their own path in a
    shell profile.
    """
    monkeypatch.setenv("RLINF_DASHBOARD_SCAN_ROOTS", "/mnt/runs")

    code = main([])

    assert code == 2
    assert "RLINF_DASHBOARD_SCAN_ROOT" in capsys.readouterr().err


def test_a_comma_separated_root_is_refused(monkeypatch, capsys):
    monkeypatch.setenv("RLINF_DASHBOARD_SCAN_ROOT", "/mnt/a,/mnt/b")

    code = main([])

    assert code == 2
    message = capsys.readouterr().err
    # The validator's own sentence, not a pydantic traceback.
    assert "single directory" in message
    assert "ValidationError" not in message


def test_incomplete_auth_is_refused_without_printing_the_secret(monkeypatch, capsys):
    monkeypatch.setenv("RLINF_DASHBOARD_AUTH_MODE", "basic")
    monkeypatch.setenv("RLINF_DASHBOARD_AUTH_PASSWORD", "do-not-print-this")

    code = main([])

    assert code == 2
    message = capsys.readouterr().err
    assert "requires both" in message
    assert "do-not-print-this" not in message
    assert "ValidationError" not in message
