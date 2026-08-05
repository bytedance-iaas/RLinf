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

"""Template selection and key binding.

The templates are the line between a product and a metrics dump, so these tests
cover both halves: that the shipped YAML files are actually well-formed and
selectable (a broken template silently falls back, which looks like a missing
feature), and that binding drops what has no data.

Binding runs server-side on purpose. The frontend renders what it is handed and
never learns which metrics a task type emits -- which is what makes a new
``task_type`` a YAML change rather than a frontend change.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from rlinf_dashboard.models import RunManifest, RunSnapshot
from rlinf_dashboard.registry import TemplateRegistry, bind_keys

SHIPPED = ["embodied", "reasoning", "sft", "fallback"]


@pytest.fixture(scope="module")
def registry() -> TemplateRegistry:
    return TemplateRegistry()


# ----------------------------------------------------------- the shipped templates


def test_every_shipped_template_loads(registry):
    """A YAML typo makes a template silently vanish into the fallback.

    That reads to a user as "the dashboard doesn't support embodied runs", with no
    error anywhere, so the presence of each shipped name is worth asserting
    directly.
    """
    assert set(SHIPPED) <= set(registry.names())


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_templates_are_well_formed(registry, name):
    template = registry.get(name)
    assert template is not None
    assert template["name"] == name
    assert template.get("step_axis_label"), "the x-axis needs a label"

    for group in template.get("groups") or []:
        assert group.get("title"), f"{name} has an untitled group"
        for chart in group.get("charts") or []:
            assert chart.get("keys"), f"{name} has a chart with no keys"


@pytest.mark.parametrize(
    ("task_type", "expected"),
    [
        ("embodied", "embodied"),
        ("reasoning", "reasoning"),
        ("coding_online_rl", "reasoning"),
        ("sft", "sft"),
        # No dedicated template; these must land on the generic one rather than
        # on an empty page.
        ("offline", "fallback"),
        ("embodied_eval", "fallback"),
        ("reasoning_eval", "fallback"),
    ],
)
def test_every_task_type_selects_a_template(registry, task_type, expected):
    """All seven ``cfg.runner.task_type`` values must resolve to something."""
    assert registry.select(task_type)["name"] == expected


def test_an_unknown_task_type_falls_back(registry):
    """A task type added to RLinf before a template is written for it.

    The fallback bins by prefix, so that run is usable on day one.
    """
    assert registry.select("some_future_task")["name"] == "fallback"
    assert registry.select(None)["name"] == "fallback"


def test_a_new_task_type_needs_only_a_yaml_file(tmp_path):
    """The requirement, stated as the thing that must not happen: a code change.

    The shipped templates are copied into a scratch directory and a file for a
    task type no template mentions is dropped alongside them. Nothing else --
    no enum entry, no registration call, no import. If selection ever grew a
    hardcoded list of known names, this is the test that fails.
    """
    import shutil

    from rlinf_dashboard.registry import TEMPLATE_DIR

    for name in os.listdir(TEMPLATE_DIR):
        if name.endswith(".yaml"):
            shutil.copy(os.path.join(TEMPLATE_DIR, name), tmp_path / name)

    # Before: no template claims it, so it lands on the generic page.
    assert TemplateRegistry(str(tmp_path)).select("world_model")["name"] == "fallback"

    _write(
        tmp_path,
        "world_model.yaml",
        """
        name: world_model
        task_types: [world_model]
        step_axis_label: Rollout step
        north_star: {key: wm/recon_loss, goal: minimize}
        groups:
          - title: Reconstruction
            charts:
              - {title: Loss, keys: [wm/recon_loss]}
    """,
    )

    template = TemplateRegistry(str(tmp_path)).select("world_model")
    assert template["name"] == "world_model"
    assert template["north_star"]["key"] == "wm/recon_loss"
    assert template["groups"][0]["charts"][0]["keys"] == ["wm/recon_loss"]


def test_embodied_north_star_is_success_rate(registry):
    """Loss says optimization works; success rate says the policy improved.

    The second is the question someone opens the page to answer, so it is the
    headline number.
    """
    north_star = registry.get("embodied")["north_star"]
    assert north_star["key"] == "env/success_once"
    assert north_star["goal"] == "maximize"


def test_reasoning_labels_its_x_axis_as_minibatch(registry):
    """A7 in the layout.

    ``reasoning_runner`` logs at a minibatch index, not an RL iteration. Labelling
    it honestly is what stops someone comparing a reasoning curve against an
    embodied one as though the steps meant the same thing.
    """
    assert registry.get("reasoning")["step_axis_label"] == "Minibatch"


def test_sft_records_its_rank_zero_caveat(registry):
    """N5: SFT eval metrics come from rank 0 only.

    A known measurement limitation of the training side, surfaced in the template
    so it reaches the person reading the number rather than living only in the
    ledger.
    """
    caveats = registry.get("sft").get("caveats") or []
    assert any("rank 0" in caveat for caveat in caveats)


def test_the_fallback_declares_prefix_groups(registry):
    fallback = registry.get("fallback")
    assert fallback["auto_group"] is True
    prefixes = {entry["prefix"] for entry in fallback["prefix_groups"]}
    assert {"env/", "train/actor/", "time/", "train/replay_buffer/"} <= prefixes
    # The bare spelling matches nothing a runner emits, and because the shorter
    # `train/` entry then claims those keys the error is invisible in the output.
    assert "replay_buffer/" not in prefixes


def test_selection_prefers_the_snapshot_but_accepts_a_manifest(registry):
    """A run that died before its first flush has only a manifest.

    It still needs a page to be shown on, so selection must work from either
    document.
    """
    manifest = RunManifest.model_validate(
        {"schema_version": 2, "run_id": "r", "task_type": "sft"}
    )
    snapshot = RunSnapshot.model_validate(
        {
            "schema_version": 2,
            "run_id": "r",
            "task_type": "embodied",
            "state": "running",
            "progress": {"step": 0, "max_steps": 1},
            "timing": {},
        }
    )
    assert registry.select_for(manifest, snapshot)["name"] == "embodied"
    assert registry.select_for(manifest, None)["name"] == "sft"
    assert registry.select_for(None, None)["name"] == "fallback"


def test_a_loss_type_refinement_wins_over_the_task_type(tmp_path):
    """How an algorithm-specific layout attaches without displacing the general one.

    The refined ``task_type:loss_type`` key is tried first, which is the extension
    point that made a three-dimensional selection key unnecessary.
    """
    _write(
        tmp_path,
        "base.yaml",
        """
        name: base
        task_types: [embodied]
        step_axis_label: Step
    """,
    )
    _write(
        tmp_path,
        "special.yaml",
        """
        name: special
        loss_types: ["embodied:actor_critic"]
        step_axis_label: Step
    """,
    )
    registry = TemplateRegistry(str(tmp_path))

    assert registry.select("embodied")["name"] == "base"
    assert registry.select("embodied", "actor_critic")["name"] == "special"
    # An unmatched loss_type falls back to the task_type template rather than to
    # the generic one.
    assert registry.select("embodied", "grpo")["name"] == "base"


def test_returned_templates_are_copies(registry):
    """A caller mutating a template must not corrupt the registry.

    ``bind_keys`` rewrites ``groups`` per request; sharing the dict would make the
    first run's key set stick for every subsequent run.
    """
    first = registry.get("embodied")
    first["groups"] = []
    assert registry.get("embodied")["groups"]


# --------------------------------------------------------------------- inheritance


def _write(directory, name: str, body: str) -> None:
    (directory / name).write_text(textwrap.dedent(body).strip() + "\n")


def test_extends_merges_groups_by_title(tmp_path):
    """The real reuse case: PPO and GRPO sharing a policy-health group.

    Merging by title lets a child add charts to an inherited group instead of
    restating it, which is what keeps the shared group from drifting between
    templates.
    """
    _write(
        tmp_path,
        "parent.yaml",
        """
        name: parent
        step_axis_label: Step
        groups:
          - title: Policy
            charts:
              - keys: [train/actor/loss]
                title: Loss
    """,
    )
    _write(
        tmp_path,
        "child.yaml",
        """
        name: child
        extends: parent
        task_types: [embodied]
        groups:
          - title: Policy
            charts:
              - keys: [train/actor/entropy]
                title: Entropy
          - title: Extra
            charts:
              - keys: [env/return]
                title: Return
    """,
    )
    template = TemplateRegistry(str(tmp_path)).select("embodied")

    assert template["step_axis_label"] == "Step", "inherited from the parent"
    policy = next(g for g in template["groups"] if g["title"] == "Policy")
    assert [chart["title"] for chart in policy["charts"]] == ["Loss", "Entropy"]
    assert any(group["title"] == "Extra" for group in template["groups"])


def test_a_child_overrides_a_scalar_field(tmp_path):
    _write(
        tmp_path,
        "parent.yaml",
        """
        name: parent
        step_axis_label: Step
    """,
    )
    _write(
        tmp_path,
        "child.yaml",
        """
        name: child
        extends: parent
        task_types: [reasoning]
        step_axis_label: Minibatch
    """,
    )
    assert (
        TemplateRegistry(str(tmp_path)).select("reasoning")["step_axis_label"]
        == "Minibatch"
    )


def test_a_cyclic_extends_does_not_hang(tmp_path):
    """A bad template must not take the server down with it.

    Which of the two wins the ``task_types`` claim is undefined -- with a cycle,
    each inherits the other's fields -- so only termination and usability are
    asserted. Loading must finish, both names must exist, and the run must still
    get a page.
    """
    _write(
        tmp_path,
        "a.yaml",
        """
        name: a
        extends: b
        task_types: [embodied]
        step_axis_label: Step
    """,
    )
    _write(
        tmp_path,
        "b.yaml",
        """
        name: b
        extends: a
    """,
    )
    registry = TemplateRegistry(str(tmp_path))

    assert registry.names() == ["a", "b"]
    selected = registry.select("embodied")
    assert selected["name"] in {"a", "b"}
    assert selected["step_axis_label"] == "Step"


def test_a_malformed_template_file_is_skipped(tmp_path):
    """One bad file must not take the rest of the directory with it."""
    _write(
        tmp_path,
        "good.yaml",
        """
        name: good
        task_types: [embodied]
        step_axis_label: Step
    """,
    )
    (tmp_path / "bad.yaml").write_text("name: bad\n  broken: [[[\n")

    registry = TemplateRegistry(str(tmp_path))
    assert registry.select("embodied")["name"] == "good"


def test_a_missing_template_dir_does_not_crash(tmp_path):
    registry = TemplateRegistry(str(tmp_path / "nope"))
    assert registry.names() == []
    assert registry.select("embodied")["name"] == "empty"


# ---------------------------------------------------------------------- bind_keys


def test_binding_drops_charts_with_no_data():
    """A template lists candidate spellings; a run has one of them.

    Without dropping, an embodied page shows a dozen empty axes and the four that
    matter get lost among them.
    """
    template = {
        "name": "t",
        "groups": [
            {
                "title": "Task",
                "charts": [
                    {"keys": ["env/success_once"], "title": "Success"},
                    {"keys": ["env/nonexistent"], "title": "Ghost"},
                ],
            }
        ],
    }
    bound = bind_keys(template, ["env/success_once"])
    assert [chart["title"] for chart in bound["groups"][0]["charts"]] == ["Success"]


def test_binding_keeps_only_the_present_spellings_within_a_chart():
    """Alternate key names in one chart: keep the ones this run wrote.

    ``[train/actor/loss, train/actor/policy_loss]`` covers two runner
    conventions. Charting the absent one would draw an empty line in a legend.
    """
    template = {
        "name": "t",
        "groups": [
            {
                "title": "Policy",
                "charts": [{"keys": ["train/actor/loss", "train/actor/pg_loss"]}],
            }
        ],
    }
    bound = bind_keys(template, ["train/actor/pg_loss"])
    assert bound["groups"][0]["charts"][0]["keys"] == ["train/actor/pg_loss"]


def test_an_empty_group_is_dropped_entirely():
    template = {
        "name": "t",
        "groups": [
            {"title": "Gone", "charts": [{"keys": ["a/b"]}]},
            {"title": "Kept", "charts": [{"keys": ["c/d"]}]},
        ],
    }
    bound = bind_keys(template, ["c/d"])
    assert [group["title"] for group in bound["groups"]] == ["Kept"]


def test_unclaimed_keys_are_reported_not_hidden():
    """A logged metric no group claimed is still a metric someone wanted.

    Silently dropping it looks like the dashboard cannot show custom metrics.
    """
    template = {"name": "t", "groups": [{"title": "G", "charts": [{"keys": ["a/b"]}]}]}
    bound = bind_keys(template, ["a/b", "my/custom/metric"])
    assert bound["unmatched"] == ["my/custom/metric"]


def test_auto_grouping_bins_real_keys_by_prefix(registry):
    """How an unknown task type gets a usable page with no template written.

    The bins mirror the grouping ``print_metrics_table`` already applies to the
    stdout table, so the web view reads the same way the console does.
    """
    available = [
        "env/return",
        "train/actor/loss",
        "train/critic/loss",
        # `train/replay_buffer/`, not `replay_buffer/`: the worker emits
        # `replay_buffer/{key}` (fsdp_sac_policy_worker.py:668) and offline_runner
        # prefixes the whole dict with `train/` (:352). The earlier bare spelling
        # here made this assertion vacuous -- the "Replay buffer" title it checked
        # for could never appear for a real run.
        "train/replay_buffer/size",
        "time/step",
        "something/odd",
    ]
    bound = bind_keys(registry.get("fallback"), available)
    titles = {group["title"] for group in bound["groups"]}

    assert {
        "Environment",
        "Actor training",
        "Critic training",
        "Replay buffer",
        "Timing",
    } <= titles
    assert "Other" in titles, "an unbinned key must still get a home"
    other = next(group for group in bound["groups"] if group["title"] == "Other")
    assert other["charts"][0]["keys"] == ["something/odd"]
    # Everything found a group, so nothing is left over.
    assert bound["unmatched"] == []


def test_the_longest_matching_prefix_wins(registry):
    """``train/actor/loss`` belongs under Actor training, not plain Training.

    Both prefixes match; the specific one is the useful one.
    """
    bound = bind_keys(registry.get("fallback"), ["train/actor/loss", "train/loss"])
    grouped = {
        group["title"]: [key for chart in group["charts"] for key in chart["keys"]]
        for group in bound["groups"]
    }
    assert grouped["Actor training"] == ["train/actor/loss"]
    assert grouped["Training"] == ["train/loss"]


def test_a_north_star_with_no_data_is_marked_unresolved():
    """An empty hero number reads as a broken run, not as an unlogged metric.

    So the UI needs to know the difference, and hide the panel rather than show a
    blank.
    """
    template = {"name": "t", "north_star": {"key": "env/success_once"}, "groups": []}
    assert bind_keys(template, ["env/return"])["north_star"]["resolved"] is False
    assert bind_keys(template, ["env/success_once"])["north_star"]["resolved"] is True


def test_the_north_star_falls_back_to_the_first_key_with_data():
    """Reward workers name their metric differently across runs.

    The candidate list makes the headline number work across those conventions
    instead of being blank for most of them.
    """
    template = {
        "name": "t",
        "north_star": {
            "key": "rollout/reward",
            "fallback_keys": ["reward/mean", "train/reward"],
        },
        "groups": [],
    }
    bound = bind_keys(template, ["train/reward"])
    assert bound["north_star"]["key"] == "train/reward"
    assert bound["north_star"]["resolved"] is True


def test_binding_a_run_with_no_metrics_yet_yields_an_empty_but_valid_page(registry):
    """Before the first log call there are no keys at all.

    The page still has to render -- that is the state a just-launched run is in,
    and it is exactly when someone is watching.
    """
    bound = bind_keys(registry.get("embodied"), [])
    assert bound["groups"] == []
    assert bound["unmatched"] == []
    assert bound["north_star"]["resolved"] is False
    assert bound["step_axis_label"] == "RL iteration"


# ------------------------------------------- binding against real emitted keys
#
# The key sets below were read off the code that builds the metric dicts, not
# invented for the test. They exist because the first drafts of embodied.yaml,
# reasoning.yaml and sft.yaml all guessed key names and all guessed wrong --
# embodied dropped 46 of 56 real keys, reasoning got 25 of 28 wrong -- and a
# wrong key is not an error: bind_keys drops it and the chart silently is not
# there. A test that feeds a template its own spellings would pass in exactly
# that case, so these lists must stay traceable to the producing code.


#: reasoning_runner.py: `actor/training/{k}` where k is already `actor/...`
#: (:690), so dual_write yields the doubled `train/actor/actor/*`. Rollout keys
#: from compute_rollout_metrics (distributed.py:288-302), time keys from the
#: timer scopes plus the handles at :654-672.
REASONING_RUNNER_KEYS = [
    "rollout/total_num_sequence",
    "rollout/prompt_length",
    "rollout/response_length",
    "rollout/average_response_length",
    "rollout/total_length",
    "rollout/reward_scores",
    "rollout/fraction_of_samples_properly_ended",
    "rollout/advantages_mean",
    "rollout/advantages_max",
    "rollout/advantages_min",
    "actor/training/actor/policy_loss",
    "train/actor/actor/policy_loss",
    "actor/training/actor/approx_kl",
    "train/actor/actor/approx_kl",
    "actor/training/actor/clip_fraction",
    "train/actor/actor/clip_fraction",
    "actor/training/actor/entropy_loss",
    "train/actor/actor/entropy_loss",
    "actor/training/actor/final_loss",
    "train/actor/actor/final_loss",
    "actor/training/actor/grad_norm",
    "train/actor/actor/grad_norm",
    "actor/training/actor/lr",
    "train/actor/actor/lr",
    "critic/training/critic/value_loss",
    "train/critic/critic/value_loss",
    "critic/training/critic/explained_variance",
    "train/critic/critic/explained_variance",
    "time/step",
    "time/prepare_data",
    "time/sync_weights",
    "time/rollout",
    "time/actor/training",
    "time/reward",
    "flops/rollout_tflops_per_gpu",
    "flops/training_tflops_per_gpu",
]

#: agent_runner.py composes `train/{k}` directly (:321), so the same worker
#: metrics arrive under the clean spelling. Rollout keys come from
#: compute_rollout_metrics_dynamic (distributed.py:167-182), a different set.
AGENT_RUNNER_KEYS = [
    "rollout/total_num_sequence",
    "rollout/response_length",
    "rollout/reward_scores_traj",
    "rollout/reward_scores_turn",
    "rollout/avg_turns_per_traj",
    "rollout/fraction_of_samples_properly_ended",
    "train/actor/policy_loss",
    "train/actor/approx_kl",
    "train/actor/clip_fraction",
    "train/actor/entropy_loss",
    "train/actor/final_loss",
    "train/actor/grad_norm",
    "train/actor/lr",
    "time/step",
    "time/prepare_data",
    "time/sync_weights",
    "time/rollout",
    "time/training",
    "agent/mean/reward",
    "agent/count/n_rollouts",
    "agent/mean/turn_count_per_rollout",
]

#: sft_runner.py:157/166 prefixing FSDPSftWorker's dict (fsdp_sft_worker.py:131,
#: :181-182) plus a VLM subclass loss (fsdp_vlm_sft_worker.py:239).
SFT_FSDP_KEYS = [
    "train/loss",
    "train/learning_rate",
    "train/grad_norm",
    "eval/eval_accuracy",
    "time/step",
    "time/training",
    "time/evaluate",
]

#: FSDPSteamSftWorker, which spells lr `lr` and accuracy `accuracy`
#: (fsdp_steam_sft_worker.py:642-648, :913) -- the disagreement the alias lists
#: in sft.yaml exist for.
SFT_STEAM_KEYS = [
    "train/loss",
    "train/accuracy",
    "train/accuracy_neighbor",
    "train/lr",
    "train/grad_norm",
    "eval/loss",
    "eval/accuracy",
    "eval/accuracy_neighbor",
    "eval/gsm8k/accuracy",
    "time/step",
    "time/training",
]


#: `LEGACY_TO_CANONICAL` from rlinf/utils/metric_naming.py:34-37, which is what
#: the training side embeds in `manifest.metric_aliases` -- copied rather than
#: imported, because the dashboard must not import rlinf.
ALIASES = {
    "actor/training/": "train/actor/",
    "critic/training/": "train/critic/",
}


def _charted(bound: dict) -> set[str]:
    return {
        key
        for group in bound["groups"]
        for chart in group["charts"]
        for key in chart["keys"]
    }


def _charted_list(bound: dict) -> list[str]:
    """Charted keys with duplicates kept, so a doubled series is visible."""
    return [
        key
        for group in bound["groups"]
        for chart in group["charts"]
        for key in chart["keys"]
    ]


@pytest.mark.parametrize(
    "keys,label",
    [(REASONING_RUNNER_KEYS, "reasoning_runner"), (AGENT_RUNNER_KEYS, "agent_runner")],
)
def test_reasoning_binds_both_runner_families(registry, keys, label):
    """`task_type: reasoning` covers two runners that spell training keys apart.

    reasoning_runner double-namespaces to `train/actor/actor/*`; the agent-style
    runners give the clean `train/actor/*`. One template serves both, so it must
    bind both -- and the failure mode is silence, not an error.
    """
    bound = bind_keys(registry.get("reasoning"), keys, ALIASES)
    charted = _charted(bound)
    missed = [key for key in keys if key not in charted]

    # Only the legacy pre-dual_write spellings may go unclaimed: they are
    # duplicates of a canonical key that is charted, so listing them would draw
    # the same line twice.
    unexpected = [
        key
        for key in missed
        if not key.startswith(("actor/training/", "critic/training/"))
    ]
    assert not unexpected, f"{label} keys no chart claims: {unexpected}"

    assert bound["north_star"]["resolved"] is True, (
        f"{label} logs a reward metric but the north star did not resolve"
    )

    # A legacy key is a duplicate, not a metric the layout forgot; reporting it
    # under `unmatched` would send the reader looking for a missing panel.
    assert bound["unmatched"] == [], f"{label} reported alias twins as unmatched"


def test_reasoning_charts_one_series_once_not_twice(registry):
    """dual_write logs both spellings, so a chart listing both would double it.

    The template has to list both -- it serves runs from either side of the
    rename -- and a real reasoning run writes both, so the naive `key in present`
    filter kept both and the chart drew one metric as two identical lines. This
    is not a template bug: dropping the legacy spelling from the YAML would
    instead lose the chart for a run written before the alias layer.
    """
    bound = bind_keys(registry.get("reasoning"), REASONING_RUNNER_KEYS, ALIASES)
    charted = _charted_list(bound)

    duplicated = sorted({key for key in charted if charted.count(key) > 1})
    assert not duplicated, f"charted more than once: {duplicated}"

    # The canonical spelling is the survivor, so the series a chart requests
    # keeps working after the deprecation window closes and the legacy write
    # goes away.
    assert "train/actor/actor/policy_loss" in charted
    assert not [key for key in charted if key.startswith("actor/training/")]


def test_a_legacy_only_run_still_gets_its_charts(registry):
    """Canonicalising must not rewrite a key to a spelling the run never wrote.

    A run captured before the dual write existed has only `actor/training/*`. If
    binding replaced that with the canonical spelling it would hand the frontend
    a key the gateway has no data for -- turning a working chart into an empty
    one, which reads as a broken run.
    """
    legacy_only = [key for key in REASONING_RUNNER_KEYS if not key.startswith("train/")]
    bound = bind_keys(registry.get("reasoning"), legacy_only, ALIASES)
    charted = _charted_list(bound)

    assert "actor/training/actor/policy_loss" in charted
    assert "critic/training/critic/value_loss" in charted
    assert not [key for key in charted if key not in legacy_only], (
        "bound a key this run never logged"
    )


def test_binding_without_aliases_is_unchanged(registry):
    """The alias argument is optional, and omitting it must not rewrite keys.

    Every caller other than the template endpoint -- and every test written
    before aliases existed -- passes two arguments, so the default has to be a
    plain presence filter.
    """
    keys = ["actor/training/actor/policy_loss", "train/actor/actor/policy_loss"]
    charted = _charted_list(bind_keys(registry.get("reasoning"), keys))
    assert sorted(charted) == sorted(keys)


def test_the_templates_keep_both_spellings_for_a_manifest_with_no_aliases(registry):
    """Why the YAML still lists legacy spellings even though binding dedupes.

    Verified against the running server: the gateway resolves a canonical request
    down to legacy-stored data, so a canonical-only template would *usually*
    work. But `metric_aliases` defaults to empty, and a manifest written before
    that field existed carries none -- with no alias map there is no resolution
    in either layer, and a canonical-only chart would simply be missing. The
    template's redundancy is what covers that run, so it is asserted here rather
    than left as a comment someone would later "simplify" away.
    """
    legacy_only = [key for key in REASONING_RUNNER_KEYS if not key.startswith("train/")]
    bound = bind_keys(registry.get("reasoning"), legacy_only, {})
    charted = _charted_list(bound)

    assert "actor/training/actor/policy_loss" in charted
    assert "critic/training/critic/value_loss" in charted


def test_reasoning_north_star_resolves_per_runner_family(registry):
    """The headline number must come from the right reward metric.

    The two families measure reward differently -- a single-turn score versus a
    per-trajectory one -- so the candidate order decides which is shown, and
    showing the per-turn number for a trajectory-level run would understate it.
    """
    assert (
        bind_keys(registry.get("reasoning"), REASONING_RUNNER_KEYS)["north_star"]["key"]
        == "rollout/reward_scores"
    )
    assert (
        bind_keys(registry.get("reasoning"), AGENT_RUNNER_KEYS)["north_star"]["key"]
        == "rollout/reward_scores_traj"
    )


def test_reasoning_does_not_promise_metrics_no_runner_logs(registry):
    """No reasoning-family runner logs eval metrics.

    reasoning_runner never calls run_eval, and reasoning_eval_runner builds a
    MetricLogger it never logs to. An `eval/pass@1` chart would therefore be
    empty on every run -- which reads as a broken run, not as a metric nobody
    writes. Guarding the template against re-acquiring it.
    """
    declared = {
        key
        for group in registry.get("reasoning").get("groups") or []
        for chart in group.get("charts") or []
        for key in chart.get("keys") or []
    }
    assert not [key for key in declared if key.startswith("eval/")]


@pytest.mark.parametrize(
    "keys,label", [(SFT_FSDP_KEYS, "FSDPSftWorker"), (SFT_STEAM_KEYS, "steam")]
)
def test_sft_binds_every_worker_spelling(registry, keys, label):
    """Six workers can sit behind `task_type: sft` and they disagree on names.

    lr is `learning_rate` or `lr`; accuracy is `eval_accuracy`, `accuracy`,
    `val_accuracy` or `cat_acc_best`. The template lists all of them, so any
    configured worker gets a populated page.
    """
    bound = bind_keys(registry.get("sft"), keys)
    charted = _charted(bound)
    missed = [key for key in keys if key not in charted]

    # Per-dataset keys are composed at runtime from config, so they cannot be
    # declared; they are expected in `unmatched` instead.
    assert missed == [key for key in missed if key.count("/") > 1], (
        f"{label} keys no chart claims: {missed}"
    )
    assert bound["north_star"]["resolved"] is True, (
        f"{label} logs an accuracy or loss metric but the north star did not resolve"
    )


def test_sft_runtime_named_keys_surface_as_unmatched(registry):
    """`eval/<dataset>/<metric>` is named from config, so it cannot be declared.

    It must still be visible: a metric someone configured a dataset to produce,
    silently dropped, reads as the dashboard not supporting per-dataset eval.
    """
    bound = bind_keys(registry.get("sft"), SFT_STEAM_KEYS)
    assert "eval/gsm8k/accuracy" in bound["unmatched"]


#: A synchronous embodied PPO + LIBERO run, which is what embodied.yaml was
#: fixed against (commit 9c767ef5, a real 4xH20 Pi0.5 run, 56 scalar tags).
#: That fix left no test, so this list is rebuilt from the emitting code rather
#: than from the template -- the template's own spellings would pass vacuously.
#:
#: `env/*`  -- libero_env.py:623-631 sets episode_info {success_once, return,
#:   episode_len, reward}; env_worker.py:527-534 copies infos["episode"] into
#:   env_info; compute_evaluate_metrics (metric_utils.py:346) appends
#:   num_trajectories; embodied_runner.py:392 prefixes `env/`.
#: `eval/*` -- the same dict through evaluate() (:219), prefixed `eval/` (:338).
#: `train/actor/*`, `train/critic/*` -- losses.py:299-311 (compute_ppo_actor_loss)
#:   and :374-375 (compute_ppo_critic_loss), both reached via loss_type
#:   `actor_critic` (:396); actor/entropy_loss and actor/total_loss from
#:   fsdp_actor_worker.py:1701/:1710; actor/grad_norm, actor/lr, critic/lr from
#:   :1577-1581; critic/explained_variance from :1596 (metric_utils.py:143).
#:   embodied_runner.py:407 prefixes `train/`, giving the clean spelling -- no
#:   alias twins here, unlike reasoning.
#: `rollout/*` -- compute_rollout_metrics (metric_utils.py:407-431), prefixed at
#:   embodied_runner.py:403. `rewards` only when the buffer has it; returns_* only
#:   for gae, which `loss_type: actor_critic` implies.
#: `time/*` -- driver scopes at embodied_runner.py:335/534-569, prefixed :363;
#:   worker sub-scopes from @Worker.timer tags, prefixed `time/env|rollout|actor/`
#:   at :373-379. The tags already carry their own `env/`/`actor/` prefix in some
#:   cases (env_worker.py:913, fsdp_actor_worker.py:1198), which is where the
#:   doubled `time/env/env/*` and `time/actor/actor/*` spellings come from.
EMBODIED_PPO_LIBERO_KEYS = [
    # env: per-iteration training episode stats
    "env/success_once",
    "env/return",
    "env/episode_len",
    "env/reward",
    "env/num_trajectories",
    # eval: the same env dict on the eval path
    "eval/success_once",
    "eval/return",
    "eval/episode_len",
    "eval/reward",
    "eval/num_trajectories",
    # actor loss metrics (compute_ppo_actor_loss)
    "train/actor/policy_loss",
    "train/actor/policy_loss_abs",
    "train/actor/ratio",
    "train/actor/ratio_abs",
    "train/actor/clipped_ratio",
    "train/actor/dual_cliped_ratio",
    "train/actor/approx_kl",
    "train/actor/clip_fraction",
    # added by the worker, not the loss
    "train/actor/entropy_loss",
    "train/actor/total_loss",
    "train/actor/grad_norm",
    "train/actor/lr",
    # critic
    "train/critic/value_loss",
    "train/critic/value_clip_ratio",
    "train/critic/explained_variance",
    "train/critic/lr",
    # rollout buffer stats
    "rollout/rewards",
    "rollout/advantages_mean",
    "rollout/advantages_max",
    "rollout/advantages_min",
    "rollout/returns_mean",
    "rollout/returns_max",
    "rollout/returns_min",
    # driver scopes
    "time/step",
    "time/sync_weights",
    "time/generate_rollouts",
    "time/cal_adv_and_returns",
    "time/actor_training",
    "time/eval",
    # env worker sub-scopes
    "time/env/interact",
    "time/env/evaluate",
    "time/env/run_interact_once",
    "time/env/env_interact_step",
    "time/env/compute_bootstrap_rewards",
    "time/env/env/bootstrap_step",
    "time/env/env/send_rollout_trajectories",
    # rollout worker sub-scopes
    "time/rollout/rollout/generate",
    "time/rollout/evaluate",
    "time/rollout/generate_one_epoch",
    "time/rollout/predict",
    "time/rollout/sync_model_from_actor",
    # actor worker sub-scopes
    "time/actor/run_training",
    "time/actor/actor/compute_adv",
    "time/actor/actor/recv_traj",
    "time/actor/actor/sync_model_to_rollout",
]


def test_embodied_binds_the_keys_a_real_ppo_libero_run_logs(registry):
    """The template must claim what an embodied PPO run actually writes.

    embodied.yaml was fixed against a real run (9c767ef5 took unmatched from 46
    of 56 keys down to 6) but that fix left no guard, and the tag list was never
    captured. This rebuilds it from the producing code so the template cannot
    silently drift back: a wrong key is not an error, the chart is just absent.
    """
    bound = bind_keys(registry.get("embodied"), EMBODIED_PPO_LIBERO_KEYS)
    charted = _charted(bound)
    missed = [key for key in EMBODIED_PPO_LIBERO_KEYS if key not in charted]

    # Every key a real run logs is charted. Writing this list first found six
    # that were not -- `train/actor/policy_loss_abs`, the two eval-path worker
    # scopes, and the three doubled `time/env/env/*` and `time/rollout/rollout/*`
    # spellings that @Worker.timer's own prefix produces -- which is the same
    # count 9c767ef5 left behind without recording which keys they were.
    assert missed == [], f"embodied keys no chart claims: {missed}"

    # Nothing may be doubled: unlike reasoning this run logs one spelling per
    # series, so any repeat would be the template listing a key twice.
    charted_list = _charted_list(bound)
    duplicated = sorted({key for key in charted_list if charted_list.count(key) > 1})
    assert not duplicated, f"charted more than once: {duplicated}"

    assert bound["north_star"]["key"] == "env/success_once"
    assert bound["north_star"]["resolved"] is True


def test_embodied_does_not_declare_keys_no_embodied_run_logs(registry):
    """The other half of the guard: charts promising data that never arrives.

    A key present in the YAML but absent from every run costs nothing at bind
    time -- it is dropped -- but it hides a typo. If a spelling here matches no
    emitter and no alias, the chart it was meant to draw is permanently empty and
    nothing says so. Only alias spellings kept for other model families and other
    RLinf versions are allowed to go unmatched.
    """
    declared = {
        key
        for group in registry.get("embodied").get("groups") or []
        for chart in group.get("charts") or []
        for key in chart.get("keys") or []
    }
    unknown = declared - set(EMBODIED_PPO_LIBERO_KEYS)

    # Each of these is a deliberate alias for a spelling this particular run does
    # not produce: `*/loss` and `*/learning_rate` for workers that name them so
    # (fsdp_sac_policy_worker, fsdp_sft_worker), `train/actor/entropy` and
    # `train/actor/clip_frac` for the SAC/DAgger families, `rollout/reward` and
    # `rollout/mean_reward` for pre-rename runs, `env/fail_once`-adjacent
    # `eval/success_at_end` for behavior_env (behavior_env.py:776), and
    # `time/actor/training` for the async runner's differently-named scope.
    assert unknown == {
        "eval/success_at_end",
        "rollout/mean_reward",
        "rollout/reward",
        "train/actor/clip_frac",
        "train/actor/entropy",
        "train/actor/learning_rate",
        "train/actor/loss",
        "train/critic/loss",
    }, f"embodied.yaml declares keys traced to no emitter: {sorted(unknown)}"


#: An offline SAC run, which has no dedicated template and so lands on the
#: fallback. Keys from fsdp_sac_policy_worker.py (:583-651 for actor/critic,
#: :668/:675 for the buffers, whose stats keys are replay_buffer.py:917-923),
#: all prefixed with `train/` by offline_runner.py:352, plus its `eval/` (:338)
#: and `time/` (:345, :349) dicts.
OFFLINE_SAC_KEYS = [
    "train/actor/loss",
    "train/actor/lr",
    "train/actor/grad_norm",
    "train/actor/entropy",
    "train/critic/loss",
    "train/critic/lr",
    "train/critic/grad_norm",
    "train/replay_buffer/num_trajectories",
    "train/replay_buffer/total_samples",
    "train/replay_buffer/cache_size",
    "train/demo_buffer/num_trajectories",
    "eval/success_once",
    "eval/return",
    "time/step",
    "time/eval",
    "time/actor/recv_traj",
]


def test_the_fallback_gives_an_offline_run_a_usable_page(registry):
    """`offline` has no template of its own, so the fallback has to carry it.

    Checked against a real offline SAC run's key spellings rather than invented
    ones, because the prefixes are what the whole template is: a prefix that
    matches nothing produces no error, and the keys quietly fall into a
    shorter-prefix group or into Other.
    """
    bound = bind_keys(registry.get("fallback"), OFFLINE_SAC_KEYS)
    grouped = {
        group["title"]: [key for chart in group["charts"] for key in chart["keys"]]
        for group in bound["groups"]
    }

    assert grouped["Replay buffer"] == [
        "train/replay_buffer/cache_size",
        "train/replay_buffer/num_trajectories",
        "train/replay_buffer/total_samples",
    ]
    assert grouped["Demo buffer"] == ["train/demo_buffer/num_trajectories"]
    # The buffers must not be swept into generic Training by the shorter prefix.
    assert not [key for key in grouped.get("Training", []) if "buffer/" in key]

    assert set(grouped["Actor training"]) == {
        "train/actor/loss",
        "train/actor/lr",
        "train/actor/grad_norm",
        "train/actor/entropy",
    }
    assert set(grouped["Critic training"]) == {
        "train/critic/loss",
        "train/critic/lr",
        "train/critic/grad_norm",
    }
    assert set(grouped["Evaluation"]) == {"eval/success_once", "eval/return"}
    assert set(grouped["Timing"]) == {"time/step", "time/eval", "time/actor/recv_traj"}

    # Nothing dropped and nothing orphaned: every key a real offline run logs
    # found a group, so "Other" is not needed here.
    assert bound["unmatched"] == []
    assert "Other" not in grouped

    # No north star: no metric is universally the headline across unknown task
    # types, and promoting an arbitrary one is worse than showing none. The
    # template declares `key: null`, so binding reports it unresolved rather than
    # absent -- which is what the Overview card keys off.
    assert bound["north_star"]["resolved"] is False
