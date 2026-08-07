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

"""Media queries and, more importantly, the access boundary.

``resolve`` is the one place in this server that turns a request parameter into a
file read, so it gets an allowlist rather than a filter: a path is served only if
it appears verbatim in this run's own media index. Video paths are absolute
strings written by env worker processes, which makes them untrusted input, and an
exact index match is a stronger guarantee than any amount of ``..``
normalization.

The traversal cases below are the ones that defeat naive filters -- an encoded
``..``, a symlink pointing out of the tree, a path that is genuinely inside the
run root but was never indexed.
"""

from __future__ import annotations

import os

import pytest

from rlinf_dashboard.discovery import RunDiscovery
from rlinf_dashboard.media import MediaService, content_type_for
from rlinf_dashboard.state import StateStore


@pytest.fixture
def media_run(tmp_path, run_tree, settings_for):
    """A run with four indexed clips across two shards and both splits."""
    videos = tmp_path / "videos"
    videos.mkdir()
    paths = {}
    for name in ("train_s1", "train_s2", "eval_s2", "train_s5"):
        path = videos / f"{name}.mp4"
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42fake-video-bytes")
        paths[name] = str(path)

    run_tree(
        "media-run",
        media={
            0: [
                {
                    "path": paths["train_s1"],
                    "step": 1,
                    "split": "train",
                    "shard": 0,
                    "num_frames": 120,
                    "fps": 30,
                    "num_success": 2,
                    "num_envs": 8,
                },
                {"path": paths["train_s2"], "step": 2, "split": "train", "shard": 0},
                {"path": paths["train_s5"], "step": 5, "split": "train", "shard": 0},
            ],
            1: [
                {"path": paths["eval_s2"], "step": 2, "split": "eval", "shard": 1},
            ],
        },
    )
    settings = settings_for()
    run = RunDiscovery(settings).find("media-run")
    assert run is not None
    return MediaService(StateStore(settings)), run, paths


# ------------------------------------------------------------------------ querying


def test_lists_every_indexed_clip(media_run):
    service, run, _ = media_run
    assert len(service.list_media(run)) == 4


def test_entries_carry_a_streaming_url(media_run):
    """The frontend gets a URL, not a filesystem path.

    It has no filesystem access, and handing it a path would push URL assembly --
    including the escaping -- into the client.
    """
    service, run, _ = media_run
    entry = service.list_media(run)[0]
    assert entry.url is not None
    assert entry.url.startswith("/api/runs/media-run/media/file?path=")


def test_filters_by_split(media_run):
    service, run, _ = media_run
    assert len(service.list_media(run, split="eval")) == 1
    assert len(service.list_media(run, split="train")) == 3


def test_filters_by_exact_step(media_run):
    """Clicking a point on a curve asks for that step's clips.

    Two shards can both have written at step 2, and both belong in the answer.
    """
    service, run, _ = media_run
    assert len(service.list_media(run, step=2)) == 2


def test_filters_by_step_range(media_run):
    service, run, _ = media_run
    entries = service.list_media(run, min_step=2, max_step=5)
    assert [entry.step for entry in entries] == [2, 2, 5]


def test_the_limit_keeps_the_newest(media_run):
    """A long run produces thousands of clips; the recent ones are the useful ones."""
    service, run, _ = media_run
    entries = service.list_media(run, limit=2)
    assert [entry.step for entry in entries] == [2, 5]


def test_filters_by_success(tmp_path, run_tree, settings_for):
    """The two queries worth having: "show me what worked" and "what did not".

    One MP4 tiles every env in a worker, so the index carries counts rather than
    a flag and "success" means *some* env succeeded. The unrecorded row is the
    interesting one: it must not answer either query, because a clip whose
    outcome was never written is not evidence of failure.
    """
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("won", "lost", "single", "unknown"):
        (videos / f"{name}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fake")

    run_tree(
        "success-run",
        media={
            0: [
                {
                    "path": str(videos / "won.mp4"),
                    "step": 1,
                    "num_success": 3,
                    "num_envs": 8,
                },
                {
                    "path": str(videos / "lost.mp4"),
                    "step": 2,
                    "num_success": 0,
                    "num_envs": 8,
                },
                # Single-env clip: the scalar is what a reader wants, and it must
                # be honoured even with no counts alongside it.
                {"path": str(videos / "single.mp4"), "step": 3, "success": True},
                {"path": str(videos / "unknown.mp4"), "step": 4},
            ]
        },
    )
    settings = settings_for()
    run = RunDiscovery(settings).find("success-run")
    assert run is not None
    service = MediaService(StateStore(settings))

    assert [
        os.path.basename(e.path) for e in service.list_media(run, success=True)
    ] == [
        "won.mp4",
        "single.mp4",
    ]
    assert [
        os.path.basename(e.path) for e in service.list_media(run, success=False)
    ] == ["lost.mp4"]
    assert len(service.list_media(run)) == 4


def test_success_counts_survive_the_read(media_run):
    """The counts are what the UI needs to say "3 of 8 succeeded"."""
    service, run, _ = media_run
    entry = next(e for e in service.list_media(run) if e.step == 1)
    assert entry.num_success == 2
    assert entry.num_envs == 8
    # A grid clip gets no scalar: collapsing 2-of-8 to `True` would read as a
    # claim about the whole clip.
    assert entry.success is None


def test_metadata_from_the_index_survives(media_run):
    """``num_frames`` and ``fps`` let the UI show duration without opening the file."""
    service, run, _ = media_run
    entry = next(e for e in service.list_media(run) if e.step == 1)
    assert entry.num_frames == 120
    assert entry.fps == 30


def test_steps_lists_which_points_have_clips(media_run):
    """The reason the index carries a step at all.

    It lets the UI mark the points on a curve that have a video behind them,
    instead of making someone guess where to click.
    """
    service, run, _ = media_run
    assert service.steps(run) == [1, 2, 5]


def test_a_run_with_no_media_is_empty_not_an_error(run_tree, settings_for):
    """Video recording is off for most runs."""
    run_tree("quiet-run")
    settings = settings_for()
    run = RunDiscovery(settings).find("quiet-run")
    service = MediaService(StateStore(settings))
    assert service.list_media(run) == []
    assert service.steps(run) == []


# ------------------------------------------------------------- the access boundary


def test_an_indexed_file_resolves(media_run):
    service, run, paths = media_run
    assert service.resolve(run, paths["train_s1"]) == os.path.realpath(
        paths["train_s1"]
    )


def test_an_unindexed_file_is_refused_even_next_to_an_indexed_one(media_run, tmp_path):
    """Reachable is not the same as allowlisted.

    This file sits in the same directory as four served clips, so any
    containment-based check would pass it. It was never indexed, so it is refused.
    """
    secret = tmp_path / "videos" / "not_indexed.mp4"
    secret.write_bytes(b"nope")
    service, run, _ = media_run
    assert service.resolve(run, str(secret)) is None


@pytest.mark.parametrize(
    "attack",
    [
        "/etc/passwd",
        "../../../../etc/passwd",
        "videos/../../../etc/passwd",
        "/etc/passwd\x00.mp4",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ],
    ids=["absolute", "relative-traversal", "mixed-traversal", "null-byte", "encoded"],
)
def test_traversal_attempts_are_refused(media_run, attack):
    """None of these appear in the index, so none of them resolve.

    An allowlist makes each of these one uninteresting case rather than a separate
    normalization bug to get right.
    """
    service, run, _ = media_run
    assert service.resolve(run, attack) is None


def test_a_symlink_out_of_the_tree_is_refused(media_run, tmp_path):
    """Real-path comparison is what closes this.

    A symlink placed inside the video directory whose target is elsewhere fails
    the comparison against the index's own real paths.
    """
    link = tmp_path / "videos" / "escape.mp4"
    target = tmp_path / "outside_secret.mp4"
    target.write_bytes(b"secret")
    link.symlink_to(target)

    service, run, _ = media_run
    assert service.resolve(run, str(link)) is None


def test_a_symlinked_log_directory_still_resolves(tmp_path, run_tree, settings_for):
    """The other side of real-path comparison: it must not break normal setups.

    Log paths are routinely symlinks to a mounted volume, and comparing both sides
    as real paths is what keeps those runs playable.
    """
    real_videos = tmp_path / "real_videos"
    real_videos.mkdir()
    clip = real_videos / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    link_dir = tmp_path / "linked_videos"
    link_dir.symlink_to(real_videos)

    run_tree("symlink-run", media={0: [{"path": str(clip), "step": 1, "shard": 0}]})
    settings = settings_for()
    run = RunDiscovery(settings).find("symlink-run")
    service = MediaService(StateStore(settings))

    # Requested through the symlinked directory, indexed under the real one.
    assert service.resolve(run, str(link_dir / "clip.mp4")) == os.path.realpath(clip)


def test_a_disallowed_extension_is_refused_even_when_indexed(
    tmp_path, run_tree, settings_for
):
    """Defence in depth against a compromised index.

    The index is written by env worker processes. If one ever wrote a path to a
    checkpoint shard or a config file, the extension check refuses it before the
    file is opened.
    """
    payload = tmp_path / "videos_x"
    payload.mkdir()
    shard = payload / "model.safetensors"
    shard.write_bytes(b"weights")

    run_tree("odd-run", media={0: [{"path": str(shard), "step": 1, "shard": 0}]})
    settings = settings_for()
    run = RunDiscovery(settings).find("odd-run")
    service = MediaService(StateStore(settings))
    assert service.resolve(run, str(shard)) is None


def test_an_indexed_file_that_was_deleted_resolves_to_none(media_run, tmp_path):
    """Clips get cleaned up while the index row stays.

    A 404 is the right answer; opening a missing file would be a 500.
    """
    service, run, paths = media_run
    os.remove(paths["train_s1"])
    assert service.resolve(run, paths["train_s1"]) is None


def test_one_run_cannot_read_another_run_s_clip(tmp_path, run_tree, settings_for):
    """The allowlist is per run, not per server.

    Two runs share a log root, and each index is checked only against its own
    rows.
    """
    videos = tmp_path / "shared_videos"
    videos.mkdir()
    a_clip = videos / "a.mp4"
    b_clip = videos / "b.mp4"
    for clip in (a_clip, b_clip):
        clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    run_tree("run-a", media={0: [{"path": str(a_clip), "step": 1, "shard": 0}]})
    run_tree("run-b", media={0: [{"path": str(b_clip), "step": 1, "shard": 0}]})

    settings = settings_for()
    discovery = RunDiscovery(settings)
    service = MediaService(StateStore(settings))
    run_a = discovery.find("run-a")

    assert service.resolve(run_a, str(a_clip)) is not None
    assert service.resolve(run_a, str(b_clip)) is None


# ------------------------------------------------------------------- content types


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/v/clip.mp4", "video/mp4"),
        ("/v/clip.webm", "video/webm"),
        ("/v/clip.GIF", "image/gif"),
        ("/v/frame.png", "image/png"),
        ("/v/frame.jpeg", "image/jpeg"),
        ("/v/mystery.bin", "application/octet-stream"),
    ],
)
def test_content_types(path, expected):
    """A wrong type makes a browser download a clip instead of playing it."""
    assert content_type_for(path) == expected
