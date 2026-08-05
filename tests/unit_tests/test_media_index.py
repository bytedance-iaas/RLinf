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

"""Tests for video media indexing in :class:`RecordVideo`.

Recorded MP4s are named only by a per-wrapper counter, so nothing on disk says
which step, split, or seed produced a given file. The index supplies that, and
it has to be written now even though playback is out of scope -- otherwise every
historical run would need backfilling later.

Two properties carry the design:

* **One row per completed MP4.** The row is appended after ``future.result()``
  returns, so a reader of the index never sees a file it cannot open. Same
  append-after-completion rule as the checkpoint index.
* **Indexing is optional and non-fatal.** The wrapper is usable standalone with
  no index attached, and a failing index must not cost a run its video.

``RecordVideo`` is driven through ``__new__`` with just the attributes
``flush_video`` touches. Constructing it for real would need a live simulator
env; what is under test is the indexing contract, not video encoding.
"""

import json
import os

import pytest

pytest.importorskip("gymnasium", reason="RecordVideo wraps a gymnasium env")
pytest.importorskip("imageio", reason="RecordVideo encodes MP4s via imageio")
numpy = pytest.importorskip("numpy")

from omegaconf import OmegaConf  # noqa: E402

from rlinf.envs.wrappers.record_video import RecordVideo  # noqa: E402
from rlinf.utils.run_state import MediaIndexWriter  # noqa: E402


class _FakeEnv:
    """Only what ``flush_video`` reads. Instantiated per test so a test that sets
    ``success_once`` cannot leak it into the next one."""

    def __init__(self):
        self.seed = 7


class _FakeFuture:
    def result(self):
        return None


@pytest.fixture
def wrapper(tmp_path, monkeypatch):
    """A ``RecordVideo`` with only what ``flush_video`` reads.

    The encode is stubbed out: it writes a placeholder file so "one row per file
    on disk" is still checkable, without needing a real codec.
    """
    instance = RecordVideo.__new__(RecordVideo)
    instance.env = _FakeEnv()
    instance._num_envs = 1
    instance.video_cfg = OmegaConf.create({"video_base_dir": str(tmp_path / "video")})
    instance.render_images = []
    instance.video_cnt = 0
    instance._fps = 30
    instance._media_index = None
    instance._global_step = None

    written = []

    def fake_submit(frames, mp4_path):
        # Stand-in for the encode: the file must exist by the time the index row
        # is appended, which is the ordering under test.
        with open(mp4_path, "wb") as handle:
            handle.write(b"fake-mp4")
        written.append(mp4_path)
        return _FakeFuture()

    monkeypatch.setattr(instance, "_submit_save", fake_submit)
    instance.written = written
    return instance


def _frames(count=3):
    return [numpy.zeros((4, 4, 3), dtype=numpy.uint8) for _ in range(count)]


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------- one row per file


def test_row_count_matches_mp4_count(wrapper, tmp_path):
    """The acceptance check: index rows == videos actually written."""
    index = MediaIndexWriter(str(tmp_path / "run"), shard=0)
    wrapper.set_media_index(index, shard=0)

    for step in range(4):
        wrapper.set_global_step(step)
        wrapper.render_images = _frames()
        wrapper.flush_video()

    rows = _rows(str(tmp_path / "run" / "media.rank0.jsonl"))
    assert len(rows) == 4 == len(wrapper.written)


def test_row_describes_the_video(wrapper, tmp_path):
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index, shard=2)
    wrapper.set_global_step(11)
    wrapper.render_images = _frames(5)
    wrapper.flush_video(video_sub_dir="eval")

    row = _rows(str(tmp_path / "run" / "media.rank0.jsonl"))[0]
    assert row["step"] == 11
    assert row["split"] == "eval"
    assert row["seed"] == 7
    assert row["num_frames"] == 5
    assert row["fps"] == 30
    assert row["shard"] == 2
    assert row["path"].endswith("0.mp4")


def test_split_defaults_to_train(wrapper, tmp_path):
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = _frames()
    wrapper.flush_video()

    assert _rows(str(tmp_path / "run" / "media.rank0.jsonl"))[0]["split"] == "train"


def test_split_can_be_labelled_without_moving_the_file(wrapper, tmp_path):
    """``split=`` labels the row; ``video_sub_dir=`` also moves the output.

    Train and eval already write to different ``video_base_dir`` roots, so the
    env worker needs the label without the relocation. Before this split existed
    the worker called ``flush_video()`` bare and every eval clip was indexed as
    ``train`` -- and passing ``video_sub_dir`` to fix that would have silently
    changed where videos land for every existing run.
    """
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = _frames()
    wrapper.flush_video(split="eval")

    row = _rows(str(tmp_path / "run" / "media.rank0.jsonl"))[0]
    assert row["split"] == "eval"
    # The path has no `eval/` component: the label did not relocate the file.
    assert os.path.dirname(row["path"]).endswith("seed_7")


def test_an_explicit_split_wins_over_the_sub_directory(wrapper, tmp_path):
    """A caller doing both gets the label it asked for, not the directory name."""
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = _frames()
    wrapper.flush_video(video_sub_dir="videos", split="eval")

    row = _rows(str(tmp_path / "run" / "media.rank0.jsonl"))[0]
    assert row["split"] == "eval"
    assert os.path.dirname(row["path"]).endswith(os.path.join("seed_7", "videos"))


# --------------------------------------------------------------------- success


def test_success_is_recorded_as_a_count_over_envs(wrapper, tmp_path):
    """One MP4 tiles every env in the worker, so success is a count.

    ``_append_frame`` tiles the per-env images into a single frame, which means a
    scalar ``success`` would be a claim the file cannot support whenever envs
    disagree. Counts let a reader ask either question.
    """
    wrapper.env.success_once = numpy.array([True, False, True, False], dtype=bool)
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = _frames()
    wrapper.flush_video()

    row = _rows(str(tmp_path / "run" / "media.rank0.jsonl"))[0]
    assert row["num_success"] == 2
    assert row["num_envs"] == 4
    assert row["success"] is None


def test_a_single_env_clip_gets_a_scalar(wrapper, tmp_path):
    """With one env the grid is one frame and a bool is what a reader wants."""
    wrapper.env.success_once = numpy.array([True], dtype=bool)
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = _frames()
    wrapper.flush_video()

    row = _rows(str(tmp_path / "run" / "media.rank0.jsonl"))[0]
    assert row["success"] is True
    assert row["num_success"] == 1
    assert row["num_envs"] == 1


def test_an_env_without_success_tracking_omits_the_fields(wrapper, tmp_path):
    """Absent, not false.

    Envs with no success notion -- and any env with ``record_metrics`` off -- have
    no ``success_once``. Writing ``false`` there would make "show me the failures"
    return every clip from every such run.
    """
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = _frames()
    wrapper.flush_video()

    row = _rows(str(tmp_path / "run" / "media.rank0.jsonl"))[0]
    assert "num_success" not in row
    assert "success" not in row


def test_an_unreadable_success_metric_still_indexes_the_clip(wrapper, tmp_path):
    """Indexing must never cost a run its video row."""

    class Hostile:
        def __array__(self, *args, **kwargs):
            raise RuntimeError("not convertible")

    wrapper.env.success_once = Hostile()
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = _frames()
    wrapper.flush_video()

    rows = _rows(str(tmp_path / "run" / "media.rank0.jsonl"))
    assert len(rows) == 1
    assert "num_success" not in rows[0]


def test_step_is_null_when_never_set(wrapper, tmp_path):
    """Readers fall back to ordering by mtime rather than getting a wrong step."""
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = _frames()
    wrapper.flush_video()

    assert _rows(str(tmp_path / "run" / "media.rank0.jsonl"))[0]["step"] is None


def test_row_written_only_after_the_file_exists(wrapper, tmp_path):
    """Append-after-completion: the path in a row is always openable."""
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = _frames()
    wrapper.flush_video()

    row = _rows(str(tmp_path / "run" / "media.rank0.jsonl"))[0]
    assert os.path.exists(row["path"])


def test_num_frames_captured_before_the_buffer_is_cleared(wrapper, tmp_path):
    """``flush_video`` empties ``render_images``; the count must be taken first."""
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = _frames(9)
    wrapper.flush_video()

    assert wrapper.render_images == []
    assert _rows(str(tmp_path / "run" / "media.rank0.jsonl"))[0]["num_frames"] == 9


# ------------------------------------------------------------------- optional


def test_no_index_attached_still_records(wrapper):
    """The wrapper stays usable standalone, which is why None is the default."""
    wrapper.render_images = _frames()
    wrapper.flush_video()

    assert len(wrapper.written) == 1


def test_a_failing_index_does_not_break_recording(wrapper, tmp_path):
    class Exploding:
        def append(self, record):
            raise RuntimeError("index gone")

    wrapper.set_media_index(Exploding())
    wrapper.render_images = _frames()

    with pytest.warns(UserWarning, match="Failed to index video"):
        wrapper.flush_video()

    assert len(wrapper.written) == 1


def test_nothing_written_when_there_are_no_frames(wrapper, tmp_path):
    """No video means no row; an empty MP4 would be a broken link in the index."""
    index = MediaIndexWriter(str(tmp_path / "run"))
    wrapper.set_media_index(index)
    wrapper.render_images = []
    wrapper.flush_video()

    assert wrapper.written == []
    assert _rows(str(tmp_path / "run" / "media.rank0.jsonl")) == []


# ---------------------------------------------------------------- multi-writer


def test_separate_shards_do_not_interleave(tmp_path):
    """Videos are encoded in env worker processes, so each writer owns a file and
    no locking is needed."""
    root = str(tmp_path / "run")
    for shard in (0, 1, 2):
        writer = MediaIndexWriter(root, shard=shard)
        for step in range(3):
            writer.append(
                {"path": f"r{shard}-s{step}.mp4", "step": step, "shard": shard}
            )

    for shard in (0, 1, 2):
        rows = _rows(os.path.join(root, f"media.rank{shard}.jsonl"))
        assert len(rows) == 3
        assert {row["shard"] for row in rows} == {shard}
