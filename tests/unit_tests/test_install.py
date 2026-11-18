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

"""Installer CLI contracts without downloading packages or requiring hardware."""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = ROOT / "requirements" / "install.sh"
DOWNLOAD_ASSETS_SH = ROOT / "requirements" / "embodied" / "download_assets.sh"
DEDUP_VENVS = ROOT / "docker" / "dedup_venvs.py"


def run_sourced(script, snippet, env):
    """Runs a bash snippet with the functions of ``script`` loaded."""
    return subprocess.run(
        ["bash", "-c", f'source "{script}"\n{snippet}'],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def ascend_install(tmp_path):
    root = Path(__file__).resolve().parents[2]
    requirements = tmp_path / "requirements"
    shutil.copytree(root / "requirements", requirements)
    script = requirements / "install.sh"
    pyproject = tmp_path / "pyproject.toml"
    original = (root / "pyproject.toml").read_text()
    pyproject.write_text(original)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").write_text("")

    # The package manager records the real installer's requirements. The fake
    # torch distribution exposes the version that those requirements select.
    (bin_dir / "uv").write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['INSTALL_LOG'], 'a') as log:\n"
        "    log.write(json.dumps({'args': sys.argv[1:], "
        "'pyproject': Path('pyproject.toml').read_text()}) + '\\n')\n"
    )
    (bin_dir / "uv").chmod(0o755)
    (bin_dir / "torch.py").write_text(
        "import re\n"
        "from pathlib import Path\n"
        '__version__ = re.search(r\'"torch==([^"]+)"\', '
        "Path('pyproject.toml').read_text()).group(1)\n"
    )
    # Model installs reuse this checkout instead of cloning, and the LIBERO
    # asset download is a no-op.
    gr00t = tmp_path / "Isaac-GR00T"
    gr00t.mkdir()
    (bin_dir / "libero-download-assets").write_text("#!/bin/bash\n")
    (bin_dir / "libero-download-assets").chmod(0o755)

    def run(
        chip_output="910B4",
        probe_exit=0,
        arch="x86_64",
        chip=None,
        torch=None,
        model=None,
        env_name="dummy",
    ):
        probe = bin_dir / "npu-smi"
        probe.write_text(
            '#!/bin/bash\nprintf "%s\\n" "$PROBE_OUTPUT"\nexit "$PROBE_EXIT"\n'
        )
        probe.chmod(0o755)
        uname = bin_dir / "uname"
        uname.write_text('#!/bin/bash\nprintf "%s\\n" "$TEST_ARCH"\n')
        uname.chmod(0o755)
        log = tmp_path / "install.jsonl"
        env = {
            **os.environ,
            "HOME": str(tmp_path),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "PYTHONPATH": str(bin_dir),
            "INSTALL_LOG": str(log),
            "PROBE_OUTPUT": chip_output,
            "PROBE_EXIT": str(probe_exit),
            "TEST_ARCH": arch,
            "GR00T_PATH": str(gr00t),
        }
        env.pop("ASCEND_CHIP", None)
        env.pop("UV_TORCH_BACKEND", None)
        if chip is not None:
            env["ASCEND_CHIP"] = chip
        command = [
            "bash",
            str(script),
            "--platform",
            "ascend",
            "--no-root",
            "--python",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "--venv",
            str(venv),
            "embodied",
            "--env",
            env_name,
        ]
        if model is not None:
            command.extend(["--model", model])
        if torch is not None:
            command.extend(["--torch", torch])
        result = subprocess.run(
            command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert pyproject.read_text() == original
        return [json.loads(line) for line in log.read_text().splitlines()]

    return run


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize(
    ("chip_name", "torch_version", "npu_requirement"),
    [
        ("910B4", "2.6.0", "torch-npu==2.6.0"),
        ("Ascend910B4", "2.6.0", "torch-npu==2.6.0"),
        ("950PR", "2.10.0", "torch-npu==2.10.0.post4"),
        ("Ascend950PR", "2.10.0", "torch-npu==2.10.0.post4"),
    ],
)
def test_ascend_installer_selects_compatible_stack(
    ascend_install, arch, chip_name, torch_version, npu_requirement
):
    calls = ascend_install(chip_output=f"| 0 | {chip_name} | OK |", arch=arch)
    assert any(npu_requirement in call["args"] for call in calls)
    project = next(call["pyproject"] for call in calls if call["args"][0] == "sync")
    suffix = "+cpu" if arch == "x86_64" else ""
    assert f'"torch=={torch_version}{suffix}"' in project
    if arch == "x86_64":
        assert "https://download.pytorch.org/whl/cpu" in project


@pytest.mark.parametrize(("output", "exit_code"), [("", 1), ("Device unavailable", 0)])
def test_ascend_installer_handles_failed_chip_probe(ascend_install, output, exit_code):
    calls = ascend_install(chip_output=output, probe_exit=exit_code)
    assert any("torch-npu==2.6.0" in call["args"] for call in calls)


def test_ascend_installer_honors_torch_override_without_chip_probe(ascend_install):
    calls = ascend_install(chip_output="", probe_exit=1, torch="2.9.0")
    assert any("torch-npu==2.9.0" in call["args"] for call in calls)


def test_ascend_installer_honors_chip_override(ascend_install):
    calls = ascend_install(chip_output="", probe_exit=1, chip="Ascend950PR")
    assert any("torch-npu==2.10.0.post4" in call["args"] for call in calls)


def test_ascend_installer_keeps_950_post_build_with_matching_torch_override(
    ascend_install,
):
    calls = ascend_install(chip="Ascend950PR", torch="2.10.0")
    assert any("torch-npu==2.10.0.post4" in call["args"] for call in calls)


def test_ascend_installer_installs_gr00t_n1d7_with_libero(ascend_install):
    calls = ascend_install(chip="Ascend950PR", model="gr00t_n1d7", env_name="libero")
    args = [call["args"] for call in calls]
    assert ["pip", "install", "rlinf-libero"] in args
    assert any(arg[-1].endswith("models/gr00t_n1d7.txt") for arg in args)
    assert any("torch-npu==2.10.0.post4" in arg for arg in args)


@pytest.fixture
def git_remote(tmp_path):
    """A local repository standing in for GitHub: branch main and annotated tag v1."""
    remote = tmp_path / "remote"
    remote.mkdir()
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }

    def git(*args):
        result = subprocess.run(
            ["git", "-C", str(remote), *args],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    git("init", "-q", "-b", "main")
    (remote / "a.txt").write_text("one")
    git("add", "a.txt")
    git("commit", "-qm", "one")
    git("tag", "-a", "v1", "-m", "v1")
    return remote, git


def clone(tmp_path, url, dest, *clone_args, skip_lfs=False):
    """Clones through clone_or_reuse_repo with the install cache under tmp_path."""
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "RLINF_INSTALL_CACHE_DIR": str(tmp_path / "cache"),
    }
    env.pop("GITHUB_PREFIX", None)
    env.pop("GIT_LFS_SKIP_SMUDGE", None)
    if skip_lfs:
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
    args = " ".join(clone_args)
    result = run_sourced(
        INSTALL_SH, f'clone_or_reuse_repo RLINF_TEST_UNSET "{dest}" "{url}" {args}', env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == os.path.realpath(dest)


def head(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.mark.parametrize("ref_args", [("-b", "main"), ("-b", "v1"), ()])
def test_git_clone_cache_copies_clone_while_ref_is_unchanged(
    git_remote, tmp_path, ref_args
):
    remote, git = git_remote
    url = f"file://{remote}"
    clone(tmp_path, url, tmp_path / "first", *ref_args)
    (entry,) = (tmp_path / "cache" / "git").iterdir()
    (entry / "from-cache").write_text("")

    clone(tmp_path, url, tmp_path / "second", *ref_args)
    assert (tmp_path / "second" / "from-cache").exists()
    assert head(tmp_path / "second") == git("rev-parse", "HEAD")
    assert git("-C", str(tmp_path / "second"), "remote", "get-url", "origin") == url


def test_git_clone_cache_fetches_again_after_the_branch_moves(git_remote, tmp_path):
    remote, git = git_remote
    url = f"file://{remote}"
    clone(tmp_path, url, tmp_path / "first", "-b", "main")
    (entry,) = (tmp_path / "cache" / "git").iterdir()
    (entry / "from-cache").write_text("")
    git("commit", "--allow-empty", "-qm", "two")

    clone(tmp_path, url, tmp_path / "second", "-b", "main")
    assert not (tmp_path / "second" / "from-cache").exists()
    assert head(tmp_path / "second") == git("rev-parse", "HEAD")
    assert head(entry) == git("rev-parse", "HEAD")


def test_git_clone_cache_keeps_clones_without_lfs_files_apart(git_remote, tmp_path):
    remote, _ = git_remote
    url = f"file://{remote}"
    clone(tmp_path, url, tmp_path / "pointers", "-b", "main", skip_lfs=True)
    (entry,) = (tmp_path / "cache" / "git").iterdir()
    (entry / "from-cache").write_text("")

    clone(tmp_path, url, tmp_path / "files", "-b", "main")
    assert not (tmp_path / "files" / "from-cache").exists()
    assert len(list((tmp_path / "cache" / "git").iterdir())) == 2


@pytest.fixture
def fake_wget(tmp_path):
    """A wget that serves one file for any URL and logs the URLs it was asked for."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    payload = tmp_path / "payload.zip"
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("libPhysX.so", "x" * 1000)
    log = tmp_path / "wget.log"
    wget = bin_dir / "wget"
    wget.write_text('#!/bin/bash\necho "$3" >> "$WGET_LOG"\ncp "$WGET_PAYLOAD" "$2"\n')
    wget.chmod(0o755)
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "WGET_LOG": str(log),
        "WGET_PAYLOAD": str(payload),
        "RLINF_INSTALL_CACHE_DIR": str(tmp_path / "cache"),
    }
    return env, log, payload


def test_fetch_zip_downloads_a_release_archive_once(fake_wget, tmp_path):
    env, log, payload = fake_wget
    release = "https://github.com/org/repo/releases/download/v1/linux-so.zip"
    # The probe can pick another GitHub proxy on the next build.
    for proxy, dest in (
        ("https://proxy-a/", "one.zip"),
        ("https://proxy-b/", "two.zip"),
    ):
        result = run_sourced(
            DOWNLOAD_ASSETS_SH,
            f'fetch_zip "{proxy}{release}" "{tmp_path / dest}"',
            {**env, "GITHUB_PREFIX": proxy},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / dest).read_bytes() == payload.read_bytes()
    assert log.read_text().splitlines() == [f"https://proxy-a/{release}"]


def test_fetch_zip_does_not_cache_an_invalid_archive(fake_wget, tmp_path):
    env, log, payload = fake_wget
    payload.write_text("<html>rate limited</html>")
    result = run_sourced(
        DOWNLOAD_ASSETS_SH,
        f'fetch_zip "https://example.com/linux-so.zip" "{tmp_path / "out.zip"}"',
        env,
    )
    assert result.returncode != 0
    assert not list((tmp_path / "cache" / "downloads").iterdir())


def make_venv(root, name, files):
    """Writes a venv whose one distribution installed ``files`` (path -> bytes)."""
    site_packages = root / name / "lib" / "python3.11" / "site-packages"
    dist_info = site_packages / "pkg-1.0.dist-info"
    dist_info.mkdir(parents=True)
    rows = []
    for rel, data in files.items():
        (site_packages / rel).parent.mkdir(parents=True, exist_ok=True)
        (site_packages / rel).write_bytes(data)
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        rows.append(f"{rel},sha256={digest.decode()},{len(data)}")
    rows.append("pkg-1.0.dist-info/RECORD,,")
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n")
    return site_packages


def dedup(root):
    result = subprocess.run(
        [sys.executable, str(DEDUP_VENVS), str(root)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_dedup_venvs_links_unchanged_installed_files_across_venvs(tmp_path):
    files = {
        "pkg/__init__.py": b"VERSION = 1\n",
        "pkg/lib.so": b"\x7fELF" * 256,
        "pkg/gone.py": b"pass\n",
    }
    a = make_venv(tmp_path, "a", files)
    b = make_venv(tmp_path, "b", files)
    # Patched in place after installation: same size, no longer the RECORD hash.
    (b / "pkg" / "lib.so").write_bytes(b"\x00" * len(files["pkg/lib.so"]))
    # Deleted under its RECORD, as uninstalling a package that shares it does.
    (b / "pkg" / "gone.py").unlink()

    dedup(tmp_path)
    assert (a / "pkg/__init__.py").stat().st_ino == (
        b / "pkg/__init__.py"
    ).stat().st_ino
    assert (a / "pkg/lib.so").stat().st_ino != (b / "pkg/lib.so").stat().st_ino
    assert "linked 0 files" in dedup(tmp_path)

    c = make_venv(tmp_path, "c", files)
    (c / "pkg" / "lib.so").chmod(0o755)
    dedup(tmp_path)
    assert (c / "pkg/__init__.py").stat().st_ino == (
        a / "pkg/__init__.py"
    ).stat().st_ino
    assert (a / "pkg/__init__.py").stat().st_nlink == 3
    assert (c / "pkg/lib.so").stat().st_ino != (a / "pkg/lib.so").stat().st_ino
