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

from packaging.version import Version


def should_set_torch_nccl_avoid_record_streams(
    torch_version: str | None = None,
) -> bool:
    """Whether PyTorch still needs the legacy NCCL record-streams opt-out.

    PyTorch 2.8 made avoiding record streams the default and deprecated
    ``TORCH_NCCL_AVOID_RECORD_STREAMS``. Older releases, including the
    supported 2.6 stack, still require the variable to retain this behavior.

    Args:
        torch_version: Version to inspect. Defaults to the installed PyTorch
            version.

    Returns:
        ``True`` when RLinf should set ``TORCH_NCCL_AVOID_RECORD_STREAMS=1``.
    """
    if torch_version is None:
        import torch

        torch_version = torch.__version__

    release = Version(torch_version).release
    return release[:2] < (2, 8)


def silence_accumulate_grad_stream_mismatch_warning() -> bool:
    """Turn off PyTorch's AccumulateGrad stream-mismatch warning.

    Under FSDP with gradient accumulation, torch warns once per rank that an
    ``AccumulateGrad`` node from a previous iteration is still alive and sits on
    a different stream. The nodes are retained by FSDP itself rather than by
    RLinf, so there is nothing to release on our side; the mismatch is expected
    here and torch offers this switch for exactly that case.

    Returns:
        Whether the warning was turned off. ``False`` on PyTorch versions
        without the switch, which are also the versions that do not warn.
    """
    import torch

    setter = getattr(
        torch.autograd.graph,
        "set_warn_on_accumulate_grad_stream_mismatch",
        None,
    )
    if setter is None:
        return False
    setter(False)
    return True
