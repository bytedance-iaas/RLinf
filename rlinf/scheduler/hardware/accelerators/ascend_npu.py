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

# Override of Ray's NPUAcceleratorManager
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

from ray._private.accelerators.npu import NPUAcceleratorManager

from .accelerator import AcceleratorManager, AcceleratorType, ProfileConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions

logger = logging.getLogger(__name__)

# The capture this process has open, if any. One per worker process: the
# profiler owns device-wide state, so a second concurrent window would
# interleave two traces into one output directory.
_capture = None

_PROFILING_ENV_VAR = "RLINF_NPU_PROFILING"


@AcceleratorManager.register_profiling_config(AcceleratorType.NPU)
@dataclass
class AscendNPUProfileConfig(ProfileConfig):
    """Capture host and device operations with ``torch_npu.profiler``.

    Select it with ``cluster.profiling.backend: torch_npu``, then pick the
    worker groups and steps through the shared ``cluster.profiling`` fields.
    Each worker and step writes its own trace directory, which is what keeps a
    multi-rank run readable -- and large, so prefer naming one actor and one
    rollout rank over profiling every rank.

    The three capture switches below are the profiler's own and all cost
    overhead: ``record_shapes`` annotates every operator with its input shapes,
    ``profile_memory`` tracks allocations, and ``with_stack`` records Python
    call stacks.
    """

    BACKEND_TYPE: ClassVar[str] = "torch_npu"
    backend: str = "torch_npu"
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False

    def check(self) -> bool:
        """Return whether ``torch_npu`` is importable on this node."""
        import importlib.util

        return importlib.util.find_spec("torch_npu") is not None


def boolean_attention_mask(attn_mask, _cache={}, *, use_cache: bool = True):
    """The boolean form of a {0, -inf} float mask, or None if it carries biases.

    Verifying the mask forces a device sync, and HF models pass one mask tensor
    through every layer, so the verdict is cached per live tensor identity.
    """
    import weakref

    import torch

    if (
        attn_mask is None
        or not attn_mask.dtype.is_floating_point
        or attn_mask.requires_grad
    ):
        return None
    use_cache = use_cache and not torch.is_inference(attn_mask)
    version = attn_mask._version if use_cache else None
    entry = _cache.get(id(attn_mask)) if use_cache else None
    if entry is not None:
        ref, cached_version, result = entry
        if ref() is attn_mask and cached_version == version:
            return result
    floor = torch.finfo(attn_mask.dtype).min * 0.5
    keep = attn_mask == 0
    # An entirely masked finite row still has a softmax distribution. A
    # boolean row instead yields zero, so only all-minus-inf rows may convert.
    safe_rows = keep.any(dim=-1) | torch.isneginf(attn_mask).all(dim=-1)
    convertible = (keep | (attn_mask <= floor)).all() & safe_rows.all()
    result = keep if bool(convertible) else None
    if use_cache:
        if len(_cache) >= 8:
            _cache.clear()
        key = id(attn_mask)
        ref = weakref.ref(
            attn_mask,
            lambda dead_ref, k=key: (
                _cache.pop(k, None) if _cache.get(k, (None,))[0] is dead_ref else None
            ),
        )
        _cache[key] = (ref, version, result)
    return result


def install_npu_sdpa_mask_cast() -> bool:
    """Cast {0, -inf} float SDPA masks to boolean so the NPU keeps its fused kernel.

    torch-npu decomposes SDPA into matmul-softmax for floating masks (4-13x
    slower); the additive causal masks HF models build are boolean in content.
    Masks carrying real biases are left untouched. Idempotent; returns whether
    the wrapper is installed.
    """
    import torch

    functional = torch.nn.functional
    if getattr(functional.scaled_dot_product_attention, "_rlinf_npu_mask_cast", False):
        return True
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return False

    original = functional.scaled_dot_product_attention

    def scaled_dot_product_attention(
        query, key, value, attn_mask=None, *args, **kwargs
    ):
        if attn_mask is not None and attn_mask.dtype.is_floating_point:
            capturing = getattr(torch.npu, "is_current_stream_capturing", None)
            # Evaluating boolean_attention_mask reads a device scalar on the
            # host, which is illegal during stream capture and cannot re-check
            # mask values copied into static buffers on replay.
            if capturing is None or not capturing():
                keep = boolean_attention_mask(attn_mask, use_cache=True)
                if keep is not None:
                    attn_mask = keep
        return original(query, key, value, attn_mask, *args, **kwargs)

    scaled_dot_product_attention._rlinf_npu_mask_cast = True
    functional.scaled_dot_product_attention = scaled_dot_product_attention
    logger.info("NPU SDPA float-mask cast installed (FlashAttentionScore path).")
    return True


def install_npu_fused_rotary() -> bool:
    """Serve HF Llama's rotary embedding from ``npu_rotary_mul`` (3x faster).

    Module-level, so every Llama-family model built afterwards picks it up.
    Idempotent; returns whether the patch is installed.
    """
    import torch

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return False
    try:
        import torch_npu
        from transformers.models.llama import modeling_llama
    except ImportError:
        return False
    if getattr(modeling_llama.apply_rotary_pos_emb, "_rlinf_npu_fused", False):
        return True

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        return (
            torch_npu.npu_rotary_mul(q, cos, sin),
            torch_npu.npu_rotary_mul(k, cos, sin),
        )

    apply_rotary_pos_emb._rlinf_npu_fused = True
    modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb
    logger.info("NPU fused rotary embedding installed for Llama models.")
    return True


@AcceleratorManager.register_manager(AcceleratorType.NPU)
class AscendNPUManager(AcceleratorManager):
    """Utility Class for Ascend NPU."""

    @staticmethod
    def get_profiling_env_vars(
        profiling_cfg: AscendNPUProfileConfig,
        output_prefix: str,
    ) -> dict[str, str]:
        """Carry the capture settings to the workers that were selected.

        The cluster calls this once per worker allocation, only for the groups
        the config profiles, so the presence of the variable is what tells a
        worker process that it may open a window at all.
        """
        return {
            _PROFILING_ENV_VAR: json.dumps(
                {
                    "output_prefix": output_prefix,
                    "record_shapes": bool(profiling_cfg.record_shapes),
                    "profile_memory": bool(profiling_cfg.profile_memory),
                    "with_stack": bool(profiling_cfg.with_stack),
                }
            )
        }

    @staticmethod
    def start_profiling(step_idx: Optional[int] = None) -> None:
        """Open a capture window for this step, if this worker was selected.

        Does nothing when the worker carries no profiling settings, or when a
        window is already open -- the runner gates steps itself, and a nested
        start would otherwise replace the handle and leak the first capture.
        """
        global _capture
        settings = os.environ.get(_PROFILING_ENV_VAR)
        if settings is None or _capture is not None:
            return

        from torch_npu import profiler

        options = json.loads(settings)
        # ``%p`` is the cluster's pid placeholder in the output prefix.
        prefix = options.pop("output_prefix").replace("%p", str(os.getpid()))
        output_dir = os.path.join(prefix, f"step_{step_idx}")
        os.makedirs(output_dir, exist_ok=True)

        capture = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.NPU,
            ],
            # One window per start/stop pair: record the step that follows and
            # nothing after it, so a long run cannot accumulate captures.
            schedule=lambda step: (
                profiler.ProfilerAction.RECORD_AND_SAVE
                if step == 0
                else profiler.ProfilerAction.NONE
            ),
            on_trace_ready=profiler.tensorboard_trace_handler(output_dir),
            **options,
        )
        try:
            capture.start()
        except Exception:
            # A failed start can still hold device-side state; release it so a
            # later step can try again instead of the worker staying wedged.
            with contextlib.suppress(Exception):
                capture.stop()
            raise
        _capture = capture
        logger.info(f"NPU profiling window open for step {step_idx}: {output_dir}")

    @staticmethod
    def stop_profiling() -> None:
        """Close the open window and write its trace. Safe to call when none is open."""
        global _capture
        if _capture is None:
            return
        capture, _capture = _capture, None
        try:
            # The schedule above acts on the step boundary, so this is what
            # makes the window eligible to be written out.
            capture.step()
        finally:
            capture.stop()

    @staticmethod
    def is_profiling_active() -> bool:
        """Return whether this process has a capture window open."""
        return _capture is not None

    @staticmethod
    def profiling_range(
        label: str,
        color: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        """Annotate a code range so it appears alongside the device operators.

        ``color`` and ``domain`` are part of the shared signature and have no
        equivalent here; the Ascend profiler has no range colouring.
        """
        if _capture is None:
            return contextlib.nullcontext()

        import torch

        return torch.profiler.record_function(label)

    @staticmethod
    def get_num_devices():
        """Get the number of Ascend NPU devices on the node."""
        return NPUAcceleratorManager.get_current_node_num_accelerators()

    @staticmethod
    def get_accelerator_type():
        """Get the type of the accelerator."""
        return AcceleratorType.NPU

    @staticmethod
    def get_accelerator_model():
        """Get the model of the Ascend NPU."""
        return NPUAcceleratorManager.get_current_node_accelerator_type()

    @staticmethod
    def get_accelerator_env_var(visible_accelerators: list[str]) -> dict[str, str]:
        """Get the environment variables related to the accelerator.

        Args:
            visible_accelerators (List[str]): A list of visible accelerator IDs.

        Returns:
            Dict[str, str]: A dictionary containing the accelerator environment variables.
        """
        env_vars = {}
        visible_accelerators_str = ",".join(visible_accelerators)

        env_vars["ASCEND_RT_VISIBLE_DEVICES"] = visible_accelerators_str
        env_vars["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
        # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L91

        return env_vars

    @staticmethod
    def get_visible_devices():
        """Get the visible device IDs."""
        visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", None)

        if visible_devices is None or visible_devices == "":
            return []
        else:
            try:
                visible_devices = [int(v.strip()) for v in visible_devices.split(",")]
            except ValueError:
                raise ValueError(
                    f"Invalid visible device IDs: {visible_devices}. "
                    "Please ensure they are integers separated by commas."
                )
            return visible_devices

    @staticmethod
    def get_ccl_backend():
        """Get the CCL backend."""
        return "hccl"

    @staticmethod
    def get_ccl_socket_ifname_env_var() -> str:
        """Get the network socket interface name environment variable.

        Returns:
            str: The network socket interface name environment variable.
        """
        return "HCCL_SOCKET_IFNAME"

    @staticmethod
    def get_torch_platform():
        """Get the PyTorch platform module."""
        import torch
        import torch_npu  # noqa: F401

        # NPU does have inter-process tensor sharing: torch_npu implements it in
        # NPUIPCTypes and torch.multiprocessing's reduce_tensor produces a real
        # NPU IPC payload (checked on CANN 9.1.0 / torch_npu 2.10.0.post4, where
        # torch.npu.ipc_collect exists). Older torch_npu builds do not export
        # ipc_collect, and the scheduler call sites (collective_group,
        # utils.utils) invoke `Worker.torch_platform.ipc_collect()`
        # unconditionally, so attach a no-op when it is missing instead of
        # guarding every call site.
        if not hasattr(torch.npu, "ipc_collect"):
            torch.npu.ipc_collect = lambda: None
        return torch.npu

    @staticmethod
    def get_device_type() -> str:
        """Get the device type."""
        return "npu"

    @staticmethod
    def setup_worker_torch() -> None:
        """Install the fused-kernel SDPA and rotary-embedding routes."""
        install_npu_sdpa_mask_cast()
        install_npu_fused_rotary()

    @staticmethod
    def get_accel_pg_options(options: Optional["CollectiveGroupOptions"]):
        """Get the accelerator CCL process group options."""
        return None
