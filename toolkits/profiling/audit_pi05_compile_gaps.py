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

"""Measure whether Pi0.5 torch.compile gains come from kernels or launch gaps.

Unlike ``bench_actor_launchbound.py``, this script sums only raw CUDA device
events. Summing CUDA time from ``key_averages()`` double-counts parent PyTorch
operators and their child kernels, which can produce impossible negative idle
percentages.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import time

import torch
from bench_pi05_denoise import build_cfg, make_env_obs
from torch.autograd import DeviceType


def synchronize() -> None:
    """Wait for the current CUDA device."""
    torch.cuda.synchronize()


def time_call(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    """Return median wall and CUDA-stream makespan in milliseconds."""
    for _ in range(warmup):
        fn()
    synchronize()

    wall_times = []
    stream_times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        fn()
        end.record()
        synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1e3)
        stream_times.append(start.elapsed_time(end))
    return statistics.median(wall_times), statistics.median(stream_times)


def profile_raw_kernels(
    fn, *, iters: int
) -> tuple[float, int, list[tuple[str, float, int]]]:
    """Return raw kernel time/call count without parent-op double counting."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
            synchronize()

    kernels = [event for event in prof.events() if event.device_type == DeviceType.CUDA]
    kernel_us = sum(event.device_time_total for event in kernels)
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for event in kernels:
        grouped[event.name].append(event.device_time_total)
    top = sorted(
        (
            (name, sum(times) / 1e3 / iters, len(times) // iters)
            for name, times in grouped.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    return kernel_us / 1e3 / iters, len(kernels) // iters, top


def report(tag: str, fn, args) -> None:
    """Benchmark and profile one execution mode."""
    wall_ms, stream_ms = time_call(fn, warmup=args.warmup, iters=args.iters)
    kernel_ms, kernel_count, top = profile_raw_kernels(fn, iters=args.profile_iters)
    gap_ms = max(stream_ms - kernel_ms, 0.0)
    print(f"\n[{tag}]")
    print(f"  median wall            : {wall_ms:9.2f} ms")
    print(f"  median CUDA makespan   : {stream_ms:9.2f} ms")
    print(f"  raw kernel duration sum: {kernel_ms:9.2f} ms")
    print(
        f"  estimated stream gaps  : {gap_ms:9.2f} ms ({100 * gap_ms / stream_ms:5.1f}%)"
    )
    print(f"  raw kernel launches    : {kernel_count:9d}")
    print("  top raw kernels (ms/call-set, launches/call):")
    for name, duration_ms, count in top:
        print(f"    {duration_ms:8.2f}  {count:5d}  {name[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--num-action-chunks", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--profile-iters", type=int, default=4)
    parser.add_argument("--compile-mode", default="default")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from rlinf.models import get_model

    model = get_model(build_cfg(args))
    model.eval()
    env_obs = make_env_obs(args.batch_size, torch.device("cuda"))

    @torch.no_grad()
    def rollout_call():
        return model.predict_action_batch(env_obs, mode="train", compute_values=True)

    report("eager rollout", rollout_call, args)
    model.enable_torch_compile(mode=args.compile_mode)
    report(f"compiled rollout ({args.compile_mode})", rollout_call, args)


if __name__ == "__main__":
    main()
