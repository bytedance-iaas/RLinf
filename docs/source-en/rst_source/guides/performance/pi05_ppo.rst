Pi0.5 PPO Performance Optimizations
===================================

Use the optimized Pi0.5 recipe to overlap async weight synchronization, compile
the rollout path, and fuse the frozen prefix VLM decoder layers. This page shows
where each optimization helps and reports the measured end-to-end gains.

Enable the Optimizations
------------------------

Start from the validated performance recipe:

.. code-block:: bash

   bash examples/embodiment/run_async.sh \
     libero_spatial_async_ppo_openpi_pi05_perf

What this does: it loads
``libero_spatial_async_ppo_openpi_pi05_perf.yaml`` and enables all three
switches below.

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Configuration
     - Default in the base Pi0.5 preset
     - Effect
   * - ``actor.sync_weight_no_wait``
     - ``False``
     - Starts actor-to-rollout synchronization in the background and coalesces
       a new request while the previous synchronization is still running.
   * - ``rollout.enable_torch_compile``
     - ``False``
     - Compiles the rollout model's hot submodules. The performance recipe uses
       ``torch_compile_mode: "default"``.
   * - ``actor.model.openpi.enable_fused_prefix``
     - ``False``
     - Replaces the standard-RMSNorm layers in the frozen PaliGemma prefix VLM
       with the fused Triton implementation. The action expert is unchanged.

Override any switch to isolate its contribution. For example, measure the
fused prefix without rollout compilation:

.. code-block:: bash

   bash examples/embodiment/run_async.sh \
     libero_spatial_async_ppo_openpi_pi05_perf \
     rollout.enable_torch_compile=False

What this does: it keeps background weight synchronization and the fused prefix
enabled, but runs rollout inference eagerly.

.. warning::

   Treat these switches as opt-in. The fused prefix path was validated with
   CUDA, Triton, FlashAttention, and NVIDIA H20 GPUs. ``torch.compile`` adds a
   first-step compilation cost, so a short run can have worse total wall time
   even when steady-state steps are faster.

Understand the Fused Prefix Path
--------------------------------

The Pi0.5 actor trains the action expert while its 18-layer PaliGemma prefix VLM
is frozen. The fused path keeps the projection GEMMs, then reduces the work
around them:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Area
     - Fused work
   * - Normalization
     - RMSNorm reduction, scale, and dtype conversion.
   * - Position encoding
     - RoPE rotation for explicit Pi0.5 position IDs.
   * - Projection epilogues
     - GELU, gated multiplication, residual addition, and dtype conversion
       around the GEMMs.
   * - Prefix cache
     - Emits the per-layer K/V tensors required by rollout prefix-cache builds.
   * - Training compatibility
     - Preserves the original submodules and parameters for checkpoint loading
       and FSDP, and supplies a backward path when gradients reach the prefix.

Large single-rank batches use 64-bit Triton pointer offsets. This avoids the
overflow found at batch 160, where the flattened first prefix-MLP output has
2,537,144,320 elements and exceeds the signed 32-bit range.

Read the Benchmark
------------------

The primary benchmark used one machine with 4 NVIDIA H20 GPUs. Each variant
ran for nine steps; the table reports the mean of steps 4 through 9. Compile
only affects rollout. Fused prefix affects both rollout and actor.

.. list-table:: LIBERO async PPO, 4-GPU collocated
   :header-rows: 1
   :widths: 23 19 19 19 20

   * - Variant
     - Actor
     - Rollout epoch
     - Step
     - Step vs baseline
   * - Baseline
     - 66.10 s
     - 78.04 s
     - 76.99 s
     - —
   * - Compile only
     - 62.95 s
     - 73.82 s
     - 72.69 s
     - **-5.59%**
   * - Fused only
     - 61.38 s
     - 74.47 s
     - 74.12 s
     - **-3.73%**
   * - Fused + compile
     - 63.63 s
     - 74.38 s
     - 72.88 s
     - **-5.34%**

The local model measurements explain the end-to-end result:

.. list-table:: Pi0.5 model path, batch 16 and three denoise steps
   :header-rows: 1
   :widths: 28 24 24 24

   * - Variant
     - Rollout
     - Actor recompute
     - Rollout vs baseline
   * - Baseline
     - 750.90 ms
     - 688.84 ms
     - —
   * - Compile only
     - 656.81 ms
     - 604.80 ms
     - **-12.53%**
   * - Fused only
     - 695.64 ms
     - 629.33 ms
     - **-7.36%**
   * - Fused + compile
     - 641.70 ms
     - 586.20 ms
     - **-14.54%**

The two compute optimizations overlap. Their local gains do not add linearly,
and environment work plus async queue timing dilute model-only gains. A matched
sync LIBERO run measured **-2.99%** for compile, **-2.66%** for fused, and
**-5.99%** for the combination. A 2+2 disaggregated ManiSkill run measured
**-6.51%**, **-3.58%**, and **-6.64%**, respectively.

Background weight synchronization used a separate 8-GPU H20 async PPO run. It
reduced the runner's blocking ``update_rollout_weights`` interval from about
3.1 seconds to 0.0018 seconds per step. Actor training stayed near 46 seconds.
End-to-end step time was dominated by 0–20 second rollout-availability noise,
so this measurement supports removal of the serial synchronization interval,
not a precise end-to-end speedup claim. Expect a larger effect when a
disaggregated or cross-node weight transfer occupies more of the step.

Check Correctness and Scope
---------------------------

The fused layer was compared against the real OpenPi Gemma decoder layer with
zero and block masks. Forward outputs, backward outputs, and all parameter
gradients stayed below ``2e-2`` relative error in BF16. The full Pi0.5
``get_log_prob_value`` comparison measured ``7.2e-8`` relative log-probability
error and about ``1e-2`` relative value error.

Interpret the numbers as steady-state results for the listed hardware and
workloads. Placement changes the bottleneck: ManiSkill GPU simulation produced
large tail latency when collocated, while its 2+2 placement was faster and much
more stable. Profile your target placement before enabling every switch.

See :doc:`Profiling <../profile>` to capture worker-level traces and
:doc:`Execution Modes <../../concepts/execution_modes>` to select a collocated
or disaggregated layout.
