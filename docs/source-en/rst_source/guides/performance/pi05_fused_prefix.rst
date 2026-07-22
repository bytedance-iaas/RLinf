Pi0.5 Fused Prefix Kernels
==========================

Enable the fused Pi0.5 prefix decoder to reduce PaliGemma VLM latency while
preserving checkpoint loading, FSDP wrapping, prefix caching, and backward
compatibility.

Enable the Fused Decoder
------------------------

Set the opt-in model switch in your Pi0.5 configuration:

.. code-block:: yaml

   actor:
     model:
       openpi:
         enable_fused_prefix: true

The base ``model/pi0_5`` preset keeps this switch ``false``. When enabled,
``get_model`` replaces only the standard-RMSNorm layers in the frozen
PaliGemma prefix VLM. The adaRMS action expert remains unchanged.

.. warning::

   Use this path with CUDA and Triton. The measurements below used NVIDIA H20
   GPUs with the OpenPi environment and FlashAttention. Keep the fallback
   disabled until you validate numerical parity and memory use on other GPU
   architectures or dependency versions.

Understand the Fused Path
-------------------------

The implementation keeps the original projection, normalization, and MLP
modules. Checkpoints therefore use the same parameter names, and FSDP wraps the
same parameters. The fused path reduces work around the projection GEMMs:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Area
     - Fused work
   * - Normalization
     - RMSNorm reduction, scaling, and dtype conversion.
   * - Position encoding
     - RoPE rotation for explicit Pi0.5 position IDs.
   * - Projection epilogues
     - GELU, gated multiplication, residual addition, and dtype conversion.
   * - Prefix cache
     - Emits the per-layer K/V tensors required by rollout prefix-cache builds.
   * - Training compatibility
     - Supplies a direct backward path when gradients reach the prefix VLM.

Large single-rank batches use 64-bit Triton row and linear offsets. At batch
160, the first flattened prefix-MLP output contains 2,537,144,320 elements,
which exceeds the signed 32-bit range. Using int64 offsets prevents the illegal
memory access observed in this configuration.

Read the Benchmark
------------------

The primary A/B measurements used one machine with 4 NVIDIA H20 GPUs. Each
variant ran for nine steps; steady-state statistics use steps 4 through 9.

.. list-table:: Pi0.5 model path, batch 16 and three denoise steps
   :header-rows: 1
   :widths: 30 24 24 22

   * - Variant
     - Rollout
     - Actor recompute
     - Rollout vs baseline
   * - Baseline
     - 750.90 ms
     - 688.84 ms
     - —
   * - Fused prefix
     - 695.64 ms
     - 629.33 ms
     - **-7.36%**

.. list-table:: LIBERO async PPO, 4-GPU collocated
   :header-rows: 1
   :widths: 24 19 19 19 19

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
   * - Fused prefix
     - 61.38 s
     - 74.47 s
     - 74.12 s
     - **-3.73%**

A masked single-layer forward pass improved from 4.521 ms to 3.803 ms
(``-15.9%``). A matched synchronous LIBERO run improved step time by
``2.66%``, and a 2+2 disaggregated ManiSkill run improved it by ``3.58%``.

Rollout compilation and fused prefix kernels optimize overlapping model work.
The combined isolated rollout measurement improved by ``14.54%``, less than
the sum of the isolated gains. Environment work, placement, and async queue
timing further dilute model-only gains.

Check Correctness and Scope
---------------------------

The fused layer was compared with the real OpenPi Gemma decoder using zero and
block masks. BF16 forward outputs, backward outputs, and all parameter
gradients stayed below ``2e-2`` relative error. The full Pi0.5
``get_log_prob_value`` comparison measured ``7.2e-8`` relative log-probability
error and about ``1e-2`` relative value error.

The int64-offset fix completed a one-step batch-160 smoke test and both fused
nine-step benchmark variants. Treat all measurements as workload-specific and
profile your target placement with :doc:`Profiling <../profile>` before making
the fused path the default.
