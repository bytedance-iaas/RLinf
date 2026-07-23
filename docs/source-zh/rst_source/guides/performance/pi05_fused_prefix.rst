Pi0.5 Fused Prefix Kernel
=========================

启用 Pi0.5 fused prefix decoder，可以在保持 checkpoint 加载、FSDP 封装、prefix cache
和 backward 兼容性的同时，降低 PaliGemma VLM latency。

启用 Fused Decoder
------------------

在 Pi0.5 配置中设置 opt-in 开关：

.. code-block:: yaml

   actor:
     model:
       openpi:
         enable_fused_prefix: true

基础 ``model/pi0_5`` preset 保持 ``false``。启用后，``get_model`` 只替换冻结
PaliGemma prefix VLM 中使用标准 RMSNorm 的层；adaRMS action expert 保持不变。

.. warning::

   请在 CUDA 和 Triton 环境中使用此路径。下述测试使用 NVIDIA H20 GPU、OpenPi
   环境和 FlashAttention。在其他 GPU 架构或依赖版本上完成数值与显存验证前，请保留
   默认 fallback。

理解 Fused 路径
---------------

实现保留原始 projection、normalization 和 MLP module，因此 checkpoint parameter
名称不变，FSDP 也会封装相同参数。融合路径减少 projection GEMM 周边的工作：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 区域
     - 融合内容
   * - Normalization
     - RMSNorm reduction、scaling 和 dtype conversion。
   * - Position encoding
     - 针对 Pi0.5 显式 position ID 的 RoPE rotation。
   * - Projection epilogue
     - GELU、gated multiplication、residual addition 和 dtype conversion。
   * - Prefix cache
     - 输出 rollout 构建 prefix cache 所需的逐层 K/V tensor。
   * - 训练兼容性
     - 当梯度经过 prefix VLM 时提供直接 backward 路径。

大规模单 rank batch 使用 64 位 Triton row 和 linear offset。batch 160 时，首个
prefix MLP 展平输出包含 2,537,144,320 个元素，超过有符号 32 位范围。使用 int64
offset 可以避免该配置下出现的 illegal memory access。

解读基准结果
------------

主要 A/B 测试使用单机 4 张 NVIDIA H20 GPU。每组运行 9 个 step，稳态统计使用第
4–9 step。

.. list-table:: Pi0.5 模型路径，batch 16、三步 denoise
   :header-rows: 1
   :widths: 30 24 24 22

   * - 变体
     - Rollout
     - Actor recompute
     - Rollout 相对 baseline
   * - Baseline
     - 750.90 ms
     - 688.84 ms
     - —
   * - Fused prefix
     - 695.64 ms
     - 629.33 ms
     - **-7.36%**

.. list-table:: LIBERO async PPO，4-GPU collocated
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - 变体
     - Actor
     - Rollout epoch
     - Step
     - 相对 baseline
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

masked 单层 forward 从 4.521 ms 降至 3.803 ms（``-15.9%``）。工作量对齐的
同步 LIBERO 测试中，step time 提升 ``2.66%``；ManiSkill 2+2 disaggregated
测试中提升 ``3.58%``。

rollout compilation 与 fused prefix kernel 会优化部分重叠的模型计算。组合后的独立
rollout 测试提升 ``14.54%``，小于两项独立收益之和。环境计算、placement 和 async
queue 时序还会进一步稀释模型局部收益。

检查正确性与适用范围
--------------------

fused layer 与真实 OpenPi Gemma decoder 在 zero mask 和 block mask 下完成对拍。
BF16 forward、backward 和所有 parameter gradient 的相对误差均小于 ``2e-2``。
完整 Pi0.5 ``get_log_prob_value`` 对拍中，log-probability 相对误差为 ``7.2e-8``，
value 相对误差约为 ``1e-2``。

int64 offset 修复完成了 batch-160 单 step smoke test 和两组 fused 9-step 基准测试。
请把所有数字视为特定 workload 的结果；将 fused 路径设为默认值前，请使用
:doc:`Profiling <../profile>` 分析目标 placement。
