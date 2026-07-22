Pi0.5 PPO 性能优化
=================

使用优化版 Pi0.5 配置，可以重叠异步权重同步、编译 rollout 路径，并融合冻结的
prefix VLM decoder layer。本页说明每项优化的适用范围和端到端实测收益。

启用优化
--------

从已验证的性能配置启动：

.. code-block:: bash

   bash examples/embodiment/run_async.sh \
     libero_spatial_async_ppo_openpi_pi05_perf

这条命令会加载 ``libero_spatial_async_ppo_openpi_pi05_perf.yaml``，并启用下表中的
三个开关。

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - 配置
     - Pi0.5 基础 preset 默认值
     - 作用
   * - ``actor.sync_weight_no_wait``
     - ``False``
     - 在后台启动 actor-to-rollout 权重同步；上一次同步尚未完成时，合并新的同步请求。
   * - ``rollout.enable_torch_compile``
     - ``False``
     - 编译 rollout 模型的热点子模块。性能配置使用
       ``torch_compile_mode: "default"``。
   * - ``actor.model.openpi.enable_fused_prefix``
     - ``False``
     - 用 Triton 融合实现替换冻结 PaliGemma prefix VLM 中使用标准 RMSNorm 的层。
       action expert 保持不变。

通过覆盖单个开关，可以隔离对应收益。例如，只测 fused prefix、不编译 rollout：

.. code-block:: bash

   bash examples/embodiment/run_async.sh \
     libero_spatial_async_ppo_openpi_pi05_perf \
     rollout.enable_torch_compile=False

这条命令保留后台权重同步和 fused prefix，但让 rollout inference 使用 eager 模式。

.. warning::

   请把这些开关视为 opt-in。fused prefix 路径只在 CUDA、Triton、FlashAttention 和
   NVIDIA H20 GPU 上完成了验证。``torch.compile`` 会增加首步编译开销，因此短任务即使
   稳态 step 更快，总墙钟时间也可能更差。

理解 Fused Prefix 路径
----------------------

Pi0.5 actor 训练 action expert，而 18 层 PaliGemma prefix VLM 保持冻结。融合路径保留
projection GEMM，并减少其周边工作：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 区域
     - 融合内容
   * - Normalization
     - RMSNorm reduction、scale 和 dtype conversion。
   * - Position encoding
     - 针对 Pi0.5 显式 position ID 的 RoPE rotation。
   * - Projection epilogue
     - GEMM 周边的 GELU、gated multiplication、residual addition 和 dtype conversion。
   * - Prefix cache
     - 输出 rollout 构建 prefix cache 所需的逐层 K/V tensor。
   * - 训练兼容性
     - 保留原始 submodule 和 parameter，兼容 checkpoint 加载与 FSDP；当梯度经过
       prefix 时提供 backward 路径。

大规模单 rank batch 使用 64 位 Triton pointer offset。batch 160 时，首个 prefix MLP
的展平输出包含 2,537,144,320 个元素，超过有符号 32 位范围；64 位 offset 修复了这个
场景中的溢出。

解读基准结果
------------

主要基准使用单机 4 张 NVIDIA H20 GPU。每组运行 9 个 step，表中统计第 4–9 step 的
均值。compile 只作用于 rollout；fused prefix 同时作用于 rollout 和 actor。

.. list-table:: LIBERO async PPO，4-GPU collocated
   :header-rows: 1
   :widths: 23 19 19 19 20

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

模型局部数据解释了端到端结果：

.. list-table:: Pi0.5 模型路径，batch 16、三步 denoise
   :header-rows: 1
   :widths: 28 24 24 24

   * - 变体
     - Rollout
     - Actor recompute
     - Rollout 相对 baseline
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

两项计算优化的覆盖范围重叠，因此局部收益不能线性相加；环境计算和 async queue 时序还会
稀释模型局部收益。工作量对齐的 sync LIBERO 测试中，compile、fused 和组合的收益分别为
**-2.99%**、**-2.66%** 和 **-5.99%**。ManiSkill 2+2 disaggregated 测试中，三者
分别为 **-6.51%**、**-3.58%** 和 **-6.64%**。

后台权重同步使用另一组单机 8 张 H20 的 async PPO 测试。runner 中阻塞的
``update_rollout_weights`` 从每 step 约 3.1 秒降到 0.0018 秒，actor training 保持在
约 46 秒。端到端 step 被 0–20 秒的 rollout 可用性抖动主导，因此该结果只能证明串行同步
区间被消除，不能用于给出精确的端到端加速比例。当 disaggregated 或跨机权重传输占据更大
step 比例时，预期收益会更明显。

检查正确性与适用范围
--------------------

fused layer 与真实 OpenPi Gemma decoder layer 在 zero mask 和 block mask 下完成了对拍。
BF16 forward、backward 和所有 parameter gradient 的相对误差均小于 ``2e-2``。完整
Pi0.5 ``get_log_prob_value`` 对拍中，log-probability 相对误差为 ``7.2e-8``，value
相对误差约为 ``1e-2``。

请把上述数字视为特定硬件和 workload 的稳态结果。placement 会改变瓶颈：ManiSkill GPU
simulation 在 collocated 模式下有明显尾延迟，而 2+2 placement 更快且更稳定。启用所有
开关前，请先 profile 目标 placement。

使用 :doc:`Profiling <../profile>` 捕获 worker 级 trace，并参考
:doc:`执行模式 <../../concepts/execution_modes>` 选择 collocated 或 disaggregated layout。
