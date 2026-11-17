启动与扩展
==========

当你需要在单机默认运行之外启动任务，或将 RLinf 接入真实硬件时，使用这些指南。

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 指南
     - 内容
   * - :doc:`多节点训练 <../multi_node>`
     - 在多个节点上启动训练。
   * - :doc:`异构集群 <../hetero>`
     - 配置混合硬件与节点分组。
   * - :doc:`远程渲染 <../remote_render>`
     - 物理仿真留在本地，把 MuJoCo 仿真器的渲染放到 GPU 机器上。
   * - :doc:`远程仿真 <../remote_env>`
     - 把整个仿真器放到 GPU 机器上运行，例如在昇腾 NPU 主机上训练 ManiSkill。
   * - :doc:`云边协同 <../cloud_edge>`
     - 在云端与边缘之间拆分推理与训练。
   * - :doc:`真机机器人 <../realworld_robot>`
     - 在真实机器人硬件上运行 RL。
   * - :doc:`遥操作 <../teleoperation>`
     - 让操作者在 rollout 中途从策略手里接管。

.. toctree::
   :hidden:

   多节点训练 <../multi_node>
   异构集群 <../hetero>
   远程渲染 <../remote_render>
   远程仿真 <../remote_env>
   云边协同 <../cloud_edge>
   真机机器人 <../realworld_robot>
   遥操作 <../teleoperation>
