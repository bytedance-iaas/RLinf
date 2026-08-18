基于 SO101 机械臂的强化学习（Real2Sim）
=======================================

`SO101 <https://github.com/TheRobotStudio/SO-ARM100>`__（SO-ARM101）是一款低成本的
6 自由度开源机械臂。本示例依据实测几何与录制的 `LeRobot
<https://github.com/huggingface/lerobot>`__ 数据集，在 ManiSkill 中重建了\ **真实**\ 的
SO101 抓取摆放工作台，并在其上对 OpenPI π₀.₅ 策略做强化学习微调。

与其他 ManiSkill 配方不同，本示例工作在\ **关节空间**\ ：机械臂由绝对关节位置驱动
（``pd_joint_pos``），策略输出的 6 维动作直接对应
``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]``。

总览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅（OpenPI）

   .. grid-item-card:: 算法
      :text-align: center

      PPO

   .. grid-item-card:: 任务
      :text-align: center

      抓取红色方块 → 放入托盘 → 回到初始位

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 4–8 GPU

| **你将完成：** 安装依赖 → 下载 ManiSkill 资产与 SO101 π₀.₅ SFT 检查点 → 先决条件探针 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>`，使用 ``--model openpi --env maniskill_libero``。

任务
~~~~

``SO101GrabRedCube-v1`` 复现了真实工作台：浅棕色木板上放置一个 2.9 cm 的红色方块，
一个蓝色方块作为视觉干扰物，木板旁是一个开口托盘。成功需要\ **同时**\ 满足两个条件，
与真实演示的语义一致：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 条件
     - 定义
   * - 方块进入托盘
     - 红色方块在托盘内部被释放（仅悬停在其上方不算）
   * - 机械臂回到初始位
     - 5 个手臂关节相对实测初始位姿的平均偏差小于 0.08 rad

初始位姿及其容差取自录制的真实回合，而非人工选定。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 字段
     - 形状 / 含义
   * - ``main_images``
     - ``[B, 480, 640, 3]`` 前视相机，内参由真实画面实测得到
   * - ``wrist_images``
     - ``[B, 480, 640, 3]`` 安装于 ``Fixed_Jaw`` 的腕部相机
   * - ``states``
     - ``[B, 6]`` 关节位置，单位为 LeRobot 归一化电机单位
   * - actions
     - ``[B, chunk, 6]`` 绝对关节目标，单位为 LeRobot 归一化电机单位

单位：检查点使用 LeRobot **归一化电机单位**\ （手臂 ``[-100, 100]``，夹爪
``[0, 100]``），而 ManiSkill 使用\ **弧度**\ 。换算实现在
``rlinf/envs/maniskill/so101_calib.py``，由从动舵机标定推导而来（Feetech，
4096 tick/圈），并在环境边界处应用：动作在
``action_utils.prepare_actions_for_maniskill``，观测在
``ManiskillEnv._wrap_obs``\ （由 ``so101_state_norm: True``\ 开启）。

.. note::

   夹爪\ **刻意不使用**\ 手臂的 tick→弧度换算。真实平行夹爪与仿真中的旋转爪连杆几何
   完全不同，套用手臂公式会让爪口停在约 8 cm 的开度上，而目标方块只有 2.9 cm，
   策略永远无法夹住它。夹爪改用单独实测的映射，闭合 1.3 cm、张开 11.2 cm。

机器人模型
~~~~~~~~~~

本任务使用 ManiSkill 内置 SO100 的放宽限位变体。原始 URDF 低估了真实硬件的行程
——它把 ``shoulder_lift`` 与 ``elbow_flex`` 都限制在 ±1.5708 rad，而真实从动臂可达
+2.48 与 −2.38——在放宽限位之前，仿真机械臂根本够不到托盘。

放宽后的 URDF 是\ **在 import 时从已安装的 ManiSkill URDF 派生**\ 的
（``rlinf/envs/maniskill/so101_agent.py``），而不是作为资产随仓库分发。限位取自
``so101_calib.JOINT_LIMITS_{LOW,HIGH}``——与动作换算裁剪所用的是同一张表——因此两者
不可能产生漂移，且全新检出无需任何额外下载。

安装
----

SO101 任务不需要标准 ManiSkill + OpenPI 环境之外的任何依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env maniskill_libero

按 :doc:`ManiSkill <maniskill>` 的说明下载 ManiSkill 资产，然后将配置指向 SO101 的
π₀.₅ SFT 检查点。强化学习的起点必须是在仿真中已具备非零成功率的策略——PPO 放大成功，
但不会凭空发现成功。

运行
----

.. code-block:: bash

   cd examples/embodiment
   bash run_embodiment.sh so101_ppo_openpi_pi05

在投入长时间训练之前，先用冻结策略在 **rollout 分布下**\ 做探针——确定性评测与带噪
rollout 的差距可以超过 50 个点，而 PPO 真正学习的是带噪的那个分布：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-path $PWD/examples/embodiment/config/ \
     --config-name so101_ppo_openpi_pi05 \
     runner.max_epochs=1 runner.val_check_interval=1 \
     actor.optim.lr=1e-9 actor.optim.value_lr=1e-9

要求 ``env/success_once`` ≥ 5%。历史上能够放大的运行起始值都落在 5–15% 区间，
而从未起来的运行起始值为 0.5–1.0%。这是必要条件，但并不充分。

.. warning::

   不要用 ``runner.only_eval=True`` 当探针。它会同时切换模型规格的来源并跳过训练
   环境的创建，走的是与训练不同的另一条代码路径。

关键配置
~~~~~~~~

有三项设置偏离了 π-RL 的默认配方。每一项都在本任务上做过对照实验，且默认组合在这里
已知会崩塌（确定性评测在第 9 步跌到约 0），因此应把它们视为任务要求而非调参旋钮：

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - 设置
     - 取值
     - 原因
   * - ``actor.model.num_action_chunks``
     - ``10``
     - 等于检查点的 ``action_horizon``。只执行 5 个会丢弃一半预测并使重规划频率翻倍；
       改用 10 单此一项就带来约 11 个点的确定性评测收益，因此评测与真机部署同样应当
       使用 10。
   * - ``actor.model.openpi.noise_logvar_range``
     - ``[0.02, 0.04]``
     - ``flow_noise`` 真正读取的字段。``noise_params`` 只供 ``flow_sde`` 使用，在
       ``noise_method: flow_noise`` 下不起任何作用。
   * - ``actor.global_batch_size``
     - ``4096``
     - 等于每轮产生的样本数
       （``num_envs × max_episode_steps / num_action_chunks × rollout_epoch``），
       即\ **每轮恰好一次更新**\ 。默认的 12 次更新会让本任务崩塌：相同起点、相同噪声、
       相同块长下，12 次更新从 61.7% 跌到 7.0%，而 1 次更新爬升到 73.4%。本任务每
       回合有 64 次带噪决策，而参考基准只有 16 次。

方块的出生范围由 ``env.*.init_params.spawn_mode`` 与 ``spawn_frac`` 控制，默认为整块
木板。可用 ``spawn_frac: [0.4294, 0.9115, 0.5142, 0.9817]`` 选择课程区域；不同出生
区域下的成绩不可比较，因此报告数字时务必说明它来自哪一种。

评测
----

.. code-block:: bash

   cd examples/embodiment
   bash run_embodiment.sh so101_eval_openpi_pi05

关注 ``eval/success_once``\ （确定性）与 ``env/success_once``\ （带噪 rollout）。两者应
分开报告：在固定评测集上挑出的峰值，相对于从未使用过的新种子是偏乐观的——本任务中
同一个检查点在前者上得到 73.4%，在后者上是 57.8%。

验证环境是否就绪
----------------

仿真器层面的冒烟测试会检查所有不需要策略检查点的部分——派生出的 URDF、两个相机、
关节空间观测、NumPy 与 Torch 两条换算路径的一致性，以及爪口几何：

.. code-block:: bash

   python -m toolkits.so101_smoke

换算契约的单元测试无需仿真器即可运行：

.. code-block:: bash

   pytest tests/unit_tests/test_so101_maniskill.py
