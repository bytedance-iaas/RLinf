Run 状态契约
============

直接从磁盘读取一个 run 的状态、存活性、当前阶段、进度和最近 checkpoint ——
无需连上 driver，也无需解析 stdout。

RLinf 现有的 logger（TensorBoard、wandb、SwanLab）属于**数据面**：标量时序。
本页规定的是**控制面**：每个 run 一份的事实，用来回答「我的任务还活着吗？跑到哪了？」

.. note::

   控制面由训练 driver 写入，当前为 ``schema_version: 2``。
   它不替代任何 metric backend —— 后者见 :doc:`日志 <../logger>`。

落盘位置
--------

.. code-block:: text

   <runner.logger.log_path>/_rlinf/
   ├── runs/<run_id>/
   │   ├── manifest.json        # 不变量：run_id、task_type、git sha、命令行、placement
   │   ├── run.json             # 当前快照，原子替换
   │   ├── events.jsonl         # append-only 生命周期与阶段事件
   │   ├── heartbeat            # 极小文件；mtime 作为兜底判活信号
   │   ├── checkpoints.jsonl    # 每次保存**完成后**追加一行
   │   └── media.rank<k>.jsonl  # 视频索引，按写者分片
   └── latest -> runs/<run_id>  # 指向最近一次启动的符号链接

用文件而非数据库：零依赖、跨虚拟环境可读、崩溃后残留仍可读。
刻意不用 SQLite —— 它在 NFS 上的锁不可靠。

读 ``run.json``
---------------

``docs/schemas/run.v2.schema.json`` 是权威定义。最关键的字段：

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 字段
     - 含义
   * - ``state``
     - ``pending`` · ``running`` · ``finished`` · ``failed`` · ``stopped``。
       仅代表写侧事实。
   * - ``phase``
     - driver 最内层的活跃阶段（``rollout``、``train``、``eval`` 等），
       由 ``ScopedTimer`` 的 scope 推导而来。
   * - ``components``
     - 异步 runner 下 ``env`` / ``rollout`` / ``actor`` 各自是否活跃。
       单个 ``phase`` 无法表达三个组件并发。
   * - ``heartbeat_at``
     - 后台心跳线程最后一次 tick。只能证明**进程**活着，仅此而已。
   * - ``last_progress_at``
     - ``step`` 最后一次推进的时刻。证明**训练循环**活着。
   * - ``last_metric_at``
     - 指标最后一次到达 backend 的时刻。
   * - ``progress``
     - ``step`` / ``max_steps`` / ``epoch``，以及 ``step_semantics``。
   * - ``timing``
     - ``elapsed_s``、``step_time_p50``、``eta_s``、``eta_confidence``。
   * - ``latest_checkpoint``
     - 镜像 ``checkpoints.jsonl`` 的最后一行。
   * - ``exit``
     - 仅 ``failed`` / ``stopped`` 时非 null：原因与 traceback 尾部。

推导存活性
----------

``state`` 刻意**不含** ``stalled`` 取值：**写侧死了就写不了自己的死讯。**
driver 被 ``kill -9`` 后，``run.json`` 会永远冻结在 ``running``。
因此存活性是*读侧*基于三个时间戳的判定：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 判定
     - 条件
   * - ``unreachable``
     - ``heartbeat_at`` 超过 ``k × max(step_time_p50, floor)``。进程已消失。
   * - ``degraded``
     - 心跳新鲜但 ``last_progress_at`` 陈旧。进程活着而训练循环没有 ——
       NCCL hang 或环境卡死。
   * - ``degraded``
     - 心跳与进度都新鲜，但 ``last_metric_at`` 陈旧。指标链路断了。
   * - ``healthy``
     - 全部新鲜，或该 run 已进入终态。

用三个时间戳而不是一个心跳，区分的正是「进程死了」与「进程好着、训练卡住了」。
后一种失败在分布式 RL 里更常见，而单心跳会把它报成健康。

.. tip::

   终态 run 本就静默。应把 ``finished`` / ``failed`` / ``stopped`` 视为健康，
   而不是让静默逐渐老化成 ``unreachable``。

跨 run 比较 step
----------------

``progress.step_semantics`` 声明一个 step 究竟指什么，因为各 runner 并不一致：

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - 取值
     - 含义
   * - ``rl_iteration``
     - 一次完整的 RL 迭代（embodied、SFT）。
   * - ``minibatch``
     - 一个 minibatch。reasoning 按 minibatch 粒度记录，
       故其 x 轴密度是 embodied run 的 ``n_minibatches`` 倍。
   * - ``optimizer_step``
     - 一个优化器步。

该字段只**如实标注**现有行为，不改变任何 runner 的 step 计数方式。
用它来标注坐标轴，避免拿不可比的 run 互相比较。

checkpoint 可见性
-----------------

只有在保存**完成后**才会向 ``checkpoints.jsonl`` 追加一行。
因此信任该文件的读者永远不会看到半写完的 checkpoint，
也就不需要额外的 ``WRITING`` / ``READY`` 协议。

每行包含 ``step``、``path``、``saved_at``、``size_bytes``、``duration_s``、
``is_best``、该步的指标，以及结构化字段 ``resume_dir`` / ``entry_script`` /
``config_name``。请用这些字段拼出 resume 命令，而不要存一个预先拼好的
shell 字符串 —— 那种字符串会过期。

人工验证步骤
------------

不启动训练也能验证契约本身 —— fixture、schema 与存活性推导：

.. code-block:: bash

   pytest tests/unit_tests/test_run_state_contract.py -v

再对着真实运行的 job 验证：

.. code-block:: bash

   # 1. 观察一个短 job 推进。
   watch -n1 cat <log_path>/_rlinf/latest/run.json

   # 2. 杀掉 driver，确认读侧判定为 unreachable：
   #    heartbeat_at 停止推进，而 state 仍是 "running"。
   kill -9 <driver_pid>

   # 3. 确认失败会记录原因。
   #    state == "failed" 且 exit.reason 非空。
   python -c "import json;print(json.load(open('<log_path>/_rlinf/latest/run.json'))['exit'])"

用 schema 校验任意快照：

.. code-block:: bash

   python -c "
   import json, jsonschema
   schema = json.load(open('docs/schemas/run.v2.schema.json'))
   jsonschema.validate(json.load(open('<path>/run.json')), schema)
   print('valid')"

schema 版本管理
---------------

``schema_version`` 是整数，目录布局或 ``run.json`` 发生破坏性变更时递增。
版本 2 新增了三时间戳存活模型、``components``、结构化 resume 字段和
``step_semantics``。

训练侧与任何读侧都对同一份已提交的 schema 做校验。
``tests/fixtures/run_state/`` 下的 fixture 由两侧原样共用，
使两个实现无法在无人察觉的情况下漂移。
