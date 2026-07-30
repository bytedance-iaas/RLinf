# RLinf 执行模式：Sync/Async 与 Colocated/Disaggregated

> 概念说明文档。与本目录下的 `PI05_RL_PERF_*` 实测报告互补：这里讲清"是什么、怎么选"，
> 那些文件给出具体数字。文中引用的单步时间均来自本目录的 Pi0.5 PPO + LIBERO 实测。

## 一、两个正交的维度，别混为一谈

RLinf 的执行模式由**两个相互独立的维度**决定，这是理解全局的前提：

- **时间维度（Sync vs Async）**：采样和训练在**时间上**是串行还是流水线并行。
  由**入口脚本 + runner** 决定（`examples/embodiment/train_embodied_agent.py` 走同步，
  `examples/embodiment/train_async.py` 走异步）。
- **空间维度（Colocated vs Disaggregated）**：actor / rollout / env 三类 worker 在
  **GPU 空间上**是共享同一批卡还是各占专属卡。由 **`cluster.component_placement`** 配置决定。

这两个维度**可以自由组合**——sync+colocated、async+colocated、async+disaggregated 等。
下面分别讲清每个维度，再讲组合选型。

## 二、Sync（同步）模式

同步模式由 `EmbodiedRunner`（`rlinf/runners/embodied_runner.py`）驱动，训练循环是**严格串行的单线**：

```
for step in range(max_steps):
    权重同步（actor → rollout）
    → generate_rollouts：env.interact + rollout.generate 采一整批轨迹
    → actor 接收轨迹、计算 advantage/returns
    → actor.run_training() 做梯度更新
    → step += 1
```

每个环节都 `.wait()` 阻塞到完成才进下一步。

- **优点**：简单、确定、数值最"干净"。训练用的永远是最新一版权重采出的 on-policy 数据，
  无陈旧度问题，调试和复现最容易。
- **代价**：资源利用率低。采样时训练卡在等，训练时采样卡在等，GPU 总有一半时间空转。
- **变体**：`runner.use_training_pipeline=True` 能把部分环节重叠以缓解，但本质仍是同步范式。

## 三、Async（异步）模式

异步模式由 `AsyncPPOEmbodiedRunner`（PPO 路径，`rlinf/runners/async_ppo_embodied_runner.py`）驱动，
结构完全不同：**env、rollout、actor 三个 worker 长驻常开**，各跑自己的无限循环，
通过 **`Channel`**（Ray 之上的队列）串成流水线：

```
env.interact  ──trajectory──▶ rollout.generate ──▶ actor.recv_rollout_trajectories
   (长驻)                        (长驻)                    (长驻，持续训练)
```

三者**同时在跑**：actor 训练第 N 版权重的同时，rollout 已在用第 N 版权重采下一批，env 并行推进仿真。
解耦的关键旋钮是 **`algorithm.staleness_threshold`**——允许 actor 消费"稍陈旧"（落后几个版本）的轨迹，
不必等最新数据，让流水线不断流。

- **优点**：吞吐高。采样与训练在时间上重叠，GPU 空转大幅减少。实测 async 稳态里
  `wait_for_rollout_store_ready≈0`，即流水线打满、两侧互相掩盖。
- **代价**：引入 off-policy 陈旧度，靠 `staleness_threshold` 和重要性权重裁剪
  `behave_weight_threshold` 控制；数值行为比同步复杂。

## 四、Colocated（合置）模式

合置由 `component_placement: actor,env,rollout: all`（或同一 GPU 区间）配置：
**三类 worker 共享同一批 GPU**，在这批卡上**时分复用**——训练时做训练，采样时做采样。

- **决定性优势**：训练和采样轮流独占**全部**卡，等效两者都能用到接近满配的算力
  （`step ≈ max(T_train(全部卡), T_sample(全部卡))`）。
- actor 与 rollout 在同一张卡上，权重同步走 **cudaIPC**，几乎零成本
  （实测 colocated `update_rollout_weights≈0`）。
- **实测**：4 卡 72.7s/step、8 卡 85.7s/step，在各自规模上均为**最优**。
- **判断**：单机（≤8 卡）场景，colocated 是默认最优选择。

## 五、Disaggregated（分置）模式

分置给每个组件指定**互不相交的 GPU 区间**（如 `actor: 0-3 / rollout: 4-7`）：
**每类 worker 独占专属卡**，物理隔离。

- 把总卡数**切分**给各组件，单步 `step ≈ max(T_train(actor卡), T_sample(rollout卡))`。
  总卡固定时，切分使训练和采样都拿不到全部卡，`max` 项必然大于 colocated 的时分复用——
  **这是实测 4 卡 disagg 2+2（99.3s）、8 卡 disagg 4+4（105.4s）都输给 colocated 的根因**。
- 权重同步不能用 cudaIPC，改走 collective broadcast：同机 NVLink 上的 NCCL 仍很快（≈0），
  跨机走网络才有明显开销。
- **真正价值在规模化与异构**：
  1. 组件**独立扩展**（训练是 compute-bound、采样是 latency-bound，按需配比）；
  2. 支持**跨机异构**（actor 放数据中心卡、渲染放消费卡，用 `node_group` 隔离；
     不同型号 GPU 不能同 NCCL 组的约束也天然满足）；
  3. 配合 async 的 staleness + 权重同步重叠（`actor.sync_weight_no_wait`），把跨机通信藏进训练。
- **判断**：单机用 colocated；到跨机（≥16 卡）、需要异构算力池、或组件需独立伸缩时，才转向 disaggregated。

## 六、组合与选型速查

| 组合 | 适用场景 | 实测/判断 |
|---|---|---|
| **Sync + Colocated** | 调试、追求数值可复现、算法验证 | 最简单干净，吞吐最低 |
| **Async + Colocated** | 单机（≤8 卡）生产训练 | **实测最优**（4 卡 72.7s / 8 卡 85.7s） |
| **Async + Disaggregated** | 跨机、异构算力池、组件独立伸缩 | 单机输给 colocated；跨机才是主场 |
| Sync + Disaggregated | 较少用 | 兼具两者缺点，一般不选 |

**一句话选型**：数值优先/调试选 Sync，吞吐优先选 Async；单机选 Colocated，跨机或异构选 Disaggregated。
二者正交，按"时间怎么排、空间怎么分"两个问题分别决策即可。

## 附：源码锚点

| 概念 | 位置 |
|---|---|
| Sync runner | `rlinf/runners/embodied_runner.py`（`EmbodiedRunner.run`） |
| Async PPO runner | `rlinf/runners/async_ppo_embodied_runner.py`（`AsyncPPOEmbodiedRunner`） |
| 入口分派 | `examples/embodiment/train_embodied_agent.py` / `train_async.py` |
| Placement 解析 | `rlinf/utils/placement.py`（`HybridComponentPlacement`） |
| Channel 数据流 | `rlinf/scheduler/channel/` |
| 权重同步（cudaIPC/NCCL 三级自适应） | `rlinf/scheduler/collective/collective_group.py` |
| 通信后端选择（NCCL/Gloo/Ray） | `rlinf/scheduler/collective/multi_channel_pg.py` |
