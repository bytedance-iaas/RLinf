# RLinf 里 PPO 的代码级全流程（以 SO101 + PI0.5 为例）

所有文件路径与行号来自 2026-08-13 的实读（分支 `henryg/pi05-maniskill-so101`）。
配置以 `examples/embodiment/config/so101_ppo_v11.yaml` 为例。

---

## 0. 一句话概览

三组 worker 并行常驻，runner 按轮驱动：

```
EnvWorker      仿真 64 路环境，执行动作块，产出 观测/奖励/done
RolloutWorker  策略推理，产出动作块 + logprob + value
ActorWorker    收轨迹 → 算优势 → PPO 更新 → 把新权重同步回 rollout
```

一轮（runner 里的一个 `_step`）= 同步权重 → 采样 → 算优势 → 训练。

---

## 1. 启动：`examples/embodiment/train_embodied_agent.py`

| 行 | 做什么 |
|---|---|
| 38 | `main(cfg)`，hydra 组装配置 |
| 45 | `HybridComponentPlacement(cfg, cluster)` —— 按 `cluster.component_placement` 决定三组 worker 各占哪些 GPU（我们是 `actor,env,rollout: all`，即三者共置在 8 卡上） |
| 97 | `actor_group = actor_worker_cls.create_group(cfg).launch(...)` |
| 103 | `rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(...)` |
| 109 | `env_group = EnvWorker.create_group(cfg).launch(...)` |
| 155 | `runner = EmbodiedRunner(...)`，随后 `runner.run()` |

**关键**：`create_group(cfg).launch(cluster, placement_strategy=...)` 是 RLinf 的 worker 抽象，底层是 Ray actor。三组 worker 之间用 `Channel` 传数据，不走磁盘。

---

## 2. 主循环：`rlinf/runners/embodied_runner.py`

`def run(self)` 在 **478 行**，循环体（484–535）：

```python
478  def run(self):
484      for _step in range(start_step, self.max_steps):
486          self.actor.set_global_step(self.global_step)
487          self.rollout.set_global_step(self.global_step)
497          with self.timer("step"):
498              with self.timer("sync_weights"):
499                  if _step % self.weight_sync_interval == 0:
500                      self.update_rollout_weights()          # ← 见 §6
501              with self.timer("generate_rollouts"):
502                  env_handle    = self.env.interact(...)     # ← 见 §3
508                  rollout_handle = self.rollout.generate(...) # ← 见 §4
518                  self.actor.recv_rollout_trajectories(...)   # 收轨迹
526              with self.timer("cal_adv_and_returns"):
528                  self.actor.compute_advantages_and_returns().wait()   # ← 见 §5
532              actor_training_handle = self.actor.run_training()        # ← 见 §5
```

注意 502 与 508：`env.interact` 和 `rollout.generate` **同时挂起**，两者通过 Channel 互相喂数据（env 发观测、rollout 回动作），是流水线而非串行。

评测在 **193 行** `def evaluate(self)`：`env.evaluate()` + `rollout.evaluate()`，按 `runner.val_check_interval` 触发。

---

## 3. 环境侧：`rlinf/workers/env/env_worker.py`

| 行 | 做什么 |
|---|---|
| 146–154 | `n_train_chunk_steps = max_steps_per_rollout_epoch // num_action_chunks`（我们：640 / 5 = **128 个动作块**每回合） |
| 1250 | `async def interact(...)` —— 训练采样入口 |
| 1265 | `for env in self.env_list:` 逐个 stage |
| 460 | `self.env_list[stage_id].chunk_step(chunk_actions)` —— **真正推进仿真的一步**，一次执行 5 个动作 |
| 1271 | `def evaluate(...)`，评测路径；1280 行 `reset()` |

`chunk_step` 在 `rlinf/envs/maniskill/maniskill_env.py`：把 `[B, num_action_chunks, action_dim]` 拆开逐步 `env.step()`，累积奖励与 done，返回下一段观测。

**SO101 特有的两处**：

- `rlinf/config.py:1068` —— `elif "so100" in robot or "so101" in robot:` → `control_mode = "pd_joint_pos"`；
- `rlinf/envs/action_utils.py:29` —— `if "so100" in policy or "so101" in policy:` → `norm_to_rad(raw_chunk_actions)`，把 LeRobot 归一化单位换成弧度。**漏了这步机械臂会直接打到关节限位。**

奖励在 `rlinf/envs/maniskill/tasks/so101_pick_place.py::compute_dense_reward`（385 行起），阶梯见 `V10_REPRODUCTION_ZH.md` §F.2。

---

## 4. 推理侧：`rlinf/workers/rollout/hf/huggingface_worker.py`

| 行 | 做什么 |
|---|---|
| 760 | `async def generate(...)` —— 训练采样入口 |
| 672 | `async def generate_one_epoch(...)`：`for _ in range(self.n_train_chunk_steps)` 循环 128 次，每次 `recv_from` 观测 → 预测 → `send_to` 动作 |
| 518/524 | `predict_action_batch(env_obs=..., mode=...)`（DAgger 模式下可能走 expert 分支） |
| 778 | `async def evaluate(...)` —— 评测入口，`mode="eval"` |
| 624 | `async def sync_model_from_actor(self)` —— 接收 actor 推来的新权重 |

### 4.1 动作是怎么采样出来的：`rlinf/models/embodiment/openpi/openpi_action_model.py`

| 行 | 做什么 |
|---|---|
| 831 | `def predict_action_batch(self, env_obs, mode="train", compute_values=True, ...)` |
| 838–845 | obs 处理链：`obs_processor` → `input_transform`（归一化，用 norm_stats）→ `precision_processor` → `Observation.from_dict` |
| 880 起 | 非 DSRL 分支：`outputs = self.sample_actions(observation, mode=mode, compute_values=compute_values)` |
| 999 | `collect_nft_state = self.config.is_nft and mode == "train"` |
| 1000–1019 | **噪声开关的真正位置**：<br>`if mode == "train":` → `denoise_inds` 取真实去噪步索引（`joint_logprob=True` 时为 `arange(num_steps)`）<br>`else:` → `denoise_inds = [-1] * num_steps` |
| 1023–1029 | 逐去噪步：`if idx == denoise_inds[0][idx]: sample_method = self.config.noise_method` —— 评测时 `denoise_inds` 全是 −1，`idx` 从 0 起，**永远不相等**，于是 `sample_method` 保持 `flow_ode`（确定性） |
| 1038 | `x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std` —— 只有训练时才走到这里 |
| 191–199 | `if self.config.noise_method == "flow_noise": self.noise_head = ExploreNoiseNet(...)` —— **可学习的噪声网络**，`noise_params` 是它的起止与退火步数 |

> **结论（读代码得出，非推测）**：`mode="eval"` 时不注入探索噪声；`mode="train"` 时按 `noise_params` 注入。所以"训练 rollout 成功率"和"评测成功率"是两个不同分布下的量，**先决条件必须用前者判断**。

---

## 5. 训练侧：`rlinf/workers/actor/fsdp_actor_worker.py`

### 5.1 收轨迹 → 算优势

| 行 | 做什么 |
|---|---|
| 1196 | `async def recv_rollout_trajectories(self, input_channel)` |
| 1296 | `def compute_advantages_and_returns(self)` |
| 1320 | `advantages_and_returns = calculate_adv_and_returns(**kwargs)` → 走注册表 |

批数据在进优势计算前会过 `rlinf/utils/utils.py:763 preprocess_embodied_batch(...)`，其中 **776 行 `batch = merge_rollout_epochs(batch, rollout_epoch)`** 把 `[rollout_epoch, B, ...]` 展平成 `[rollout_epoch*B, ...]`。

> **这就是"每轮样本数 = 环境数 × 块数 × rollout_epoch"的出处。**
> 我们：64 × 128 × 3 = 24,576；官方 LIBERO：64 × 48 × 8 = 24,576。二者不变量相同。

### 5.2 GAE：`rlinf/algorithms/advantages.py:25`

```python
24  @register_advantage("gae")
25  def compute_gae_advantages_and_returns(
        rewards, gamma=1.0, gae_lambda=1.0, values=None, dones=None, ...
    )
```

配置里 `algorithm.adv_type: gae` 通过 `rlinf/algorithms/registry.py::calculate_adv_and_returns` 分发到这里。`gamma 0.99 / gae_lambda 0.95`。

### 5.3 PPO 更新：`run_training`（1492 行）

关键循环（1528 起）：

```python
1531  batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
1536  update_epoch = self.cfg.algorithm.get("update_epoch", 1)
1537  for _ in range(update_epoch):                      # ← 样本复用次数
1542      for train_global_batch in rollout_dataloader_iter:   # ← 每个 global batch 一次更新
1544          train_global_batch_size = ...
1550          assert train_global_batch_size % micro_batch_size == 0
              # 内层按 micro_batch 切分做梯度累积
1686          loss, metrics_data = policy_loss(**loss_kwargs)
```

**每轮更新次数 = (样本数 ÷ global_batch_size) × update_epoch**：

- 我们 / 官方 LIBERO：24,576 ÷ 2048 × 1 = **12 次**，每条样本用 1 次
- 官方 ManiSkill：5,120 ÷ 5,120 × 5 = **5 次**，每条样本用 5 次

`micro_batch_size` 只切分梯度累积粒度，**不改变更新次数，也不改变数学结果**（损失按 1/累积数缩放，模型无 BatchNorm）。显存不够只能动它。

### 5.4 损失：`rlinf/algorithms/losses.py`

| 行 | 内容 |
|---|---|
| 396–397 | `@register_policy_loss("actor_critic")` → `compute_ppo_actor_critic_loss(**kwargs)` |
| 170 | `compute_ppo_actor_loss(...)` —— 裁剪目标，用 `clip_ratio_low/high`、`clip_ratio_c` |
| 315 | `compute_ppo_critic_loss(...)` —— 用 `value_clip`、`huber_delta` |

配置 `algorithm.loss_type: actor_critic` 选中 396 行那个。

价值头：`actor.model.add_value_head: True` + `openpi.value_after_vlm: True`（价值头读 VLM 输出）。价值头的学习率单列：`actor.optim.value_lr: 1.0e-4`，是策略 `lr: 5.0e-6` 的 20 倍——**官方靠这个解决"新价值头冷启动"，不需要 warmup 阶段**。

---

## 6. 权重同步：actor → rollout

| 位置 | 内容 |
|---|---|
| `embodied_runner.py:187` | `def update_rollout_weights(self)` |
| 188–189 | `rollout.sync_model_from_actor()` 与 `actor.sync_model_to_rollout()` 成对挂起 |
| `fsdp_actor_worker.py:344 / 1141` | `sync_model_to_rollout` 的两个实现（同步 / 异步） |
| `rlinf/hybrid_engines/weight_syncer/patch_syncer.py` | `WeightPatch`（138 行）、`EmptyWeightPatch`（98 行）—— 只传**变化的权重**（稀疏 patch，`as_coo_2d_view` 把张量转成 COO 视图） |

配置 `defaults: - weight_syncer/patch_syncer@weight_syncer` 选中 patch 方式；另有 `bucket_syncer.py`（整桶传输）可选。

`runner.weight_sync_interval` 控制多少轮同步一次（默认每轮）。

---

## 7. 配置到代码的映射速查

| 配置键 | 代码位置 | 作用 |
|---|---|---|
| `algorithm.adv_type` | `advantages.py` 注册表 | 选优势函数 |
| `algorithm.loss_type` | `losses.py:396` | 选损失 |
| `algorithm.update_epoch` | `fsdp_actor_worker.py:1536` | 样本复用次数 |
| `actor.global_batch_size` | `fsdp_actor_worker.py:1531` | 每次更新的样本数 → 决定每轮更新次数 |
| `actor.micro_batch_size` | 1550 行附近 | 梯度累积粒度（不改数学） |
| `env.train.rollout_epoch` | `utils.py:776` | 乘进每轮样本数 |
| `env.*.max_steps_per_rollout_epoch` ÷ `num_action_chunks` | `env_worker.py:146` | 每回合动作块数 |
| `openpi.noise_method` / `noise_params` | `openpi_action_model.py:191,1023` | 探索噪声（**仅训练模式**） |
| `add_value_head` / `value_after_vlm` | 模型构建 | 价值头 |
| `actor.optim.value_lr` | 优化器构建 | 价值头单独学习率 |
| `rollout.model` vs `actor.model` | **`rlinf/config.py:830`** | `model_cfg = cfg.rollout.model if only_eval else cfg.actor.model` —— **训练读 actor.model，只评测读 rollout.model** |

最后一行是个坑：官方配置的 `rollout.model` 只写 `model_path` + `precision`，训练时没问题；一旦 `only_eval=True`，同一份配置就会因为缺键而崩。解法是 `model: ${actor.model}` 整块继承。

---

## 8. 一轮的完整时序（把上面串起来）

```
_step 开始
├─ sync_weights          actor 把权重 patch 推给 rollout        (runner:499)
├─ generate_rollouts
│   ├─ env.interact      128 次 chunk_step，每次推进 5 个仿真步  (env_worker:460)
│   └─ rollout.generate  128 次 predict_action_batch(mode=train) (hf worker:672)
│        └─ 每次注入 flow noise，记录 logprob 与 value
│      ×  rollout_epoch=3  →  共 24,576 条样本
├─ recv_rollout_trajectories                                    (actor:1196)
├─ cal_adv_and_returns   merge_rollout_epochs → GAE            (utils:776, adv:25)
└─ run_training          update_epoch=1 × 12 个 global batch
    └─ 每个 batch 内按 micro_batch=32 梯度累积 → policy_loss  (actor:1537,1686)
每 val_check_interval 轮：evaluate()（mode=eval，无噪声）        (runner:193)
每 save_interval 轮：存检查点
```
