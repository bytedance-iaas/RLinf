# PPO 成功配方（v13）—— 详细步骤

**这是这个项目第一次让 PPO 真正放大策略。** 此前多次尝试全部失败或崩塌，失败清单见 §6。

| | 值 |
|---|---|
| 起点 | `so101_sft_v10/.../global_step_1000`（阶段 E 的 SFT 产物） |
| 产物 | `so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30`（19 GB） |
| 训练区域 | 环 1，`SO101_SPAWN_FRAC="0.4294,0.9115,0.5142,0.9817"` |
| 门评口径 | 61.7% → **73.4%**（同一套固定评测局面，峰值 @ step 29） |
| 诚实口径（从未用过的种子 4141/4242） | **57.8%**（58.6 / 57.0），起点同条件对照 52.0% → **+5.9 点** |
| 框内（legacy）参照 | **77.3%** |
| 全板参照 | **14.8%** |
| 耗时 | 约 2 小时（55 轮，每轮约 10 分钟） |

> **两个口径要分清**：
>
> | 口径 | 起点 | PPO 后 | 增益 |
> |---|---|---|---|
> | 门评（同一套固定评测局面） | 61.7% | 73.4% | +11.7 |
> | **诚实（从未用过的种子 4141/4242，同块长）** | **52.0%** | **57.8%** | **+5.9** |
>
> 门评口径偏乐观，因为峰值是在那套局面上挑出来的。**对外只引用诚实口径的 +5.9。** 起点对照用 `tools_so101_session/verify_baseline_control.sh` 单独测过（48.4/55.5%），不是估算。

---

## 1. 三个决定成败的参数

其余全部照抄 πRL 官方配方。这三个每一个都有对照实验，不是猜的。

### 1.1 `num_action_chunks`: 5 → **10**

| 指标 | chunks=5 | chunks=10 |
|---|---|---|
| 带噪 rollout 成功率 | 1.0% | **4.7%** |
| 确定性 eval | 51–55% | **66.4%** |

**为什么**：模型在 SFT 时按 `action_horizon: 10` 训练，每次预测 10 个动作；而我们一直只执行前 5 个就重新推理——**丢掉一半预测，并且过度频繁地重规划**。改成 10 之后：

- 每回合的带噪决策次数从 128 降到 64（噪声是逐决策注入的，留在 BC 窄脊上的概率是**累乘**的）；
- **确定性成绩本身涨了 11 点**——这一项与 RL 无关，是白捡的，**评测和真机部署都该用 10**。

`num_action_chunks: 5` 的来源是官方参考配置，从未针对本任务推导过。

### 1.2 `openpi.noise_logvar_range`: 默认 `[0.08,0.16]` → **`[0.02,0.04]`**

| 指标 | 默认 | `[0.02,0.04]` |
|---|---|---|
| 带噪 rollout 成功率 | 4.7% | **39.1%** |

> ⚠️ **不要去调 `noise_params`**。看 `rlinf/models/embodiment/openpi/openpi_action_model.py:47-58`：
> ```python
> noise_method: str = "flow_sde"
> # noise config for flow-sde
> noise_params: list = [0.7, 0.3, 400]
> # noise config for flow-noise
> noise_logvar_range: list = [0.08, 0.16]
> ```
> 我们的 `noise_method` 是 `flow_noise`，**它不读 `noise_params`**。我曾用它扫了 4 个档位（跨度 8 倍），结果几乎完全一样——因为调的是空参数。
> 连带纠正一条历史记载：早期"把噪声减半"那条经验**从来没有生效过**——成功与失败的历史运行，实际噪声完全相同。

### 1.3 每轮更新次数: 12（官方）→ **1**

**这是决定性的那一个。** 同样的起点、同样的噪声与块长，只改这一项：

| 每轮更新次数 | 结果 |
|---|---|
| 12（官方配方） | step 4 从 61.7% 掉到 35.9%，step 9 掉到 **7.0%**，被守卫自动停机 |
| **1** | 稳定爬到 **73.4%** |

**怎么算**：

```
samples  = num_envs × (max_episode_steps ÷ num_action_chunks) × rollout_epoch
updates  = samples ÷ global_batch_size
```

要做到 1 次/轮，令 `global_batch_size = samples`：

```
64 环境 × (640 ÷ 10) × rollout_epoch 1 = 4096 样本
global_batch_size = 4096         → 1 次更新/轮
per-rank 512 ÷ micro_batch 32     = 16 次梯度累积
```

**推测的原因**：本任务每回合 64 次带噪决策，官方 ManiSkill 基准只有 16 次、LIBERO 48 次。同样的更新强度作用在更长时域、更脆的 BC 脊上。官方配方在它自己的基准上成立，**不能照搬到长时域任务**。

---

## 2. 启动前必做：冻结探针

**先决条件必须在 rollout 分布下测，不是确定性评测下测。** 这两个数可以差 50 个点以上。

用**冻结测试**：跑真实训练路径，`lr=1e-9`，权重实质不变，一轮就同时给出两个数。

```bash
export SO101_SPAWN_FRAC="0.4294,0.9115,0.5142,0.9817"
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=$RESULTS/probe \
  runner.val_check_interval=1 runner.save_interval=1000 runner.max_epochs=1 \
  actor.optim.lr=1e-9 actor.optim.value_lr=1e-9 \
  env.train.rollout_epoch=1 \
  actor.model.num_action_chunks=10 actor.global_batch_size=4096 \
  "+actor.model.openpi.noise_logvar_range=[0.02,0.04]"
```

读 tensorboard：

| 标签 | 含义 | 门槛 |
|---|---|---|
| `env/success_once` | **带噪 rollout**——PPO 真正学习的分布 | **≥ 5%** |
| `eval/success_once` | 确定性评测 | 不应比起点低太多 |

5% 这条线来自本项目四次历史运行的实测分界：**两次放大成功的起始带噪成功率是 5–15%，两次从未起来的是 0.5–1.0%**。**必要不充分**——本次探索中有一个变体带噪成功率 39% 仍然崩塌，因为更新次数不对。

**不要用 `runner.only_eval=True` 当探针**：它同时切换模型规格来源（`rlinf/config.py:830`）并跳过训练环境创建（`env_worker.py:108`、`huggingface_worker.py:70`），是另一条代码路径。我为此连撞三次才换用冻结测试。

---

## 3. 正式启动

配置文件 `examples/embodiment/config/so101_ppo_v11.yaml`（= πRL 官方配方，起点换成 v10），命令行只覆盖 §1 那三项：

```bash
export SO101_SPAWN_FRAC="0.4294,0.9115,0.5142,0.9817"
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=$RESULTS/so101_ppo_v13 \
  runner.val_check_interval=5 runner.save_interval=5 runner.max_epochs=300 \
  actor.model.num_action_chunks=10 \
  env.train.rollout_epoch=1 \
  actor.global_batch_size=4096 \
  actor.optim.lr=2e-6 \
  "+actor.model.openpi.noise_logvar_range=[0.02,0.04]"
```

带自动停机守卫的启动器：`tools_so101_session/ppo_train.sh`（推荐用它，见 §4）。

**其余参数**（全部来自官方配方，未改）：

| 参数 | 值 | 说明 |
|---|---|---|
| `update_epoch` | 1 | 每条样本只用一次 |
| `lr` / `value_lr` | **2e-6** / 1e-4 | lr 由官方 5e-6 降下来配合 1 次/轮；value_lr 保持官方——价值头靠它自己快速学，**不需要 warmup 阶段** |
| `clip_ratio_low/high` | 0.2 / 0.2 | |
| `value_clip` / `huber_delta` | 0.2 / 10.0 | |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | |
| `entropy_bonus` | 0 | 熵由 flow 噪声提供 |
| `adv_type` / `loss_type` | gae / actor_critic | |
| `env.train.total_num_envs` | 64 | |
| `env.*.ignore_terminations` | True | 成功即终止会让"成功"反而放弃后续稠密奖励 |
| `micro_batch_size` | 32 | 显存；纯梯度累积，不改数学 |
| `add_value_head` / `value_after_vlm` | True / True | |
| norm_stats | v4 那份 | 血统冻结，全程不变 |

---

## 4. 自动停机守卫（必须写在启动器里）

**不能写在会话里**——会话结束守卫就没了，历史上因此白烧过 180 轮。

`tools_so101_session/ppo_train.sh` 每 5 分钟读一次 tensorboard 的 `eval/success_once`：

| 规则 | 条件 | 动作 |
|---|---|---|
| 崩塌 | 某次评测低于历史峰值 **20 点** | 立即杀训练 |
| 无收益 | 连续 **3 次**低于第一次评测 5 点以上 | 停 |

本次探索中那个 12 次更新/轮的变体，就是被第一条规则自动停掉的（step 9，7.0% vs 峰值 35.9%）。

**采收律**：`save_interval` 必须 ≤ `val_check_interval`，否则峰值检查点存不下来。峰值就是交付物。

---

## 5. 实测曲线

| step | 4 | 9 | 14 | 19 | 24 | **29** | 34 | 39 | 44 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval | 60.9 | 58.6 | 62.5 | 58.6 | 70.3 | **73.4** | 69.5 | 70.3 | 68.0 | 68.8 |

前 20 轮在 58–62% 徘徊（**不要在这里判死刑**），step 24 起上台阶，step 29 到峰值，之后进入 68–70% 平台。在 step 55 附近手动停机采收。

训练侧健康指标：

| 指标 | 值 | 读法 |
|---|---|---|
| `env/success_once` | 30–41% | 带噪成功率没塌，一直有成功样本 |
| `approx_kl` | 0.010–0.014 | 策略移动平稳 |
| `value_loss` | 360–470 | 高但平稳（长回合、大回报量级），无发散 |

---

## 6. 失败配置对照表（不要重走）

| # | 配置 | 带噪 rollout | 结果 |
|---|---|---|---|
| 1 | 真机 SFT 起点直接 RL | — | 独立评测 **0.0%**；两次运行分别为全程 0 与峰值 3–4% |
| 2 | 仿真 SFT 起点 + 默认参数 | — | 起点约 50%，**10 轮内塌到 0**，三次重现 |
| 3 | 冻结测试 `lr=1e-9` | — | 行为存活 66 轮 → 凶手是**更新步骤本身** |
| 4 | 早期手搓的保守参数组 | — | 46.9% → 75.0%（第 100 轮），随后衰减到 10.9%（第 320 轮）——先升后降 |
| 5 | 高起点（81.6%）上再用同一套保守参数 | — | **30 轮掉 53 点** |
| 6 | chunks=5、默认噪声、12 次更新/轮（官方配方原样） | 1.0% | step 9 时 eval 归零 |
| 7 | chunks=10、logvar 0.02/0.04、**12 次更新/轮** | 39.1% | step 4 掉到 35.9%，step 9 掉到 7.0%，自动停机 |
| **8** | **chunks=10、logvar 0.02/0.04、1 次更新/轮** | **39.1%** | **61.7% → 73.4%** ✅ |

第 6 与第 7 的对比说明：**先决条件（带噪成功率）修好了也不够**；第 7 与第 8 的对比说明：**更新次数是那个决定性变量**。

---

## 7. 验证与归档

```bash
# 诚实验证：从未用过的种子，且用它训练时的 chunks=10
bash tools_so101_session/verify_honest_seeds.sh
# 起点对照（证明增益不是评测集选择效应）
bash tools_so101_session/verify_baseline_control.sh
```

**产物清单**：

| 内容 | 路径 |
|---|---|
| 峰值检查点 | `results/so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30` |
| 配置 | `examples/embodiment/config/so101_ppo_v11.yaml` |
| 启动器（含守卫） | `tools_so101_session/ppo_train.sh` |
| 探针编排器 | `tools_so101_session/ppo_param_search.sh` |
| 冻结测试模板 | `tools_so101_session/ppo_freeze_probe.sh` |
| 诚实验证 / 起点对照 | `tools_so101_session/verify_honest_seeds.sh` / `verify_baseline_control.sh` |
| 代码级流程说明 | `PPO_CODE_WALKTHROUGH_ZH.md` |

---

## 8. 三条可迁移的教训

1. **先决条件要在 rollout 分布下测**（`env/success_once`），不是确定性评测下测。两者可以差 50 点以上；冻结探针一轮就能同时拿到。
2. **调参数前先确认它是否生效**。`noise_params` 对 `flow_noise` 是空参数——我扫了一整晚才发现，而代码注释就写在那里。
3. **官方配方不能整体照搬**。它在自己基准的时域上标定（16–48 次决策/回合），本任务是 64–128 次；**每轮更新次数**这一项必须重新标定，其余可以照抄。
