# SO101 + PI0.5：从真机数据集到 RL 微调策略的完整可复现流程

**输入**：`henry-guo/so101-pick-place-v2`（87 集真机遥操作数据）+ PI0.5 基座权重。
**输出**：一个在仿真里抓取红方块并放入托盘、随后回到初始位姿的策略。
**总耗时**：约 35 小时（8×H200），其中约 12 小时是纯 CPU 的示范生成与视频编码。

---

## 0. 这条流程在做什么

真机数据只有 87 集。**只用它做 SFT、跳过仿真示范训练、直接放进仿真环境**的策略，独立评测为 **0.0%**（`success_once` 与 `success_at_end` 均为 0）；从这个检查点起跑的两次 RL，一次全程为 0，另一次峰值仅 3–4%。原因是视觉域完全不同——同一个策略在**真机上**大概率可用，只是在仿真里为零。所以流程分三段：

```
真机数据 SFT ──→ 仿真示范 SFT（阶段 B–E，监督学习）──→ PPO 在线微调（阶段 G）
   建立语义        把地板从 0 抬到 57.8%              再抬 5.9 点
```

**三个贯穿全程的原理**，每一个都由对照实验确立：

| 原理 | 内容 | 证据 |
|---|---|---|
| **密度律** | BC 的成绩下限由**示范间距** = √(生成面积 ÷ 示范条数) 决定，不是由总量决定。方块 2.9 cm、抓取容差 ±0.7 cm | 三个实测点：间距 1.01 cm→12.5%、0.91 cm→7.0%、**0.44 cm→56.7%**。是阈值不是渐变 |
| **血统冻结** | 同一条血统内**绝不重算** norm_stats | A/B 实测：只换统计量、其余全不变，19.5%→9.4% |
| **RL 是放大器** | PPO 放大已有的成功，不发现新成功；先决条件必须在 **rollout 分布**下测 | 起点为 0 时 RL 起不来（两次运行，0% 与峰值 3–4%）；带噪成功率 1% 时 PPO 9 轮摧毁策略 |

---

## 1. 前置条件

### 1.1 硬件

| 项 | 要求 |
|---|---|
| GPU | 8 × H200（141 GB） |
| 磁盘 | ≥ 500 GB |
| `/dev/shm` | **≥ 8 GB**（容器默认 64 MB 会让 NCCL 直接失败，报错看起来像显存不足）：`mount -o remount,size=16G /dev/shm` |
| CPU | ≥ 32 核（示范生成 8 进程并行） |

### 1.2 代码与数据

```bash
git clone https://github.com/bytedance-iaas/RLinf.git && cd RLinf
git checkout henryg/pi05-maniskill-so101
bash requirements/install.sh embodied --model openpi --env maniskill
```

| 资源 | 落地路径 |
|---|---|
| PI0.5 基座（LeRobot 格式） | `checkpoints/lerobot_pi05_base`（14 GB） |
| 真机数据集 87 集 | `$HF_LEROBOT_HOME/henry-guo/so101-pick-place-v2` |
| ManiSkill 资产 | `hf download --repo-type dataset RLinf/maniskill_assets` → 复制到 `rlinf/envs/maniskill/assets/`（`.gitignore` 排除了这个目录，clone 后一定没有） |

### 1.3 环境变量（每个终端）

```bash
export REPO_PATH=$PWD PYTHONPATH=$PWD HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p $XDG_RUNTIME_DIR
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99     # Ray 默认阈值会因 / 盘满而阻塞对象创建
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 1.4 注册表条目（`rlinf/models/embodiment/openpi/dataconfig/__init__.py`）

本流程用到 5 个，结构完全相同（`action_horizon=10`、`discrete_state_input=True`、`extra_delta_transform=False`），只有 `repo_id` 不同：

| 条目 | repo_id | 阶段 |
|---|---|---|
| `pi05_so101` | `henry-guo/so101-pick-place-v2` | A |
| `pi05_so101_v4` | `so101-sim-demos-v4` | B（**血统 norm_stats 在此确立**） |
| `pi05_so101_v8` | `so101-sim-demos-v8` | C |
| `pi05_so101_v9` | `so101-sim-demos-v9` | D |
| `pi05_so101_v10` | `so101-sim-demos-v10` | E、G |

**`config_name` 选数据变换管线，`norm_stats_path` 单独由 yaml 指定**——正因为分离，各阶段才能用自己的数据集条目却共享同一份统计量。

### 1.5 每次启动训练前

```bash
.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/sft/config/ --config-name <配置名> \
  runner.logger.log_path=<输出目录>
```

必须看到 `PREFLIGHT OK`。它校验路径存在性、批量算术、模型-数据一致性。**批量算术那一项对 PPO 是命门**（见 §7.1）。

---

## 2. 工具清单

全部在 `tools_so101_session/`（59 个脚本，`README.md` 是按用途分类的索引）。本流程直接用到的：

| 工具 | 做什么 | 用在 |
|---|---|---|
| `gen_so101_demos.py` | 规划器：抓取→抬起→搬运→入盒→回位，成功才留 | B、C、E |
| `v10_gen.sh` | 8 worker 并行生成**环形带**示范（只补新区域） | E |
| `v10_collect.sh` | 采集策略自己的成功轨迹（`SO101_COLLECT_DIR` 打开录制器） | D、E |
| `convert_v10_demos.py` | 在上一版数据集副本上**追加**新集，不重编码旧集（省 2.75 h） | E |
| `convert_v4/v8/v9_demos.py` | 各阶段的转换 | B、C、D |
| `v9_expert_iter.sh` | 专家迭代完整流水线 | D |
| `v10_rest.sh` | 转换 → SFT → 门评 | E |
| `freeze_v11.sh` | **冻结探针**（`lr=1e-9`，跑真实训练路径不改权重） | G 的先决条件测量 |
| `rl_v13.sh` | PPO 启动器，**含自动停机守卫** | G |
| `v13_verify.sh` / `v13_baseline.sh` | 诚实验证 / 起点对照 | G |
| `offline_replay_check.py` | 用真机数据离线检验策略（sim2real 上机前的门） | 部署前 |
| `toolkits/preflight_config.py` | 启动前配置校验 | 全程 |
| `toolkits/invariant_audit.py` | 9 项静默错误检查 | 换规格/换数据集时 |

---

## 3. 阶段 A —— 真机数据 SFT

```bash
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.lerobot.calculate_norm_stats --config-name pi05_so101
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_openpi_pi05 \
  runner.logger.log_path=$RESULTS/so101_sft_openpi_pi05
```

| 参数 | 值 | 依据 |
|---|---|---|
| 热启动 | `checkpoints/lerobot_pi05_base` | PI0.5 基座 |
| `lr` | 2.5e-5 | 全新任务的标准 BC 学习率 |
| `max_steps` | 20000（取用 **step_8000**） | |
| micro/global batch | 16 / 128 | 128 必须是 world_size(8)×micro 的整数倍 |

**耗时** 约 3 h。这一步的产物**在仿真里独立评测为 0.0%**（实测），它只是阶段 B 的热启动权重，不是可用策略。

---

## 4. 阶段 B —— 全板仿真示范 + 冻结统计量

**B1 规划器探针**（先证明任务可解）：

```bash
.venv/bin/python tools_so101_session/gen_so101_demos.py --num 12 --seed0 79000 --out $DATA/probe
```
门槛：≥8/12 成功、中位长度 ≤530 步。**不过就不要往下走**——规划器做不到的，BC 和 RL 都做不到。

**B2 分层生成 420 条全板示范**（4×4 格、每格 45 次尝试、8 worker）：

```bash
SEED=80000
for XI in 0 1 2 3; do for YI in 0 1 2 3; do
  SO101_SPAWN_FRAC="$(echo "$XI*0.25"|bc -l),$(echo "($XI+1)*0.25"|bc -l),$(echo "$YI*0.25"|bc -l),$(echo "($YI+1)*0.25"|bc -l)" \
  .venv/bin/python tools_so101_session/gen_so101_demos.py --num 45 --seed0 $SEED \
      --out $DATA/v4_demos_cell_${XI}_${YI} &
  SEED=$((SEED+100)); done; done; wait
```

**B3 转换 + 计算 norm_stats（整条血统只算这一次）**：

```bash
.venv/bin/python tools_so101_session/convert_v4_demos.py
.venv/bin/python -m toolkits.lerobot.calculate_norm_stats --config-name pi05_so101_v4
```

> **此后 C/D/E/G 一律沿用 `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json`，绝不重算。**

**B4/B5 SFT + 门评**（`so101_sft_v4`，lr 2.5e-5、4000 步、save 1000）：

**验收：最优点是 `global_step_1000`，全板约 12.5%。** 之后单调下降（2.0→2.3→0.0）——SFT loss 降到 0.002 却越训越差，是**过拟合规划器习惯**，所以最优点在最早期。

> ⚠️ 历史上阶段 B 是从一个 pp 时代检查点热启动的，那套规格已废弃。本文档改为从阶段 A 热启动，**该替换已用对照实验验证**：全板 10.2% vs 对照 12.5%（差 2.3 点，阈值 3 点），且同期用 v4 原检查点重测今天的环境得到 12.5%（与两天前完全一致，证明环境未漂移）。

---

## 5. 阶段 C —— 收窄生成区（用密度换成绩）

全板 426 cm² 要达到 0.44 cm 间距需要约 2200 条示范；先收窄区域，用同样的数据量换密度。

```bash
export SO101_SPAWN_MODE=legacy      # 唯一的收窄项：6×8 cm = 48 cm²
for W in 0 1 2 3 4 5 6 7; do
  .venv/bin/python tools_so101_session/gen_so101_demos.py \
    --num 32 --seed0 $((90000 + W*1000)) --out $DATA/v8_demos_w$W & done; wait
.venv/bin/python tools_so101_session/convert_v8_demos.py     # 不重算 norm_stats
.venv/bin/python examples/sft/train_vla_sft.py --config-name so101_sft_v8 ...
```

| 参数 | 值 | 依据 |
|---|---|---|
| 热启动 | v4 的 `global_step_1000` | 阶段 B 最优点 |
| `lr` / 步数 / 存点 | 2.5e-5 / 4000 / **250** | 250 是教训值：最优点在 step_2500，按 1000 存会错过 |

**验收**：247 条示范、间距 **0.44 cm**；最优 `global_step_2500`，**诚实值 57.8/55.5%**（种子 1313/1414），全板 9.4%。

> **判读纪律**：step_250 只有 7.8%、step_500 只有 0.8%——**前两个点低不代表方向错**。

---

## 6. 阶段 D/E —— 专家迭代与扩域

**D（专家迭代，+20 点）**：用当前策略在 8 个**从未用过**的种子上采集自己的成功轨迹（`v10_collect.sh` 的做法），与原始规划器示范**混合**后轻量 SFT。

```bash
export SO101_COLLECT_DIR=$DATA/v9_rollouts
for SEED in 2001 2002 2003 2004 2005 2006 2007 2008; do
  SO101_SPAWN_MODE=legacy .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-name so101_eval_openpi_pi05 rollout.model.model_path=$V8 ... env.eval.seed=$SEED
done
.venv/bin/python tools_so101_session/convert_v9_demos.py    # 247 规划器 + 425 策略 = 672 集
.venv/bin/python examples/sft/train_vla_sft.py --config-name so101_sft_v9 ...   # lr 1e-5, 2000 步
```

**必须混合**：纯自蒸馏会让策略越练越窄（历史上掉 53 点）。
**单位不对称陷阱**：录制器 npz 的 `state` 已归一化而 `action` 是弧度；h5 两者都是弧度。
**验收**：最优 `global_step_1250`，**诚实值 77.3/75.8%**，全板 19.5%（翻倍）。峰值在中段——最后一个点只有 7.8%，**必须全部检查点都评**。

**E（环 1 扩域，负结果但产出了 PPO 的起点）**：生成区扩到 96 cm²（`SO101_SPAWN_FRAC="0.4294,0.9115,0.5142,0.9817"`），自采 429 条 + 环形带规划器 204 条，追加进 v9 数据集得 1292 集。

**验收**：`so101_sft_v10/.../global_step_1000`，环 1 诚实 55.1%、框内 75.0%、全板 10.2%。

> **这是负结果**：扩域没有在目标区带来提升（v9 在环 1 上本来就有 58.6%），全板还掉了一半。**密度律不能外推成"扩域只要补够密度"**——v9 外环一条示范都没有却已经有 58.6%，说明外环从来不缺示范。
> 但 v10 是 PPO 的起点，因为它是在**环 1 数据上训练**的，与 PPO 的训练区域一致。

---

## 7. 阶段 G —— PPO 在线微调

完整细节见 `PPO_V13_RUNBOOK_ZH.md`；这里是要点。

### 7.1 三个决定成败的参数

| 参数 | 官方/原值 | 改成 | 实测依据 |
|---|---|---|---|
| `num_action_chunks` | 5 | **10** | 带噪 rollout 1.0%→**4.7%**，且**确定性成绩 55%→66.4%**（白捡 11 点）。模型 SFT 时 `action_horizon=10`，只执行 5 步等于丢一半预测 |
| `openpi.noise_logvar_range` | `[0.08,0.16]` | **`[0.02,0.04]`** | 带噪 rollout 4.7%→**39.1%**。⚠️ **别调 `noise_params`**——那是 flow-SDE 的参数，`flow_noise` 不读（`openpi_action_model.py:47-58`） |
| **每轮更新次数** | 12 | **1** | 12 次/轮：step 9 从 61.7% 掉到 **7.0%**；1 次/轮：稳定爬到 **73.4%**。**决定性的那一个** |

`updates = num_envs × (max_episode_steps ÷ chunks) × rollout_epoch ÷ global_batch_size`
→ `64 × 64 × 1 = 4096 样本`，`global_batch_size=4096` → **1 次更新/轮**。

### 7.2 启动前必测：先决条件在 rollout 分布下

用**冻结探针**（`lr=1e-9`，真实训练路径，权重不变），一轮同时给出两个数：

- `env/success_once`（**带噪**，PPO 真正学习的分布）：**门槛 ≥5%**
- `eval/success_once`（确定性）：不应比起点低太多

5% 来自四次运行的实测分界（pp4 5–9%、v10 10–15% 成功；v6 0.5%、v11 1.0% 失败）。**必要不充分**——v12 有 39% 仍崩，因为更新次数不对。

**不要用 `runner.only_eval=True` 当探针**：它同时切换模型规格来源并跳过训练环境创建（`config.py:830`、`env_worker.py:108`），是另一条代码路径。

### 7.3 启动

```bash
export SO101_SPAWN_FRAC="0.4294,0.9115,0.5142,0.9817"
bash tools_so101_session/rl_v13.sh        # 含自动停机守卫，推荐
```

守卫每 5 分钟读 `eval/success_once`：低于峰值 20 点 → 停；连续 3 次低于起点 5 点 → 停。**必须写在启动器里**，写在会话里会随会话消失（历史上因此白烧 180 轮）。

### 7.4 实测结果

| step | 4 | 9 | 14 | 19 | 24 | **29** | 34 | 39 | 44 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval | 60.9 | 58.6 | 62.5 | 58.6 | 70.3 | **73.4** | 69.5 | 70.3 | 68.0 | 68.8 |

前 20 轮徘徊（不要在这里判死刑），step 24 起上台阶。峰值检查点 `global_step_30` 即交付物。

**诚实口径**（从未用过的种子 4141/4242，同块长，与起点同条件对照）：

| | 4141 | 4242 | 均值 |
|---|---|---|---|
| 起点 v10 | 48.4% | 55.5% | 52.0% |
| **PPO v13** | **58.6%** | **57.0%** | **57.8%** |
| 增益 | +10.2 | +1.5 | **+5.9** |

框内参照 77.3%、全板参照 14.8%。

---

## 8. 全流程验收表

**任何一步显著低于预期就停下排查**，后面每一步都建立在前一步之上。

| 阶段 | 产物 | 验收数字 | 耗时 |
|---|---|---|---|
| A 真机 SFT | `global_step_8000` | 不评测 | ~3 h |
| B 全板 SFT | `v4/global_step_1000` | 全板 **12.5%** | ~7 h |
| C 收窄 | `v8/global_step_2500` | 诚实 **56.7%**，间距 0.44 cm | ~5 h |
| D 专家迭代 | `v9/global_step_1250` | 诚实 **76.6%**，全板 19.5% | ~5 h |
| E 环 1 扩域 | `v10/global_step_1000` | 环 1 **55.1%**（负结果） | ~7 h |
| **G PPO** | `so101_ppo_v13/.../global_step_30` | 环 1 诚实 **57.8%**（+5.9） | ~2 h |

---

## 9. 会浪费一整夜的坑

| 症状 | 真因 | 处置 |
|---|---|---|
| NCCL 直接失败，像显存不足 | `/dev/shm` 只有 64 MB | remount 16G；先看日志里 `Last error:` 那行 |
| 评测卡住、驱动进程永远等待 | Ray worker `SYSTEM_ERROR` 猝死 | 每次评测加 `timeout 1800` + 3 次重试 + 完整清理 |
| 改了参数但行为没变 | 参数在**三处**独立定义（env yaml、生成器自己的 `gym.make`、转换器硬编码） | 改任何参数前 `grep` 出所有消费方 |
| 调了噪声却没有任何变化 | `noise_params` 对 `flow_noise` 是空参数 | 用 `noise_logvar_range` |
| 换数据集后成绩掉一半 | 重算了 norm_stats | 血统内绝不重算 |
| 流水线在离终点很近处被杀 | 超时按整数拍脑袋 | 按**实测速率**定，留 ≥50% 余量 |
| 上一阶段跑完，下一阶段没启动，GPU 空转数小时 | 等待脚本先查完成标记再查上游存活；两者只差微秒 | 发现上游消失后**必须再查一次完成标记**；别把哨兵串写进自己的日志文案 |
| 监工从不报告"任务已结束" | `pgrep -f` **匹配到了自己** | 用启动时记下的 pid 读 `/proc/<pid>/stat`（僵尸进程 `/proc/<pid>` 仍在） |
| 多个后继同时启动、互相杀 Ray | 后继之间没有串行 | 后继必须串成链，尤其当各阶段都做全局清理时 |

---

## 10. 相关文档

| 文档 | 内容 |
|---|---|
| `PPO_V13_RUNBOOK_ZH.md` | PPO 成功配方的详细步骤与失败对照表 |
| `PPO_CODE_WALKTHROUGH_ZH.md` | PPO 的代码级全流程（带文件行号） |
| `SIM2REAL_PLAN_ZH.md` | 真机部署方案 + **一个已量化的阻断缺陷** |
| `V10_REPRODUCTION_ZH.md` | 阶段 A–E 的更详细版本 |
| `tools_so101_session/README.md` | 59 个脚本的用途索引 |
| `.claude/skills/rlinf-embodied-training/SKILL.md` | 工程纪律（含已证伪诊断清单） |

**下一步不是继续训练，而是 sim2real**：离线检验已证明当前策略读不懂真实观测（策略动作误差比"完全不动"还差 4.5 倍，最坏的是腕部两个自由度）。原因是仿真的腕部相机拍的是机械臂自己。详见 `SIM2REAL_PLAN_ZH.md`。
