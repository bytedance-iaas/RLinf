# V10 从零复现指南

**目标**：在另一台同规格机器上，只凭 `henry-guo/so101-pick-place-v2` 这一个真机数据集和 PI0.5 基座权重，重建出当前的 SO101 pick-and-place 策略。

**产物**：一个在 96 cm² 生成区（环 1）内成功率约 60-75% 的策略检查点，以及沿途每一步的可验收中间产物。

**总耗时**：约 30-35 小时（8×H200 连续占用），其中约 12 小时是纯 CPU 的示范生成与视频编码。

---

## 0. 先读这一节：这份文档里唯一未验证的一步

真实历史的血统是：

```
lerobot_pi05_base
  └─ 真机数据 SFT  (global_step_8000)
       └─ pp 时代的一串仿真 SFT  ← 旧任务规格：160×120 相机、20 Hz、无回位判据、窄生成区
            └─ pp6b_step_1000
                 └─ v4 → v8 → v9 → v10
```

中间那段 pp 时代的检查点，是在**已经被废弃的任务规格**下训出来的。今天的代码渲染 640×480、跑 30 Hz、成功判据要求回位——**用现在的代码复现 pp 时代已经不可能，也没有意义**。

因此本文档把那一段替换为：**阶段 B 直接从真机数据 SFT 的检查点热启动**。

> **风险 R1（本文档最大的不确定性）**：这个替换**没有被实测验证过**。v4 当初是从 pp6b 热启动的，不是从真机 SFT。替换之后阶段 B 的成绩可能与历史不同。
>
> **如何处置**：阶段 B 有一个明确的验收数字（全板 ~12.5%）。如果你测出来显著低于它，**停下来排查，不要继续往下走**——因为后面每一步都建立在它之上。若确实达不到，退路是先复现 pp 时代的窄任务 SFT 作为跳板（本文档不含该路径）。

其余所有步骤的参数、命令、验收数字都是实测值。

---

## 1. 前置条件

### 1.1 硬件与系统

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | 8 × H200（141 GB） | SFT 用 FSDP 占满 8 卡；评测同样 8 卡 |
| 磁盘 | ≥ 500 GB 可用 | 基座权重 14 GB、各阶段检查点各约 32 GB、数据集合计约 2 GB |
| `/dev/shm` | **≥ 8 GB** | 容器默认 64 MB 会让 NCCL 直接失败，且报错信息看起来像显存不足。`mount -o remount,size=16G /dev/shm` |
| CPU | ≥ 32 核 | 示范生成 8 进程并行；视频编码单进程约 3.3 集/分钟 |

### 1.2 代码

全部代码、配置、工具已在分支上：

```bash
git clone https://github.com/bytedance-iaas/RLinf.git
cd RLinf
git checkout henryg/pi05-maniskill-so101
```

这个分支包含（本文档所有命令都依赖它们）：

- `rlinf/envs/maniskill/tasks/so101_pick_place.py` —— 仿真任务
- `rlinf/envs/maniskill/so101_agent.py`、`so101_calib.py` —— 机器人接入与关节标定
- `rlinf/models/embodiment/openpi/policies/so101_policy.py`、`dataconfig/so101_dataconfig.py` —— 模型侧数据变换
- `dataconfig/__init__.py` 里注册的 TrainConfig 条目。**本流程只用到 5 个**（其余 SO101 条目是历史遗留，可忽略）：

  | 条目 | repo_id | 用在哪一步 |
  |---|---|---|
  | `pi05_so101` | `henry-guo/so101-pick-place-v2` | 阶段 A（真机 SFT + 真机 norm_stats） |
  | `pi05_so101_v4` | `so101-sim-demos-v4` | 阶段 B（全板示范 SFT + **血统 norm_stats**） |
  | `pi05_so101_v8` | `so101-sim-demos-v8` | 阶段 C（收窄生成区） |
  | `pi05_so101_v9` | `so101-sim-demos-v9` | 阶段 D（专家迭代） |
  | `pi05_so101_v10` | `so101-sim-demos-v10` | 阶段 E（环 1 扩域） |

  五个条目结构完全相同（`action_horizon=10`、`discrete_state_input=True`、`extra_delta_transform=False`），只有 `repo_id` 不同——新增一个阶段就是复制一段改 `repo_id`。
  **注意 `config_name` 与 `norm_stats` 是分开的**：前者选数据变换管线，后者由 yaml 的 `norm_stats_path` 显式指定（全流程一律指向 v4 那份）。正因为分离，各阶段才能用自己的数据集条目却共享同一份统计量。
  忽略即可的历史条目：`pi05_so101_sim`、`pp`、`pp5`、`pp6`、`pp7`（pp 时代旧规格，160×120 相机）、`v3`（频率分叉版）、`v5`（纯自蒸馏，掉 53 点）、`v7`（分带课程，已证伪）。
- `examples/sft/config/so101_sft_*.yaml`、`examples/embodiment/config/so101_*.yaml`
- `tools_so101_session/` —— 生成器、转换器、流水线脚本（见其 `README.md`）
- `toolkits/preflight_config.py`、`toolkits/invariant_audit.py`

### 1.3 安装

```bash
bash requirements/install.sh embodied --model openpi --env maniskill
```

需要的额外资产（ManiSkill 会自动下载到 `~/.maniskill/data`）：`bridge_v2_real2sim_dataset`、`widowx` 机器人（仅官方基准需要；SO101 任务自带 URDF）。

### 1.4 基础权重与数据

| 名称 | 来源 | 落地路径 |
|---|---|---|
| PI0.5 基座（LeRobot 格式） | `lerobot/pi05_base` | `checkpoints/lerobot_pi05_base`（14 GB） |
| 真机数据集 | `henry-guo/so101-pick-place-v2` | `$HF_LEROBOT_HOME/henry-guo/so101-pick-place-v2` |
| 分词器 / 视觉塔 | `RLinf/openpi_tokenizer`、`google/paligemma-3b-pt-224` | HF 缓存 |

### 1.5 每个终端都要导出的环境变量

```bash
export REPO_PATH=$PWD PYTHONPATH=$PWD HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p $XDG_RUNTIME_DIR
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_LEROBOT_HOME=/data08/henryg/pai/data        # 改成你的数据目录
export RAY_local_fs_capacity_threshold=0.99            # Ray 默认阈值会因 / 盘满而阻塞对象创建
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

`RLINF_MASTER_ADDR_OVERRIDE` 是必需的：本机节点 IP 若是 IPv6，未加方括号的 `tcp://` 会报 "Port could not be cast to integer"。

### 1.6 每次启动训练前

```bash
export EMBODIED_PATH=$PWD/examples/sft        # 或 examples/embodiment（评测/RL）
.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/sft/config/ --config-name <配置名> \
  runner.logger.log_path=<输出目录>
```

必须看到 `PREFLIGHT OK`。它检查路径存在性、批量整除性、模型-数据一致性——这些错误如果漏到运行时，通常表现为跑了一整夜才发现方向错了。

---

## 2. 阶段 A —— 真机数据 SFT（起点）

**目的**：让 PI0.5 学会这个任务的语义和真机视觉域。这是整条血统的根。

```bash
export EMBODIED_PATH=$PWD/examples/sft

# A1. 真机数据集的归一化统计量
.venv/bin/python -m toolkits.lerobot.calculate_norm_stats --config-name pi05_so101
# 产出 assets/pi05_so101/henry-guo/so101-pick-place-v2/norm_stats.json

# A2. SFT
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ \
  --config-name so101_sft_openpi_pi05 \
  runner.logger.log_path=$RESULTS/so101_sft_openpi_pi05
```

| 参数 | 值 | 依据 |
|---|---|---|
| 热启动 | `checkpoints/lerobot_pi05_base` | PI0.5 基座 |
| 数据 | `henry-guo/so101-pick-place-v2` | 真机 87 条示范 |
| `config_name` | `pi05_so101` | 注册表里对应真机数据集的条目 |
| `lr` | 2.5e-5 | 全新任务的标准 BC 学习率 |
| `max_steps` | 20000 | 实际取用的是 **global_step_8000** |
| `micro_batch_size` / `global_batch_size` | 16 / 128 | 128 必须是 world_size(8) × micro 的整数倍；micro 只是梯度累积粒度，不影响训练正确性 |

**产出**：`$RESULTS/so101_sft_openpi_pi05/checkpoints/global_step_8000`
**耗时**：约 3 小时
**验收**：这一步的产物**在仿真里独立评测为 0.0%**（`success_once`/`success_at_end` 均为 0，见 `eval_sft_fixed.out`）；从它起跑的两次 RL 分别为全程 0 与峰值 3–4%。它只是阶段 B 的热启动权重，不是可用策略。

---

## 3. 阶段 B —— 全板仿真示范 + 冻结归一化统计量

**目的**：把策略从真机视觉域迁到仿真视觉域，并**确立整条血统的 norm_stats**。

### B1. 规划器探针（先验证任务可解）

```bash
.venv/bin/python tools_so101_session/gen_planner_demos.py \
  --num 12 --seed0 79000 --out $DATA/probe
```

**门槛**：成功 ≥8/12，示范中位长度 ≤530 步（回合预算 640 留 10% 余量）。
**若不过**：不要往下走。规划器做不到的任务，RL 与 BC 都做不到。历史上这一步的失败都指向环境事实错误（复位姿态、物体质量、夹爪行程）。

### B2. 分层生成 420 条全板示范

```bash
# 4×4 = 16 个格子，每格 45 次尝试，8 worker 各跑 2 格
SEED=80000
for XI in 0 1 2 3; do for YI in 0 1 2 3; do
  X0=$(echo "$XI*0.25" | bc -l); X1=$(echo "($XI+1)*0.25" | bc -l)
  Y0=$(echo "$YI*0.25" | bc -l); Y1=$(echo "($YI+1)*0.25" | bc -l)
  SO101_SPAWN_FRAC="$X0,$X1,$Y0,$Y1" \
    .venv/bin/python tools_so101_session/gen_planner_demos.py \
      --num 45 --seed0 $SEED --out $DATA/v4_demos_cell_${XI}_${YI} &
  SEED=$((SEED+100))
done; done
wait
```

| 参数 | 值 | 依据 |
|---|---|---|
| 生成区 | 全棕区（默认，不设 `SO101_SPAWN_MODE`） | 真实任务：方块可出现在棕区任何位置（用户确认 + 87 帧真机首帧实测） |
| 分层 | 4×4 格，每格独立配额 | 均匀随机会让远端格子样本过少；分层保证覆盖 |
| 每格 `--num` | 45 | 目标全板约 420 条 |
| 种子 | 80000 + 100·格号 | 各格种子区间不重叠 |

**产出**：420 条成功示范
**耗时**：约 4 小时（纯 CPU，8 进程）

### B3. 转换 + 计算 norm_stats（**整条血统只算这一次**）

```bash
.venv/bin/python tools_so101_session/convert_fullboard.py     # -> so101-sim-demos-v4
.venv/bin/python -m toolkits.lerobot.calculate_norm_stats --config-name pi05_so101_v4
# 产出 assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
```

> **此后所有阶段（C、D、E）一律沿用这份 norm_stats，绝不重算。**
> 依据：A/B 实测——只换统计量、其余全不变，成绩从 19.5% 掉到 9.4%。归一化统计量是策略输入分布的定义，中途更换等于悄悄换掉了模型看到的世界。

**验收**：420 集、约 14.3 万帧、30 fps、图像 (480,640,3)。

### B4. SFT

```bash
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_v4 \
  runner.logger.log_path=$RESULTS/so101_sft_v4
```

| 参数 | 值 | 依据 |
|---|---|---|
| 热启动 | **阶段 A 的 global_step_8000**（历史上是 pp6b_1000，见 §0 风险 R1） | |
| `lr` | 2.5e-5 | 跨视觉域迁移，需要标准 BC 学习率而非微调率 |
| `max_steps` / `save_interval` | 4000 / 1000 | |
| `norm_stats` | B3 产出，路径写死 | 血统冻结 |

### B5. 门评

对**每一个**检查点评测，不要只看最后一个。

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
for CK in $RESULTS/so101_sft_v4/so101_sft_openpi_pi05/checkpoints/global_step_*; do
  mkdir -p $CK/so101-sim-demos-v4 && cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json $CK/so101-sim-demos-v4/
  .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=$RESULTS/so101_eval_v4 \
    rollout.model.model_path=$CK \
    rollout.model.openpi.config_name=pi05_so101_v4 \
    rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    env.eval.total_num_envs=128 env.eval.seed=777
done
```

**验收（关键）**：最优检查点应为 **global_step_1000**，全板成功率 **约 12.5%**。

历史实测：step_1000 = 10.9%（验证 12.5%），之后 2.0% → 2.3% → 0.0% **单调下降**。这不是异常——SFT loss 降到 0.002（几乎完美拟合示范），成功率却在跌，说明**过拟合到规划器的个人习惯**。因此最优点在最早期，`save_interval` 必须足够密才不会错过。

**若显著低于 12.5%**：这是 §0 风险 R1 兑现了，停下排查。

---

## 4. 阶段 C —— 收窄生成区（密度律）

**目的**：把 BC 的地板从 12.5% 抬到 57%。手段不是改模型，是**提高示范密度**。

> **密度律（本项目最核心的发现）**：BC 的成绩由**示范间距** = √(生成面积 ÷ 示范条数) 决定，与总量无关。方块边长 2.9 cm，抓取容差约 ±0.7 cm。三个实测点：间距 1.01 cm → 12.5%；0.91 cm → 7.0%；**0.44 cm → 56.7%**。这是**阈值**不是渐变：间距小于容差时 BC 只需插值，大于时必须泛化，而它泛化不了。

全板 426 cm² 要达到 0.44 cm 需要约 2200 条示范；因此先收窄生成区，用同样的数据量换密度。

### C1. 生成 247 条窄框示范

```bash
export SO101_SPAWN_MODE=legacy     # 唯一的收窄项：6×8 cm = 48 cm²
for W in 0 1 2 3 4 5 6 7; do
  .venv/bin/python tools_so101_session/gen_planner_demos.py \
    --num 32 --seed0 $((90000 + W*1000)) --out $DATA/v8_demos_w$W &
done; wait
```

| 参数 | 值 | 依据 |
|---|---|---|
| `SO101_SPAWN_MODE=legacy` | x ∈ [−0.534,−0.474]、y ∈ [0.020,0.100] | 48 cm²；**其余一切保持真任务取值**（640×480 相机、30 Hz、真实几何、8 g 方块、回位判据） |
| `--num` 合计 | 256 | 目标约 175 条；实测 **247/256 = 96.5%** |
| 种子 | 90000 + 1000·W | 进程间不重叠 |

**验收**：≥120 条，中位长度 ≤530 步。实测 247 条、中位 357 步、间距 **0.44 cm**。

### C2. 转换（**不重算 norm_stats**）

```bash
.venv/bin/python tools_so101_session/convert_narrow_box.py     # -> so101-sim-demos-v8
```

### C3. SFT + 门评

```bash
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_v8 \
  runner.logger.log_path=$RESULTS/so101_sft_v8
```

| 参数 | 值 | 依据 |
|---|---|---|
| 热启动 | v4 的 **global_step_1000** | 阶段 B 的最优点 |
| `lr` | 2.5e-5 | 仍在教新东西（新的空间分布） |
| `max_steps` / `save_interval` | 4000 / **250** | 250 是吸取教训后的值：v8 的最优点在 step_2500，若按 1000 存点会错过 |

门评：16 个检查点全部过种子 777 → 前三名过 888 → 最优点在**从未用过的** 1313/1414 上出诚实值。

**验收**：最优 = **global_step_2500**，门评 59.0%（777: 61.7 / 888: 56.3），**诚实值 57.8% / 55.5%**，全板参照 9.4%。

**判读纪律**：step_250 只有 7.8%、step_500 只有 0.8%——**前两个点低不代表方向错**。历史上我在这里误判过一次，差点砍掉一条正确的路线。

---

## 5. 阶段 D —— 专家迭代（零风险放大器）

**目的**：57% → 77%。手段是让策略**给自己造数据**。

> 依据：iRe-VLA（arXiv 2501.16664）。关键在于**必须混入原始规划器示范**——纯自蒸馏会让策略越练越窄，历史上纯自蒸馏的一版掉了 53 个点。

### D1. 采集策略自己的成功轨迹

```bash
export SO101_COLLECT_DIR=$DATA/v9_rollouts; mkdir -p $SO101_COLLECT_DIR
V8=$RESULTS/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_2500
for SEED in 2001 2002 2003 2004 2005 2006 2007 2008; do
  SO101_SPAWN_MODE=legacy .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=$RESULTS/so101_eval_v9 \
    rollout.model.model_path=$V8 \
    rollout.model.openpi.config_name=pi05_so101_v8 \
    rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    env.eval.total_num_envs=128 env.eval.seed=$SEED
done
```

| 参数 | 值 | 依据 |
|---|---|---|
| `SO101_COLLECT_DIR` | 目录 | 打开 `ManiskillEnv` 的录制器：每路环境**第一次成功**时，把 (前视, 腕视, 状态, 动作) 写成 `.npz` 文件存盘；128 路环境是异步重置的，缓冲区按路单独清空 |
| 种子 | **2001-2008**（全新） | 与筛选(777/888)、验收(1313/1414)完全不重叠——采过的局面等于进了训练集 |
| 评测模式 | 确定性（无探索噪声） | 录的是策略的最好水平，不是抖出来的偶然成功 |

**产出**：477 条成功轨迹（8 个种子成功率 57.0-65.6%，均值 61.3%——这同时是 v8 的一次无偏复测）
**耗时**：约 55 分钟

### D2. 混合转换

```bash
.venv/bin/python tools_so101_session/convert_expert_iter.py    # -> so101-sim-demos-v9
```

混合内容：247 条规划器示范 + 425 条策略轨迹（477 条里 52 条被 80-580 帧的长度闸挡掉）= **672 集**，间距 **0.27 cm**。

> **单位不对称陷阱**：录制器 npz 的 `state` **已经是归一化值**，而 `action` 是**弧度**；h5 规划器示范则两者都是弧度。转换时必须分别处理，否则不报错、只是模型学错。

### D3. 轻量 SFT + 门评

```bash
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_v9 \
  runner.logger.log_path=$RESULTS/so101_sft_v9
```

| 参数 | 值 | 依据 |
|---|---|---|
| 热启动 | v8 的 **global_step_2500** | |
| `lr` | **1.0e-5**（比阶段 C 的 2.5e-5 更轻） | 这是在**打磨一个已经不错的策略**，不是教新行为 |
| `max_steps` / `save_interval` | 2000 / 250 | |

门评：8 个检查点过 777 → 前三名过 888 → 最优点在**从未用过的 2323/2424** 上出诚实值。

**验收**：最优 = **global_step_1250**，门评 74.2%，**诚实值 77.3% / 75.8%**，全板参照 **19.5%**（v8 是 9.4%，翻倍）。

**注意检查点形状**：最后一个点 step_2000 只有 7.8%——**塌了**。只看最后一个点会得出"专家迭代失败"的相反结论。八个点全评是必须的。

---

## 6. 阶段 E —— 环 1 扩域（V10 本体）

**目的**：在保持密度的前提下把生成区扩大一倍。

> **扩域的唯一硬约束**：面积涨多少，示范就得涨多少，间距不能变。否则地板立刻塌回去。

### E0. 环 1 的定义

原 6×8 cm 框以自身中心按 √2 各向外扩 → **8.48 × 11.31 cm = 96.0 cm²**（正好 2 倍）。

用**全板比例**表达，一个字符串供生成、采集、评测三处共用（避免三处定义分叉——历史上频率参数就是这么分叉的，静默毁掉一整天）：

```bash
RING1="0.4294,0.9115,0.5142,0.9817"
```

（全板生成范围是 17.6 × 24.2 cm = 426 cm²，已扣除 2 cm 边距。）

### E1. 采集环 1 内策略自己的成功轨迹

```bash
export SO101_COLLECT_DIR=$DATA/v10_rollouts; mkdir -p $SO101_COLLECT_DIR
V9=$RESULTS/so101_sft_v9/so101_sft_openpi_pi05/checkpoints/global_step_1250

# 先量基线（不录制），这是决定后面生成量的依据
SO101_SPAWN_FRAC="$RING1" .venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
  runner.logger.log_path=$RESULTS/so101_eval_v10 \
  rollout.model.model_path=$V9 rollout.model.openpi.config_name=pi05_so101_v9 \
  rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
  env.eval.total_num_envs=128 env.eval.seed=3000

# 再采集
for SEED in 3001 3002 3003 3004 3005 3006 3007 3008; do
  SO101_SPAWN_FRAC="$RING1" SO101_COLLECT_DIR=$DATA/v10_rollouts \
  .venv/bin/python evaluations/eval_embodied_agent.py ... env.eval.seed=$SEED   # 其余同上，模型仍是 $V9
done
```

**实测**：基线（种子 3000）**51.6%**；采集 8 个种子均值 59.4%，得 **429 条**轨迹。

**这个基线数是用来算生成量的**：设框内成功率 p_in = 76.6%、环 1 成功率 p_ring = 51.6%，则外圈成功率 ≈ 2·p_ring − p_in = **27%**。也就是说自采轨迹里只有约 136 条落在外圈（48 cm²），间距 √(48/136) = 0.59 cm——**高于 0.44 cm 的目标，不够**。

### E2. 只在环形带补规划器示范

自采数据天然偏向策略已经会做的内圈，所以补数据只补外圈。四条边带（环 1 减去内框）：

```bash
unset SO101_SPAWN_MODE          # 全板模式，由边带自己收窄
LEFT="0.4294,0.5000,0.5142,0.9817"    # 1.24 × 11.31 cm = 14.0 cm²
RIGHT="0.8410,0.9115,0.5142,0.9817"   # 1.24 × 11.31 cm = 14.0 cm²
BOTTOM="0.5000,0.8410,0.5142,0.5826"  # 6.00 ×  1.66 cm =  9.9 cm²
TOP="0.5000,0.8410,0.9132,0.9817"     # 6.00 ×  1.66 cm =  9.9 cm²

SEED=110000; W=0
for FRAC in "$LEFT" "$LEFT" "$RIGHT" "$RIGHT" "$BOTTOM" "$BOTTOM" "$TOP" "$TOP"; do
  SO101_SPAWN_FRAC="$FRAC" .venv/bin/python tools_so101_session/gen_planner_demos.py \
    --num 30 --seed0 $SEED --out $DATA/v10_demos_w$W &
  W=$((W+1)); SEED=$((SEED+1000))
done; wait
```

**需要多少条**：外圈 48 cm² 要达到 0.44 cm 需 48/0.44² = 248 条，自采已贡献约 136 条，故需补约 **112 条**，按面积配额分到四条带（左33/右33/下23/上23）。

**实测产出 204 条**（下 60/60、右 53/60、左 48/60、上 43/60）——高于目标，因为每条示范最多试 3 个变体，按示范计的产出率是 74-100%。

> 注意：**按尝试计**的成功率（35% 左右）与**按示范计**的产出率（74-100%）差一倍。我曾把前者当后者，得出"规划器在外圈几乎不行"的错误判断。

**验收**：合计 ≥112 条。外圈实际间距 √(48/(136+204)) = **0.376 cm**，优于目标。

### E3. 转换（在上一版数据集副本上追加）

```bash
.venv/bin/python tools_so101_session/convert_append_region.py    # -> so101-sim-demos-v10
```

不从零重建：复制 `so101-sim-demos-v9` 后只追加新集。v9 那 672 集重新编码要约 2.75 小时而产出**逐字节相同**的视频，纯浪费。

**产出**：**1292 集** = v9 底座 672 + 环 1 自采 416 + 环形带规划器 204。环 1 整体间距 **0.27 cm**。
**耗时**：2 小时 25 分（若从零重建约 6 小时）

### E4. 轻量 SFT

```bash
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_v10 \
  runner.logger.log_path=$RESULTS/so101_sft_v10
```

| 参数 | 值 | 依据 |
|---|---|---|
| 热启动 | v9 的 **global_step_1250** | 阶段 D 的最优点 |
| `lr` / 步数 / 存点 | 1.0e-5 / 2000 / 250 | 与阶段 D 完全一致——那一组产生了 +20 点 |
| `norm_stats` | 仍是 v4 那份 | 血统冻结 |

**耗时**：43 分钟（8 卡，loss 收敛到 9e-4）

### E5. 门评

```bash
# 1) 8 个检查点全部过种子 777（环 1 区域）
# 2) 前三名过 888
# 3) 最优点在从未用过的 3131 / 3232 上出诚实值
# 4) 另加两个参照：框内（SO101_SPAWN_MODE=legacy）与全板
```

直接用脚本：`tools_so101_session/pipeline_region_expand.sh`（含 full clean、超时 1800、3 次重试——Ray worker 曾经猝死卡住整条流水线 24 分钟）。

> **框内参照必须用 `SO101_SPAWN_MODE=legacy`**，不能用等价的比例矩形：legacy 模式下**蓝色干扰块的分布不同**，两者的数字不可比。

---

## 7. 全流程验收表

每一步都有一个可证伪的数字。**任何一步显著低于预期就停下排查，不要往下走**——后面每一步都建立在前一步之上。

| 阶段 | 产出 | 验收数字（实测） | 耗时 |
|---|---|---|---|
| A 真机 SFT | `global_step_8000` | 不评测（仿真里为 0） | ~3 h |
| B1 规划器探针 | — | ≥8/12 成功，中位 ≤530 步 | ~20 min |
| B2 全板示范 | 420 条 | 间距 1.01 cm | ~4 h |
| B4/B5 全板 SFT | `v4/global_step_1000` | 全板 **12.5%** | ~2.5 h |
| C1 窄框示范 | 247 条 | 间距 **0.44 cm**，中位 357 步 | ~1.8 h |
| C3 窄框 SFT | `v8/global_step_2500` | 诚实 **56.7%**，全板 9.4% | ~3.5 h |
| D1 自采 | 477 条 | 8 个种子 57.0-65.6% | ~1 h |
| D3 专家迭代 | `v9/global_step_1250` | 诚实 **76.6%**，全板 **19.5%** | ~4.5 h |
| E1 环 1 自采 | 429 条 | 环 1 基线 **51.6%** | ~1 h |
| E2 环形带示范 | 204 条 | 外圈间距 0.376 cm | ~1.9 h |
| E4/E5 环 1 SFT | `v10/global_step_????` | 环 1 应 > 51.6%（v9 水平） | ~3 h |

---

## 8. 风险与已知缺陷

| 编号 | 内容 | 影响 | 处置 |
|---|---|---|---|
| **R1** | 阶段 B 的热启动由 pp 时代检查点替换为真机 SFT，**未经验证** | 可能整条血统偏移 | 用 B5 的 12.5% 作为判据；不过则停 |
| **R2** | **腕部相机指向机械臂本体**，全程看不到工作区（`so101_pick_place.py:86-88`，FOV 86° vs 规格 106°） | **不影响仿真成绩**（前视为主），但**会毁掉真机迁移**：上线时真机腕视看得到工作区，与训练分布完全不同 | 真机部署前必修：重新指向 → 用真机腕视做数值标定（板面掩膜 IoU 为判据）→ 重录数据 → 重训。修了就要从阶段 B 重来 |
| **R3** | GPU 仿真 + 流匹配噪声导致不完全确定 | 各阶段数字有 ±2-6 点波动 | 验收一律用**两个从未用过的种子**取平均；单个种子的数字不作结论（同一策略同一区域，换一批局面差过 8 个点） |
| **R4** | 生成区边距 2 cm 排除了约 11% 的真机方块起始位置 | 真机上有一小部分开局是训练分布外的 | 待定：要么缩小边距重生成，要么接受 |
| **R5** | 规划器放置误差中位 3.9 cm、46% 超过 5 cm（箱子内半宽约 5.1×3.8 cm） | 限制远端示范产出率 | 已验证**不是** IK 精度问题（细网格改动实测无效）。误差产生在**松爪之后**（脱手动力学），要改先动放置高度与松爪时机 |

---

## 9. 会让你浪费一整夜的坑（都真实发生过）

| 症状 | 真因 | 处置 |
|---|---|---|
| NCCL 直接失败，看起来像显存不足 | `/dev/shm` 只有 64 MB（容器默认），且残留大量 `cuda.shm.*` | `mount -o remount,size=16G /dev/shm` + 清理残留。**先看日志里 `Last error:` 那一行**，它会点名 `/dev/shm/nccl-...` |
| 评测卡住不动、驱动进程永远等待 | Ray worker `SYSTEM_ERROR: Worker unexpectedly exits` | 每次评测加 `timeout 1800` + 3 次重试 + 完整清理 |
| 示范长度改了频率却没变 | 频率在**三处**独立定义（env yaml、生成器自己的 `gym.make`、转换器硬编码 fps） | 改任何参数前 `grep` 出**所有**消费方 |
| 成绩莫名其妙上不去，不报错 | 转换时 fps 标错（生成器实跑 20 Hz、数据标 15 fps） | 转换器的 `FPS` 必须等于生成器实跑的 `control_freq` |
| 换了数据集后成绩掉一半 | 重算了 norm_stats | 血统内绝不重算 |
| 流水线某阶段被杀在离终点很近处 | 超时按整数拍脑袋（3.3 集/分 × 724 集 = 3h20m，却写了 3h） | 超时按**实测速率**定，留 ≥50% 余量，把算术写进注释 |
| 上一阶段跑完了，下一阶段没启动，GPU 空转数小时 | 等待脚本先查完成标记、再查上游存活——上游写标记和退出只差微秒，于是被判成"没到终点就死了" | 发现上游消失后**必须再查一次完成标记**；且别把哨兵串写进自己的日志文案 |
| 监工从不报告"任务已结束" | `pgrep -f 'bash .*run.sh'` **匹配到了监工自己** | 用启动时记下的 pid 读 `/proc/<pid>/stat` 状态位（setsid 下退出的 bash 会变**僵尸**，`/proc/<pid>` 目录还在） |

---

## 10. 相关文档

- `tools_so101_session/README.md` —— 59 个脚本的用途索引
- `V8_COMMANDS_ZH.md` —— V8/V9 每条命令的参数与理由
- `SO101_PP_80PCT_RUNBOOK.md` —— 含每一步的代码改动
- `RLINF_PI05_REAL2SIM_BEST_PRACTICES.md` —— real2sim 通用经验
- `.claude/skills/rlinf-embodied-training/SKILL.md` —— 工程纪律（含已证伪诊断清单）

---

## 附录 P —— PPO 接线参考（早期版本，主线未采用）

> **本阶段在当前主线中未被采用**（v8→v9→v10 全是监督学习）。这里完整记录接线方式、参数与判读方法，含**失败记录**，供后续需要时直接使用，也避免重复已经踩过的坑。

### P.0 先决条件（不满足就不要启动）

**PPO 是放大器，不是发现器。** 启动前必须满足以下之一：

- 起点策略在**目标环境里已经偶尔成功**（success_once > 几个百分点），或
- 探索足够强，能靠随机撞出成功（从零策略 + 高熵噪声 + 巨量环境步数）。

预训练 VLA **两者都不满足**：它的探索极窄（flow-noise 下 approx_kl 约 0.01–0.04/步）。

> 实测：真机数据 SFT 的 PI0.5 直接扔进仿真跑 RL，**5 次运行、约 2000 轮、零抓取**；后来有一版成功率出现了，但卡在 1–2/128 持续 740 轮，稀疏到无法放大。
> 因此本文档的阶段 B–E（仿真示范 SFT + 专家迭代）是 RL 的**前置条件**，不是可选项。

**启动前的必做检查清单**：

| 检查 | 怎么做 | 不通过怎么办 |
|---|---|---|
| 任务物理可解 | 规划器探针跑到完整成功（阶段 B1） | 修环境，别调 RL |
| 起点有非零成功率 | 用起点检查点在**训练用的那个区域**评一次 | 回到阶段 C/D 抬地板 |
| 奖励阶梯无刷分路径 | 手算每个状态的**每步**收益（见 P.2） | 改奖励，别指望 PPO 绕过去 |
| 批量算术自洽 | `global_batch = num_envs × (预算/动作块长) × rollout_epoch ÷ 每轮更新数` | 见 P.3 |

### P.1 代码接线（本仓库已全部就位，无需改动）

RL 与 SFT 共用同一套环境与模型，只多三处：

**(1) 控制模式分支** —— `rlinf/config.py:1068`

```python
elif "so100" in robot or "so101" in robot:
    control_mode = "pd_joint_pos"      # 绝对关节位置
```

**(2) 动作单位换算** —— `rlinf/envs/action_utils.py:29`

```python
if "so100" in policy or "so101" in policy:
    # 数据集里的动作是 LeRobot 归一化单位（臂 ±100、爪 0-100），
    # 而 ManiSkill 的 pd_joint_pos 要弧度。少了这一步，机械臂会
    # 直接打到关节限位或朝反方向动。
    from rlinf.envs.maniskill.so101_calib import norm_to_rad
    return norm_to_rad(raw_chunk_actions)
```

**(3) 价值头** —— 配置里 `actor.model.add_value_head: True`，`value_after_vlm: True`（价值头挂在 VLM 之后）。SFT 时是 `False`。

### P.2 奖励函数与防刷分算术

奖励在 `rlinf/envs/maniskill/tasks/so101_pick_place.py::compute_dense_reward`，是一条阶梯：

```python
reward = 1 - tanh(5 · d_tcp→cube)                      # 接近，[0,1]
reward += 0.5 · (d < 0.04) · closedness                # 夹爪梯度桥（见下）
reward += is_grasped                                    # 抓住，二值 +1
reward += lift · is_grasped                             # 抬起，[0,1]
reward += 2.0 · (1 - tanh(5 · d_cube→box)) · is_grasped # 搬运，[0,2]
reward  = where(placed, 6.0 + 1.5 · home_term, reward)  # 入盒 + 回位
reward[success] = 8
```

**每一项的依据**：

| 项 | 权重 | 为什么 |
|---|---|---|
| 接近 `1−tanh(5d)` | 1 | 经过验证的抓取奖励配方（ManiSkill PickCube / lerobot-sim2real） |
| **夹爪梯度桥** | 0.5，且只在 `d<4cm` 时给 | 二值的 `is_grasped` 在第一次成功抓取前对夹爪维度**没有任何梯度**，而这个 VLA 的探索**实测在约 130 万次 rollout 转移里一次都没闭合过**。所以补一条"靠近时闭合就给分"的连续上坡路。**只在近处给**——早期版本给了"远处保持张开 0.3"，结果策略学会了**永久悬停张爪**：那一项每回合能领约 60 步，而近处的闭合项只有约 5 步，**时间积分完全压倒**。评估任何整形项都要算**每回合时间积分**，不是看每步权重。 |
| 抓住 | +1（二值跳变） | 让抓取严格优于悬停 |
| 抬起 / 搬运 | 1 / 2，都被 `is_grasped` 门控 | 门控防止空手刷分 |
| 入盒 6.0 + 回位 1.5 | | 成功判据是"入盒**且**回位" |
| 成功 | 8 | |

**防刷分算术（启动前必须手算一遍）**：

```
握着方块悬在盒子上方（不松手）：1 + 1 + 1 + 2 ≈ 5.4/步
放进盒子但没回位：            6.0/步
放进盒子且回位：              7.5/步
成功：                        8.0/步
```

**松手严格优于握着，回位严格优于赖着，成功优于一切。**

> **成功即终止的陷阱**：若成功会终止回合，则"成功"反而**放弃**了后续的稠密奖励。实测：悬停不放手的回合回报约 **196**，而成功并终止只有约 **35** —— PPO 会理性地学出"抓住就不放"。指纹是**奖励上升而成功率趋零**。
> 解法：`env.train.ignore_terminations: True`，让成功状态**每步都付钱**。

### P.3 配置文件：`examples/embodiment/config/so101_ppo_v6_official.yaml`

这份配置是**逐字对齐 πRL 官方配方**（`libero_spatial_ppo_openpi_pi05.yaml`），只做了四处任务性适配，每处都标了 `[TASK]`。

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ \
  --config-name so101_ppo_v6_official \
  runner.logger.log_path=$RESULTS/so101_ppo_v6
```

**算法参数**

| 参数 | 值 | 依据 |
|---|---|---|
| `adv_type` / `loss_type` | `gae` / `actor_critic` | 标准 PPO |
| **`update_epoch`** | **1** | **官方高起点配方的关键**：每个样本只用一次。离策略复用是毒药——我手搓的那版用 2，配上更少的数据，每条新样本承受的离策略压力反而更大 |
| `clip_ratio_low/high` | 0.2 / 0.2 | 官方（我手搓版收到 0.1，无益） |
| `clip_ratio_c` | 3.0 | 官方 |
| `value_clip` / `huber_delta` | 0.2 / 10.0 | 官方 |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | 官方 |
| `entropy_bonus` | 0 | 官方；熵靠 flow 噪声提供，不靠奖励项 |
| `entropy_type` | `token_level` | 官方 |
| `reward_type` / `logprob_type` | `chunk_level` | 动作块级 |
| `normalize_advantages` | True | |
| `rewards_lower_bound` / `upper_bound` | 0.1 / 0.9 | 官方 |
| `kl_beta` | 0.0 | 官方不加 KL 惩罚 |

**采样与批量（这段算术错了会静默毁掉整个运行）**

| 参数 | 值 | 依据 |
|---|---|---|
| `env.train.total_num_envs` | 64 | 官方 |
| `env.train.rollout_epoch` | 3 | **保持官方不变量：每轮 24,576 条全新样本**。算式：64 环境 × (640 步预算 ÷ 5 步动作块 = 128 块) × 3 = 24,576 |
| `global_batch_size` | 2048 | 官方 → 每轮 12 次 minibatch 更新，全部在策略上 |
| `micro_batch_size` | 32 | **[显存]** 官方 128 是给 128×128 图像调的；我们是 640×480（激活约 25 倍）会 OOM。这是**纯梯度累积粒度**，损失按 1/累积数缩放、无 BatchNorm，**数学上等价** |
| `env.train.ignore_terminations` | **True** | **[TASK]** 见 P.2 的终止陷阱 |
| `max_episode_steps` | 640 | **[TASK]** 30 Hz 真任务预算（官方 240 是 LIBERO 的） |
| `num_action_chunks` | 5 | 官方 |

> **每轮更新次数是这套保守配方的真正标定量，不是某个旋钮的数值。** 保持 `global_batch_size = num_envs × (预算 ÷ 动作块长)` 才能做到"每轮恰好一次更新"。历史上把回合预算调成三倍，**静默地把每轮更新数也变成三倍**，重新触发了 BC 侵蚀：训练成功率在涨、评测却从 71.9% 掉到 30%（低于 46.9% 的零样本起点）。**train↑ / eval↓ 同时出现就是这个指纹。**

**优化器**

| 参数 | 值 | 依据 |
|---|---|---|
| `lr` | 5.0e-6 | 官方策略学习率 |
| **`value_lr`** | **1.0e-4** | **官方给价值头单独的、快 20 倍的学习率**——新价值头靠设计解决，**不需要 warmup 阶段**。（我曾把"冷 critic"当成崩塌主因去加 warmup，已证伪：warmup 确实压低了 value_loss，策略照塌） |
| `critic_warmup_steps` | 0 | 同上 |
| `weight_decay` / `clip_grad` | 0.01 / 1.0 | 官方 |
| `sharding_strategy` | `no_shard` | 官方 |

**探索噪声**

| 参数 | 值 | 依据 |
|---|---|---|
| `noise_method` | `flow_noise` | 本仓库 SO101 链路用的是 flow_noise（LIBERO 官方变体用 flow_sde） |
| `noise_params` | `[0.16, 0.12, 200]` | **maniskill 官方值，未减半**——即使起点已有 77% 成功率也不减 |
| `num_steps` | 4 | 与 flow_noise 配套 |

**起点与统计量**

```yaml
actor.model.model_path:   <阶段 D 或 E 的最优检查点>
rollout.model.model_path: <同上，必须一致>
actor.model.openpi.config_name: pi05_so101_v4      # 数据变换管线
actor.model.openpi_data.norm_stats_path: assets/pi05_so101_v4/.../norm_stats.json   # 血统冻结
actor.model.add_value_head: True
```

### P.4 监控与判读

**把奖励值翻译成行为等级再看**，不要盯原始数字：

| 平均奖励 | 对应行为 |
|---|---|
| ~0.20 | 只会接近 |
| ~0.28 | 开始闭合夹爪 |
| >0.4 | 出现抓取 |
| >5 | 抓住并搬运 |
| →8 | 成功 |

**必须监控的三个指纹**：

| 指纹 | 含义 | 处置 |
|---|---|---|
| 奖励**上升**而成功率**趋零** | 终止陷阱（成功即终止，放弃了后续稠密奖励） | 检查 `ignore_terminations` |
| 训练成功率**上升**、评测成功率**下降** | 快速 BC 侵蚀：每轮更新数超了 | 核对批量算术 |
| 训练成功率**持平**、评测从峰值**单调衰减** | 慢速侵蚀。**训练指标被噪声税盖住了，看不见衰减；评测是唯一真信号** | 采收峰值检查点并停 |

**采收律（RL-from-BC 是一个有限的改进窗口）**：即使配方完全正确，也是**先升后降**。实测一次成功放大：46.9% → 75.0%（第 100 轮）→ 维持 65–72% 到第 140 轮 → 第 320 轮衰减到 10.9%。**峰值检查点就是交付物**：

- `save_interval ≤ val_check_interval`，保证峰值一定被存下（历史上峰值在第 10 轮而存点间隔 50，峰值直接丢了）；
- **自动停机写进启动脚本本身**（评测低于最佳值 20 点、或连续 3 次低于零样本基线就停）。写在会话里的守卫会随会话消失——那次崩塌阈值在第 250 轮被跨过，无人叫停，白烧了 180 轮。

### P.5 历史结果（含失败）

| 尝试 | 配方 | 结果 |
|---|---|---|
| 真机 SFT 起点直接 RL | 默认 | **零抓取**（5 次运行、约 2000 轮） |
| 仿真 SFT 起点 + 默认参数 | 默认 | 起点约 50%，**10 轮内塌到 0**，三次重现 |
| 冻结测试 | `lr=1e-9` | 行为**存活 66 轮** → 凶手是更新步骤本身，不是环境/奖励/critic |
| 手搓保守组 | 噪声减半、lr 2e-6、`update_epoch` 2、clip 0.1 | **唯一一次成功放大**：46.9% → 75.0%（第 100 轮），随后单调衰减到 10.9%（第 320 轮） |
| 高起点（81.6%）再用手搓保守组 | 同上 | **30 轮掉 53 点** |
| 同一天改用专家迭代 | 阶段 D | **+18 点** |
| **官方配方（本节配置）** | `so101_ppo_v6_official.yaml` | **尚未运行** |

**结论性判断**：高起点崩塌**不是必然的**，是配方问题——πRL 论文用同一框架把 PI0.5 在 ManiSkill 25 任务上从 40.1% 推到 90.9%。所以在高能力起点上的优先级是：**(1) 再做一轮专家迭代 → (2) 严格对齐官方配方的 PPO → (3) 永远不要在验证区间之外使用手搓的保守参数组**（它只在规划器-BC 起点约 45–50% 的区间被验证过）。

### P.6 已知陷阱清单

| 陷阱 | 症状 | 处置 |
|---|---|---|
| 从"坏奖励"训出的检查点热启动 | 后代全部收敛到同一个悬停行为 | **血统被污染，整条作废**。实测：两次用修好的奖励、但热启动自坏奖励检查点，都收敛到同样的悬停；而未污染的 SFT 起点配同样奖励，14 轮内出现首次成功 |
| 相邻动作块看到的画面几乎没动 | 高频小块 | 核对动作块时长 vs 实际运动尺度 |
| `enable_offload: True` | —— | 官方是 False；我当初开它是在追一个误诊 |
| 显存不足 | OOM | 只动 `micro_batch_size`（纯梯度累积），**不要动 `global_batch_size`**（会改变每轮更新数） |

---

## 阶段 F —— PPO 在线微调（**已跑通，2026-08-14**）

> 附录 P 记录的是"怎么接线"（那一版从未跑通）。这一节记录的是**实际跑通的那次**：起点、每个改动的**实测依据**、完整命令、以及为什么其它配置会失败。
> 结果：环 1 上 eval **61.7% → 73.4%**（峰值 @ step 29），检查点 `so101_ppo_v13/.../global_step_30`。

### F.0 三个决定成败的参数（其余抄官方即可）

这三个都不是猜的，每个都有对照实验：

| 参数 | 官方/原值 | **改成** | 实测依据 |
|---|---|---|---|
| `actor.model.num_action_chunks` | 5 | **10** | 冻结探针：带噪 rollout 成功率 1.0% → **4.7%**，且**确定性成绩 55% → 66.4%**（白捡 11 点）。模型 SFT 时 `action_horizon=10`，只执行 5 步等于丢掉一半预测并过度重规划 |
| `openpi.noise_logvar_range` | 默认 `[0.08,0.16]` | **`[0.02,0.04]`** | 冻结探针：带噪 rollout 4.7% → **39.1%**。⚠️ **不要去调 `noise_params`**——那是 flow-SDE 的参数，`flow_noise` 不读它（见 `openpi_action_model.py:47-58`）。我扫了 4 个档位（跨度 8 倍）结果完全一样，就是因为调的是空参数 |
| **每轮更新次数** | 12（官方） | **1** | v12（12 次/轮）9 轮内从 61.7% 掉到 7.0%；v13（1 次/轮）稳定爬到 73.4%。**这是决定性的那一个** |

**每轮更新次数怎么算**：`samples = num_envs × (max_episode_steps ÷ num_action_chunks) × rollout_epoch`，`updates = samples ÷ global_batch_size`。
要做到 1 次/轮，就令 `global_batch_size = samples`：

```
64 环境 × (640 ÷ 10) × rollout_epoch 1 = 4096 样本
global_batch_size = 4096  →  1 次更新/轮
per-rank 512 ÷ micro_batch 32 = 16 次梯度累积
```

### F.1 先决条件：必须在 **rollout 分布**下测

启动前用**冻结测试**（真实训练路径，`lr=1e-9`，权重不变）测一轮，它同时给出两个数：

```bash
export SO101_SPAWN_FRAC="0.4294,0.9115,0.5142,0.9817"   # 环 1
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=$RESULTS/probe \
  runner.val_check_interval=1 runner.save_interval=1000 runner.max_epochs=1 \
  actor.optim.lr=1e-9 actor.optim.value_lr=1e-9 \
  env.train.rollout_epoch=1 \
  actor.model.num_action_chunks=10 actor.global_batch_size=4096 \
  "+actor.model.openpi.noise_logvar_range=[0.02,0.04]"
```

然后从 tensorboard 读 **`env/success_once`（带噪，PPO 真正学习的分布）** 与 **`eval/success_once`（确定性）**。

| 门槛 | 值 | 依据 |
|---|---|---|
| `env/success_once` | **≥ 5%** | 历史四次运行的分界线：pp4 5–9%、v10 10–15% 都放大成功；v6 0.5%、v11 1.0% 从未起来。**必要不充分**——v12 有 39% 仍然崩，因为更新次数不对 |
| `eval/success_once` | 不低于起点太多 | 防止为了抬高 rollout 成功率而把策略本身改坏（例如动作块开太长） |

**不要用 `runner.only_eval=True` 当探针**：它同时切换模型规格来源（`config.py:830`）并跳过训练环境创建（`env_worker.py:108`），是另一条代码路径。

### F.2 正式启动（v13 实际用的命令）

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

配置文件 `examples/embodiment/config/so101_ppo_v11.yaml`（πRL 官方配方 + 起点换成 v10），命令行只覆盖上面这几项。
起点：`so101_sft_v10/.../global_step_1000`。启动器（含自动停机守卫）：`tools_so101_session/ppo_train.sh`。

| 其余参数 | 值 | 来源 |
|---|---|---|
| `update_epoch` | 1 | 官方（每条样本只用一次） |
| `lr` / `value_lr` | **2e-6** / 1e-4 | lr 降自官方 5e-6（配合 1 次/轮）；value_lr 保持官方，价值头靠它自己快速学，不需要 warmup |
| `clip_ratio` / `value_clip` / `huber_delta` | 0.2 / 0.2 / 10.0 | 官方 |
| `entropy_bonus` | 0 | 官方（熵由 flow 噪声提供） |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | 官方 |
| `ignore_terminations` | True | 任务性：成功即终止会让"成功"反而放弃后续稠密奖励 |
| `micro_batch_size` | 32 | 显存；纯梯度累积，不改数学 |

### F.3 实测曲线（每 5 轮一次评测）

| step | 4 | 9 | 14 | 19 | 24 | **29** | 34 | 39 | 44 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval | 60.9 | 58.6 | 62.5 | 58.6 | 70.3 | **73.4** | 69.5 | 70.3 | 68.0 | 68.8 |

约 10 分钟一轮（采样 + 1 次更新 + 每 5 轮一次评测）。峰值在 step 29，之后进入 68–70% 平台。**峰值检查点 = 交付物**，`save_interval` 必须 ≤ `val_check_interval`，否则峰值存不下来。

训练侧健康指标：`env/success_once` 30–41%（未塌）、`approx_kl` 0.010–0.014（策略移动平稳）、`value_loss` 360–470（平稳无发散）。

### F.4 自动停机守卫（必须写在启动器里，不能写在会话里）

`tools_so101_session/ppo_train.sh` 每 5 分钟读一次 tensorboard 的 `eval/success_once`：

- **崩塌**：某次评测低于历史峰值 **20 点** → 立即杀训练；
- **无收益**：连续 **3 次**低于第一次评测 5 点以上 → 停。

v12 就是被第一条在 02:25 自动停掉的（step 9，7.0% vs 峰值 35.9%），没有重演历史上"白烧 180 轮"。

### F.5 失败配置对照（不要重走）

| 配置 | 带噪 rollout | 结果 |
|---|---|---|
| chunks=5、默认噪声、12 次更新/轮（= 官方配方原样） | 1.0% | step 9 时 eval 归零 |
| chunks=10、logvar 0.02/0.04、**12 次更新/轮** | 39.1% | step 4 掉到 35.9%，step 9 掉到 7.0%，自动停机 |
| **chunks=10、logvar 0.02/0.04、1 次更新/轮** | 39.1% | **61.7% → 73.4%** ✅ |
| chunks=20 | 探针失败 | 未查明，暂不用 |

**结论**：在这个任务上，官方配方的 `12 次更新/轮` 是致命项。原因推测是时域长度——每回合 128（chunks=5）或 64（chunks=10）次带噪决策，而官方 ManiSkill 只有 16 次；同样的更新强度作用在更脆的 BC 脊上。
