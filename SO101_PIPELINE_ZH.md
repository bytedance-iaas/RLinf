# SO101 + PI0.5：从真机数据集到 RL 微调策略的完整可复现流程

**输入**：`henry-guo/so101-pick-place-v2`（87 集真机遥操作数据）+ PI0.5 基座权重。
**输出**：一个在仿真里抓取红方块、放入托盘、并回到初始位姿的策略。
**总耗时**：约 35 小时（8×H200），其中约 12 小时是纯 CPU 的示范生成与视频编码。
**代码**：分支 `henryg/pi05-maniskill-so101`（github.com/bytedance-iaas/RLinf）。

---

## 摘要

| 阶段 | 手段 | **测试区域** | 成绩（诚实口径） |
|---|---|---|---|
| 起点 | 仅用 87 集真机数据 SFT | 全板 426 cm² | **0.0%** |
| B | 全板仿真示范 SFT（420 条） | 全板 426 cm² | 12.5% |
| C | 生成区收窄，示范间距降到 0.44 cm | 框内 48 cm² | **56.7%** |
| D | 专家迭代（策略自采成功 + 规划器示范混合） | 框内 48 cm² | **76.6%** |
| E | 生成区扩到 96 cm² | 环 1 96 cm² | 55.1%（负结果） |
| **G** | **PPO 在线微调** | 环 1 96 cm² | **57.8%**（起点 52.0%，**+5.9**） |

> ⚠️ **各行的测试区域不同，不能直接纵向比较。** 区域越大越难：同一个最终策略在框内是 **77.3%**、环 1 是 **57.8%**、全板是 **14.8%**。
> 按同一区域看：框内 0% → 76.6%；环 1 52.0% → 57.8%。

**任务判据**：红方块放入托盘 **且** 手臂回到初始位姿——比常见的"放进去即成功"更严。所有数字都来自**从未参与任何挑选的种子**。

**三个可复用的结论**：

1. **示范密度决定 BC 的地板**，而不是示范总量。间距 = √(生成面积 ÷ 示范条数)，与抓取容差（±0.7 cm）比较；实测 1.01 cm→12.5%、0.44 cm→56.7%，是阈值不是渐变。
2. **专家迭代是零风险的放大器**（+20 点），而 PPO 有条件才成立。
3. **PPO 能否放大，取决于一个此前没人量过的量**：策略在**带探索噪声的 rollout 分布**下的成功率（不是确定性评测下的）。二者在本任务上可以差 50 个点以上。

**当前状态**：仿真内的训练已完成并归档；**真机部署被一个已量化的视觉域差距阻断**（见 §11）。

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
| **血统冻结** | **血统 = 权重的继承链**：每个阶段都从上一阶段的检查点热启动，所以它们必须用同一套输入解释方式，也就是同一份 `norm_stats`（把关节数值缩放到模型习惯范围的均值/标准差）。**链内绝不重算** | A/B 实测：只换统计量、其余全不变，19.5%→9.4%——上游传下来的权重突然看到被换了刻度的输入，它学到的"关节角 0.3 意味着什么"整个错位 |
| **RL 是放大器** | PPO 放大已有的成功，不发现新成功；先决条件必须在 **rollout 分布**下测 | 起点为 0 时 RL 起不来（两次运行：全程 0、峰值 3–4%）；带噪成功率仅 1% 时，PPO 9 轮内摧毁策略 |

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

全部在 `tools_so101_session/`（79 个脚本；`SO101_TOOLS_RUNBOOK_ZH.md` 是完整索引，含每个脚本"要解决什么问题"）。
下面只列本流程直接用到的，并说清**它做什么、怎么用、有什么坑**。

### 2.1 造数据

**`gen_planner_demos.py` —— 主力示范生成器（阶段 B、C、E）**

脚本化的运动规划器，不含任何神经网络。一条示范的完整流程：复位 → **抬臂到就绪位姿**（24 步插值）→ 用方块的定向包围盒算抓取位姿 → 上方 3cm 下降到 1cm → 闭合夹爪并**保持 10 帧让它物理上真的合拢** → **微提验证**（抬 3cm，方块必须升高 >1.5cm，否则判定抓取失败重试）→ 两段式 FK 搬运（先垂直提升再平移）→ 闭环修正落点 → 松爪 → 30 步插值回到初始位姿。**只有成功的才留下**。

```bash
SO101_SPAWN_FRAC="x0,x1,y0,y1" python tools_so101_session/gen_planner_demos.py     --num 30 --seed0 90000 --out $DATA/demos_w0
```

| 参数 | 含义 |
|---|---|
| `--num` | 要**多少条示范**，不是尝试多少次——每条最多试 3 个变体（投放点微调、投放高度、抓取点抖动） |
| `--seed0` | 起始种子；多进程并行时各进程种子区间必须不重叠，否则生成重复轨迹 |
| `SO101_SPAWN_MODE=legacy` | 把方块限制在 6×8cm 小框 |
| `SO101_SPAWN_FRAC="x0,x1,y0,y1"` | 按全板比例限制到任意子矩形 |

**两个关键设计**，都是踩坑后加的：**微提验证**——ManiSkill 的 `is_grasping` 靠接触力，夹住一个角、其实拿不起来也会返回 True，没有这一步会有一批"看起来抓住了、一搬运就掉"的假示范混进训练集，而模型会老实把这种失败模式学下来。**抬臂前缀**——真实复位姿态是手臂折叠、夹爪近闭合，而规划器的抓取例程假设张开的夹爪和腕滚 π/2；不加这个前缀，成功率从 45/64 掉到 1/60。

**`gen_demos_annulus.sh` —— 只补新增区域（阶段 E）**

扩大生成区时，旧区域已经有足够示范，重新生成整个区域是浪费。这个脚本把**环形带**（新区域减去旧区域）切成四条边带，按**面积配额**分配生成量（左33/右33/下23/上23），8 个 worker 并行。配额来自密度律：需要多少条 = 面积 ÷ 目标间距²。

**`collect_policy_successes.sh` —— 让策略给自己造数据（阶段 D、E）**

不是新写的采集程序，而是**在评测回路里挂一个录制器**：设 `SO101_COLLECT_DIR` 后，`ManiskillEnv` 每步把 (前视图, 腕视图, 状态, 动作) 追加进**内存缓冲区**；某一路环境**第一次成功**时，才把整段缓冲区**写成一个 `.npz` 文件存到磁盘**。失败的回合一直留在内存里，随下一次 reset 被丢弃，不产生文件。

```bash
SO101_COLLECT_DIR=$DATA/rollouts SO101_SPAWN_MODE=legacy python evaluations/eval_embodied_agent.py --config-name so101_eval_openpi_pi05     rollout.model.model_path=$CKPT env.eval.seed=2001 env.eval.total_num_envs=128
```

三条纪律：**只在第一次成功时写文件**（成功之后策略还会继续动，只保留到成功那一刻）；**必须用从未用过的种子**（采过的局面等于进了训练集）；**评测是确定性的**（录的是策略的最好水平，不是抖出来的偶然成功）。

### 2.2 转成训练数据集

**先说三种文件格式**，后面反复出现：

| 格式 | 谁产生 | 是什么 |
|---|---|---|
| **`.h5`**（HDF5） | 规划器（ManiSkill 原生） | 一个文件里装一个"文件系统"：`traj_0/obs/agent/qpos`、`traj_0/obs/sensor_data/3rd_view_camera/rgb`、`traj_0/actions`……一个文件放几十条轨迹，图像**未压缩**（一条 391 帧的双相机轨迹就 720 MB）。外带一个同名 `.json` 记录每条轨迹的 `episode_id` / `episode_seed` / **`success`**，转换器靠它挑出成功的那些 |
| **`.npz`** | 策略自采的录制器（numpy 原生） | 扁平的数组打包，一个文件一条轨迹 |
| **LeRobot**（parquet + mp4） | 转换器的产出 | 训练器只认这个。状态/动作进 parquet，图像编码成视频，体积小两个数量级 |

转换做四件事：**统一单位** → **长度过滤** → **编码成视频** → **写索引**。视频编码占 90% 的时间（约 3.3 集/分钟）。

| 脚本 | 用在 | 它的特殊之处 |
|---|---|---|
| `convert_fullboard.py` | 阶段 B | 纯规划器 h5；**同时产出整条血统共用的 norm_stats**（见下） |
| `convert_narrow_box.py` | 阶段 C | 窄框规划器 h5；不重算 norm_stats |
| `convert_expert_iter.py` | 阶段 D | **混合两种来源**：规划器 h5 + 策略 npz |
| `convert_append_region.py` | 阶段 E | **在上一版数据集副本上追加**，不重编码旧集 |
| `convert_cotrain_simreal.py` | sim2real | 把真机集追加进仿真数据集 |

**它们不能互相取代**：`convert_append_region.py` 的追加技巧只在"已有数据集可追加"时成立，从零复现时前面没有可追加的东西。

**单位不对称是这里最容易出错的地方**：

| 来源 | `state` | `action` |
|---|---|---|
| 规划器 h5 | 弧度 → 要转 | 弧度 → 要转 |
| 策略自采 npz | **已归一化** → 直接用 | 弧度 → 要转 |
| 真机数据 | 已归一化 | 已归一化 |

搞错不会报错，只会让模型学错。

**长度过滤的上限要按数据来源定**：仿真示范用 580（挡掉超时挣扎的轨迹），但真机是人类遥操作、更慢，87 集是 395–825 帧、中位 575——**用 580 会静默丢掉 46% 的真机数据**。这个坑在 sim2real 那步实际发生过，靠启动前先量分布挡住。

### 2.3 成套流水线

| 脚本 | 串起哪几步 |
|---|---|
| `pipeline_expert_iteration.sh` | 采集策略成功 → 与规划器示范混合 → 轻量 SFT → 门评（阶段 D） |
| `pipeline_region_expand.sh` | 转换 → SFT → 环形门评 → 检查点清理（阶段 E） |
| `pipeline_cotrain_simreal.sh` | 建混合数据集 → SFT → **双轴门评**（sim2real） |

这些脚本都是 `setsid nohup bash <脚本> &` 启动、无人值守跑完。它们内部都遵守同一套约定：**超时按实测速率定**、**阶段间用哨兵串行**、**每次评测前完整清理 Ray 与 /dev/shm**、**评测失败重试 3 次**。

### 2.4 PPO 专用

**`ppo_freeze_probe.sh` —— 启动前的先决条件测量**

跑**真实的训练路径**（同样的 worker、环境创建、模型构建、权重同步），但把学习率设成 `1e-9`，权重实质不变。一轮就同时给出两个数：

- `env/success_once`：**带探索噪声的 rollout 成功率**——PPO 真正学习的分布，**门槛 ≥5%**；
- `eval/success_once`：确定性评测成功率——用来确认为了抬高前者没有把策略本身改坏。

这两个数在本任务上**可以差 50 个点以上**。不测这个直接启动 PPO，就是七次失败的原因。

**`ppo_train.sh` —— 启动器 + 自动停机守卫**

守卫每 5 分钟读 tensorboard 的 `eval/success_once`：低于历史峰值 20 点 → 立即杀训练；连续 3 次低于第一次评测 5 点以上 → 停。**必须写在启动器里**，写在会话里会随会话消失（历史上因此白烧 180 轮）。

### 2.5 验证与体检

| 脚本 | 回答什么问题 |
|---|---|
| `verify_honest_seeds.sh` | "这个成绩能对外说吗"——用**从未参与挑选**的种子重测 |
| `verify_baseline_control.sh` | "涨的是真本事还是评测集选择效应"——同种子同条件测**起点** |
| `offline_replay_check.py` | "策略在真实观测下还能不能用"——**上机前的门**，不碰机器人 |
| `toolkits/preflight_config.py` | "这次启动会不会白跑一夜"——路径存在性、**批量算术**、模型-数据一致性 |
| `toolkits/invariant_audit.py` | "有没有不报错但结果是错的地方"——9 项静默错误检查 |

`preflight_config.py` 的**批量算术**那一项对 PPO 是命门：`updates/epoch = num_envs × (预算÷块长) × rollout_epoch ÷ global_batch`，这个数决定 PPO 是放大还是摧毁策略。这个工具自己曾漏算 `rollout_epoch` 因子、把 12 次报成 4 次——守门的工具算错，等于没有门。

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
.venv/bin/python tools_so101_session/gen_planner_demos.py --num 12 --seed0 79000 --out $DATA/probe
```
门槛：≥8/12 成功、中位长度 ≤530 步。**不过就不要往下走**——规划器做不到的，BC 和 RL 都做不到。

**B2 分层生成 420 条全板示范**（4×4 格、每格 45 次尝试、8 worker）：

```bash
SEED=80000
for XI in 0 1 2 3; do for YI in 0 1 2 3; do
  SO101_SPAWN_FRAC="$(echo "$XI*0.25"|bc -l),$(echo "($XI+1)*0.25"|bc -l),$(echo "$YI*0.25"|bc -l),$(echo "($YI+1)*0.25"|bc -l)" \
  .venv/bin/python tools_so101_session/gen_planner_demos.py --num 45 --seed0 $SEED \
      --out $DATA/v4_demos_cell_${XI}_${YI} &
  SEED=$((SEED+100)); done; done; wait
```

**B3 转换 + 计算 norm_stats（仿真域只算这一次）**：

```bash
.venv/bin/python tools_so101_session/convert_fullboard.py
.venv/bin/python -m toolkits.lerobot.calculate_norm_stats --config-name pi05_so101_v4
```

> **这是第二次调用 `calculate_norm_stats`，但不是重复劳动。** 归一化统计量定义"模型看到的数值世界"，而真机与仿真是两个不同的分布：
>
> | 调用 | 产出 | 谁用 |
> |---|---|---|
> | 阶段 A 的 `--config-name pi05_so101` | `assets/pi05_so101/henry-guo/so101-pick-place-v2/norm_stats.json` | **只有阶段 A** |
> | 这一次 `--config-name pi05_so101_v4` | `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json` | **阶段 B 之后全部**，且从此冻结 |
>
> **此后 C/D/E/G 一律沿用 v4 那份，绝不重算**（各阶段 yaml 里都硬写着这个路径，注释是 `lineage frozen: continuing v4`）。

**B4/B5 SFT + 门评**（`so101_sft_v4`，lr 2.5e-5、4000 步、save 1000）：

**验收：最优点是 `global_step_1000`，全板约 12.5%。** 之后单调下降（2.0→2.3→0.0）——SFT loss 降到 0.002 却越训越差，是**过拟合规划器习惯**，所以最优点在最早期。

> ⚠️ 历史上阶段 B 曾从一个更早的、任务规格已废弃的检查点热启动。本文档改为从阶段 A 热启动，**该替换已用对照实验验证**：全板 10.2% vs 对照 12.5%（差 2.3 点，阈值 3 点），且同期用 v4 原检查点重测今天的环境得到 12.5%（与两天前完全一致，证明环境未漂移）。

---

## 5. 阶段 C —— 收窄生成区（用密度换成绩）

全板 426 cm² 要达到 0.44 cm 间距需要约 2200 条示范；先收窄区域，用同样的数据量换密度。

```bash
export SO101_SPAWN_MODE=legacy      # 唯一的收窄项：6×8 cm = 48 cm²
for W in 0 1 2 3 4 5 6 7; do
  .venv/bin/python tools_so101_session/gen_planner_demos.py \
    --num 32 --seed0 $((90000 + W*1000)) --out $DATA/v8_demos_w$W & done; wait
.venv/bin/python tools_so101_session/convert_narrow_box.py     # 不重算 norm_stats
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

**D（专家迭代，+20 点）**：用当前策略在 8 个**从未用过**的种子上采集自己的成功轨迹（`collect_policy_successes.sh` 的做法），与原始规划器示范**混合**后轻量 SFT。

```bash
export SO101_COLLECT_DIR=$DATA/v9_rollouts
for SEED in 2001 2002 2003 2004 2005 2006 2007 2008; do
  SO101_SPAWN_MODE=legacy .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-name so101_eval_openpi_pi05 rollout.model.model_path=$V8 ... env.eval.seed=$SEED
done
.venv/bin/python tools_so101_session/convert_expert_iter.py    # 247 规划器 + 425 策略 = 672 集
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

5% 来自本项目四次历史运行的实测分界：**两次放大成功的起始带噪成功率是 5–15%，两次从未起来的是 0.5–1.0%**。**必要不充分**——本次探索中有一个变体带噪成功率 39% 仍然崩塌，因为更新次数不对。

**不要用 `runner.only_eval=True` 当探针**：它同时切换模型规格来源并跳过训练环境创建（`config.py:830`、`env_worker.py:108`），是另一条代码路径。

### 7.3 启动

```bash
export SO101_SPAWN_FRAC="0.4294,0.9115,0.5142,0.9817"
bash tools_so101_session/ppo_train.sh        # 含自动停机守卫，推荐
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
| 起点（阶段 E 产物） | 48.4% | 55.5% | 52.0% |
| **PPO 后** | **58.6%** | **57.0%** | **57.8%** |
| 增益 | +10.2 | +1.5 | **+5.9** |

框内参照 77.3%、全板参照 14.8%。

> **报告里请引用 +5.9 这个诚实增益。** 门评口径上是 61.7%→73.4%（+11.7），但峰值检查点正是在那套固定评测局面上挑出来的，会偏乐观。

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
| `SIM2REAL_PLAN_ZH.md` | 真机部署方案与阻断项 |
| `V10_REPRODUCTION_ZH.md` | 阶段 A–E 的更详细版本 |
| `tools_so101_session/README.md` | 59 个脚本的用途索引 |

---

## 11. 局限与下一步（写报告时必须一并说明）

### 11.1 真机迁移尚未打通，且原因已量化

**不能直接上机。** 用 87 集真机数据做的**离线检验**（不碰机器人）：把真实的（前视、腕视、关节状态）喂给策略，比较它预测的动作与人类当时的实际动作。两个对照让数字可解释——仿真数据（分布内）与"完全不动"（动作幅度的尺度）。

| 输入配置 | 仿真比值 | 真机比值 |
|---|---|---|
| 前视 + 腕视（训练时的配置） | **0.10** | **4.47** |
| 只喂前视 | 0.34 | **4.59** |

比值 <1 表示策略优于"什么都不做"。**仿真 0.10（好 10 倍），真机 4.5（差 4.5 倍）。**

**砍掉腕部那一路后真机比值几乎不变**（4.47→4.59），所以这**不只是**"仿真腕部相机指向错误"这一个缺陷——即使不用它，策略仍读不懂真实的前视图像。**这是一个更广的视觉域差距**。

复现命令：

```bash
python tools_so101_session/offline_replay_check.py \
  --ckpt <检查点> --real-root <真机数据集> --episodes 5 --frames 12
```

### 11.2 三条可能的出路（均未验证）

| 路线 | 做法 | 代价 |
|---|---|---|
| 真机数据协同训练 | 把 87 集真机数据与仿真示范混合再微调 | 约 1 h，但真机数据量很小 |
| 域随机化 | 仿真渲染时随机化光照/材质/相机位姿 | 需改环境 + 重新生成示范并重训 |
| 提高仿真保真度 | 先修腕部相机指向（仿真里它拍的是机械臂自己），再逐项对齐外观 | 约 30 h 机时，全流程重来 |

**任何一条都必须重新过 §11.1 那个离线检验，比值 <1 才允许接机械臂。**

### 11.3 其它已知局限

| 项 | 内容 |
|---|---|
| 覆盖区域 | 策略只在 96 cm² 的环 1 内训练；全板（426 cm²）成绩仅 14.8%。真机摆放必须限制在对应矩形内 |
| 生成区边距 | 2 cm 的边距排除了约 11% 的真实方块起始位置 |
| 扩域未解决 | 把生成区从 48 扩到 96 cm² 没有带来提升（阶段 E 是负结果）；密度律不能外推成"扩域只要补够密度" |
| 物理参数 | 摩擦系数等未经真机标定，属声明的默认值；方块质量按用户给的上界（<10 g）设定 |
| 单一任务 | 只验证了"抓红方块放托盘并回位"这一个任务 |
