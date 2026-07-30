# RLinf × Isaac Lab × SO101 × π₀.₅ 开工文档

> 版本：v3.0（FINAL，可开工） · 2026-07-28
> 代码基线：`backup/fix-volcengine-downloads-dirty-worktree-20260726`
> 目标：以 Isaac Lab 为仿真后端，用 RLinf 对 π₀.₅ 做 PPO 训练，最终跑在 SO101 上
>
> 所有带 `文件:行号` 的引用均已在对应仓库中核实。

## v3.0 的两个决策输入（相对 v2.0 的实质变化）

| 输入 | 影响 |
| --- | --- |
| **SO101 上基于 π₀.₅ 的 SFT checkpoint 已有** | 原最高风险项 Phase 4a 从「训 SFT」变为「验证 ckpt 与目标任务的匹配度」。**关键后果：协议不再由我们定义，而是由 ckpt 反推**（见 [3.1](#31-动作--状态协议表以-sft-ckpt-为准)、[3.4](#34-必须从-sft-checkpoint-确认的清单-phase-0-第一项)）。总工期由 2–4 周降至 **1–2.5 周** |
| **Isaac Lab 基座采用 LeIsaac** | 版本组合直接锁定 LeIsaac 官方验证过的一套，**torch 2.11 vs 2.7 的分歧消除**（见 [3.2](#32-版本冻结表已锁定)）。RLinf/IsaacLab fork 退化为「Franka reward 任务的参考样板」，不再是基座（见 [2.2](#22-rlinfisaaclab-fork-的新定位参考样板) 与 [11 节](#11-基座决策的落地与团队协作)） |

**最重要的一条新洞察**：LeIsaac 自带 openpi 推理链路（`scripts/evaluation/policy_inference.py --policy_type=openpi`）。**这意味着在写任何 RLinf 代码之前，就能先评测已有 SFT ckpt 在目标任务上的成功率** —— 全项目风险最高的一步被前移到第一周。见 [Phase 1.5](#phase-15--提前评测-sft-ckpt-最重要的-gate1-2-天)。

---

## 目录

- [0. 决策摘要](#0-决策摘要)
- [1. 范围界定](#1-范围界定)
- [2. 现状盘点](#2-现状盘点)
- [3. Phase 0：冻结接口](#3-phase-0冻结接口)
- [4. Gap 详解与改动清单](#4-gap-详解与改动清单)
- [5. 路线图与验收标准](#5-路线图与验收标准)
- [6. 已核实的高危坑](#6-已核实的高危坑)
- [7. 工期与风险](#7-工期与风险)
- [8. 实机部署（本期不承诺）](#8-实机部署本期不承诺)
- [9. 待确认清单](#9-待确认清单)
- [10. 参考](#10-参考)
- [11. 基座决策的落地与团队协作](#11-基座决策的落地与团队协作)

---

## 0. 决策摘要

**这条链路可行，且随着 SFT ckpt 到位，主要风险已从「能不能学会」转为「协议对不对齐」。**

```
RLinf 已完成 RL 算法与分布式主干（不用动）
+ LeIsaac 已完成 SO101 仿真资产、相机、数据采集、openpi 契约（可复用）
+ SO101 π₀.₅ SFT checkpoint 已有（可用）
──────────────────────────────────────────────────
仍需： observation / action 适配层（协议以 ckpt 为准）
     + 稀疏 reward（~100 行）
     + ckpt 与目标任务的匹配度验证   ← 现在的最高风险
     + （若要实机）sim-to-real 与安全控制
```

**已定决策（不再讨论）**

| 决策项 | 结论 | 依据 |
| --- | --- | --- |
| Isaac Lab 基座 | **LeIsaac**（含其 `dependencies/IsaacLab` submodule，stock v2.3.0） | 用户决策。附带收益：版本组合被官方验证过，消除 torch 分歧 |
| 动作空间 | **6 维关节空间**（5 臂关节 + 夹爪），不用 7 维 EEF delta | SO101 只有 5 个臂部 DoF，任意 6-DoF 位姿 IK 不可达；LeRobot / LeIsaac / NVIDIA 官方一律走关节控制。**但最终以 SFT ckpt 的实际定义为准** |
| 夹爪 | **连续控制**，不做 `sign()` 二值化 | LeIsaac `so101leader` 分支与臂关节同走 `JointPositionActionCfg`；现有 `IsaacLabOutputs` 的 `np.sign()` 是 Franka 约定 |
| 首个任务 | **`LeIsaac-SO101-LiftCube-v0`**（若与 ckpt 不匹配则改选，见 [3.4](#34-必须从-sft-checkpoint-确认的清单-phase-0-第一项)） | 成功条件简单、episode 短、已有 success termination |
| SFT 来源 | **用户提供的现成 ckpt**，不自己训 | 用户输入 |
| articulation 物理参数 | 先用 LeIsaac 的；sim2real 阶段再考虑切到 isaac_so_arm101 的逐关节精调值 | 见 [4.2](#42-g2--reward稀疏-success) |
| reward 定义位置 | **Isaac Lab 侧**（LeIsaac 的 `RewardsCfg`） | RLinf 的 Isaac Lab 路径不调用 `_calc_step_reward()`，见 [6.1](#61-use_rel_reward-对-isaac-lab-不生效) |
| PPO 前置门槛 | **SFT ckpt 在目标任务上成功率显著 > 0** | sparse 0/1 reward，成功率为 0 时 PPO 不动 |
| 本期交付边界 | **到 PPO 曲线上升为止**；实机作为独立后续项目 | 实机不确定性远大于仿真 |

**工期**：仿真最小可跑 **3–6 天**；可信的仿真 PPO 结果 **1–2.5 周**。实机另计 4–8 周以上。

---

## 1. 范围界定

```
① SO101 Isaac Lab 仿真可用
        ↓
② RLinf 中 π₀.₅ 完成 PPO 训练并提升成功率      ← 本期交付边界
        ↓
③ 部署到真实 SO101                            ← 独立后续项目
```

若把"完全从零"记为 10：

| 目标 | v2.0 Gap | v3.0 Gap | 变化原因 |
| --- | ---: | ---: | --- |
| ① 在 RLinf 中 reset/step SO101 Isaac Lab 环境 | 3/10 | **2/10** | LeIsaac 作基座，obs 结构与 RLinf 高度对应 |
| ② π₀.₅ 在 SO101 仿真中完成 PPO 训练 | 4–6/10 | **3–4/10** | SFT ckpt 已有；openpi 契约可复用 |
| ③ 稳定部署到真实 SO101 | 7–8/10 | 7–8/10 | 未变 |

**本文档只对 ①② 给出承诺。**

---

## 2. 现状盘点

### 2.1 RLinf 侧已现成（不需要动）

| 能力 | 位置 | 说明 |
| --- | --- | --- |
| Isaac Lab 后端 | `rlinf/envs/isaaclab/`（465 行） | 已注册为 `SupportedEnvType.ISAACLAB` |
| Isaac Sim 进程隔离 | `rlinf/envs/isaaclab/venv.py` | `SubProcIsaacLabEnv`，spawn 子进程独占；`AppLauncher` 单进程只能起一次 |
| π₀.₅ + Isaac Lab PPO 参考实现 | `examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml` | Franka 版，作为改写模板 |
| CI e2e 参考 | `tests/e2e_tests/embodied/isaaclab_ppo_openpi_pi05.yaml` | 同上 |
| PPO / GAE / value head / chunk-level reward | `rlinf/algorithms/` | 与机器人无关 |
| 视频录制 | `rlinf/workers/env/env_worker.py:392` → `RecordVideo` | 通用 wrapper，新 env 自动获得 |
| FSDP / Ray placement / checkpoint / eval | 主干 | 无需改动 |
| norm stats 加载 | `rlinf/models/embodiment/openpi/__init__.py:95-129` | 支持 `data_config.asset_id`（`:112`）或显式 `norm_stats_path`（`:97-106`） |

### 2.2 `RLinf/IsaacLab` fork 的新定位：参考样板

基座既已定为 LeIsaac，这个 fork **不再作为 Isaac Lab 的安装来源**，但它的 90 行 reward 任务仍是我们写 SO101 `RewardsCfg` 的最佳样板。

fork 与上游 `isaac-sim/IsaacLab` 的全部差异（7 commit、3 文件）：

| 文件 | 改动 | 对我们的意义 |
| --- | --- | --- |
| `.../franka/stack_ik_rel_visumotor_rewarded_env_cfg.py` | **新增 90 行**：`RewardsCfg`（`success = RewTerm(func=mdp.cubes_stacked, weight=20.0)`）+ env cfg + `TiledCameraCfg` + `sim.render_interval = 5` | ⭐ **抄这个写 SO101 的 reward** |
| `.../manipulation/stack/config/franka/__init__.py` | +10 行，注册 gym id | 注册写法参考 |
| `isaaclab.sh` | torch pin 2.11.0 / 0.26.0；去掉 docker 内 `update_vscode_settings` | ⚠️ **不采用**，改用 LeIsaac 的 2.7.0（见 [3.2](#32-版本冻结表已锁定)） |

> **代价**：不用这个 fork，RLinf 现有的 Franka `isaaclab` e2e 测试跑不了。处理方式见 [11.2](#112-rlinf-现有-franka-e2e-怎么办)。

### 2.3 LeIsaac 侧已现成（基座）

[LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)（Apache-2.0，v0.4.0）：

| 能力 | 位置（LeIsaac 仓库内） | 对我们的价值 |
| --- | --- | --- |
| SO101 Follower USD + articulation cfg | `assets/robots/lerobot.py`（`SO101_FOLLOWER_CFG`） | 直接用 |
| 6+ 操作任务：`PickOrange` / `LiftCube` / `CleanToyTable` / `FoldCloth` / `AssembleHamburger` / `LeKiwi-CleanupTrash`（多含 `-Direct-v0` 变体） | `tasks/` | 按 ckpt 匹配度选 |
| **双路 `TiledCameraCfg`**：`front` + `wrist`，640×480@30FPS，含 focal/aperture 标定 | `tasks/template/single_arm_env_cfg.py:51-84` | ⭐ 省掉"换 TiledCamera" |
| **VLA 友好的 obs 字典**（`concatenate_terms=False`） | `single_arm_env_cfg.py:109-136` | ⭐ 4 个字段直接对应 RLinf |
| `task_description` 语言指令 | `SingleArmTaskEnvCfg.task_description`；LiftCube = `"Lift the red cube up."` | VLA 必需 |
| success termination | `tasks/lift_cube/lift_cube_env_cfg.py`：`cube_height_above_base(height_threshold=0.20)` | 包成 RewTerm 即可 |
| **原生 domain randomization** | `randomize_object_uniform` + `randomize_camera_uniform`（LiftCube 已启用） | sim2real 直接受益 |
| 遥操作：SO101Leader / 键盘 / 手柄 / 远程 / LeKiwi | `devices/`（含 vendored Feetech SDK） | 补数据时用 |
| HDF5 → LeRobot Dataset 转换 | `scripts/convert/isaaclab2lerobot.py`（v2）/ `isaaclab2lerobotv3.py`（v3） | 补数据时用 |
| **openpi 推理链路** | `scripts/evaluation/policy_inference.py --policy_type=openpi`；`policy/service_policy_clients.py:338` | ⭐⭐ **直接用来评测已有 ckpt**，见 [Phase 1.5](#phase-15--提前评测-sft-ckpt-最重要的-gate1-2-天) |
| **关节单位双向转换**（弧度 ↔ 度 ↔ USD limit ↔ 真机 motor limit） | `utils/robot_utils.py:96` 与 `:119` | ⭐ 直接解掉 [6.2](#62-关节顺序--归一化错位) 那个坑 |
| GR00T N1.5 / N1.6、LeRobot policy 推理 | `policy/` | 对照基线 |
| `datagen` 状态机、MimicGen、Digital Twin、Cosmos、Marble | `datagen/`、`scripts/mimic/`、`enhance/` | 本期按需 |

**体量**（非文档部分 ≈ 577 KB 纯 Python + cfg）：`devices/` 136 KB·24 文件；`tasks/` 103 KB·43；`enhance/` 64 KB·23；`policy/` 54 KB·13；`utils/` 23 KB·9；`datagen/` 17 KB·4；`assets/` 6.8 KB·7。

> ⚠️ **USD 资产不在 repo 里**：`assets/robots/.gitkeep` 与 `assets/scenes/.gitkeep` 是空占位，USD 需单独下载。资产根目录由 `utils/constant.py` 的 `_resolve_assets_root()` 决定：**优先读环境变量 `LEISAAC_ASSETS_ROOT`**，否则回退到「git root / assets」。见 [6.6](#66-leisaac_assets_root-未设置会找错资产目录)。

**唯一的实质缺失：RL reward。** `single_arm_env_cfg.py:143` 的 `SingleArmRewardsCfg` **是个空类**（LeIsaac 定位是 IL / 遥操作 playground）。但 LiftCube 已有 `success = DoneTerm(cube_height_above_base, height_threshold=0.20)` —— 把它包成稀疏 `RewTerm` 就是我们要的 reward，参照 [2.2](#22-rlinfisaaclab-fork-的新定位参考样板) 的 90 行样板。

### 2.4 LeIsaac 的接入性质：independent extension

**它对 Isaac Lab 零改动**，所以「用 LeIsaac 作基座」不等于换了一个魔改版 Isaac Lab：

- `.gitmodules` 只有一条：`dependencies/IsaacLab → https://github.com/isaac-sim/IsaacLab.git`，**stock 上游未改**
- `source/leisaac/pyproject.toml` 把 Isaac Lab 声明为普通依赖：`isaaclab = ["isaaclab[isaacsim,all]==2.3.0"]`
- 所有扩展走干净的子类化，无 monkeypatch / 无 `setattr` / 无 `__class__` 替换：
  ```python
  class RecorderEnhanceDirectRLEnv(DirectRLEnv)            # enhance/envs/direct_rl_env.py:13
  class StreamingRecorderManager(RecorderManager)          # enhance/managers/recorder_manager.py:35
  class ManagerBasedRLDigitalTwinEnv(ManagerBasedRLEnv)    # enhance/envs/manager_based_rl_digital_twin_env.py:10
  ```
  （目录名 `enhance/` 听起来像打补丁，实际全是继承。）

**含义**：基座 = stock Isaac Lab v2.3.0 + LeIsaac extension。若日后需要 RLinf fork 的 Franka reward 任务，作为 3 文件 patch 叠加即可，冲突面极小。

### 2.5 为什么不能只改 task ID

1. **注册表只有一个 task**：`rlinf/envs/isaaclab/__init__.py` 的 `REGISTER_ISAACLAB_ENVS` 只有 Franka stack-cube，且 `get_env_cls()` 会 assert task id 在表内（`rlinf/envs/__init__.py`）
2. **observation schema 硬编码 Franka**：`rlinf/envs/isaaclab/tasks/stack_cube.py:78` 的 `_wrap_obs()` 读 `policy.wrist_cam` / `table_cam` / `eef_pos` / `eef_quat` / `gripper_pos`，拼成 7 维 EEF 状态
3. **动作语义不兼容**：见 [4.1](#41-g1--动作空间7-维-eef--6-维关节)

---

## 3. Phase 0：冻结接口

**协议方向已反转**：v2.0 是「从 LeIsaac 正推协议」，v3.0 是「**从 SFT ckpt 反推协议**」。ckpt 是既成事实，环境必须适配它，不是反过来。

### 3.1 动作 / 状态协议表（以 SFT ckpt 为准）

下表左列是 LeIsaac 侧的既有事实（已核实），**右列必须与 SFT ckpt 核对**。任何一行不一致，都要决定「改环境适配 ckpt」还是「改 ckpt 侧 transform」——**优先改环境**，因为 ckpt 已经训好了。

| 项 | LeIsaac 侧事实（✅ 已核实） | 与 ckpt 核对 |
| --- | --- | :---: |
| 关节名称与顺序 | `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`（`utils/constant.py` `SINGLE_ARM_JOINT_NAMES`） | ⬜ |
| 维度 | 6 | ⬜ |
| LeRobot feature 名 | `<joint>.pos`（如 `shoulder_pan.pos`） | ⬜ |
| 单位（仿真侧） | 弧度（Isaac Lab 原生） | — |
| 单位（模型侧） | **真机 motor position** —— `robot_utils.py:96` 做 `弧度 → 度 → USD joint limit 归一化 → motor limit` | ⬜ **最关键** |
| 绝对 or 相对 | **绝对关节位置**（`so101leader` 分支 `JointPositionActionCfg(scale=1.0)`） | ⬜ |
| 夹爪表示 | 连续，与臂关节同 action term | ⬜ |
| action 上下限 | `SO101_FOLLOWER_USD_JOINT_LIMLITS` / `SO101_FOLLOWER_MOTOR_LIMITS`（`assets/robots/lerobot.py:56,66`，如 `shoulder_pan`：USD ±110° / motor ±100°） | ⬜ |
| 相机 key | `front` / `wrist`（`single_arm_env_cfg.py:51,69`） | ⬜ |
| 相机路数 | LiftCube 默认 `delete_attribute(self, "wrist")` **只留 front** | ⬜ ckpt 要 1 路还是 2 路？ |
| 相机原生分辨率 | 640×480 @ 30 FPS，`TiledCameraCfg` | — |
| 送进 π₀.₅ 的分辨率 | **224×224，`resize_with_pad`**（pad 不 crop，`service_policy_clients.py:367-370`） | ⬜ |
| state dtype | `float64`（`service_policy_clients.py:375`） | ⬜ |
| prompt 文本 | LiftCube = `"Lift the red cube up."` | ⬜ **须与 ckpt 训练时一致** |
| decimation / episode 长度 | `decimation=1`，`episode_length_s=25.0` | — |
| chunk 长度 | LeIsaac openpi 示例返回 `(10, 6)` | ⬜ ckpt 的 `action_horizon` |

> ⚠️ **两个必须注意的 LeIsaac 约定**
>
> 1. **action 配置按遥操作设备分支**（`devices/action_process.py` 的 `init_action_cfg()`）：只有 `so101leader` / `bi-so101leader` 走 6 维 `JointPositionActionCfg`；`keyboard` / `gamepad` 走 `DifferentialInverseKinematicsActionCfg`，`mimic_*` 又是另一套。**RLinf 接入与任何数据采集都必须走 leader 分支的配置**，不能依赖 `use_teleop_device()` 的默认行为。
> 2. **数据导出有静默降级**：`utils/robot_utils.py:55-61` 的 `build_feature_from_env()`，当 `action_dim != len(default_feature_joint_names)` 时会把 feature 名降级成 `dim_0..dim_N` 并设 `dataset_cfg.action_align = False`（源码注释自称 "A bit tricky"）。若需补采数据，务必核对 `meta/info.json` 里 feature 名不是 `dim_*`。

### 3.2 版本冻结表（已锁定）

基座定为 LeIsaac，直接采用它官方验证过的组合（Isaac Sim 5.1 一列）：

| 组件 | 版本 | 状态 |
| --- | --- | :---: |
| Isaac Sim | 5.1.0 | ✅ 锁定 |
| Isaac Lab | **v2.3.0**（LeIsaac 的 submodule，stock 上游） | ✅ 锁定 |
| Python | 3.11 | ✅ 锁定 |
| CUDA | 12.8 | ✅ 锁定 |
| **PyTorch** | **2.7.0 / torchvision 0.22.0** | ✅ 锁定 |
| LeIsaac | 固定 commit | ⬜ Phase 0 填 |
| openpi | `5bff19b0c0c447c7a7eaaaccf03f36d50998ec9d`（LeIsaac 验证过的） | ⬜ 与 ckpt 训练时的 openpi 版本核对 |

**✅ torch 分歧已消除。** v2.0 的最大不确定性（RLinf fork 的 2.11.0 vs LeIsaac 的 2.7.0）随基座决策自然消失。RLinf 侧用 `--torch` 参数对齐：

```bash
bash requirements/install.sh embodied --model openpi --env isaaclab --torch 2.7.0
```

（`requirements/install.sh:104-105,167,724-764` 会自动推导 torchvision/torchaudio 并 patch `pyproject.toml` 的 override-dependencies。）

> ⚠️ **需回归验证**：RLinf 其他组件（openpi、flash-attn）此前跑在 2.11 上，降到 2.7.0 后需确认能正常编译与运行。这是 Phase 0 的唯一版本风险，且有明确的验证方法（跑一次现有的 LIBERO π₀.₅ e2e）。

### 3.3 安装步骤（以 LeIsaac 为基座）

```bash
# ── 1. 装 LeIsaac（含它的 stock Isaac Lab submodule）
git clone --recursive https://github.com/LightwheelAI/leisaac.git
cd leisaac && git checkout <pinned-commit>          # commit 填入 3.2 表

# ── 2. 建 venv 并装 Isaac Sim + torch 2.7.0（按 LeIsaac 官方安装文档）
#      Python 3.11 / CUDA 12.8 / torch==2.7.0 torchvision==0.22.0 (cu128)
#      isaacsim[all,extscache]==5.1.0 from https://pypi.nvidia.com

# ── 3. 装 Isaac Lab（LeIsaac 的 submodule，stock v2.3.0）
sudo apt install cmake build-essential
cd dependencies/IsaacLab && ./isaaclab.sh --install && cd ../..

# ── 4. 装 LeIsaac extension 本体
pip install -e "source/leisaac[openpi]"            # openpi extra：评测 ckpt 用

# ── 5. 下载 SO101 USD 与场景资产，显式指定资产根目录
export LEISAAC_ASSETS_ROOT=/path/to/shared/leisaac_assets

# ── 6. 在同一 venv 里装 RLinf（torch 已是 2.7.0，勿让它改回 2.11）
cd /path/to/RLinf
bash requirements/install.sh embodied --model openpi --env isaaclab --torch 2.7.0
```

要点：

1. **步骤 3 现在要执行**（与 v2.0 相反）—— 基座就是 LeIsaac 的 Isaac Lab，不再跳过
2. **不要装 `RLinf/IsaacLab` fork**，否则又变成两份 Isaac Lab（见 [6.4](#64-装出两份-isaac-lab)）
3. `LEISAAC_ASSETS_ROOT` **必须显式设置**，且要让 Ray worker 继承到（`env_configs.env_vars`，或在 `ray start` 前 export）
4. **LeRobot 数据转换用独立 venv**：`[lerobot]` extra 会拉 `lerobot==0.4.2`，v2 转换路径还要求 `numpy==1.26.0`，可能与主 venv 冲突（见 [6.5](#65-lerobot-转换的-numpy-降级)）
5. RLinf env 的**子进程内 `import leisaac`** 触发 task 注册，必须在 `AppLauncher` 之后（见 [4.1](#41-g1--动作空间7-维-eef--6-维关节)）
6. **长期**：在 `requirements/install.sh` 加 `install_leisaac_env()`（参考 skill `add-install-docker-ci-e2e`），并加 Dockerfile stage

### 3.4 必须从 SFT checkpoint 确认的清单（Phase 0 第一项）

**这是 v3.0 新增的最重要一节。** ckpt 是既成事实，下列每一项都会决定环境侧怎么改：

| # | 要确认的 | 为什么关键 |
| --: | --- | --- |
| 1 | **训练数据来源**：真机 SO101？LeIsaac 仿真？其他仿真？ | ⚠️ **最高风险项**。若是真机或别的仿真场景，与 LeIsaac LiftCube 的相机位姿 / 光照 / 物体外观差异会直接压低成功率 |
| 2 | **任务内容**：具体做什么？ | 决定选哪个 LeIsaac 任务。若 ckpt 是「pick orange」，就该选 `PickOrange` 而非 `LiftCube` |
| 3 | **prompt 文本** | π₀.₅ 对 prompt 敏感；须与训练时逐字一致 |
| 4 | **openpi `TrainConfig` / `config_name`** | 决定 `rlinf/models/embodiment/openpi/dataconfig/` 里怎么注册；data transform 要对齐 |
| 5 | **norm_stats 的位置与 `asset_id`** | RLinf 从 checkpoint 目录按 `asset_id` 加载（`openpi/__init__.py:112`），或走 `norm_stats_path`（`:97-106`） |
| 6 | **`action_dim`** | 预期 6，需确认 |
| 7 | **`action_horizon`（chunk 长度）** | 决定 RLinf 的 `num_action_chunks` |
| 8 | **相机路数与 key** | 决定 `num_images_in_input`，以及 LiftCube 要不要把 wrist 加回来 |
| 9 | **state 的单位与 joint 顺序** | 与 [3.1](#31-动作--状态协议表以-sft-ckpt-为准) 逐行核对 |
| 10 | **action 绝对 / 相对** | 若 ckpt 是相对，环境侧 action term 要改 |
| 11 | **是否含 value head** | RLinf PPO 需要 `add_value_head: True`；若 ckpt 无 value head，首轮 critic 从零学（可接受，但 `critic_warmup_steps` 要调） |
| 12 | **训练时的 openpi 版本 / commit** | 与 [3.2](#32-版本冻结表已锁定) 的 pin 核对，避免权重加载不兼容 |

> **第 1 项的处置预案**：若 ckpt 与 LeIsaac 场景差异大导致成功率为 0，有三条路（按代价排序）：
> ① 换成与 ckpt 更匹配的 LeIsaac 任务；
> ② 调整 LeIsaac 场景（相机位姿 / 光照 / 物体）去贴近 ckpt 的训练分布；
> ③ 用 LeIsaac 补采少量数据做二次 SFT（回到 v2.0 的路径，但起点比从零好得多）。
>
> **这个判断必须在 [Phase 1.5](#phase-15--提前评测-sft-ckpt-最重要的-gate1-2-天) 完成，不要等到接完 RLinf。**

---

## 4. Gap 详解与改动清单

### 4.1 G1 · 动作空间：7 维 EEF → 6 维关节

**现状**：RLinf 的 Franka 版是 7 维 EEF 增量（xyz + rpy + gripper），走 `IK-Rel` 差分 IK。

**SO101 约束**：5 关节 + 夹爪（`shoulder_pan / shoulder_lift / elbow_flex / wrist_flex / wrist_roll` + gripper，6× Feetech STS3215）。5 个臂部 DoF 无法实现任意 6-DoF 位姿，`xyz+rpy` 的 IK 通常不可达。

**⭐ π₀.₅ 的 obs/action 契约不用自己推，LeIsaac 已写好**

`policy/service_policy_clients.py:365-397` 的 `OpenPIServicePolicyClient.get_action()`：

```python
obs_dict = {
    f"images/{key}": image_tools.convert_to_uint8(
        image_tools.resize_with_pad(observation_dict[key].cpu().squeeze().numpy(), 224, 224))
    for key in self.camera_keys                                    # front / wrist
}
if self.task_type == "so101leader":
    joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
    obs_dict["state"] = joint_pos.squeeze().astype(np.float64)     # 6 维，已转真机 motor 单位
obs_dict["prompt"] = observation_dict["task_description"]

action_chunk = self.infer(obs_dict)["actions"]                     # (10, 6)
processed_action = convert_lerobot_action_to_leisaac(action_chunk) # 反向转回仿真弧度
```

移植对应关系：

| LeIsaac `get_action()` | RLinf 落点 |
| --- | --- |
| `images/front`、`images/wrist` → 224×224 uint8 | `SO101Inputs`：`base_0_rgb` / `left_wrist_0_rgb` |
| `state` = `convert_leisaac_action_to_lerobot(joint_pos)`，float64 | `SO101Inputs`：`state` |
| `prompt` = `task_description` | `transforms.InjectDefaultPrompt` |
| `convert_lerobot_action_to_leisaac(action_chunk)` | `SO101Outputs` |

> **能抄什么、不能抄什么**：抄**映射逻辑**。**不能**复用整个 `OpenPIServicePolicyClient` 类 —— 它继承 `WebsocketServicePolicy`，为「训好的模型跑离线评测」设计；RLinf 训练时模型在 FSDP 里、要反传梯度，走不了 websocket。传输层丢掉，契约留下。
>
> 但在 [Phase 1.5](#phase-15--提前评测-sft-ckpt-最重要的-gate1-2-天) 评测阶段，**整个 client 可以直接用**。

**改动清单**

| 文件 | 改动 | 关键细节 |
| --- | --- | --- |
| `rlinf/envs/isaaclab/tasks/so101.py` | **新建** ~120 行 | 照 `stack_cube.py`。① 子进程 `_make_env_function()` 内 `import leisaac`（**必须在 `AppLauncher` 之后**）② `_wrap_obs()` 做 4 个 key 重命名（见下表）③ 相机 key 做成 config 项 |
| `rlinf/envs/isaaclab/__init__.py` | 注册 | `REGISTER_ISAACLAB_ENVS` 加 `"LeIsaac-SO101-LiftCube-Rewarded-v0": IsaaclabSO101Env` |
| `rlinf/envs/action_utils.py:82` | 分支 | `prepare_actions_for_isaaclab()` 现对 openpi 是直通；SO101 需单独分支做 `convert_lerobot_action_to_leisaac` 等价转换 |
| `rlinf/models/embodiment/openpi/policies/so101_policy.py` | **新建**（移植 LeIsaac 映射） | `SO101Inputs` / `SO101Outputs`。**不要**复用 `IsaacLabOutputs` 的 `data["actions"][:, :7]` 与 `np.sign()`（见 [6.3](#63-isaaclaboutputs-的-npsign-陷阱)） |
| `rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py` | **新建** | 照 `isaaclab_dataconfig.py`；**`TrainConfig` 须与 ckpt 训练时的一致**（[3.4](#34-必须从-sft-checkpoint-确认的清单-phase-0-第一项) 第 4 项） |
| `rlinf/models/embodiment/openpi/dataconfig/__init__.py:468` 附近 | 加 `TrainConfig` | 指向新 DataConfig + `repo_id` + `assets` |
| `examples/embodiment/config/env/isaaclab_so101_lift_cube.yaml` | **新建** | 照 `env/isaaclab_stack_cube.yaml` |
| `examples/embodiment/config/isaaclab_so101_ppo_openpi_pi05.yaml` | **新建** | 照 Franka 版，**`actor.model.action_dim: 6`**、`model_path` 指向用户的 ckpt |
| `examples/embodiment/config/model/pi0_5.yaml` | **不改** | `action_dim: 7` 是默认，在具体 config override |

**`_wrap_obs()` 映射**（LeIsaac obs 字典 → RLinf 统一 obs）：

| LeIsaac `PolicyCfg` 产出 | RLinf 需要（`stack_cube.py:94`） |
| --- | --- |
| `front` | `main_images` |
| `wrist` | `wrist_images` |
| `joint_pos` | `states` |
| `task_description`（env cfg 属性，非 obs term） | `task_descriptions` |

**已核实：`action_dim` 只改 config 就够，模型代码不用动。** 传导链：config → `rlinf/workers/actor/fsdp_actor_worker.py:1489,1528`（async 版 `async_ppo_fsdp_worker.py:449,488`）→ `openpi.action_env_dim` → `openpi_action_model.py:402,740` 对 loss / logprob 做 `[..., :action_env_dim]` 裁剪。

最小 config 片段：

```yaml
actor:
  model:
    model_type: openpi
    action_dim: 6                        # ← 关键
    num_action_chunks: <ckpt action_horizon>
    add_value_head: True
    model_path: <用户的 SO101 SFT ckpt>
    openpi:
      config_name: <与 ckpt 训练时一致>
      action_env_dim: ${..action_dim}
      num_images_in_input: <1 或 2，按 ckpt>
```

### 4.2 G2 · reward（稀疏 success）

**LeIsaac 唯一的实质缺失。** `single_arm_env_cfg.py:143` 的 `SingleArmRewardsCfg` 是空类。

**要做的**

1. LiftCube 已有 `success = DoneTerm(func=mdp.cube_height_above_base, params={..., "height_threshold": 0.20})`，把同一判定包成稀疏 `RewTerm` 填进 `SingleArmRewardsCfg`。参照 `RLinf/IsaacLab` fork 的 `stack_ik_rel_visumotor_rewarded_env_cfg.py`（90 行样板）
2. 决定相机路数：LiftCube 默认 `delete_attribute(self, "wrist")` 只留 front。**按 ckpt 需要**决定是否把 wrist 加回来（template 里本来就定义好了）
3. 确认 `sim.render_interval` 与 `decimation=1` 的关系
4. 注册新 gym id（如 `LeIsaac-SO101-LiftCube-Rewarded-v0`）
5. 明确 reward / termination 语义（见 [6.1](#61-use_rel_reward-对-isaac-lab-不生效)）：
   - 成功时返回 1
   - 失败与 timeout 如何区分（LeIsaac 有独立 `time_out` DoneTerm → RLinf 的 `truncations`；`success` → `terminations`）
   - episode 结束前是否继续执行剩余 action chunk（`chunk_step()` 的 `ignore_terminations` 语义）
   - auto-reset 后如何保存 `final_observation` / `final_info`

~~相机换 `TiledCameraCfg`~~ —— **已满足**，LeIsaac 原生就是。

**关于 articulation 参数（sim2real 阶段再处理）**

`MuammerBay/isaac_so_arm101` 的 `robots/trs_so101/so_arm101.py:44-70` 有逐关节精调值，与 LeIsaac 的统一值差异很大：

| | isaac_so_arm101 | LeIsaac |
| --- | --- | --- |
| stiffness | 逐关节 `200 / 170 / 120 / 80 / 50`（源码注释写明按各关节承载质量推算） | 统一 `17.8` |
| damping | 逐关节 `80 / 65 / 45 / 30 / 20` | 统一 `0.60` |
| `effort_limit_sim` | `1.9`（接近 STS3215 真实扭矩） | `10`（放宽） |

isaac_so_arm101 的更贴近真机物理（利于 sim2real）；LeIsaac 更宽松（利于遥操作不打架，且有 `dynamic_reset_gripper_effort_limit`）。**本期先用 LeIsaac 的**（与 ckpt 的训练环境更可能一致）；sim2real 阶段再做对照实验。这是一个文件级移植，不需要换底座。

> 背景：[IsaacLab Discussion #3934](https://github.com/isaac-sim/IsaacLab/discussions/3934) 有人因 SO101 的 stiffness/damping 不当，抓住方块后其他关节乱抖。这类参数踩过坑才调得出来，不建议自己从零调。

### 4.3 G3 · SFT checkpoint（已有，转为验证工作）

**v3.0 变化**：从「采数据 + 训 SFT」（3–10 天，最高风险）变为「**验证 ckpt 与目标任务的匹配度**」（1–2 天，仍是最高风险但可极早暴露）。

**要做的**

1. 按 [3.4](#34-必须从-sft-checkpoint-确认的清单-phase-0-第一项) 清单摸清 ckpt 的 12 项属性
2. 按 [3.1](#31-动作--状态协议表以-sft-ckpt-为准) 逐行核对协议，环境侧适配 ckpt
3. **用 LeIsaac 的 openpi 链路直接评测**（见 [Phase 1.5](#phase-15--提前评测-sft-ckpt-最重要的-gate1-2-天)）—— 不需要先接 RLinf
4. 接 norm stats：`rlinf/models/embodiment/openpi/__init__.py:95-129`

**若成功率为 0**，走 [3.4](#34-必须从-sft-checkpoint-确认的清单-phase-0-第一项) 的三条处置预案。

**norm stats 必须与以下全部一致**：关节顺序、动作单位、absolute/relative 定义、gripper 表示、图像预处理。错了会导致模型能 forward 但动作完全越界。

### 4.4 G4 · GR00T N1.7（可选，本期不做）

若改用 GR00T N1.7 需额外：

- `rlinf/models/embodiment/gr00t/embodiment_tags.py`：无 SO101 tag，走 `NEW_EMBODIMENT`（`:58`，projector index 10 见 `:75`）。注：代码注释指出 N1.7 因新 processor 加载方式实际不用该 mapping，需实测
- `rlinf/models/embodiment/gr00t/simulation_io.py:190` 的 `convert_to_isaaclab_stack_cube_action()` 硬编码 7 分量并 `assert shape[-1] == 7`，需新写 converter 并注册进 `OBS_CONVERSION` / `ACTION_CONVERSION_N1D7`（`:214-232`）
- `obs_converter_type` 目前仅 `libero` / `maniskill` / `isaaclab_stack_cube`
- SO101 / new-embodiment modality metadata + 数据集 statistics + SO101 SFT ckpt

**战略价值**：LeIsaac 原生支持 GR00T N1.5 / N1.6 推理，LeRobot 也有 GR00T N1.7 用 `new_embodiment` 在 SO101 上训练并真机 rollout 的公开流程。可作为**协议基线与对照模型**，但不要拉进本期交付。

---

## 5. 路线图与验收标准

**顺序已重排**：把 ckpt 评测（原 Phase 4a，最高风险）前移为 Phase 1.5，在写 RLinf 代码之前就 gate。

### Phase 0 · 冻结接口（1–2 天）

- [ ] 完成 [3.4](#34-必须从-sft-checkpoint-确认的清单-phase-0-第一项) 的 12 项 ckpt 属性确认
- [ ] 用 ckpt 属性填完 [3.1](#31-动作--状态协议表以-sft-ckpt-为准) 协议表右列，逐行标记「一致 / 需适配」
- [ ] 按 [3.3](#33-安装步骤以-leisaac-为基座) 装 LeIsaac + Isaac Lab v2.3.0 + torch 2.7.0
- [ ] 确认 SO101 USD 资产下载源与体积，设好 `LEISAAC_ASSETS_ROOT`
- [ ] **回归验证 torch 2.7.0**：跑一次 RLinf 现有的 LIBERO π₀.₅ e2e，确认 openpi / flash-attn 正常
- ✅ **验收**：协议表填完并 commit；`python -c "import leisaac, isaaclab"` 成功；torch 2.7.0 下 RLinf 现有 e2e 通过

### Phase 1 · 验证 LeIsaac 环境（1–2 天）

- [ ] 选定任务（按 ckpt 匹配度，默认 `LeIsaac-SO101-LiftCube-v0`）能 reset / step
- [ ] `zero_agent` / `random_agent` 能下发 6 维 action
- [ ] front（及按需加回的 wrist）RGB 在 headless 下正常
- [ ] success termination 能触发
- [ ] 单环境 + 4～16 并行环境，渲染吞吐可接受
- ✅ **验收**：obs key / shape / dtype 与 [3.1] 表一致；并行环境下无渲染瓶颈

### Phase 1.5 · 提前评测 SFT ckpt（最重要的 gate）（1–2 天）

**不写任何 RLinf 代码**，直接用 LeIsaac 自带的 openpi 链路：

```bash
# 1) 起 openpi policy server（openpi 官方 remote_inference），加载用户的 SO101 ckpt
# 2) 跑 LeIsaac 评测
python scripts/evaluation/policy_inference.py \
    --task=<选定任务> \
    --policy_type=openpi \
    --policy_host=localhost --policy_port=8000 \
    --policy_language_instruction='<与 ckpt 训练时一致的 prompt>' \
    --eval_rounds=20 \
    --device=cuda --enable_cameras
```

- [ ] ckpt 能被 policy server 正常加载，norm stats 正确
- [ ] 输出 action shape 为 `[chunk, 6]`，无 NaN / Inf
- [ ] action 不持续撞 joint limit
- [ ] 记录 20 轮的成功率
- ✅ **验收（硬门槛）**：**成功率显著 > 0**
- ❌ **不达标**：走 [3.4](#34-必须从-sft-checkpoint-确认的清单-phase-0-第一项) 的三条处置预案，**不要继续往下做**

> 这一步的价值：如果 ckpt 与目标任务不匹配，此时的沉没成本只有 2–4 天，且 RLinf 侧一行代码没写。

### Phase 2 · 加 reward（1–2 天）

- [ ] 把 success termination 包成稀疏 `RewTerm`，填进当前为空的 `SingleArmRewardsCfg`
- [ ] 注册新 gym id
- [ ] 确认 `sim.render_interval` 与 `decimation` 关系
- ✅ **验收**：`random_agent` 下 reward 能触发且 success 判定正确；用 Phase 1.5 的 ckpt 跑一轮，成功时 reward = 1

### Phase 3 · 接入 RLinf（2–4 天）

- [ ] 新建 `IsaaclabSO101Env` + 注册（obs 映射见 [4.1](#41-g1--动作空间7-维-eef--6-维关节)）
- [ ] **移植** LeIsaac `get_action()` 的映射逻辑 → `so101_policy.py`
- [ ] 新建 `so101_dataconfig.py` + `TrainConfig`（与 ckpt 训练时一致）
- [ ] 新建两个 config yaml，`action_dim: 6`，`model_path` 指向 ckpt
- [ ] 加 env smoke test
- ✅ **验收**（分层）：
  - **环境层**：obs key / shape / dtype 正确；6 维 action 能执行；success / timeout / partial reset 行为正确；CUDA tensor 能过 RLinf 子进程通信；partial reset 不影响其他环境
  - **RLinf 层**：`only_eval` 模式下跑通，**`env/success_once` 与 Phase 1.5 的成功率相当** ← 这是协议对齐的最强验证

> 最后一条很重要：如果 RLinf 里的成功率明显低于 Phase 1.5，说明移植的 obs/action 映射有偏差，而不是模型问题。

### Phase 4 · PPO（3–8 天）

- [ ] one-step e2e：rollout → reward → advantage → actor/value loss → optimizer step → checkpoint save/load
- [ ] 短训练曲线
- [ ] 按需调 reward / action horizon / KL / lr / `critic_warmup_steps`（若 ckpt 无 value head）
- ✅ **验收**：
  - `env/success_once` 不永久为 0
  - value loss 不发散；PPO ratio / KL 不异常；action 分布不快速坍缩
  - **eval 成功率相对 SFT baseline 有提升**

### Phase 5 · 实机（独立项目，本期不承诺）

见 [第 8 节](#8-实机部署本期不承诺)。

---

## 6. 已核实的高危坑

### 6.1 `use_rel_reward` 对 Isaac Lab 不生效

`examples/embodiment/config/env/isaaclab_stack_cube.yaml` 里配了 `use_rel_reward: True`，**但对 Isaac Lab 实际无效**。

`rlinf/envs/isaaclab/isaaclab_env.py:119` 的 `step()` 直接使用 Isaac Lab 返回的 `step_reward`。基类虽在 `:256` 定义了 `_calc_step_reward()`（读 `use_rel_reward` / `reward_coef`），但**全仓库没有一处从 Isaac Lab 路径调用它** —— 对比 libero（`libero_env.py:730`）、maniskill（`:299`）、metaworld（`:341`）、robocasa（`:398`）都调了。

**影响**：reward shaping 必须写在 Isaac Lab 侧的 `RewardsCfg`；`use_rel_reward` / `reward_coef` 在 Isaac Lab env config 里是**死配置**。

**处理**：写 SO101 env 时明确选一种 —— 要么接受此约定（推荐，reward 全在 Isaac Lab 侧），要么在 `step()` 里显式调用 `_calc_step_reward`。**不要两边各写一半。**

### 6.2 关节顺序 / 归一化错位

社区已有案例：joint 顺序与数据集归一化时不一致，导致归一化值爆掉、**策略直接"忽略"图像**、退化为固定轨迹（把 state 置零后模型反而"恢复视力"）。

**好消息**：LeIsaac 的 `utils/robot_utils.py:96` `convert_leisaac_action_to_lerobot()` 按 `SO101_FOLLOWER_USD_JOINT_LIMLITS` 与 `SO101_FOLLOWER_MOTOR_LIMITS` 做 `弧度 → 度 → USD limit 归一化 → 真机 motor limit` 的重映射，`:119` 有反向。**这套映射同时服务仿真与真机**，等于把 sim2real 的单位对齐一并解决。走它、不自己另写一套，这个坑基本可控。

**仍需的防御**：
- 与 ckpt 的 state / action 定义逐维核对（[3.1](#31-动作--状态协议表以-sft-ckpt-为准)）
- 零 state 对照实验：若置零后行为反而变好，几乎必定是归一化问题
- **Phase 3 验收对比 Phase 1.5 的成功率** —— 这是发现映射偏差最有效的手段
- 把协议表写进代码注释和 config 注释，不要只存在于人脑里

### 6.3 `IsaacLabOutputs` 的 `np.sign()` 陷阱

`rlinf/models/embodiment/openpi/policies/isaaclab_policy.py` 的 `IsaacLabOutputs` 做两件 Franka-specific 的事：`data["actions"][:, :7]` 截断到 7 维；`actions[..., -1] = np.sign(...)` 把夹爪二值化到 `{-1, +1}`。

**关节空间下这两条都是错的。** 新建 `SO101Outputs`，不要继承或复用。

### 6.4 装出两份 Isaac Lab

基座定为 LeIsaac 后，**不要再装 `RLinf/IsaacLab` fork**。`requirements/install.sh:1899` 的 `install_isaaclab_env()` 会 clone 那个 fork 并执行 `isaaclab.sh --install`。

**处理**：装 RLinf 时避免触发 `install_isaaclab_env()`（或先装 LeIsaac 的 Isaac Lab，并确认 `ISAAC_LAB_PATH` 指向它 —— `clone_or_reuse_repo` 会复用已有路径）。Phase 0 验收要确认 `python -c "import isaaclab; print(isaaclab.__file__)"` 指向 LeIsaac 的 submodule。

### 6.5 LeRobot 转换的 numpy 降级

`pip install -e "source/leisaac[lerobot]"` 会拉 `lerobot==0.4.2`；v2 转换路径（`isaaclab2lerobot.py`）还要求 `numpy==1.26.0`，可能与主 venv 冲突。

**建议**：若需补采数据，转换放独立 venv，产物（LeRobot dataset）跨 venv 传递。

### 6.6 `LEISAAC_ASSETS_ROOT` 未设置会找错资产目录

`utils/constant.py` 的 `_resolve_assets_root()`：优先读 `LEISAAC_ASSETS_ROOT`，**否则回退到「当前 git root / assets」**。在 RLinf 进程里 git root 是 RLinf，会指向不存在的 `RLinf/assets/`。

**必须在启动前显式 export**，并确保 Ray worker 继承（`env_configs.env_vars`，或 `ray start` 前 export）。

### 6.7 Isaac Sim 与 π₀.₅ 显存挤占

Isaac Sim 本身吃显存，叠加 π₀.₅（~3B）+ value head + FSDP 容易 OOM。参考 Franka 版 config 的调法：`cluster.component_placement`、`actor.enable_offload`、`rollout.pipeline_stage_num`、`fsdp_config.sharding_strategy`。

> 注意 Franka 版 π₀.₅ config 用 `sharding_strategy: no_shard` 且 `gradient_checkpointing: False`（openpi 不支持 gradient checkpointing，config 里有明确注释禁止修改）。

### 6.8 LeIsaac 的 action term 按遥操作设备分支

`devices/action_process.py` 的 `init_action_cfg()` 按设备给出**完全不同的 action 定义**：`so101leader` → 6 维 `JointPositionActionCfg`；`keyboard`/`gamepad` → `DifferentialInverseKinematicsActionCfg`；`mimic_*` → 又一套。

**RLinf 接入必须显式走 leader 分支的配置**，不能依赖 `use_teleop_device()` 默认行为。搞错会得到 IK 语义的动作空间，与 ckpt 完全不匹配。

---

## 7. 工期与风险

### 7.1 工期估算

假设：一名熟悉 RLinf / Isaac Lab / VLA 的工程师；LeIsaac task 与资产可正常运行；有 8-GPU 机器；SFT ckpt 由用户提供。

| 阶段 | v2.0 估算 | v3.0 估算 | 变化原因 |
| --- | ---: | ---: | --- |
| Phase 0 冻结接口 | 1–3 天 | **1–2 天** | 版本组合已锁定，torch 分歧消除 |
| Phase 1 验证 LeIsaac 环境 | 2–4 天 | **1–2 天** | 不含采数据（ckpt 已有） |
| **Phase 1.5 评测 ckpt** | — | **1–2 天** | 新增（原 Phase 4a 的一部分，前移） |
| Phase 2 加 reward | 1–3 天 | **1–2 天** | 只需包一层 RewTerm |
| Phase 3 接入 RLinf | 2–4 天 | 2–4 天 | 未变 |
| Phase 4 PPO | 1–3 周 | **3–8 天** | 不含 SFT 训练；起点已知可用 |
| *(Phase 5 实机)* | *2–6 周* | *2–6 周* | 未变 |

**综合**：
- 仿真最小可跑（跑到 Phase 3 验收）：**3–6 天**
- 可信的仿真 PPO 结果：**1–2.5 周**  ← 本期目标
- 真实 SO101 可用：**4–8 周以上**（不承诺）

> 工期的主导项已从「SFT 数据与微调」转为「**ckpt 与目标任务的匹配度**」。若 Phase 1.5 一次通过，整体会落在乐观区间；若不通过，按处置预案可能回到 2–4 周。

### 7.2 风险清单

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| **ckpt 与 LeIsaac 场景不匹配 → 成功率 0** | 现在的**最高风险** | Phase 1.5 极早 gate（沉没成本仅 2–4 天）；三条处置预案见 [3.4](#34-必须从-sft-checkpoint-确认的清单-phase-0-第一项) |
| torch 降到 2.7.0 后 RLinf 其他组件异常 | 阻塞 Phase 0 | Phase 0 用现有 LIBERO π₀.₅ e2e 回归验证 |
| Isaac Lab v2.3.0 与 RLinf 现有代码 API 不兼容 | 阻塞 Phase 3 | RLinf 的 Isaac Lab 代码只有 465 行且用标准 API，风险低；Phase 1 就能暴露 |
| 协议映射偏差（obs/action 转换写错） | PPO 学不动，且难定位 | Phase 3 验收**对比 Phase 1.5 成功率**；走 LeIsaac 的 `convert_*` 不自写 |
| 走错 LeIsaac 的 action term 分支 | 动作空间语义完全错 | [6.8](#68-leisaac-的-action-term-按遥操作设备分支) |
| ckpt 无 value head | critic 从零学，PPO 前期不稳 | 调 `critic_warmup_steps`；Phase 0 第 11 项先确认 |
| `LEISAAC_ASSETS_ROOT` 未设 / Ray worker 未继承 | env 起不来 | [6.6](#66-leisaac_assets_root-未设置会找错资产目录) |
| USD 资产下载源不明或体积过大 | 阻塞 Phase 0/1 | Phase 0 待确认项 |
| 装出两份 Isaac Lab | 行为诡异，难排查 | [6.4](#64-装出两份-isaac-lab)；Phase 0 验收查 `isaaclab.__file__` |
| Isaac Sim + π₀.₅ 显存挤占 OOM | 训练起不来 | 参考 Franka 版 placement / offload 配置 |
| 放弃 `RLinf/IsaacLab` fork 导致 Franka e2e 失效 | CI 缺口 | [11.2](#112-rlinf-现有-franka-e2e-怎么办) |
| numpy 降级污染主 venv | 环境损坏 | 数据转换用独立 venv |

---

## 8. 实机部署（本期不承诺）

RLinf 已有 RealWorld 环境框架（`rlinf/envs/realworld/`），但注册的是 Franka、GimArm、DOSW1、Turtle2 等，**没有 SO101**。

### 路线 A（推荐）：Policy server + LeRobot SO101 runtime

```
RLinf 训练 checkpoint → OpenPI-compatible policy server → LeRobot SO101 client → 真实 SO101
```

优点：复用成熟的 SO101 串口 / calibration / camera 接入；训练机与控制机可分离；不需要立即把 SO101 驱动并入 RLinf。**且 LeIsaac 的单位转换（`robot_utils.py`）本来就是为真机设计的，协议天然对齐。**

缺口：需导出或直接加载 RLinf FSDP checkpoint；policy server 的 obs/action schema 必须与训练一致；需加实机 action clipping、速度限制、watchdog。

### 路线 B：新增 `SO101RealWorldEnv`

在 `rlinf/envs/realworld/` 下直接接 LeRobot / Feetech 驱动（LeIsaac 已 vendored Feetech SDK 可参考）。好处是 SFT / RL / HIL / 实机评估都留在 RLinf 内；代价是实现与测试成本高，需处理串口断连、电机过载、calibration 文件、相机同步、控制频率、action chunk 中断、人工介入、急停与安全复位。

### Sim-to-real 主要风险

相机位姿与 FOV；曝光 / 白平衡 / 背景 / 光照；关节零位误差；servo dead zone 与 backlash；摩擦、负载与夹爪接触；控制频率与网络延迟；物体质量 / 尺寸 / 纹理差异；action chunk 预测期间真实环境已变化。

建议至少加入：相机 pose / FOV / 光照 / 纹理随机化（**LeIsaac 已有 object + camera 随机化，可扩**）；物体 pose / 质量 / 摩擦随机化；joint bias 与 action latency 随机化；action clipping + 速度加速度限制；少量真机数据混合 SFT；必要时 HIL / DAGGER 收集失败状态。**此阶段可考虑切到 isaac_so_arm101 的 articulation 精调参数**（见 [4.2](#42-g2--reward稀疏-success)）。

**纯仿真 SFT + RL 后直接 zero-shot 上实机，不应作为第一版的成功标准。**

---

## 9. 待确认清单

### 9.1 已查实（不必再查）

| 项 | 答案 | 依据 |
| --- | --- | --- |
| LeIsaac 任务用 `CameraCfg` 还是 `TiledCameraCfg`？ | **`TiledCameraCfg`**，640×480@30FPS，双路 front+wrist | `tasks/template/single_arm_env_cfg.py:51-84` |
| observation 具体 key？ | `joint_pos, joint_vel, joint_pos_rel, joint_vel_rel, actions, wrist, front, ee_frame_state, joint_pos_target`，`concatenate_terms=False` | `single_arm_env_cfg.py:109-136` |
| action 绝对还是 delta？ | **绝对关节位置**（`so101leader` 分支） | `devices/action_process.py` |
| gripper 编码？ | 与臂关节同走 `JointPositionActionCfg`（连续） | 同上 |
| joint 顺序？ | `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` | `utils/constant.py` |
| LiftCube 有 success 判定吗？ | 有：`DoneTerm(cube_height_above_base, height_threshold=0.20)`，但**是 termination 不是 reward** | `tasks/lift_cube/lift_cube_env_cfg.py` |
| LeIsaac 支持 OpenPI 吗？ | **支持**，有 `openpi` extra + `OpenPIServicePolicyClient` + `policy_inference.py --policy_type=openpi` | `pyproject.toml`、`policy/service_policy_clients.py:338`、`docs/resources/available_policy.md` |
| LeIsaac 需要 port 进别的 fork 吗？ | **不需要**，是独立 extension，对 Isaac Lab 零改动 | `.gitmodules`、`pyproject.toml`、`enhance/` 全为子类化 |
| torch 用哪个版本？ | **2.7.0**（LeIsaac 官方组合，随基座决策锁定） | LeIsaac 安装文档兼容表 |

### 9.2 Phase 0 必须落实

**A. 关于 SFT ckpt** —— 见 [3.4](#34-必须从-sft-checkpoint-确认的清单-phase-0-第一项) 的 12 项完整清单。最关键的三项：

- [ ] **训练数据来源**（真机 / LeIsaac 仿真 / 其他）—— 决定 Phase 1.5 的成败概率
- [ ] **任务内容与 prompt 文本** —— 决定选哪个 LeIsaac 任务
- [ ] **openpi `TrainConfig` + norm_stats 位置** —— 决定 RLinf 侧怎么注册

**B. 环境与依赖**

- [ ] SO101 USD 资产的下载源与体积（repo 里只有 `.gitkeep` 空占位）
- [ ] torch 2.7.0 下 RLinf 现有组件（openpi / flash-attn）回归验证
- [ ] Isaac Lab v2.3.0 与 RLinf 的 `rlinf/envs/isaaclab/` 代码兼容性
- [ ] `control rate`：由 `decimation=1` × `sim.dt` 推出，并决定 `sim.render_interval`
- [ ] 相机路数决策（按 ckpt 定 1 路还是 2 路）
- [ ] 确认 `isaaclab.__file__` 指向 LeIsaac 的 submodule，而非 RLinf fork

---

## 10. 参考

### 本仓库

- `rlinf/envs/isaaclab/` · `tasks/stack_cube.py` · `venv.py`
- `rlinf/envs/__init__.py`（`SupportedEnvType` / `get_env_cls`）· `rlinf/envs/action_utils.py:82`
- `rlinf/models/embodiment/openpi/policies/isaaclab_policy.py` · `dataconfig/isaaclab_dataconfig.py` · `dataconfig/__init__.py:468`
- `rlinf/models/embodiment/openpi/__init__.py:95-129`（norm stats 加载）
- `rlinf/models/embodiment/openpi/openpi_action_model.py:402,740`（`action_env_dim` 裁剪）
- `examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml` · `config/env/isaaclab_stack_cube.yaml`
- `requirements/install.sh:1899`（`install_isaaclab_env`）· `:104-105,724-764`（`--torch` 参数）
- `docs/source-{en,zh}/rst_source/examples/embodied/isaaclab.rst`
- `AGENTS.md` → *Extending RLinf: algorithms, models, envs*
- Skills：`.claude/skills/add-install-docker-ci-e2e`、`add-example-doc-model-env`、`install-check`

### LeIsaac（基座）

- [仓库](https://github.com/LightwheelAI/leisaac) · [安装与版本兼容](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/) · [可用环境](https://lightwheelai.github.io/leisaac/resources/available_env/) · [可用机器人](https://lightwheelai.github.io/leisaac/resources/available_robots/) · [**可用 policy（含 openpi）**](https://lightwheelai.github.io/leisaac/resources/available_policy/)
- 关键源码位置（本文引用）：
  - `source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py` —— TiledCamera（`:51-84`）、obs 字典（`:109-136`）、空 `RewardsCfg`（`:143`）
  - `source/leisaac/leisaac/tasks/lift_cube/lift_cube_env_cfg.py` —— success termination、domain randomization
  - `source/leisaac/leisaac/policy/service_policy_clients.py:338` —— **OpenPI obs/action 契约**
  - `source/leisaac/leisaac/utils/robot_utils.py:96,119` —— 关节单位双向转换；`:43,55-61` —— `build_feature_from_env` 与静默降级
  - `source/leisaac/leisaac/assets/robots/lerobot.py` —— `SO101_FOLLOWER_CFG`、`:56,66` joint/motor limits
  - `source/leisaac/leisaac/devices/action_process.py` —— **按遥操作设备分支的 action cfg**
  - `source/leisaac/leisaac/utils/constant.py` —— `SINGLE_ARM_JOINT_NAMES`、`LEISAAC_ASSETS_ROOT` 解析
  - `scripts/evaluation/policy_inference.py` —— **Phase 1.5 的评测入口**

### SO101 硬件与生态

- [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)（硬件规格）
- [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) —— `src/isaac_so_arm101/robots/trs_so101/so_arm101.py:44-70`（articulation 精调参数，sim2real 阶段可借）
- [isaac-sim/Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) · [NVIDIA 教程](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/)
- [IsaacLab Discussion #3934](https://github.com/isaac-sim/IsaacLab/discussions/3934)（SO101 stiffness/damping 导致关节乱抖）

### 其他 fork 与模型

- [RLinf/IsaacLab](https://github.com/RLinf/IsaacLab)（**已不作基座**，仅取其 90 行 reward 样板）· [与上游 diff](https://github.com/isaac-sim/IsaacLab/compare/main...RLinf:IsaacLab:main)
- [bytedance-iaas/IsaacLab](https://github.com/bytedance-iaas/IsaacLab) —— 工作在 `dev` 分支，见 [11.3](#113-与同事的传统-rl-路线如何共存)
- [OpenPI 自定义数据微调](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data) · LeIsaac 验证的 commit：`5bff19b0c0c447c7a7eaaaccf03f36d50998ec9d`
- [LeRobot π₀.₅ 文档](https://huggingface.co/docs/lerobot/pi05) · [LeRobot GR00T N1.7 文档](https://huggingface.co/docs/lerobot/groot)
- 参考数据集：`LightwheelAI/leisaac-pick-orange`（可查 `info.json` 确认 SO101 schema）· `LightwheelAI/leisaac-pick-orange-v0`（policy）

---

## 11. 基座决策的落地与团队协作

### 11.1 决策：基座 = LeIsaac（含其 stock Isaac Lab v2.3.0）

**含义**：

```
Isaac Sim 5.1.0
  └─ Isaac Lab v2.3.0（LeIsaac 的 submodule，stock 上游未改）
       └─ LeIsaac extension（pip install -e，对 Isaac Lab 零改动）
            └─ SO101 任务 / 相机 / openpi 契约 / 单位转换
  └─ RLinf（torch 对齐到 2.7.0）
```

**收益**：
- 版本组合是 LeIsaac 官方验证过的唯一组合，**torch 2.11 vs 2.7 分歧自动消除**
- LeIsaac 的所有能力（相机、DR、openpi 评测链路）零适配可用
- 不需要维护第三个 Isaac Lab fork

**代价**：
- 放弃 `RLinf/IsaacLab` fork 的 torch 2.11 pin → 需回归验证 RLinf 其他组件（Phase 0）
- Franka reward 任务不在基座里 → 见 11.2

### 11.2 RLinf 现有 Franka e2e 怎么办

`tests/e2e_tests/embodied/isaaclab_ppo_openpi_pi05.yaml` 依赖 `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0`，该任务只存在于 `RLinf/IsaacLab` fork。

三个选项（按推荐度）：

1. **把 fork 的 3 个文件作为 patch 叠加到 LeIsaac 的 Isaac Lab 上**（推荐）。改动只有 90 行 env cfg + 10 行注册 + torch pin（torch pin 不要），冲突面极小，两个任务可共存
2. **暂时接受 Franka e2e 在新环境不可用**，只跑 SO101 e2e。若 CI 是独立环境，影响可控
3. 双环境并存（两个 venv）—— 维护成本最高，不推荐

**Phase 0 需明确选哪个**，因为它决定是否要向 RLinf 主仓提交 e2e 相关改动。

### 11.3 与同事的传统 RL 路线如何共存

同事的 `bytedance-iaas/IsaacLab@dev` 走的是 **rsl_rl / skrl + MLP + 低维状态 + 密集 reward** 路线（`3a9a6f7` vendor 了 `isaac_so_arm101`，`31cd1cd` 加了 sim2real DR，另有训练服务 + frontend + SAC 接入）。

**两条路线不该合并**：我核实过他们 vendor 后的 `soarm101/src/isaac_so_arm101/tasks/lift/lift_env_cfg.py:118-136`，observation 与上游一致，仍是扁平状态向量 + 物体位姿特权信息，**vendor 版也没有加相机**。对 VLA 不可用 —— π₀.₅ 吃图像 + 语言，而它一张图都没有，且策略靠特权信息才学得会。

**能共享的三样**：

| 可共享 | 说明 |
| --- | --- |
| SO101 URDF / USD 资产 | 同一个机器人 |
| **articulation 物理参数** | 建议 sim2real 阶段统一到 isaac_so_arm101 那套，否则两条路线的 sim2real 结论无法互相印证 |
| domain randomization 配置 | 他们的 `31cd1cd` 可借，补 LeIsaac 原生 DR 之外的部分 |

**不共享**（本来就该各走各的）：任务定义、observation、reward、RL 栈。

**建议向同事同步的一句话**：VLA 路线以 LeIsaac 为基座（需要相机 + 语言 + 无特权信息），与他们的 isaac_so_arm101 路线并行；两边共用 SO101 资产与 articulation 参数，sim2real 阶段再对齐物理参数以便结论互证。
