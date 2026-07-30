# RLinf + Isaac Lab + SO101 + π₀.₅ 链路 Gap 分析

> 分析时间：2026-07-26  
> 分析范围：当前 RLinf 工作区、LeIsaac、OpenPI、LeRobot 与 GR00T N1.7 的公开实现和文档

## 1. 结论

目标链路可以拆成三段：

```text
SO101 Isaac Lab 仿真
        ↓
RLinf 中使用 π₀.₅ 进行强化学习
        ↓
将训练结果部署到真实 SO101
```

这条链路并不是从零开始。RLinf 已经打通了 Isaac Lab、π₀.₅、PPO、FSDP、Ray
worker、action chunk 和 checkpoint 等训练主干；LeIsaac 也已经提供了 SO101 的
仿真资产、任务、遥操作和数据转换能力。

如果把“完全从零”记为 10：

| 目标 | Gap | 判断 |
| --- | ---: | --- |
| 在 RLinf 中 reset/step SO101 Isaac Lab 环境 | 3/10 | 主要是环境 wrapper 和依赖接入 |
| π₀.₅ 在 SO101 仿真中完成 PPO 训练 | 4–6/10 | 需要动作协议、数据转换、SFT 与 norm stats |
| 训练结果稳定部署到真实 SO101 | 7–8/10 | sim-to-real、控制安全和实机数据是主要难点 |

因此：

- **仿真最小可跑版本：中等工作量。**
- **仿真训练稳定并提升成功率：中高工作量。**
- **真实 SO101 可靠运行：较大工作量和较高不确定性。**

从纯软件功能看，现有组件约覆盖链路的 70%；但从最终实机效果看，尚未完成的
数据闭环和 sim-to-real 往往是最困难的 30%。

## 2. 当前已经具备的能力

### 2.1 RLinf 已有 Isaac Lab + π₀.₅ + PPO

当前仓库已经包含：

- `examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml`
- `tests/e2e_tests/embodied/isaaclab_ppo_openpi_pi05.yaml`
- `.github/workflows/embodied-e2e-tests.yml` 中的 OpenPI Isaac Lab E2E job
- Isaac Lab GPU 子进程、并行环境与 action chunk 执行
- π₀.₅ 的 stochastic flow、log-prob、value head 和 PPO actor-critic 训练
- FSDP、Ray placement、checkpoint 和 evaluation 主干

这意味着无需重新实现 RL 算法、模型训练框架或分布式系统。

### 2.2 LeIsaac 已有 SO101 仿真能力

[LeIsaac](https://github.com/LightwheelAI/leisaac) 已经提供：

- 单臂和双臂 SO101 Follower 的 USD 资产
- `PickOrange`、`LiftCube`、`CleanToyTable`、`FoldCloth` 等环境
- front camera、wrist camera 和关节状态
- 任务成功条件和部分 domain randomization
- SO101 Leader 遥操作
- HDF5 数据采集
- HDF5 到 LeRobot Dataset 的转换
- GR00T、LeRobot policy 和 OpenPI 的 simulation inference 接口

相关资料：

- [LeIsaac 支持的机器人](https://lightwheelai.github.io/leisaac/resources/available_robots/)
- [LeIsaac 支持的环境](https://lightwheelai.github.io/leisaac/resources/available_env/)
- [LeIsaac 数据和 policy 流程](https://lightwheelai.github.io/leisaac/docs/getting_started/policy_support/)

### 2.3 上游已有 SO101 的模型训练与实机基线

LeRobot 当前已经提供：

- π₀.₅ 的自定义数据集 fine-tuning
- SO101 硬件驱动和数据采集
- GR00T N1.7 的 `new_embodiment` 训练
- GR00T N1.7 在真实 SO101 上的 rollout 示例

这些能力不能直接替代 RLinf 的 PPO 适配，但可以用于验证 SO101 的数据格式、
关节顺序、相机设置和实机控制链路。

参考：

- [LeRobot π₀.₅ 文档](https://huggingface.co/docs/lerobot/pi05)
- [LeRobot GR00T N1.7 文档](https://huggingface.co/docs/lerobot/groot)

## 3. 为什么不能只修改 Isaac Lab task ID

RLinf 当前虽然将 `isaaclab` 注册为一种环境类型，但具体实现仍然是 Franka
stack-cube 专用适配。

### 3.1 只注册了一个 Isaac Lab task

`rlinf/envs/isaaclab/__init__.py` 当前只注册：

```text
Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0
```

因此 LeIsaac 的 task ID 不会自动进入 RLinf 的环境注册表。

### 3.2 环境 wrapper 硬编码了 Franka observation

`rlinf/envs/isaaclab/tasks/stack_cube.py` 直接读取：

```text
policy.wrist_cam
policy.table_cam
policy.eef_pos
policy.eef_quat
policy.gripper_pos
```

并将状态拼成：

```text
3D EE position + 3D axis-angle + gripper
```

即 7 维 Franka 末端执行器状态。

LeIsaac SO101 的主要字段则包括：

```text
front
wrist
joint_pos
joint_vel
joint_pos_target
ee_frame_state
```

其模型数据和实机控制通常以 SO101 关节空间为核心。两者的 observation schema
和控制语义不一致。

### 3.3 Reward 与 success 也需要重新确认

RLinf 的 Isaac Lab wrapper 当前通过 `step_reward > 0` 记录
`success_once`。LeIsaac 的部分任务以 termination/check-success 表示成功，
环境自身未必提供适合 PPO 的稀疏 reward。

SO101 wrapper 需要明确：

- 成功时是否返回 1；
- 失败和 timeout 如何区分；
- 是否使用一次性相对奖励；
- episode 结束前是否继续执行剩余 action chunk；
- auto-reset 后如何保存 final observation 和 final metrics。

## 4. 需要新增的 Isaac Lab SO101 适配

推荐新增一个独立 wrapper，例如：

```text
rlinf/envs/isaaclab/tasks/so101.py
```

而不是继续扩展 Franka 的 `IsaaclabStackCubeEnv`。

该 wrapper 至少需要完成：

1. 在 Isaac Lab 子进程启动后导入 LeIsaac，触发 task 注册。
2. 根据 `init_params.id` 加载指定的 LeIsaac task。
3. 将相机名做成配置项，例如 `front` 和 `wrist`。
4. 将 SO101 `joint_pos` 转成 RLinf 的 `states`。
5. 输出 RLinf 统一 observation：

   ```python
   {
       "main_images": front,
       "wrist_images": wrist,
       "states": joint_pos,
       "task_descriptions": prompts,
   }
   ```

6. 将成功 termination 转成 PPO reward。
7. 验证 partial reset、auto-reset、headless camera 和多环境并行。
8. 在 `rlinf/envs/isaaclab/__init__.py` 注册所需 task ID。

建议第一个 MVP 使用单臂 `LiftCube`，而不是直接使用长时序的 `PickOrange`：

- 成功条件简单；
- episode 较短；
- reward 更容易验证；
- 更容易区分策略问题和 sim-to-real 问题；
- 更适合做 E2E smoke test。

## 5. SO101 动作协议是核心接口

必须在开始训练前确定唯一的动作定义，并贯穿以下所有阶段：

```text
LeIsaac 数据采集
→ π₀.₅ SFT
→ RLinf rollout
→ RLinf PPO actor update
→ 仿真 evaluation
→ 真实 SO101 deployment
```

需要明确的内容包括：

- 6 个关节的名称与顺序；
- 单位是 degree、radian、motor position 还是归一化值；
- action 是绝对位置还是相对增量；
- gripper 是连续位置、百分比还是二值开关；
- action 的上下限；
- 仿真和实机 calibration 的互相转换；
- policy control rate；
- action chunk 的预测长度和实际执行长度。

建议第一版采用：

- 6D SO101 joint-space state；
- 6D joint-space action；
- gripper 保持连续控制；
- 训练、仿真和实机共享同一套关节顺序；
- 如果采用 relative action，只对 arm joints 求差分，gripper 保持 absolute。

不要复用当前 Franka 的 7D EE delta action，也不要复用
`IsaacLabOutputs` 中对最后一维执行 `sign()` 的逻辑。

## 6. π₀.₅ 侧的缺口

### 6.1 当前 Isaac Lab transform 是 Franka 专用

`rlinf/models/embodiment/openpi/policies/isaaclab_policy.py` 当前：

- 使用 7 维 state/action；
- 输出 action 时只保留前 7 维；
- 将最后一维二值化到 `{-1, +1}`。

SO101 需要新增独立的 policy transform，例如：

```text
rlinf/models/embodiment/openpi/policies/so101_policy.py
```

建议包含：

- `SO101Inputs`
- `SO101Outputs`
- front/wrist camera mapping
- 6D joint state mapping
- 6D joint action mapping
- 必要的 absolute/relative action transform

### 6.2 需要新的 OpenPI data config

建议新增：

```text
rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py
```

并在 data config registry 中加入：

```text
pi05_isaaclab_so101_lift_cube
```

模型配置至少需要覆盖：

```yaml
actor:
  model:
    model_type: openpi
    action_dim: 6
    openpi:
      config_name: pi05_isaaclab_so101_lift_cube
      action_env_dim: 6
      num_images_in_input: 2
```

### 6.3 Normalization stats 必须与 SO101 数据匹配

π₀.₅ 的 input/output transform 会加载 state/action normalization stats。
这些统计量必须与以下内容一致：

- SO101 的关节顺序；
- 动作单位；
- absolute/relative action 定义；
- gripper 表示；
- 仿真数据和真实数据的预处理方式。

如果使用错误的 norm stats，即使模型能够正常 forward，执行出来的动作也可能
完全不在合理范围内。

OpenPI 官方自定义机器人流程同样要求：

1. 将数据转换成 LeRobot Dataset；
2. 定义输入和输出 transform；
3. 计算 norm stats；
4. SFT；
5. 再启动 policy server 或环境 inference。

参考：[OpenPI 自定义数据微调](https://github.com/Physical-Intelligence/openpi#fine-tuning-base-models-on-your-own-data)。

## 7. 为什么建议先 SFT，再进行 RL

RLinf 当前的 Isaac Lab π₀.₅ 示例本身也是从 task-specific SFT checkpoint
开始，而不是直接从 `pi05_base` 做 PPO。

推荐链路：

```text
LeIsaac SO101 demonstrations
        ↓
SO101 task-specific π₀.₅ SFT
        ↓
确认仿真中 SFT success rate > 0
        ↓
RLinf PPO
```

如果 SFT 在目标环境中的成功率始终为零，则 PPO 面临：

- 极低的有效奖励密度；
- 长时序 credit assignment；
- 大模型在线探索成本高；
- 早期 action distribution 可能持续越界；
- value head 很难从全零回报中学习。

因此第一阶段的验收标准不应是“PPO 能启动”，而应是：

```text
SFT policy 在 SO101 仿真中能稳定产生合法动作，
并在简化任务上具有可测量的非零成功率。
```

## 8. 安装、版本和资产管理

RLinf 当前 `requirements/install.sh` 的 Isaac Lab 安装逻辑拉取：

```text
https://github.com/RLinf/IsaacLab
```

它不会安装 LeIsaac，也不会自动下载 LeIsaac 的 SO101 和场景资产。

推荐做法：

1. 保留 RLinf 使用的 Isaac Lab fork。
2. 在同一个 venv 中以 package 形式安装固定版本的 LeIsaac。
3. 将 SO101 和场景资产放到明确的共享目录。
4. 通过环境变量或 Hydra config 指定 asset root。
5. 固定 Isaac Sim、IsaacLab、LeIsaac、PyTorch 和 Python 版本。
6. 在 Dockerfile 中加入独立的 SO101 build stage 或现有 Isaac Lab stage 扩展。

LeIsaac 当前文档给出的兼容组合包括：

| Isaac Sim | IsaacLab | Python | PyTorch |
| --- | --- | --- | --- |
| 5.1 | 2.3.0 | 3.11 | 2.7.0 |

这与 RLinf 当前使用 Isaac Sim 5.1 的方向基本一致，但仍应固定具体 commit，
避免 Isaac Lab registry、camera API 或 action term API 随上游更新发生变化。

参考：[LeIsaac 安装与版本兼容](https://lightwheelai.github.io/leisaac/docs/getting_started/installation/)。

## 9. E2E 与验收测试

建议按以下顺序增加测试。

### 9.1 环境 smoke test

- LeIsaac task 可以被 registry 找到；
- 1 个环境可以 reset；
- observation key、shape、dtype 正确；
- 6D action 可以执行；
- success、timeout 和 reset 行为正确。

### 9.2 Vectorized environment test

- 4–16 个并行环境；
- TiledCamera 在 headless 模式工作；
- CUDA tensor 可以通过 RLinf 子进程通信；
- partial reset 不影响其他环境。

### 9.3 π₀.₅ inference test

- 加载 SO101 SFT checkpoint；
- norm stats 正确加载；
- 输出 action shape 为 `[B, chunk, 6]`；
- action 不包含 NaN/Inf；
- action 不持续撞击 joint limit；
- SFT 在简化任务上有非零成功率。

### 9.4 PPO one-step E2E

- rollout；
- reward；
- advantage；
- actor/value loss；
- optimizer step；
- checkpoint save/load。

### 9.5 短训练曲线

至少确认：

- `env/success_once` 不下降为永久 0；
- value loss 不发散；
- PPO ratio 和 KL 不异常；
- action distribution 没有快速坍缩；
- evaluation success rate 相对 SFT baseline 有提升。

## 10. 实机部署的额外 Gap

RLinf 已有 RealWorld 环境框架，但当前注册的是 Franka、GimArm、DOSW1、
Turtle2 等机器人，没有 SO101。

### 路线 A：Policy server + LeRobot SO101 runtime

推荐优先采用：

```text
RLinf 训练 checkpoint
        ↓
RLinf/OpenPI-compatible policy server
        ↓
LeRobot SO101 client/runtime
        ↓
真实 SO101
```

优点：

- 复用成熟的 SO101 串口、calibration 和 camera 接入；
- 仿真/训练机和机器人控制机可以分离；
- 不需要立即将 SO101 驱动并入 RLinf。

缺口：

- 需要导出或直接加载 RLinf FSDP checkpoint；
- policy server 的 observation/action schema 必须与训练一致；
- 需要加入实机 action clipping、速度限制和 watchdog。

### 路线 B：新增 RLinf SO101RealWorldEnv

另一种做法是在 `rlinf/envs/realworld/` 下直接接入 LeRobot/Feetech 驱动。

这会使 SFT、RL、HIL 和实机评估都能留在 RLinf 中，但实现和测试成本更高，
且需要特别处理：

- 串口断连；
- 电机过载；
- calibration 文件；
- 相机同步；
- 控制频率；
- action chunk 中断；
- 人工介入；
- 急停和安全复位。

## 11. Sim-to-real 的主要风险

即使仿真 PPO 成功，真实 SO101 仍可能失败。主要差异包括：

- 相机位置和视场角；
- 曝光、白平衡、背景和光照；
- 关节零位误差；
- servo dead zone 和 backlash；
- 摩擦、负载与夹爪接触；
- 控制频率和网络延迟；
- 仿真物体和真实物体的质量、尺寸、纹理差异；
- action chunk 预测期间真实环境已经发生变化。

建议至少加入：

- 相机 pose、FOV、光照、纹理随机化；
- 物体 pose、质量和摩擦随机化；
- joint bias 和 action latency 随机化；
- action clipping、速度和加速度限制；
- 少量真实 SO101 数据混合 SFT；
- 必要时使用 HIL/DAGGER 收集失败状态。

纯仿真 SFT + RL 后直接 zero-shot 上实机，不应作为第一版的成功标准。

## 12. GR00T N1.7 的定位

当前 RLinf 已经支持 `gr00t_n1d7`，但主要验证环境是 LIBERO。其
`rlinf/models/embodiment/gr00t/simulation_io.py` 仍然只包含：

- LIBERO；
- ManiSkill；
- Isaac Lab Franka stack-cube；

没有 SO101 observation/action converter。

如果在 RLinf 中使用 GR00T N1.7，同样需要：

- SO101 converter；
- SO101/new-embodiment modality metadata；
- 目标数据集的 statistics；
- SO101 SFT checkpoint；
- Isaac Lab SO101 wrapper。

不过 LeRobot 已给出 GR00T N1.7 使用 `new_embodiment` 在 SO101 数据上训练，
并在真实 SO101 上 rollout 的公开流程。因此可以先把它作为：

- SO101 数据格式基线；
- 仿真 action/state 协议基线；
- 实机控制与相机基线；
- π₀.₅ 结果的对照模型。

它能够降低 SO101 链路本身的不确定性，但不会自动完成 RLinf 的 π₀.₅ PPO
适配。

## 13. 推荐实施顺序

### Phase 0：冻结接口

- 选择 `LeIsaac-SO101-LiftCube-v0`；
- 固定 Isaac Sim/IsaacLab/LeIsaac commit；
- 确定 6D 关节顺序；
- 确定 absolute 或 relative action；
- 确定 camera key、分辨率和 control rate。

### Phase 1：独立验证 LeIsaac

- reset/step；
- scripted/random action；
- success detection；
- front/wrist RGB；
- 单环境和少量并行环境；
- 遥操作采集一批 demonstration。

### Phase 2：接入 RLinf 环境

- 新建 `IsaaclabSO101Env`；
- 注册 task；
- 增加 Hydra env config；
- 增加 env smoke test。

### Phase 3：π₀.₅ SFT

- 转换 LeRobot 数据；
- 新建 SO101 OpenPI transform；
- 计算 norm stats；
- 训练 SFT；
- 在 LeIsaac 中 evaluation。

### Phase 4：RLinf PPO

- 新建 SO101 π₀.₅ PPO config；
- 先跑 one-step E2E；
- 再跑短曲线；
- 调整 reward、action horizon、KL、learning rate 和 value head。

### Phase 5：实机

- 先做 open-loop action 检查；
- 再做低速、限幅的 closed-loop；
- 对齐相机和 calibration；
- 收集真实失败状态；
- real-data SFT/HIL；
- 最后评估 sim-to-real 收益。

## 14. 工期估算

假设条件：

- 一名熟悉 RLinf、Isaac Lab 和 VLA 的工程师；
- LeIsaac task 和资产可正常运行；
- 有可用的 8-GPU 机器；
- 有一套可操作的 SO101 Leader/Follower；
- 不需要重新制作复杂场景。

| 工作项 | 估算 |
| --- | ---: |
| LeIsaac 和 RLinf 版本/依赖对齐 | 1–3 天 |
| SO101 Isaac Lab wrapper 与配置 | 2–5 天 |
| π₀.₅ SO101 transform、data config、norm stats | 3–7 天 |
| 数据采集与 SFT baseline | 3–10 天 |
| PPO E2E 和训练稳定性 | 1–3 周 |
| 实机 runtime、校准与安全控制 | 1–2 周 |
| sim-to-real、真实数据回流与 HIL | 1–4 周 |

综合估算：

- **仿真最小可跑：1–2 周**
- **可信的仿真 PPO 结果：2–4 周**
- **真实 SO101 可用：4–8 周以上**

任务复杂度、demonstration 质量和硬件稳定性可能显著扩大最后一个阶段的工期。

## 15. 最终判断

这条链路最大的误区，是把它理解成：

```text
RLinf 已支持 Isaac Lab
+ Isaac Lab 已有 SO101
= 修改 task ID 后即可训练
```

实际情况是：

```text
RLinf 已完成 RL 和系统主干
+ LeIsaac 已完成 SO101 仿真资产与数据采集
+ 仍需实现两者之间的 observation/action/reward adapter
+ 仍需建立 SO101-specific π₀.₅ SFT 和 normalization
+ 仍需完成实机部署和 sim-to-real
```

因此项目是可行的，且仿真部分风险可控；但如果最终目标是“训练后的 π₀.₅
可靠运行在真实 SO101 上”，应把它作为一个完整的机器人学习项目，而不是一个
简单的模拟器后端适配任务。

