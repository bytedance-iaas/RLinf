# SO101 sim2real 方案（含一个会阻断部署的已知缺陷）

**结论先行**：现阶段**不能直接上机**。离线检验（不碰机器人）已量化证明策略读不懂真实观测——见 §2。本文档给出产物清单、缺陷分析、三条修复路线，以及缺陷解决之后的完整部署步骤。

硬件约束：真机 SO101 只接在你的 Mac 上；推理跑在 8×H200 训练节点。架构就是 **Mac 采图/读关节 → H200 推理 → 返回动作块 → Mac 执行**。

---

## 1. 现阶段产物

| 产物 | 路径 | 说明 |
|---|---|---|
| **策略检查点** | `results/so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30` | PPO 峰值点，19 GB |
| 归一化统计量 | `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json` | **全血统冻结**，部署必须用这一份 |
| 数据变换条目 | `pi05_so101_v10`（`dataconfig/__init__.py`） | 决定图像/状态如何进模型 |
| 仿真环境 | `rlinf/envs/maniskill/tasks/so101_pick_place.py` | 真任务规格 |
| 训练/复现文档 | `V10_REPRODUCTION_ZH.md`、`PPO_V13_RUNBOOK_ZH.md` | |

**策略的输入输出契约**（部署时必须逐项对齐）：

| 项 | 值 |
|---|---|
| 输入图像 | 前视 + 腕视，各 **640×480 RGB** |
| 输入状态 | 6 维关节位置，**LeRobot 归一化单位**（臂 ±100、爪 0–100） |
| 提示词 | `"Grab the red cube"` |
| 输出 | `[10, 6]` 动作块，同样是归一化单位，**直接发给 LeRobot follower** |
| 控制频率 | **30 Hz** |
| 每次推理执行 | **10 步**（`num_action_chunks=10`），执行完再推理 |

**仿真内成绩**（供参考，不是真机预期）：

| 区域 | 成功率 |
|---|---|
| 环 1（8.48×11.31 cm，训练区） | 57.8%（从未用过的种子） |
| 框内（6×8 cm） | 77.3% |
| 全板（21.6×30.1 cm 的棕区） | 14.8% |

---

## 2. 阻断项：腕部相机在仿真里拍的是机械臂自己

### 2.1 证据

渲染训练数据的真实帧（2026-08-13）：一条示范的 15%/45%/75% 三个时刻，**腕视全是机械臂自身的白色外壳**，看不到板、方块或夹爪咬合处；而前视一切正常。配置在 `so101_pick_place.py:86-88`：

```python
WRIST_CAM_EYE    = [0.0,  0.0,   0.04]
WRIST_CAM_TARGET = [0.0, -0.05, -0.18]
WRIST_CAM_FOV    = 1.5      # 86°，真机规格是 106°
```

这组值当初是**目视调的**，注释写着"能看到工作区+方块+夹爪"——**被那三帧证伪**。它也解释了长期未决的"腕部板面掩膜 IoU ≈ 0"。

### 2.2 量化：离线回放检验

把 87 集真机数据的（前视、腕视、关节状态）喂给策略，比较它预测的动作与人类当时的实际动作。两个对照让数字可解释：**仿真数据**（分布内参照）和**"完全不动"**（动作幅度的尺度）。

脚本：`tools_so101_session/offline_replay_check.py`

```
=== 仿真（分布内对照，60 帧）===
  MEAN            策略 MAE 0.34    不动 MAE 3.26    比值 0.10
=== 真机（sim2real 的问题所在，60 帧）===
  shoulder_pan          1.76            0.67     2.62
  shoulder_lift        14.85            3.45     4.30
  elbow_flex           12.77            3.74     3.41
  wrist_flex            6.66            0.85     7.82
  wrist_roll           17.40            0.78    22.38
  gripper              26.14            8.30     3.15
  MEAN                 13.26            2.97     4.47
```

**读法**：比值 <1 表示策略优于"什么都不做"，≥1 表示不如。

- 仿真 **0.10** —— 策略确实在读观测；
- 真机 **4.47** —— 策略输出的动作**比原地不动还差 4.5 倍**；
- 最坏的两个关节是 **`wrist_roll`（22.4）与 `wrist_flex`（7.8）**，正是腕部相机对应的自由度。

**现在部署 = 机械臂乱动。** 这不是推测，是测量。

（用另一份真机数据集重复此检验得到 4.50，两份独立数据同一结论。）

---

## 3. 三条修复路线

| 路线 | 做法 | 代价 | 判据 |
|---|---|---|---|
| **C. 只喂前视** | 部署时 `num_images_in_input=1`，不送腕视 | **零训练成本**，先离线量 | 用 `offline_replay_check.py --no-wrist` 测比值；**<1 才有意义** |
| **B. 修相机重训** | 重新指向腕部相机 → 用真机腕视做数值标定（板面掩膜 IoU 为判据）→ 重录示范 → 从阶段 B 重训 | **约 30 小时机时**，全流程重来 | 重训后再跑同一个离线检验，比值须 <1 |
| **A. 加真机数据微调** | 用 87 集真机数据对现策略做轻量 SFT（真机+仿真混合） | 约 1 小时，但真机数据只有 87 集 | 同上 |

**推荐顺序：C → A → B。** C 已排队测量（结果会写进本文档）。C 的风险是模型按两张图训练，砍掉一路同样是分布外——所以必须**测**而不是假设。

> **在任何一条路线把离线比值降到 <1 之前，不要接机械臂。**

---

## 4. 部署架构（缺陷解决后执行）

```
┌─ Mac（真机侧）────────────┐          ┌─ H200 节点（推理侧）────────┐
│ LeRobot SO101 follower    │  图像+   │ openpi websocket_policy_    │
│ 2 × USB 相机（640×480）    │  状态 →  │ server                      │
│ 30 Hz 控制回路            │  ← 动作  │ + v13 检查点 + v4 norm_stats │
└───────────────────────────┘  块[10,6] └─────────────────────────────┘
                      SSH 隧道 / 内网
```

仓库里三个部件都现成，不用自己写：

| 部件 | 位置 |
|---|---|
| 检查点转换 | `rlinf/utils/ckpt_convertor/openpi/sft2deploy.py` |
| 推理服务端 | `.venv/lib/python3.11/site-packages/openpi/serving/websocket_policy_server.py` |
| 客户端 | `.venv/lib/python3.11/site-packages/openpi_client/websocket_client_policy.py` |

### 4.1 H200 侧：转换 + 起服务

```bash
cd /data08/henryg/pai/RLinf
# 1) 训练格式 -> 部署格式
.venv/bin/python -m rlinf.utils.ckpt_convertor.openpi.sft2deploy \
  --ckpt results/so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30 \
  --output /data08/henryg/pai/models/so101_v13_deploy

# 2) norm_stats 必须随行（部署侧按 <model_path>/<repo_id>/ 查找）
mkdir -p /data08/henryg/pai/models/so101_v13_deploy/so101-sim-demos-v4
cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
   /data08/henryg/pai/models/so101_v13_deploy/so101-sim-demos-v4/

# 3) 起 websocket 推理服务（端口自选）
#    服务端每次返回 [10, 6] 的动作块
```

Mac 侧建隧道：

```bash
ssh -N -L 8000:localhost:8000 <user>@<h200-host>
```

### 4.2 Mac 侧：控制回路骨架

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy
# LeRobot 的 SO101 follower 与相机按你采数据时的配置初始化

policy = WebsocketClientPolicy(host="localhost", port=8000)
PROMPT = "Grab the red cube"
CHUNK  = 10          # 必须与训练一致：执行完 10 步再推理
HZ     = 30

while not done:
    front = cam_front.read()        # 640x480 RGB
    wrist = cam_wrist.read()        # 640x480 RGB（若走路线 C 则不发）
    state = robot.get_observation() # 6 维，LeRobot 归一化单位

    actions = policy.infer({
        "observation/image":       front,
        "observation/wrist_image": wrist,
        "observation/state":       state,
        "prompt":                  PROMPT,
    })["actions"]                   # [10, 6]

    for a in actions[:CHUNK]:       # 30 Hz 逐步下发
        robot.send_action(a)
        sleep(1 / HZ)
```

**动作单位不需要换算**：策略输出的就是 LeRobot 归一化单位（`shoulder_pan.pos` 等），与 follower 接受的一致。

### 4.3 真机摆放必须限制在环 1

策略只在环 1（96 cm²）训练过。全板成绩只有 14.8%，所以**红方块必须放在这个矩形内**，用胶带在板上标出来：

| 方向 | 位置 |
|---|---|
| x（朝托盘方向） | 距**托盘那一侧板边** 3.6 cm 到 12.0 cm（宽 8.5 cm） |
| y | 距**黑色窄条对面那条板边** 3.4 cm 到 14.7 cm（长 11.3 cm） |

蓝色干扰块可放在棕区任意位置。

### 4.4 参数对齐表（任何一项不一致都会掉成绩）

| 项 | 必须是 | 训练时的依据 |
|---|---|---|
| 相机分辨率 | 640×480 两路 | 与真机数据集一致 |
| 控制频率 | 30 Hz | 真机数据集 fps |
| 每次执行步数 | **10** | `num_action_chunks=10`（v13 训练值） |
| 状态/动作单位 | LeRobot 归一化 | 数据集原生单位 |
| norm_stats | v4 那份 | 血统冻结 |
| 提示词 | `Grab the red cube` | 训练用的任务串 |
| 方块 | 2.9 cm、<10 g | 仿真物理参数按此设 |
| 初始姿态 | 手臂折叠、夹爪近闭合 | 87 帧真机首帧的中位姿态 |

---

## 5. 上机流程（安全）

**只有离线比值 <1 之后才执行。**

1. **空跑**：机械臂断电/脱机，只跑回路，确认 30 Hz 稳定、每次拿到 `[10,6]`、延迟可接受（往返 >100 ms 就要考虑把服务放更近）。
2. **限幅**：首次通电，把每步动作变化量夹到很小的范围（例如每步 ≤2 归一化单位），人手放在急停上。
3. **固定位置**：方块固定放在环 1 正中，跑 5 次。这是**最容易的**局面，若这里都不成功，不必往下试。
4. **环 1 随机**：在标出的矩形内随机放 20 次，记录成功率。与仿真的 57.8% 对比——**差距就是 sim2real gap**，这才是真正要报告的数字。
5. **失败归类**：抓不住 / 抓住掉 / 放偏 / 不回位，四类分别计数。它们指向不同的修复方向。

---

## 6. 当前状态与下一步

| 项 | 状态 |
|---|---|
| 策略训练 | ✅ 完成（仿真环 1 诚实 57.8%） |
| 部署链路 | ✅ 仓库已有全部部件 |
| **离线检验** | ❌ **未通过**（比值 4.47，应 <1） |
| 路线 C（只喂前视）测量 | 🔄 排队中 |
| 上机 | ⛔ 阻断 |

**下一步只有一个**：等路线 C 的测量结果。

- 若 C 的比值 <1 → 按 §4 部署，但只送前视；
- 若 C 仍 ≥1 → 走路线 A（真机数据微调）或 B（修相机重训），两者都要重新过这个离线检验。
