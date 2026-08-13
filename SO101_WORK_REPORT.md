# SO101 + PI0.5 仿真 RL 全过程工作报告

日期:2026-08-04 ~ 2026-08-05 · 机器:8×H200(IPv6 网络)· 仓库:RLinf · 执行:Claude(用户:henry-guo)

---

## 0. 目标与总体路线

**目标**:对已用 LeRobot 在真机数据集 `henry-guo/so101-pick-place-v2`(87 集,双相机 front+wrist,任务 "Grab the red cube")上训练的 PI0.5 策略做 **RL 微调**,最终在真机 SO101 上评估。

**约束**:真机 SO101 只能接 Mac M4 Pro(无 CUDA);RLinf 无 SO101 真机栈 → 唯一可行路线:**ManiSkill 仿真(内置 SO100 机器人)中搭建等价任务 → sim RL → sim2real**。

**路线图**:搭 sim 环境 → 场景/相机对齐真机 → 单位制标定 → SFT(native)→ PPO RL → (round 2) sim2real 部署桥。

---

## 1. 阶段详述

### 阶段 A:环境搭建(machine setup)

| 项 | 内容 |
|---|---|
| 为什么 | 机器上无 openpi/ManiSkill 环境 |
| 做了什么 | `bash requirements/install.sh embodied --model openpi --env maniskill_libero` 建 `.venv`;配置 apt/wget 代理(`http://[fdbd:dc61:d:297::16]:8888`,写入 `/etc/apt/apt.conf.d/99proxy`、`/root/.wgetrc`) |
| 踩坑 1 | wget 不认大写代理变量,安装卡 34 分钟 → 补小写 `http_proxy/https_proxy` |
| 踩坑 2 | **Vulkan ErrorIncompatibleDriver**:计算卡驱动(580.105.08)缺 GL/Vulkan 用户态库 → 从 `/data08/yichen/NVIDIA-Linux-x86_64-580.105.08/` 软链匹配版本库到 `.venv/nvidia_gl/`,私有 `nvidia_icd.json`。运行时必须:`VK_ICD_FILENAMES`、`LD_LIBRARY_PATH`、`XDG_RUNTIME_DIR=/tmp/xdg-runtime`、`MUJOCO_GL=egl` |
| 结果 | ManiSkill GPU 渲染可用 |

### 阶段 B:SO101 仿真任务搭建 + 场景/相机标定

| 项 | 内容 |
|---|---|
| 为什么 | RLinf 无 SO101 资产;ManiSkill v3.0.0b22 内置 SO100(与 SO101 机械通用) |
| 做了什么 | 新建任务 `SO101GrabRedCube-v1`(`rlinf/envs/maniskill/tasks/so101_pick_place.py`),复用 `env_type: maniskill`;场景按真实照片迭代:奶白桌面、8"×12" 牛皮纸板、2.9cm 红/蓝方块、开口托盘(黑边浅底)、黑边条;front 相机近俯视(eye [-0.50,0,0.559],fov 0.84),wrist 相机挂 `Fixed_Jaw`(9 位姿扫描找到 [-y 朝向工作区]) |
| 用户纠正(多轮) | "盒子不像盒子"→改开口托盘;"木板尺寸/方向/距离"→12"×8" 长边沿 y、贴近底座;"桌子奶白色";"红块总在木板上";"盒子旋转90度" |
| 结果 | front/wrist 双视角与真机帧结构性对齐(对比图已验证) |

### 阶段 C:单位制标定(第一个重大 bug,用户发现)

| 项 | 内容 |
|---|---|
| 症状 | 用户看视频:"so101 的运动方向都是错的" |
| 根因 | 数据集是 LeRobot **归一化单位**(臂 [-100,100],爪 [0,100]),ManiSkill `pd_joint_pos` 是**弧度** —— 归一化值被当弧度直接喂给环境 |
| 修复 | 新建 `rlinf/envs/maniskill/so101_calib.py`:用用户的 follower 舵机标定(feetech 4096 tick/圈)做 `norm_to_rad` / `rad_to_norm`;接线:`action_utils.prepare_actions_for_maniskill`(so100/so101 分支)+ `maniskill_env._wrap_obs`(`so101_state_norm: True`) |
| 中间错误 | 第一版在软件里减 homing_offset —— 错:LeRobot 已把 homing 写进舵机寄存器,读数在 homed 坐标系(中心 2048)。修正后 812/812 关键帧吻合,往返误差 ~2e-6 |
| 遗留(后爆) | 夹爪沿用了手臂的 tick→rad 线性换算 —— **物理机构不同,埋下阶段 F 的致命 bug** |
| 方向微调 | wrist_flex 需 `OFFSET +0.6 rad`(朝下抓);经用户三轮反馈("向上/更上了/好多了")确定符号 |

### 阶段 D:SFT(native)

| 项 | 内容 |
|---|---|
| 为什么 | LeRobot pi05 权重与 openpi "old" 布局不兼容(需适配),选择 B 方案:从 `lerobot/pi05_base`(本地快照,812/812 键直载)做 RLinf 原生 SFT |
| 准备 | 数据集打 v2.1 tag;norm_stats 子采样 15000 帧计算(`assets/pi05_so101/.../norm_stats.json`);LeRobot 0.6.1 列名是 `action`(单数)→ `action_sequence_keys=("action",)` |
| 结果 | ✅ 8×H200 训练健康(loss→0.001),产出 `so101_sft_openpi_pi05/checkpoints/global_step_8000` |

### 阶段 E:RL 第一次全量跑(750 epochs,12 小时)

| 项 | 内容 |
|---|---|
| 配置 | `so101_ppo_openpi_pi05.yaml`:PPO(gae + actor_critic),从 SFT-8000 起步 |
| 途中修的三个框架问题 | ① `get_robot_control_mode` 不认 so100 → 加分支;② IPv6 master_addr 未加括号 → `collective_group.py` 修;③ Ray 因 /tmp 96% 满卡传输 → `RAY_local_fs_capacity_threshold=0.99` |
| 规模坑 | 320 envs 首个 rollout 挂死 → 降到 **128 envs 稳定**(H200 无关) |
| 结果 | ❌ reward 有涨(reach 学会了:18cm→1.6cm)但 **success_once 永远 0** |

### 阶段 F:诊断 → 夹爪标定致命 bug(本项目最重要发现)

| 项 | 内容 |
|---|---|
| 用户的问题 | "现在摩擦系数是多少?夹爪和方块的最近距离是多少?" |
| 测量 | 摩擦:爪 2.0 / 方块 0.3;TCP-方块最近 **7.6mm**(已贴上)但 `is_grasping` 恒 0 |
| 用户追问 | "夹爪有闭合动作吗?最大闭合幅度是多少?" → 实测两指间距 vs 关节角 |
| **根因** | 标定把夹爪指令映射到 **8.1~12.5cm 间距** —— 方块只有 2.9cm,**物理上永远合不拢**。真机平行爪(tick 行程≈0.80rad)与 sim 旋转爪几何完全不同,不能沿用手臂换算 |
| 用户的方法论纠正 | "你是不是应该查一下其它关节?不能我问一个你查一个" → 全 6 关节审计:wrist_roll 超限 0.51rad(follower 标定是未标定整圈 0..4095),shoulder_lift/elbow 轻微超限 |
| 修复 | `so101_calib.py`:夹爪专用映射 `norm0→-1.0rad(合,1.4cm)/norm100→+0.5rad(开,11cm)` + 全关节 clip 到 URDF 限位。实测爪能合到 1.3cm ✓ |

### 阶段 G~J:四次 RL 迭代(两次设计错误 + 血统污染)

| 版本 | 起点 | Reward | 结果 | 教训 |
|---|---|---|---|---|
| resume-750 | RL-750 ckpt | +近距合爪 0.5 | 86 epochs 平台 0.20(纯 reach) | 750 的夹爪已彻底塌缩到"开"(norm65→90),探索跨不过 63 个归一化单位的鸿沟 |
| sft_restart | SFT-8000 | **双侧 shaping(错误②)**:远开 0.3 + 近合 0.5 | 192 epochs 平台 0.185;悬停不下探 | **设计错误①时间失衡**:远(60 步)×0.3 ≫ 近(5 步)×0.5,"保持张开"赢;**错误②方向反了**:用户演示是"运输时合、到位上方才开"(12 步分解),far-open 正好教反 → **step_150 权重被烤入"悬停"吸引子(血统污染源头)** |
| v3 | step_150(污染) | proven 配方(调研后,无爪 shaping)+ entropy 0.01 | 128 epochs 死平 0.16,130 万步探索 0 次闭合 | 案例调研正确结论:成功 reward 都不管夹爪、靠 `is_grasped` 跳变;但窄分布 flow-VLA(KL 0.01-0.04)探索撞不出闭合 |
| v4 | v3-step_100(同一污染血统) | proven + **close 梯度桥** `0.5·(d<4cm)·closedness` | 140 epochs 0.186→0.20 平台,eval 视频依旧悬停 | **权重选择 > reward 选择**:PPO 挪不动已丧失的行为;污染血统整条死 |
| **v5(现役)** | **SFT-8000(未污染,会下探+闭合)** | v4 同款(close 桥反制侵蚀) | 起点 0.057(低=未污染的标志)→12 epochs 0.133,恢复中 | 三条件首次同时成立:会闭合的起点+能闭合的爪+奖励闭合的 reward;裁决点 epoch 250 |

### 阶段 K:运维事故系列(浪费最多算力的部分)

| 事故 | 根因 | 纠正 |
|---|---|---|
| 三次"首 rollout 挂死"(51min/46min/5min+浪费) | **误归因**为脏 Ray 状态;真根因是 `Cluster.find_free_port` 用 **AF_INET(IPv4)** 探测端口而通信走 **IPv6** → 端口在 v6 上被占 → TCPStore 连错/超时 → 一 rank 死等,概率性发作("清理后成功"纯属运气) | v4 日志栈回溯定位;修 `cluster.py:143` 双栈探测(AF_INET6+V6ONLY=0);修后连过此前必死点,零复发。**值得回馈上游** |
| 训练随 Claude 会话退出被杀 | 后台命令是会话子进程 | `setsid` 全脱离启动 |
| 清理脚本"自杀"(×3) | `pgrep -f`/`ps|grep` 字符串匹配命中**自己的命令行**;kill 循环 `case *bash*` 杀掉自己中途退出 → "已清理"是假象(191 孤儿健在) | 全部改 `/proc/<pid>/exe` 精确校验 + 排除 `$$`;清理后**紧邻启动前**再验证一次 |
| 启动链里放 `sleep` 导致静默失败 | harness 阻断前台 sleep | 启动链禁 sleep |
| 挂死无人发现(51 分钟) | 守卫只盯 reward 里程碑,不盯"训练是否启动" | 每次启动配**启动期限检查**(30 分钟内出完整第一步,否则判死) |
| 僵尸进程 1.1 万 | 容器 PID1 是 `sleep infinity` 不收尸;一个挂了一天的 normstats 脚本泄漏 | 杀源头;僵尸无害(PID 上限 4.2M) |

### 阶段 L:框架级反思(用户触发)+ 双线并进

| 项 | 内容 |
|---|---|
| 用户的问题 | "你有没有怀疑过你训错了?对比了其他成功的案例吗?" |
| 补齐的对比 | 成功案例的**前提条件**(此前只对比了 reward):lerobot-sim2real 靠从零 CNN + 海量暴力探索;RLinf 官方 pi05 示例 **先在 sim 演示上 SFT**(RL 起点 success>0)。**PPO 是已有成功的放大器,不是从零发现器** —— 我们的配方(真机 SFT 的窄分布 VLA 直入 sim RL)两个前提都不占 |
| 行动 | v5 跑到裁决点(现配方最后一张牌);**并行**建正确配方管线:ManiSkill 自带 SO100 运动规划器适配到我们环境 → **首条脚本演示即成功抓取(同时首次实证环境可解)** → 150 条生成 100 条成功 → 转 LeRobot v2.1 数据集(100 集/8415 帧,单位对齐,夹爪语义验证 66.7开→13.3合)|
| 若 v5 败 | 从 SFT-8000 用 sim 演示继续 SFT → RL 起点 success>0 → 复刻官方成功配方 |

---

## 2. 错误总清单(为什么错)

| # | 错误 | 为什么会犯 | 预防机制(已落地) |
|---|---|---|---|
| 1 | 归一化单位当弧度 | 没核对数据集的单位约定 | 新环境先做单位/量纲审计 |
| 2 | 标定减 homing_offset | 没读透 LeRobot 舵机标定语义 | 用整数据集回验(812/812) |
| 3 | **夹爪沿用手臂换算(8.1cm 合不拢)** | 未意识到夹爪机构不同;未做行程实测 | 任何执行器:先实测行程-指令曲线 |
| 4 | 只查被问的关节 | 头痛医头 | 同类问题一次全量审计 |
| 5 | Reward 时间失衡(远开压倒近合) | 只算每步权重,没算时长积分 | 设计 reward 必须做逐项**每集积分**估算 |
| 6 | far-open 与演示语义相反 | 没对照真实演示的动作分解 | reward 设计先对齐演示语义,交用户 review |
| 7 | 污染血统当 warm-start(v3/v4 两次) | 低估 PPO-on-flow-VLA 的行为惯性(KL~0.02) | 起点选择先查"目标行为是否还在权重里"(实测 grip_q) |
| 8 | 挂死误归因脏状态 | 相关当因果;没拿到错误栈就下结论 | 不见 traceback 不定根因;让失败可复现地暴露 |
| 9 | IPv4 探测 IPv6 通信端口 | 上游 bug,但我三次没深挖 | 已修 + 上游可提 PR |
| 10 | pgrep 字符串自匹配(×3) | 便捷但不可靠的验证手段 | 一律 `/proc/<pid>/exe` + 排除 `$$` |
| 11 | 验证不在启动紧邻前 | 流程顺序错误 | 清理→**验证→立即启动**,顺序固定 |
| 12 | 守卫不盯启动本身 | 只监控计划内指标 | 启动期限检查为强制项 |
| 13 | 只对比 reward 不对比配方前提 | 查到"够回答"就停 | 框架级预检(premortem)强制先行 |
| 14 | "不会再犯"式承诺 | —— | 只给可验证的有界损失(30 分钟上限),不给承诺 |

---

## 3. 产出文件清单

**代码(仓库内,长期资产)**
| 文件 | 用途 |
|---|---|
| `rlinf/envs/maniskill/tasks/so101_pick_place.py` | 任务环境:场景/相机/成功判据/奖励(现役:proven+close桥);`SO101_LOG_DIST` 诊断开关 |
| `rlinf/envs/maniskill/so101_calib.py` | 单位换算 + 夹爪专用映射 + 关节限位 clip(核心标定资产) |
| `rlinf/models/embodiment/openpi/policies/so101_policy.py` | SO101Inputs/Outputs(双相机,6 维) |
| `rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py` + `__init__.py` | `pi05_so101` TrainConfig |
| `rlinf/config.py` | so100/so101 → pd_joint_pos |
| `rlinf/envs/action_utils.py` | so100/so101 → norm_to_rad |
| `rlinf/envs/maniskill/maniskill_env.py` | wrist_camera 输出 + so101_state_norm |
| `rlinf/scheduler/collective/collective_group.py` | IPv6 master_addr 加括号 |
| `rlinf/scheduler/cluster/cluster.py` | **find_free_port 双栈修复(建议上游 PR)** |
| `examples/embodiment/config/env/maniskill_so101_pick_place.yaml`、`so101_ppo_openpi_pi05.yaml`、`so101_eval_openpi_pi05.yaml`、`examples/sft/config/so101_sft_openpi_pi05.yaml`、`.vscode/launch.json` | 配置与调试入口 |

**数据/模型资产**
| 路径 | 内容 |
|---|---|
| `/data08/henryg/pai/results/so101_sft_openpi_pi05/checkpoints/global_step_8000` | SFT 权重(唯一未污染起点) |
| `/data08/henryg/pai/results/so101_ppo_{run,run_fixed,sft_restart,v3,v4,v5}` | 各次 RL 输出(ckpt/tensorboard/eval 视频) |
| `assets/pi05_so101/.../norm_stats.json` | 真机数据 norm stats |
| `/data08/henryg/pai/data/so101_sim_demos/`(h5) | 150 条脚本演示原始轨迹 |
| `/data08/henryg/pai/data/so101-sim-demos/`(LeRobot v2.1) | 100 条成功演示数据集(fps15,128²,双相机) |

**脚本(scratchpad,会话级)**:`rl_v3_step150.sh` / `rl_v4.sh` / `rl_v5.sh`(启动模板)、`gen_so101_demos.py`(演示生成)、`convert_demos_to_lerobot.py`(格式转换)、各诊断/渲染脚本

---

## 4. 关键命令模板

```bash
# ── 启动 RL(全套环境变量,setsid 脱离会话)──
setsid bash <launcher>.sh </dev/null >/dev/null 2>&1 &
# launcher 内必备:VK_ICD_FILENAMES / LD_LIBRARY_PATH=.venv/nvidia_gl / XDG_RUNTIME_DIR
#   MUJOCO_GL=egl / RAY_local_fs_capacity_threshold=0.99 / HF_HUB_OFFLINE=1
#   128 envs + global_batch 2048(320 会挂)

# ── 停止(顺序不可变)──
.venv/bin/ray stop --force
# 杀残留:仅凭 /proc/<pid>/exe 判定,排除 $$,绝不用字符串 pgrep 判定
rm -rf /tmp/ray/session_*
# 验证(必须紧邻启动前):real live ray = 0,GPU 显存空

# ── 启动后强制:期限检查 ──
# 30 分钟内 log 出现完整第一步(success_once= 行);否则判死、找根因,不盲目重试
```

---

## 5. 后续进展(2026-08-05 深夜 ~ 08-06,报告初稿之后)

### 5.1 v5 首次崩溃 = 史上第一次抓取(第 15 号 bug)
v5 epoch 28 崩溃:`tensor 16 vs 15` —— `_red_start_z` 在 **partial reset** 时被部分批量覆盖。关键:partial reset 只在环境**提前终止(=success)**时触发,此前 1300+ epochs 全是失败所以从未走到这条路径。**即:策略第一次抓取成功的那一刻,把训练炸了。**修复:全批量缓冲 + `buf[env_idx]=` 下标更新;用 `env.reset(options=dict(env_idx=...))` 显式验证。教训:per-episode 缓冲必须全批量(已入 skill §5b)。

### 5.2 v5r2:首次记录在案的 SUCCESS,但 PPO 放大失败
修复后重启,**epoch 14 出现 `success_once=0.0078`(1/128)—— 项目首个正式 success,且修复扛住了**。但此后 **740 epochs 成功率卡死在 0.8%~1.6%**,reward 平台 0.21-0.22。结论:初始成功率太稀薄(1-2/128),PPO 的优势信号被稀释,放大机制带不动 —— 定量验证了"PPO 是放大器"的前提条件判断。checkpoints 已存至 step_650。

### 5.3 转向:sim 演示 SFT(正确配方的缺失环节,进行中)
按升级阶梯第 5 级自主执行:停 v5r2 → 注册 `pi05_so101_sim` TrainConfig → 计算 sim 数据集 norm_stats(`assets/pi05_so101_sim/so101-sim-demos/`)→ 建 `examples/sft/config/so101_sft_sim.yaml`(从 SFT-8000 **继续**训 4000 步,数据 = 100 条 sim 脚本演示)→ 启动。
- 途中错误(第 16 号):克隆配置漏改 `train_data_paths`(仍指真机数据集)→ 首启失败,修正后正常训练
- **与第一次 SFT 的区别**:上次教"怎么抓"(真机画面),这次教"在 sim 里怎么抓"(sim 画面)—— 补齐视觉域适配,让 RL 开局就有可观成功率(= RLinf 官方 pi05+ManiSkill 成功配方的做法)
- 下一步:SFT 完成(~2h)→ eval 实测 sim 初始成功率(对照 1.5% 天花板)→ 以新权重为起点重启 RL

### 5.4 计划不变项
- **Phase 2**(用户确认):抓取跑通后,扩展到完整 pick-and-place(运送→盒上方→释放,用户 12 步演示语义)
- **Round 2(暂缓)**:sim2real 部署桥(Mac↔H200)、桌沿对齐等视觉保真项、相机外参/关节零位精调

---

## 附录:完整问题登记表(50 条,症状→根因→预防)

Symptom → root cause → prevention. Scan this table BEFORE similar work.

**Install / machine**
| # | Symptom | Root cause | Prevention |
|---|---|---|---|
| 1 | apt "network unreachable" | box needs IPv6 proxy | write `/etc/apt/apt.conf.d/99proxy` (`http://[fdbd:dc61:d:297::16]:8888`) |
| 2 | install hung 34 min on wget | wget ignores UPPERCASE proxy vars | set lowercase `http_proxy/https_proxy` + `/root/.wgetrc` |
| 3 | `vk::createInstanceUnique ErrorIncompatibleDriver` | compute-only NVIDIA driver, no GL/Vulkan userland | symlink EXACT-version `.run` libs into `.venv/nvidia_gl/` + private ICD json; never apt's mismatched libnvidia-gl |
| 4 | zombie procs accumulate forever (11k+) | container PID 1 is `sleep infinity`, never reaps | kill the leaking parent; zombies themselves harmless (pid_max 4.2M) |

**Sim scene / cameras**
| # | Symptom | Root cause | Prevention |
|---|---|---|---|
| 5 | "盒子不像盒子" (solid black block) | built box as solid cuboid | real prop = OPEN tray: black rim walls + light floor |
| 6 | board size/orientation/distance wrong ×3 | guessed instead of measured | 12"×8", long edge along y, near edge at robot base; confirm each prop vs real photo |
| 7 | desk color mismatch | ManiSkill default wooden table | thin cream kinematic cover box |
| 8 | red cube sometimes off board | spawn region too wide | constrain spawn to board area |
| 9 | `sapien.Pose(q=look_at().q)` crash (tensor [1,4]) | look_at returns batched pose | use look_at() result directly |
| 10 | wrist camera saw nothing useful | mount-frame direction unknown | 9-pose sweep grid, workspace is Fixed_Jaw **-y** |
| 11 | wrist view broke after joint recalibration | camera mounts on Fixed_Jaw → moves with wrist_flex OFFSET | re-point wrist cam after ANY wrist joint offset change |

**Units / calibration**
| # | Symptom | Root cause | Prevention |
|---|---|---|---|
| 12 | arm slams to limits, "运动方向都是错的" | LeRobot NORMALIZED units fed as radians | audit unit conventions FIRST for any new dataset/env pair |
| 13 | round-trip error 355, joints clamped | subtracted homing_offset in software | LeRobot homing lives in servo HW register; homed frame centers at tick 2048 |
| 14 | gripper "grabs air" pointing forward/up | URDF zero ≠ real mount for wrist_flex | OFFSET +0.6 rad (down-forward); verify sign via user/video, negative was UP |
| 15 | **jaws never below 8.1cm gap (cube 2.9cm)** | gripper linkage ≠ arm servo → arm tick→rad invalid | dedicated map norm0→−1.0rad / norm100→+0.5rad; ALWAYS measure command→travel curve for a new actuator |
| 16 | wrist_roll commands −3.65 rad (limit −3.14) | follower calib = uncalibrated full turn 0..4095 | clip all joints to URDF limits in norm_to_rad |
| 17 | gripper bug found late | only audited the joint user asked about | when one joint is wrong, audit ALL siblings immediately |

**Dataset / SFT / config**
| # | Symptom | Root cause | Prevention |
|---|---|---|---|
| 18 | dataset load fails | missing v2.1 tag | `HfApi().create_tag` |
| 19 | norm_stats pickle error / OOM-slow | runpy wrapper + full dataset | run as module `-m toolkits.lerobot.calculate_norm_stats`, subsample (15k frames), `__main__` guard |
| 20 | KeyError actions | LeRobot 0.6.1 column is `action` singular | `action_sequence_keys=("action",)` |
| 21 | `lerobot/pi05_base` not found by openpi | maybe_download takes local/gs/s3 only | snapshot_download to local dir first |
| 22 | "Robot so100 not supported" | get_robot_control_mode missing branch | add so100/so101 → pd_joint_pos |
| 23 | rollout worker can't find norm stats | it looks INSIDE the ckpt dir | copy norm_stats.json into `<ckpt>/so101-pick-place-v2/` for every new warm-start |
| 24 | eval crashes on rollout.model.model_type | eval-only reads rollout.model, not actor.model | dedicated eval yaml with `model/pi0_5@rollout.model` |
| 25 | hydra "Primary config directory not found" | relative --config-path from another cwd | absolute --config-path always |

**Distributed / Ray / launch ops**
| # | Symptom | Root cause | Prevention |
|---|---|---|---|
| 26 | "Port could not be cast to integer" | unbracketed IPv6 in tcp:// URL | bracket fix in collective_group.py (in-tree) |
| 27 | rollout→actor transfer hang, GPUs 0% | Ray blocks object store, /tmp 96% full | `RAY_local_fs_capacity_threshold=0.99` |
| 28 | 320-env first rollout hang | render/scale limit with colocated groups | 128 envs + batch 2048 validated; scale cautiously |
| 29 | **flaky first-rollout hangs (×3, 51/46 min)** | find_free_port probed IPv4, comms on IPv6 → TCPStore dials dead/wrong port | dual-stack AF_INET6 probe fix (in-tree, cluster.py:143); never accept "cleanup fixed it" without a traceback |
| 30 | training died at session boundary | bg child of Claude process | `setsid bash launcher </dev/null >/dev/null 2>&1 &` |
| 31 | launcher silently never ran | `sleep` in the chain (harness blocks foreground sleep) | no sleep in launch chains |
| 32 | "cleanup done" but 191 orphans alive | kill loop matched own shell (`*bash*` case) and killed ITSELF | /proc exe check, exclude `$$`, never kill by string match |
| 33 | relaunch hung on "clean" state | verified mid-cleanup, then re-dirtied with more kills | verify-clean must be the LAST step before launch |
| 34 | hang undetected for 51 min | monitors watched reward milestones only | mandatory startup deadline: full first step (`success_once=` line) within 30 min |
| 35 | monitor flooding every epoch | filter pattern matched per-epoch progress lines | filters match only actionable signals |
| 36 | stale monitor reported old EXIT | tail -f on a log from a previous attempt | fresh log file per attempt, or reset monitors on relaunch |
| 37 | false "train alive" / phantom PIDs | pgrep -f matches your own command line | /proc/<pid>/exe verification, always |

**RL design / training**
| # | Symptom | Root cause | Prevention |
|---|---|---|---|
| 38 | 12h RL: reward up, success 0 forever | gripper physically couldn't close (see #15) | before ANY long run: verify the success condition is physically reachable (motion-planner probe) |
| 39 | policy hovers, gripper drifts open | shaping #1: far-open (0.3×~60 steps) ≫ near-close (0.5×~5) | compute per-episode TIME INTEGRAL of every reward term |
| 40 | shaping taught the OPPOSITE of demos | far-open contradicts demo (closed in transport, open above cube) | map reward semantics to the demonstrated step sequence first |
| 41 | ckpts from bad-shaping runs never recover | hover attractor baked in; PPO on flow-VLA barely moves behavior (KL 0.01–0.04) | NEVER warm-start from a checkpoint trained under a bad reward; audit target behavior (grip_q) first |
| 42 | 1.3M transitions, zero exploratory closes | narrow flow-VLA exploration can't jump 63 norm units | binary-jump rewards need either initial success or brute exploration; else add a gradient bridge |
| 43 | recipe doubt raised only by user | compared references' REWARDS but not their PRECONDITIONS | frame-level premortem (section 1) before every launch |
| 44 | crash at the FIRST-ever success (16 vs 15) | `_red_start_z` overwritten by partial reset batch | full-batch per-episode buffers + explicit partial-reset test (section 5b) |
| 45 | cube friction 0.3 (low) suspected early | ManiSkill default; ruled non-blocking (jaws never touched cube) | revisit friction only AFTER contact happens; don't fix invisible problems |

**Verification / reporting**
| # | Symptom | Root cause | Prevention |
|---|---|---|---|
| 46 | wrong claim about arm motion | judged from 3-4 frames | read ALL frames before behavioral claims |
| 47 | "为什么不能显示图像了" | Read shows image only to the model | every cited image must go through SendUserFile |
| 48 | reported run "alive" when dead | own grep matched itself (see #37) | evidence = GPU util + advancing epochs, never "process exists" |
| 49 | "won't happen again", then it did | promise instead of mechanism | offer bounded damage (deadlines) + falsifiable criteria, never promises |
| 50 | user had to trigger every frame-level insight | plan-continuation bias | premortem + verdict points; idle GPU while thinking < wrong 4h run |

