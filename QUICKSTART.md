# RLinf 快速上手

本文介绍如何在火山引擎容器服务上部署 RLinf，并跑通一次完整的具身强化学习训练。全文以
π₀.₅ 为例，主线是单机 4 卡的 LIBERO async PPO 训练，第二部分第 9 节另给出 SO101 +
ManiSkill 的跑法。

内容分为两部分：**部署**只做一次，**运行训练**每次实验重复。源码、模型、日志与
checkpoint 都保存在 `/workspace` 持久卷中，Pod 重建后不会丢失。

开始之前需要准备一个可用的 Kubernetes 命名空间和相应的 GPU 配额。后续命令统一用 `${NS}`
指代该命名空间：

```bash
export NS=<你的命名空间>
```

---

# 一、部署

部署通过 Helm Chart 完成。Chart 创建一个 StatefulSet，其中包含两个容器：

| 容器 | 职责 |
|---|---|
| `rlinf` | 训练容器。常驻运行，进入容器后手动启动训练任务 |
| `dashboard` | 可视化面板 Sidecar。以只读方式挂载 `/workspace`，扫描日志目录并提供 Web 界面 |

面板做成 Sidecar 是为了隔离故障域。面板进程异常时只重启该容器，训练进程不受影响。

## 1. 准备面板访问凭据

Dashboard 支持 HTTP Basic 认证。通过 API 网关对外发布时必须启用认证，否则 Chart 会在渲染
阶段拒绝安装，以免把无鉴权页面挂到公网上。

Chart 只接受引用已有的 Kubernetes Secret，不支持在 values 里直接填写用户名密码。Helm 会把
values 原样保存在 release 历史中，任何能执行 `helm get values` 的人都可以读到明文。

凭据本身按集群统一的一份来管理，与具体某次 RLinf 部署无关，本文使用 `physical-ai-auth`
这个名字。Kubernetes 不允许 Pod 引用其他命名空间下的 Secret，因此部署所在的命名空间里需要
有一份同名同内容的 Secret：

```bash
kubectl create secret generic physical-ai-auth -n "${NS}" \
  --from-literal=username=<用户名> \
  --from-literal=password='<密码>'
```

Chart 默认读取 `username` 与 `password` 两个 key。Secret 若使用其他 key 名，通过
`dashboard.auth.usernameKey` 与 `dashboard.auth.passwordKey` 指定。

## 2. 编写 values 文件

Chart 不预设镜像版本与存储类型，以下两项必填，缺失时安装会在渲染阶段失败：

```yaml
image:
  repository: ai-containers-cn-beijing.cr.volces.com/physicalai/rlinf
  tag: <镜像版本>

persistence:
  storageClass: <集群块存储 StorageClass>    # kubectl get storageclass
  size: 500Gi

# 4 卡训练的实测取值。变更卡数时，按每卡约 4 core / 32Gi 申请量等比缩放。
resources:
  requests: { cpu: "16", memory: 128Gi, nvidia.com/gpu: "4" }
  limits:   { cpu: "60", memory: 512Gi, nvidia.com/gpu: "4" }

dashboard:
  image:
    repository: ai-containers-cn-beijing.cr.volces.com/physicalai/rlinf-dashboard
    tag: ""            # 留空表示与 image.tag 保持一致
  auth:
    enabled: true
    existingSecret: physical-ai-auth
```

其余可选配置：

| 配置项 | 说明 |
|---|---|
| `nodeSelector` | 默认不锁定节点，由 GPU 申请量决定调度结果。需要指定机型或特定节点时填写 |
| `dashboard.logsPath` | 面板扫描的日志目录，默认 `/workspace/RLinf/logs`。须与训练时的 `log_path` 一致，否则界面中不会出现对应的 run |
| `dshmSize` | 共享内存大小，默认 `256Gi`。容器默认的 64Mi 会让多进程仿真报 Bus error |
| `persistence.size` | 持久卷容量，默认 `500Gi`。创建后不可修改，扩容需直接调整 PVC |

## 3. 配置公网入口（API 网关）

不需要公网访问时可跳过本节，保持 `apig.enabled=false`（默认值），用端口转发访问面板。

需要公网 HTTPS 入口时，有两种方式。

### 方式 A：接入已有网关

```yaml
apig:
  enabled: true
  create: false
  existingId: <网关实例 ID>          # 于 API 网关控制台查询
  ingressClassName: <网关声明的 class>
  host: rlinf.apig.local            # 内部占位域名，同一网关下不可重复
```

一次部署即可生效，`helm uninstall` 也不会影响网关本身，适合需要长期稳定访问地址的场景。

网关在控制台中显示的名称通常与 Kubernetes 中的对象名称不一致，匹配时以实例 ID 为准。

### 方式 B：新建网关

```yaml
apig:
  enabled: true
  create: true
  subnetIds:
    - <集群 VPC 内的子网 ID>
  host: rlinf.apig.local
```

一次安装即可，无需回填任何信息。网关开通需要几分钟，用以下命令观察：

```bash
kubectl get apiginstance rlinf-apig -n "${NS}"
```

状态变为 `Running` 后，实例 ID 会写入该资源的 `status.id`，Ingress 通过 ingress class 自动
完成绑定。

这个 ID 不要回填到 `apig.existingId`。`existingId` 会写入 `spec.id`，而该字段在 CRD 中不可
变，准入校验会拒绝后续每一次 `helm upgrade`：

```text
spec.id: Forbidden: forbidden to update, old: , new: <id>
```

此后 release 会一直处于 `failed`，直到把该值清空。`existingId` 仅用于方式 A。

网关规格（`instanceSpecCode`、`clbSpecCode` 等）都是可选的，留空时由平台选择默认值，实测
默认即 1c2g / small_1 / 2 副本。显式指定时注意 `publicNetworkBandwidth` 与 `replicas` 在
CRD 中是整数类型，写成带引号的字符串会被校验拒绝。

方式 B 创建的网关会随 `helm uninstall` 一并删除，其自动分配的 `*.volceapi.com` 域名同时
失效，重新部署将得到一个新域名。

## 4. 执行安装

```bash
helm install rlinf oci://ai-containers-cn-beijing.cr.volces.com/physicalai/rlinf \
  --version <Chart 版本> \
  -n "${NS}" -f values.yaml
```

## 5. 验证部署结果

```bash
kubectl rollout status statefulset/rlinf -n "${NS}" --timeout=10m
kubectl get pod -n "${NS}"
```

部署成功时 Pod 状态为 `2/2 Running`，表示训练容器与面板容器均已就绪：

```text
NAME      READY   STATUS    RESTARTS   AGE
rlinf-0   2/2     Running   0          2m
```

面板容器的日志可以单独查看：

```bash
kubectl logs rlinf-0 -n "${NS}" -c dashboard
```

## 6. 访问 Dashboard

**已启用 API 网关**：平台会为每个 host 分配独立的 `*.volceapi.com` 域名。该域名无法通过
`kubectl` 查询，需要在 API 网关控制台的服务列表中获取。打开域名后用第 1 步的凭据登录。

**未启用 API 网关**：使用端口转发访问。

```bash
kubectl port-forward -n "${NS}" pod/rlinf-0 8420:8420
```

随后在浏览器中打开 `http://localhost:8420`。

此时还没有训练任务，页面为空是正常的。训练启动后面板每 5 秒自动发现一次，无需重启。页面
结构与各项指标的含义见第二部分第 7 节。

---

# 二、运行训练

## 1. 进入训练容器

```bash
kubectl exec -it rlinf-0 -n "${NS}" -c rlinf -- bash
```

Pod 中有两个容器，`-c rlinf` 不能省略。后续命令都在容器内执行。

## 2. 准备 π₀.₅ 模型

镜像内置了 `oniond`，直接从对象存储拉取 SFT 检查点：

```bash
mkdir -p /workspace/models
oniond download model RLinf-Pi05-LIBERO-SFT --dir /workspace/models
```

模型会落在 `/workspace/models/RLinf-Pi05-LIBERO-SFT`，训练命令中 actor 与 rollout 的
`model_path` 都指向这个路径。下载支持断点续传，中断后重跑同一条命令即可。

模型已存在时跳过本节，可以先确认一下：

```bash
test -d /workspace/models/RLinf-Pi05-LIBERO-SFT && echo "model exists"
```

## 3. 本文使用的训练配置

基础配置文件为
`examples/embodiment/config/libero_spatial_async_ppo_openpi_pi05.yaml`，并通过命令行覆盖为
经过验证的 4 卡配置：

- async + colocated 模式，actor、rollout 与 env 共用 4 张 GPU；
- 64 个训练环境，horizon 120；
- `group_size=2`、`update_epoch=2`；
- global batch 128、micro batch 32；
- 每 40 step 保存一次 checkpoint，并在最后一步保存。

## 4. 性能优化特性

RLinf 在标准训练流程之外提供三项自研性能优化，默认均为关闭状态。本节说明各自的作用，
第 5 节给出按场景的推荐组合与实测收益。

| 特性 | 配置项 | 作用侧 |
|---|---|---|
| 异步权重同步 | `+actor.sync_weight_no_wait=true` | actor |
| Fused Prefix Kernel | `actor.model.openpi.enable_fused_prefix=true` | actor（默认同时作用于 rollout） |
| Rollout 图编译 | `+rollout.enable_torch_compile=true` `+rollout.torch_compile_mode=default` | rollout |

### 4.1 异步权重同步

actor 完成参数更新后无需等待权重同步至 rollout，可以立即继续下一步计算，把同步开销与训练
重叠。框架保证同一时刻只有一次同步在执行，权重不会乱序或积压。该行为适用于 async 训练：
rollout 使用落后一个版本的权重，本身就是 off-policy 训练的预期行为。

### 4.2 Fused Prefix Kernel

将 π₀.₅ 中 PaliGemma 视觉语言模型侧的 decoder layer 替换为算子融合实现，包含融合的前向
计算与手写的反向实现，用于加速 actor 训练。action expert 部分不受影响。

rollout 的模型配置由 actor 深拷贝而来，因此这个开关默认同时作用于 actor 与 rollout。需要
rollout 单独关闭时，追加 `+rollout.model.openpi.enable_fused_prefix=false`。

### 4.3 Rollout 图编译

对 rollout 推理启用 torch.compile，加速 π₀.₅ 的动作预测。首步存在一次性编译开销，实测增加
约 53 至 59 秒，不应据此判断稳态性能。

两处写法需要注意：

- `sync_weight_no_wait` 与 `rollout.enable_torch_compile` 不在 YAML 中预定义，覆盖时必须带
  `+` 前缀，否则 Hydra 会报 `Key not in struct`。
- `torch_compile_mode` 需显式写成 `default`。省略时会走代码兜底值
  `max-autotune-no-cudagraphs`，编译开销明显更高，得到的也不是第 5 节的数字。

## 5. 优化组合推荐与实测收益

最优组合取决于瓶颈侧，没有通用最优解。判断方法是比较 `time/actor_training` 与
`time/rollout/generate_one_epoch`，数值大的一侧是瓶颈。加速非瓶颈侧换不到收益，在 colocated
下还会因为破坏流水线平衡而变慢：同样开图编译，LIBERO colocated 是 +8.85%（更慢），
LIBERO disaggregated 是 −4.76%（更快），两者只差一个 placement。

| 场景 | 瓶颈侧 | 推荐 | 端到端收益 |
|---|---|---|---|
| LIBERO colocated（本文 4 卡配置） | actor | 只开 fused | **−7.97%** |
| LIBERO disaggregated 2+2 | rollout | 只开 compile | −4.76% |
| ManiSkill disaggregated 2+2 | rollout | 只开 compile | −6.99% |
| 瓶颈侧不确定 | — | split（见下） | 距当场最优 1 个百分点以内 |

各优化项的稳态收益（单机 4×H20，async 模式，四场景实测）：

| 优化项 | 观察指标 | 相对 baseline |
|---|---|---|
| Fused Prefix Kernel | `time/actor_training` | −5.8% ~ −6.9% |
| Rollout 图编译 | `time/rollout/predict` | −11.8% ~ −12.9% |
| 异步权重同步 | — | 单机无可测收益（同机同步仅 1–2 s），面向跨机场景 |

fused 与 compile 不要同时落在 rollout 上。两者直接叠加会互相抵消，rollout 推理收益从 −12%
掉到 −0% 至 −2%。正确用法是两项各管一侧，即 split：

```text
actor.model.openpi.enable_fused_prefix=true \
+rollout.model.openpi.enable_fused_prefix=false \
+rollout.enable_torch_compile=true \
+rollout.torch_compile_mode=default
```

> 完整的测试方法、四场景逐项数据与测量口径，见
> [π₀.₅ 强化学习性能报告](docs/performance/pi05_rl_performance.md)。

## 6. 启动训练

RLinf 提供 `examples/embodiment/run_async.sh` 作为标准入口，可自动完成环境变量设置：

```bash
bash examples/embodiment/run_async.sh libero_spatial_async_ppo_openpi_pi05 LIBERO
```

该脚本不支持追加额外的配置覆盖参数。需要调整 batch、环境数量或启用上述优化特性时，直接
调用训练脚本：

```bash
cd /workspace/RLinf
source switch_env openpi

export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO

export RUN_NAME="$(date +%Y%m%d-%H%M%S)-pi05-libero-4gpu"
export LOG_DIR="/workspace/RLinf/logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

setsid python examples/embodiment/train_async.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name libero_spatial_async_ppo_openpi_pi05 \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name="${RUN_NAME}" \
  +runner.run_id="${RUN_NAME}" \
  runner.val_check_interval=-1 \
  algorithm.group_size=2 \
  algorithm.update_epoch=2 \
  env.train.total_num_envs=64 \
  env.train.max_episode_steps=120 \
  env.train.max_steps_per_rollout_epoch=120 \
  actor.global_batch_size=128 \
  actor.micro_batch_size=32 \
  actor.model.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
  rollout.model.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
  +actor.sync_weight_no_wait=true \
  actor.model.openpi.enable_fused_prefix=true \
  > "${LOG_DIR}/run.log" 2>&1 < /dev/null &
echo $! > "${LOG_DIR}/driver.pid"
```

命令末尾两行是第 4 节的**自研优化参数**，其余为常规训练配置。按第 5 节的场景建议增删：

- **`+actor.sync_weight_no_wait=true`**，异步权重同步（4.1）；
- **`actor.model.openpi.enable_fused_prefix=true`**，Fused Prefix Kernel（4.2）。本文的
  LIBERO colocated 场景中，它是端到端收益最高的一项，−7.97%；
- 改用 disaggregated placement 时，换成 **`+rollout.enable_torch_compile=true`**
  **`+rollout.torch_compile_mode=default`**，即 Rollout 图编译（4.3），并按 4.2 关闭
  rollout 侧的 fused。

几个环境变量的用途：`EMBODIED_PATH` 是配置文件解析所必需的，缺失时 Hydra 在插值阶段就会
失败；`MUJOCO_GL` 与 `PYOPENGL_PLATFORM` 用于 LIBERO 的离屏渲染；`ROBOT_PLATFORM` 决定动作
维度与归一化方式。GPU 可见性由容器的资源申请决定，无需额外指定。

每次训练建议使用独立的 `log_path`，避免 TensorBoard 指标与面板 run 相互覆盖。`LOG_DIR` 须
位于面板扫描目录（默认 `/workspace/RLinf/logs`）之下，否则界面中不会出现该 run。

命令会依次完成 Hydra 配置解析、Ray 集群连接、actor/rollout/env worker 创建，然后开始 PPO
训练。面板会自动发现这次运行。

只想快速验证环境时，在训练命令末尾（重定向之前）追加：

```text
runner.max_epochs=2 runner.max_steps=2 runner.save_interval=-1
```

`runner.max_epochs` 与 `runner.max_steps` 需要同时设置。embodied runner 将每个 epoch 固定为
一个 RL step，只设置 `max_steps` 无法突破较小的 `max_epochs` 限制。

调整训练规模时不要单独修改 `env.train.total_num_envs`。actor 侧有
`total_num_envs % (global_batch_size / world_size) == 0` 的约束，只缩小环境数量会在模型加载
与首轮 rollout 完成之后才触发断言失败。控制单步耗时应通过 `max_episode_steps` 与
`max_steps_per_rollout_epoch`。

## 7. 查看训练进度

有两种查看方式，看的是同一批指标。Dashboard 读取训练进程写出的 TensorBoard 曲线与 run 元
数据，与容器内 `metrics.log` 同源同名。面板适合看趋势和横向对比，命令行适合确认进程是否在
推进。

### 7.1 Dashboard

访问方式见部署第 6 节。训练启动后面板每 5 秒自动发现一次，无需重启。

| 页面 | 内容 |
|---|---|
| 运行列表 | run 状态、健康度、训练进度与心跳。选中两个及以上可进入对比 |
| 概览 | 当前阶段、耗时、checkpoint、异常信号与核心指标 |
| 指标 | TensorBoard 曲线，按 embodied 模板分组（Task performance / Policy optimization / Off-policy lag / Value function / Rollout / Throughput / Evaluation） |
| 视频 | 训练与评测录像，`video_cfg.save_video` 打开时才有 |
| 事件 | 生命周期、阶段切换与退出原因 |
| 对比 | 多个 run 的同一指标叠加对比 |

面板右上角可切换中英文界面。核心指标为 `env/success_once`。

### 7.2 命令行

在新的 shell 中需要重新指定本次运行的目录：

```bash
export RUN_NAME=<RUN_NAME>
export LOG_DIR=/workspace/RLinf/logs/${RUN_NAME}
```

查看主日志与结构化指标：

```bash
tail -f "${LOG_DIR}/run.log"
tail -f "${LOG_DIR}/metrics.log"
```

`metrics.log` 在第一个训练 step 完成后生成。模型与 LIBERO 环境初始化期间先看 `run.log`。
只看最新进度：

```bash
grep "Global Step:" "${LOG_DIR}/metrics.log" | tail -n 5
```

本次运行的 Hydra 最终配置保存在 `${LOG_DIR}/tensorboard/config.yaml`。查看 GPU 与 Ray 状态：

```bash
nvidia-smi
ray status
```

需要注意，`metrics.log` 是 rich 渲染的表格，较长的数值会被截断，不能用于精确取数。需要精确
数值时请读 `${LOG_DIR}/tensorboard/` 下的 event 文件，或者使用面板。

### 7.3 主要指标

面板与 `metrics.log` 使用同一套指标名：

| 指标 | 含义 |
|---|---|
| `env/success_once` | 每批 episode 中至少成功一次的比例，训练效果的核心观察指标 |
| `env/return` | 环境回报 |
| `env/episode_len` | episode 长度。它逼近步数上限而 return 不涨，通常说明策略学会了拖延而不是完成任务 |
| `num_trajectories` | 每步完成的 episode 数，会有波动。它是 env 侧的产出，不是 actor 的训练 batch |
| `time/step` | 端到端每个 RL step 的耗时 |
| `time/actor_training` | actor 训练耗时 |
| `time/rollout/predict` | π₀.₅ rollout 推理耗时 |
| `time/rollout/generate_one_epoch` | rollout 生成一轮的耗时。与 `time/actor_training` 比较可判断瓶颈侧，见第 5 节 |
| `time/env/run_interact_once` | 完整 rollout/env 交互耗时 |
| `time/actor/wait_for_rollout_store_ready` | actor 等待可用轨迹的时间，用于区分 actor 慢和 actor 被饿着 |
| `rollout/discarded_unused_trajs` | 因超出 `algorithm.staleness_threshold` 被丢弃的轨迹数。持续非零说明 rollout 快于训练 |
| `train/actor/policy_loss` | policy loss |
| `train/critic/value_loss` | value loss |
| `train/critic/explained_variance` | critic 解释方差。小于等于 0 表示价值头不如直接预测均值，此时 actor 侧曲线不能按表面读 |
| `eval/success_once` | 确定性评测成功率，`runner.val_check_interval` 触发时才有 |

## 8. Checkpoint

当前配置每 40 step 及最后一个 step 保存一次。embodied runner 会在 `log_path` 下附加一层
`experiment_name`，因此完整路径为：

```text
${LOG_DIR}/${RUN_NAME}/checkpoints/global_step_<N>/actor/
```

主要内容：

```text
actor/dcp_checkpoint/                         # 模型、optimizer 与 scheduler 状态
actor/model_state_dict/full_weights.pt        # rank 0 汇总的完整模型权重
```

恢复训练时，先记录原有 checkpoint 路径：

```bash
export RESUME_DIR=/workspace/RLinf/logs/<OLD_RUN_NAME>/<OLD_RUN_NAME>/checkpoints/global_step_40
```

然后在训练命令末尾追加：

```text
runner.resume_dir="${RESUME_DIR}"
```

`runner.resume_dir` 须指向 `global_step_<N>` 目录，而不是其下的 `actor` 子目录。恢复后的
训练可以使用新的 `RUN_NAME` 与 `LOG_DIR`，`RESUME_DIR` 仍指向原有运行的绝对路径。

## 9. 变体：在 SO101 上用 ManiSkill 做 RL 训练

这是 LIBERO 之外的另一条路径：把真实 SO101（SO-ARM101）机械臂的抓取摆放工作台在 ManiSkill
中重建，并在其上对 π₀.₅ 做 PPO 微调，即 real2sim。部署、容器操作、面板与 checkpoint 流程
完全复用前文，本节只给出差异部分。

| 项 | 内容 |
|---|---|
| 环境 | ManiSkill，任务 id `SO101GrabRedCube-v1` |
| 机器人 | SO101，6 自由度，关节空间绝对位置控制（`pd_joint_pos`） |
| 动作 | 6 维 `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`，LeRobot 归一化电机单位 |
| 观测 | 前视与腕部两路 640×480 相机，外加 6 维关节位置 |
| 成功判据 | 红色方块被放入托盘，且机械臂回到初始位（5 个手臂关节平均偏差小于 0.08 rad） |
| 配置 | `examples/embodiment/config/so101_ppo_openpi_pi05.yaml`，同步 PPO，单机 4 至 8 卡，actor/rollout/env 共卡 |

工作台几何、相机内参、初始位姿与容差都由实测的真实回合与录制的 LeRobot 数据集导出。任务
原理与单位标定细节见
[SO101 示例文档](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/so101.html)。

### 9.1 准备

镜像中已包含 ManiSkill 与 openpi 环境，容器内 `source switch_env openpi` 即可。自建环境用
`bash requirements/install.sh embodied --model openpi --env maniskill_libero`。运行时若提示
缺少 ManiSkill 资产，在容器内执行 `download_assets --assets maniskill`，默认写入
`~/.maniskill`。

还需要两样东西：SO101 的 π₀.₅ SFT 检查点，以及该检查点训练时所用的 `norm_stats.json`。强化
学习的起点必须是在仿真中已具备非零成功率的策略，PPO 放大成功，但不会凭空发现成功。

仿真侧冒烟检查不需要检查点：

```bash
python -m toolkits.so101_smoke
```

### 9.2 启动训练

```bash
cd /workspace/RLinf
source switch_env openpi

export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export SO101_CKPT=/workspace/models/so101_sft_openpi_pi05
export SO101_NORM_STATS=/workspace/assets/so101/norm_stats.json

export RUN_NAME="$(date +%Y%m%d-%H%M%S)-pi05-so101"
export LOG_DIR="/workspace/RLinf/logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

setsid python examples/embodiment/train_embodied_agent.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name so101_ppo_openpi_pi05 \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name="${RUN_NAME}" \
  +runner.run_id="${RUN_NAME}" \
  actor.model.model_path="${SO101_CKPT}" \
  rollout.model.model_path="${SO101_CKPT}" \
  actor.model.openpi_data.norm_stats_path="${SO101_NORM_STATS}" \
  > "${LOG_DIR}/run.log" 2>&1 < /dev/null &
echo $! > "${LOG_DIR}/driver.pid"
```

该任务使用同步 PPO 入口 `train_embodied_agent.py`，LIBERO 一节用的是 async 入口
`train_async.py`。`LOG_DIR` 同样须位于面板扫描目录之下，面板会自动发现这次运行。

投入长训练之前，建议先用冻结策略在带噪 rollout 分布下探针一轮，要求 `env/success_once`
不低于 5%：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name so101_ppo_openpi_pi05 \
  runner.max_epochs=1 runner.val_check_interval=1 \
  actor.optim.lr=1e-9 actor.optim.value_lr=1e-9 \
  actor.model.model_path="${SO101_CKPT}" \
  rollout.model.model_path="${SO101_CKPT}" \
  actor.model.openpi_data.norm_stats_path="${SO101_NORM_STATS}"
```

评测使用同目录下的 `so101_eval_openpi_pi05`（`runner.only_eval: True`），同样通过命令行覆盖
路径，把 `model_path` 指向要评测的检查点：

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name so101_eval_openpi_pi05 \
  actor.model.model_path="${SO101_CKPT}" \
  rollout.model.model_path="${SO101_CKPT}" \
  actor.model.openpi_data.norm_stats_path="${SO101_NORM_STATS}"
```

`eval/success_once`（确定性）与 `env/success_once`（带噪 rollout）应分开报告，两者可以相差
十几个百分点。

配置中有三项刻意偏离 π-RL 默认配方，默认值在本任务上已知会崩塌，请不要还原：
`actor.model.num_action_chunks: 10`（须等于检查点的 action horizon）、
`actor.model.openpi.noise_logvar_range: [0.02, 0.04]`、`actor.global_batch_size: 4096`
（等于每轮样本数，即每轮恰好一次更新）。

第 4 节的优化开关同样适用于该任务，同为 openpi π₀.₅。不过第 5 节的数据是在 LIBERO 与
ManiSkill 的 async 场景下测得的，SO101 上未做测量。

---

# 三、常见问题

## 部署阶段

| 现象 | 原因与处理方式 |
|---|---|
| 安装时提示缺少 `image.tag` 或 `persistence.storageClass` | Chart 不预设这两项，需在 values 中补充 |
| 提示 `dashboard.auth.enabled=true is required` | 通过 API 网关发布面板时必须启用认证，参见部署第 1 节 |
| Pod 停留在 `1/2`，`dashboard` 容器反复重启 | 面板镜像版本过低，不支持注入的认证配置。请使用支持 `RLINF_DASHBOARD_AUTH_MODE` 的镜像版本。该拦截是预期行为，用于避免旧版本忽略凭据后对外提供无鉴权服务 |
| 容器报 `CreateContainerConfigError` | `existingSecret` 指定的 Secret 不存在。Helm 不校验集群中是否存在该资源，安装会成功但容器无法启动 |
| `helm upgrade` 执行成功，但 Pod 仍运行旧镜像 | Pod 处于非 Ready 状态时，StatefulSet 滚动更新不会推进。执行 `kubectl delete pod rlinf-0 -n "${NS}"` 使其按新模板重建 |
| Ingress 长时间没有访问地址 | 查看网关是否已 `Running`：`kubectl get apiginstance rlinf-apig -n "${NS}"`。若为 `Pending` 或创建失败，见下一行 |
| 网关状态为创建失败 | 常见原因是账号余额不足或子网可用 IP 不足。CR 与 controller 日志都不会给出原因，需到 API 网关控制台查看实例状态 |
| `helm upgrade` 报 `spec.id: Forbidden: forbidden to update` | `apig.create=true` 时填了 `apig.existingId`。清空该值后重新 upgrade |
| 无法获取公网访问域名 | 域名只能在 API 网关控制台的服务列表中查询，`kubectl` 无法获取 |

## 训练阶段

| 现象 | 原因与处理方式 |
|---|---|
| 面板中不显示训练任务 | `LOG_DIR` 须位于 `dashboard.logsPath`（默认 `/workspace/RLinf/logs`）之下 |
| 任务长时间处于 pending 状态 | 模型与环境仍在初始化，可查看 `${LOG_DIR}/run.log` |
| `metrics.log` 不存在 | 第一个 RL step 尚未完成，可继续查看 `run.log` |
| Hydra 报 `Key ... is not in struct` | 配置中未预定义的键需保留 `+` 前缀，如 `+actor.sync_weight_no_wait=true` |
| 首个 step 耗时明显偏长 | 模型与环境初始化的固有开销。启用图编译时还包含首次编译时间，不应据此判断稳态性能 |
| 训练异常退出后残留 Ray 进程 | 执行 `ray stop --force` 后重新启动训练 |

## 面板运维

面板以独立容器方式运行，日志通过 kubectl 查看：

```bash
kubectl logs -f rlinf-0 -n "${NS}" -c dashboard
```

更新 Secret 中的凭据后，只需重启该 Sidecar，正在运行的训练不受影响：

```bash
kubectl exec -n "${NS}" rlinf-0 -c dashboard -- sh -c 'kill -TERM 1'
```

启用或关闭认证、修改 Secret 名称或 key 名称、变更镜像版本，都会改动 Pod 模板并触发整个 Pod
重建，建议在没有训练任务运行时执行。
