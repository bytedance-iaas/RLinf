# RLinf Pi0.5 + LIBERO 快速上手

本文档介绍如何在火山引擎容器服务上部署 RLinf 训练环境，并在单机 4 卡上运行 Pi0.5 +
LIBERO 的 async + colocated PPO 训练。

RLinf 是面向具身智能的强化学习训练框架。本文覆盖两部分内容：**部署**（一次性完成）与
**运行训练**（每次实验重复执行）。源码、模型、日志与 checkpoint 均保存在 `/workspace`
持久卷中，Pod 重建后不会丢失。

---

# 一、部署

部署通过 Helm Chart 完成。Chart 会创建一个 StatefulSet，其中包含两个容器：

| 容器 | 职责 |
|---|---|
| `rlinf` | 训练容器。常驻运行，用户进入容器后手动启动训练任务 |
| `dashboard` | 可视化面板 Sidecar。以只读方式挂载 `/workspace`，扫描日志目录并提供 Web 界面 |

采用 Sidecar 架构的目的是隔离故障域：面板进程异常时仅重启该容器，训练进程不受影响。

## 1. 创建面板访问凭据

RLinf Dashboard 支持 HTTP Basic 认证。**通过 API 网关对外发布时必须启用该认证**，否则
Chart 会在渲染阶段直接拒绝安装，以避免将无鉴权页面暴露至公网。

建议将凭据预先写入 Kubernetes Secret，使其不会进入 Helm 的 release 历史记录：

```bash
kubectl create namespace rlinf

kubectl create secret generic rlinf-dashboard-auth -n rlinf \
  --from-literal=username=<用户名> \
  --from-literal=password='<密码>'
```

Chart 默认读取 `username` 与 `password` 两个 key。若已有 Secret 使用其他 key 名称，可通过
`dashboard.auth.usernameKey` 与 `dashboard.auth.passwordKey` 指定。

> 凭据亦可直接写入 values 文件，由 Helm 自动创建 Secret。但该方式会将明文保留在 release
> 历史中（通过 `helm get values` 可读取），仅建议用于私有测试环境。

## 2. 编写 values 文件

Chart 不预设镜像版本与存储类型，以下两项为必填，缺失时安装会在渲染阶段失败：

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
    existingSecret: rlinf-dashboard-auth
```

其余可选配置：

| 配置项 | 说明 |
|---|---|
| `nodeSelector` | 默认不锁定节点，由 GPU 申请量决定调度结果。需要指定机型或特定节点时填写 |
| `dashboard.logsPath` | 面板扫描的日志目录，默认 `/workspace/RLinf/logs`。**须与训练时的 `log_path` 保持一致**，否则界面中不会出现对应的 run |
| `dshmSize` | 共享内存大小，默认 `256Gi`。容器默认的 64Mi 会导致多进程仿真出现 "Bus error" |
| `persistence.size` | 持久卷容量，默认 `500Gi`。**创建后不可修改**，扩容需直接调整 PVC |

## 3. 配置公网入口（API 网关）

若无需公网访问，可跳过本节，保持 `apig.enabled=false`（默认值），通过端口转发访问面板。

需要公网 HTTPS 入口时，可选择以下两种方式。

### 方式 A：接入已有网关

```yaml
apig:
  enabled: true
  create: false
  existingId: <网关实例 ID>          # 于 API 网关控制台查询
  ingressClassName: <网关声明的 class>
  host: rlinf.apig.local            # 内部占位域名，同一网关下不可重复
```

该方式一次部署即可生效，且 `helm uninstall` 不会影响网关本身，适用于需要长期稳定访问
地址的场景。

> 网关在控制台中显示的名称通常与 Kubernetes 中的对象名称不一致，**建议以实例 ID 为准进行
> 匹配**。

### 方式 B：新建网关

```yaml
apig:
  enabled: true
  create: true
  subnetIds:
    - <集群 VPC 内的子网 ID>
  host: rlinf.apig.local
```

一次安装即可，无需回填任何信息。网关开通需要几分钟，可通过以下命令观察：

```bash
kubectl get apiginstance rlinf-apig -n rlinf
```

状态变为 `Running` 后，实例 ID 会写入该资源的 `status.id`，Ingress 通过 ingress class
自动完成绑定。

> ⚠️ **不要把该 ID 回填到 `apig.existingId`。** `existingId` 会写入 `spec.id`，而该字段在 CRD
> 中不可变，准入校验会拒绝后续每一次 `helm upgrade`：
>
> ```text
> spec.id: Forbidden: forbidden to update, old: , new: <id>
> ```
>
> 此后 release 会一直处于 `failed`，直到把该值清空。`existingId` 仅用于方式 A。

网关规格（`instanceSpecCode`、`clbSpecCode` 等）均为可选，留空时由平台选择默认值（实测默认
即 1c2g / small_1 / 2 副本）。如需显式指定，注意 `publicNetworkBandwidth` 与 `replicas` 在
CRD 中为整数类型，写成带引号的字符串会被校验拒绝。

⚠️ 通过方式 B 创建的网关会随 `helm uninstall` 一并删除，其自动分配的 `*.volceapi.com`
域名同时失效，重新部署将获得新的域名。

## 4. 执行安装

```bash
helm install rlinf oci://ai-containers-cn-beijing.cr.volces.com/physicalai/rlinf \
  --version <Chart 版本> \
  -n rlinf -f values.yaml
```

## 5. 验证部署结果

```bash
kubectl rollout status statefulset/rlinf -n rlinf --timeout=10m
kubectl get pod -n rlinf
```

部署成功时 Pod 状态为 **`2/2 Running`**，表示训练容器与面板容器均已就绪：

```text
NAME      READY   STATUS    RESTARTS   AGE
rlinf-0   2/2     Running   0          2m
```

面板容器的日志可单独查看：

```bash
kubectl logs rlinf-0 -n rlinf -c dashboard
```

## 6. 访问 Dashboard

**已启用 API 网关**：平台会为每个 host 分配独立的 `*.volceapi.com` 域名。该域名**无法通过
`kubectl` 查询，需在 API 网关控制台的服务列表中获取**。访问该域名后，使用第 1 步设置的
凭据登录。

**未启用 API 网关**：使用端口转发访问。

```bash
kubectl port-forward -n rlinf pod/rlinf-0 8420:8420
```

随后在浏览器中打开 `http://localhost:8420`。

面板包含以下页面：

| 页面 | 内容 |
|---|---|
| Runs | run 状态、健康度、训练进度与心跳 |
| Overview | 当前阶段、耗时、checkpoint、异常信号与关键指标 |
| Metrics | TensorBoard 曲线，按 embodied 模板分组 |
| Media | LIBERO 训练录像 |
| Events | 生命周期、阶段切换与退出原因 |

此时尚无训练任务，页面为空属正常现象。训练启动后面板每 5 秒自动发现一次，无需重启。

---

# 二、运行训练

## 1. 进入训练容器

```bash
kubectl exec -it rlinf-0 -n rlinf -c rlinf -- bash
```

由于 Pod 中包含两个容器，`-c rlinf` 参数不可省略。后续命令均在容器内执行。

## 2. 准备 Pi0.5 模型

若模型已存在则跳过本节。可先行检查：

```bash
test -d /workspace/models/RLinf-Pi05-LIBERO-SFT && echo "model exists"
```

否则从 HuggingFace 将模型下载至持久卷。模型仓库为
[RLinf/RLinf-Pi05-LIBERO-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT)：

```bash
mkdir -p /workspace/models

# 国内网络可先指向镜像站，可显著提升下载速度
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download RLinf/RLinf-Pi05-LIBERO-SFT \
  --local-dir /workspace/models/RLinf-Pi05-LIBERO-SFT
```

下载完成后，确认模型目录为 `/workspace/models/RLinf-Pi05-LIBERO-SFT`。训练命令会将 actor
与 rollout 的 `model_path` 均指向该路径。

## 3. 推荐配置

基础配置文件为
`examples/embodiment/config/libero_spatial_async_ppo_openpi_pi05.yaml`，并通过命令行覆盖
为经过验证的 4 卡配置：

- async + colocated 模式，actor、rollout 与 env 共用 4 张 GPU；
- 64 个训练环境，horizon 120；
- `group_size=2`、`update_epoch=2`；
- global batch 128、micro batch 32；
- 每 40 step 保存一次 checkpoint，并在最后一步保存。

## 4. RLinf 性能优化特性

以下三项为 RLinf 在标准训练流程之外提供的性能优化能力，默认均为关闭状态，可按需启用。

### 4.1 异步权重同步

| | |
|---|---|
| 配置项 | `+actor.sync_weight_no_wait=true` |
| 默认值 | `false` |

标准流程中，actor 完成更新后需等待权重同步至 rollout 才能继续，同步耗时期间 actor 处于
空闲状态。启用该特性后，actor 发起同步即继续执行后续计算，将同步开销与下一步训练重叠。

该优化适用于异步训练场景：rollout 使用落后一个版本的权重，本身即是 off-policy 训练的预期
行为。框架保证同一时刻仅有一次同步在执行，新请求会被合并而非排队，因此传输较慢时不会
形成积压；权重只会被丢弃，不会出现乱序。

> 该配置项不在 YAML 中预定义，覆盖时**必须使用 `+` 前缀**，否则 Hydra 会报
> `Key not in struct`。

### 4.2 Fused Prefix Kernel

| | |
|---|---|
| 配置项 | `actor.model.openpi.enable_fused_prefix=true` |
| 默认值 | `false` |

将 Pi0.5 中 PaliGemma 视觉语言模型侧的 decoder layer 替换为算子融合实现，包含融合的前向
计算与手写的反向实现，支持任意 additive attention mask。

该替换仅作用于使用标准 RMSNorm 的 prefix 侧，action expert（adaRMS）部分保持不变。

### 4.3 Rollout 图编译

| | |
|---|---|
| 配置项 | `+rollout.enable_torch_compile=true` |
| 默认值 | `false` |

对 rollout 推理过程启用 torch.compile 图编译。需注意首次运行存在编译开销。

## 5. 性能数据

> **TODO**：补充最新一轮性能测试数据与结论链接。

## 6. 启动训练

RLinf 提供 `examples/embodiment/run_async.sh` 作为标准入口，可自动完成环境变量设置：

```bash
bash examples/embodiment/run_async.sh libero_spatial_async_ppo_openpi_pi05 LIBERO
```

该脚本不支持追加额外的配置覆盖参数。因此在需要调整 batch、环境数量或启用上述优化特性时，
应直接调用训练脚本：

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

关于环境变量：`EMBODIED_PATH` 为配置文件解析所必需，缺失时 Hydra 在插值阶段即会失败；
`MUJOCO_GL` 与 `PYOPENGL_PLATFORM` 用于 LIBERO 的离屏渲染；`ROBOT_PLATFORM` 决定动作维度
与归一化方式。GPU 可见性由容器的资源申请决定，无需额外指定。

每次训练建议使用独立的 `log_path`，以避免 TensorBoard 指标与面板 run 相互覆盖。`LOG_DIR`
须位于面板扫描目录（默认 `/workspace/RLinf/logs`）之下，否则界面中不会出现该 run。

该命令会完成 Hydra 配置解析、Ray 集群连接、actor/rollout/env worker 创建，随后开始 PPO
训练。面板将自动发现该 run。

如需快速验证环境，可在训练命令末尾（重定向之前）追加：

```text
runner.max_epochs=2 runner.max_steps=2 runner.save_interval=-1
```

`runner.max_epochs` 与 `runner.max_steps` 需同时设置。embodied runner 将每个 epoch 固定为
一个 RL step，仅设置 `max_steps` 无法突破较小的 `max_epochs` 限制。

> ⚠️ 调整训练规模时请勿单独修改 `env.train.total_num_envs`。actor 侧存在
> `total_num_envs % (global_batch_size / world_size) == 0` 的约束，仅缩小环境数量会在模型
> 加载与首轮 rollout 完成后才触发断言失败。建议通过 `max_episode_steps` 与
> `max_steps_per_rollout_epoch` 控制单步耗时。

## 7. 查看训练进度

在新的 shell 中需重新指定本次运行的目录：

```bash
export RUN_NAME=<RUN_NAME>
export LOG_DIR=/workspace/RLinf/logs/${RUN_NAME}
```

查看主日志与结构化指标：

```bash
tail -f "${LOG_DIR}/run.log"
tail -f "${LOG_DIR}/metrics.log"
```

`metrics.log` 在第一个训练 step 完成后生成。模型与 LIBERO 环境初始化期间，可先查看
`run.log`。

仅查看最新进度：

```bash
grep "Global Step:" "${LOG_DIR}/metrics.log" | tail -n 5
```

主要观察指标：

| 指标 | 含义 |
|---|---|
| `env/success_once` | 每批环境是否至少成功一次，训练效果的主要观察指标 |
| `env/return` | 环境回报 |
| `time/step` | 端到端每个 RL step 的耗时 |
| `time/rollout/predict` | Pi0.5 rollout 推理耗时 |
| `time/actor_training` | actor 训练耗时 |
| `time/env/run_interact_once` | 完整 rollout/env 交互耗时 |
| `time/actor/wait_for_rollout_store_ready` | actor 等待 rollout batch 的时间 |
| `train/actor/policy_loss` | policy loss |
| `train/critic/value_loss` | value loss |

本次运行的 Hydra 最终配置保存于 `${LOG_DIR}/tensorboard/config.yaml`。

查看 GPU 与 Ray 状态：

```bash
nvidia-smi
ray status
```

## 8. Checkpoint

当前配置每 40 step 及最后一个 step 保存一次。embodied runner 会在 `log_path` 下附加一层
`experiment_name`，因此完整路径为：

```text
${LOG_DIR}/${RUN_NAME}/checkpoints/global_step_<N>/actor/
```

主要内容包括：

```text
actor/dcp_checkpoint/                         # 模型、optimizer 与 scheduler 状态
actor/model_state_dict/full_weights.pt        # rank 0 汇总的完整模型权重
```

恢复训练时，先记录原有 checkpoint 路径：

```bash
export RESUME_DIR=/workspace/RLinf/logs/<OLD_RUN_NAME>/<OLD_RUN_NAME>/checkpoints/global_step_40
```

随后在训练命令末尾追加：

```text
runner.resume_dir="${RESUME_DIR}"
```

`runner.resume_dir` 须指向 `global_step_<N>` 目录，而非其下的 `actor` 子目录。恢复后的训练
可使用新的 `RUN_NAME` 与 `LOG_DIR`，`RESUME_DIR` 仍指向原有运行的绝对路径。

---

# 三、常见问题

## 部署阶段

| 现象 | 原因与处理方式 |
|---|---|
| 安装时提示缺少 `image.tag` 或 `persistence.storageClass` | Chart 不预设这两项，需在 values 中补充 |
| 提示 `dashboard.auth.enabled=true is required` | 通过 API 网关发布面板时必须启用认证，参见部署第 1 节 |
| Pod 停留在 `1/2`，`dashboard` 容器反复重启 | 面板镜像版本过低，不支持注入的认证配置。请使用支持 `RLINF_DASHBOARD_AUTH_MODE` 的镜像版本。该拦截为预期行为，用于避免旧版本忽略凭据后对外提供无鉴权服务 |
| 容器报 `CreateContainerConfigError` | `existingSecret` 指定的 Secret 不存在。Helm 不校验集群中是否存在该资源，安装会成功但容器无法启动 |
| `helm upgrade` 执行成功，但 Pod 仍运行旧镜像 | Pod 处于非 Ready 状态时，StatefulSet 滚动更新不会推进。执行 `kubectl delete pod rlinf-0 -n rlinf` 使其按新模板重建 |
| Ingress 长时间没有访问地址 | 查看网关是否已 `Running`：`kubectl get apiginstance rlinf-apig -n rlinf`。若为 `Pending` 或创建失败，见下一行 |
| 网关状态为创建失败 | 常见原因是账号余额不足或子网可用 IP 不足。CR 与 controller 日志都不会给出原因，需到 API 网关控制台查看实例状态 |
| `helm upgrade` 报 `spec.id: Forbidden: forbidden to update` | `apig.create=true` 时填了 `apig.existingId`。清空该值后重新 upgrade |
| 无法获取公网访问域名 | 域名仅可在 API 网关控制台的服务列表中查询，`kubectl` 无法获取 |

## 训练阶段

| 现象 | 原因与处理方式 |
|---|---|
| 面板中不显示训练任务 | `LOG_DIR` 须位于 `dashboard.logsPath`（默认 `/workspace/RLinf/logs`）之下 |
| 任务长时间处于 pending 状态 | 模型与环境仍在初始化，可查看 `${LOG_DIR}/run.log` |
| `metrics.log` 不存在 | 第一个 RL step 尚未完成，可继续查看 `run.log` |
| Hydra 报 `Key ... is not in struct` | 配置中未预定义的键需保留 `+` 前缀，如 `+actor.sync_weight_no_wait=true` |
| 首个 step 耗时明显偏长 | 模型与环境初始化的固有开销；启用图编译时还包含首次编译时间，不应据此判断稳态性能 |
| 训练异常退出后残留 Ray 进程 | 执行 `ray stop --force` 后重新启动训练 |

## 面板运维

面板以独立容器方式运行，日志通过 kubectl 查看：

```bash
kubectl logs -f rlinf-0 -n rlinf -c dashboard
```

更新 Secret 中的凭据后，仅需重启该 Sidecar，正在运行的训练不受影响：

```bash
kubectl exec -n rlinf rlinf-0 -c dashboard -- sh -c 'kill -TERM 1'
```

⚠️ 启用或关闭认证、修改 Secret 名称或 key 名称、变更镜像版本，均会改动 Pod 模板并触发
**整个 Pod 重建**，建议在没有训练任务运行时执行。
