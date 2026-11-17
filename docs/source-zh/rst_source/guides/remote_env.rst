远程仿真
========

本页介绍如何把仿真器放到一台 GPU 机器上运行，同时让 env worker、policy 和训练留在训练机器上。它适用于训练加速卡自己跑不了仿真器的集群，例如昇腾 NPU 主机；目前支持的仿真器是 ManiSkill 和 LIBERO。

ManiSkill 的 GPU 物理需要 CUDA。没有 CUDA 时，ManiSkill 只能退回 CPU 物理，每个进程只跑一个环境，而一张 GPU 上可以同时跑几百个。改用远程仿真后，EnvWorker 在原本构造仿真器的地方构造一个代理。代理在 env 服务上开一个会话，服务在自己的一张 GPU 上构造真正的环境；之后每次 chunk step，代理把动作发过去，再把环境在 chunk 末尾返回的结果取回来。下面我们先启动服务，再让一次训练连上它，然后看看网络上传的是什么、怎样保证训练可复现，以及 GPU 机器该怎么配。

LIBERO 还有一个更轻的选择。它的物理在 CPU 上很便宜，需要 GPU 的只是渲染，所以 :doc:`remote_render` 把物理留在 policy 旁边，只把「画」这一步交给 GPU 机器。这种模式传的数据更少，视频录制和数据采集也照常可用；只有当训练机器的 CPU 撑不住 LIBERO 的物理时，才需要把整个 LIBERO 环境放到远程。

启动 env 服务
-------------

env 服务是 GPU 机器上的一个独立进程。它不加入 Ray 集群，也不需要训练加速卡。它需要和训练机器相同的 RLinf 代码，两端都要装上承载会话的 HeteroPulse（``pip install heteropulse``），并且像本地运行时一样装好仿真器和它的资产：ManiSkill 需要 SAPIEN 和 ``MS_ASSET_DIR``，LIBERO 需要它的场景文件。这台机器的 NVIDIA 驱动还要能运行这台机器上 PyTorch 编译时所用的 CUDA 版本。如果这台机器不是用 ``requirements/install.sh`` 装的，还要补上它会装的系统包。NVIDIA 上跑 ManiSkill 尤其需要 ``libvulkan1`` 和 ``libegl1``：NVIDIA 的 Vulkan 驱动启动时会加载 ``libEGL.so.1``，缺了它 SAPIEN 建不起 Vulkan 实例。

.. code-block:: bash

   python -m rlinf.envs.remote.env_server --port 18991 --gpus 0,1

这条命令会：

1. 监听 18991 端口。每有一个代理连上来，就在 ``--gpus`` 里挑一张 GPU 启动一个进程：先看这张卡上与该代理同一 split 的会话最少，再看总会话数最少，这样训练会话会先铺满各张 GPU，再两个共用一张。这个进程把 ``CUDA_VISIBLE_DEVICES`` 设为这张 GPU；对基于 MuJoCo 的仿真器，还会设置 ``MUJOCO_GL=egl`` 和与之对应的 EGL 设备。``--max-sessions-per-gpu`` 设定每张 GPU 的会话上限，所有 GPU 都满了之后拒绝新会话。
2. 在这个进程里，按 EnvWorker 传给代理的 split 配置和参数，用 RLinf 自己的 env 类构造环境，一直服务到代理断开。之后进程退出，释放 GPU。训练机器如果没有关闭连接就失联了，服务会在约三分钟内发现。
3. 启动时在日志里打印 ``rlinf/envs`` 下 Python 代码的哈希。哈希不同的代理在连接时就会被拒绝（除非它所在的 split 设置了 ``remote.require_same_code: false``），否则两端对环境行为的理解会不一致。

每个会话都会在日志里记下它分到了哪张 GPU、环境构造用了多久。

.. warning::

   env 服务没有任何认证，而且会话会构造代理要求的任何环境。请把它绑定在只有训练机器能访问的网络上，例如用 ``--host`` 指定一个内网地址。

让训练连上 env 服务
-------------------

是否使用远程仿真按 env 的 split 单独设置。``env_type`` 仍然写真实的仿真器，所以动作格式化等依赖它的逻辑都不变；``remote.address`` 指定服务地址：

.. code-block:: yaml

   cluster:
     num_nodes: 1
     component_placement:
       actor: 0
       rollout: 0
       env:
         node_group: sim
         placement: "0:0"
     node_groups:
       - label: sim
         node_ranks: 0
         ignore_hardware: true

   env:
     train:
       env_type: maniskill
       remote:
         address: "192.168.3.2:18991"
         streams: 16               # 可选，每个环境的数据连接数
         connect_timeout_s: 60     # 可选，每条连接最多等多久
         request_timeout_s: 900    # 可选，单次调用最长多久，包括构造环境
         require_same_code: true   # 可选，拒绝代码不同的服务

在命令行上，同样的开关写作 ``++env.train.remote.address=192.168.3.2:18991``；``env.eval.remote`` 对评测起同样的作用。上面的 placement 把 env worker 放在一个忽略硬件的节点组上：仿真器在服务端时，env worker 只负责转发动作和观测，不需要加速卡。

远程环境藏在代理后面，并且每个 chunk 只返回最后一步，所以 ``validate_cfg`` 会拒绝需要更多信息的功能：要读取每一步的 ``video_cfg.save_video`` 和 ``data_collection.enabled``，要直接调用仿真器对象的 ``enable_offload``，外部奖励模型（``reward.use_reward_model``），以及 train split 上的在线 LeRobot 采集。同时设置了 ``remote`` 和 ``remote_render`` 的 split 也会被拒绝。

网络上传的是什么
----------------

每个代理在环境构造时建立连接，并在环境的整个生命周期里保持：一条用于请求的控制连接，加上 ``streams`` 条数据连接。所以服务如果在训练中途重启，训练会在下一次调用时失败。

请求里是动作，一个 chunk 只有几 KB。chunk step 的回复里是 chunk 末尾的观测和 info，以及 chunk 内每一步的奖励、termination 和 truncation。chunk 前面几步的观测留在服务端：只要上面那些功能都关着，EnvWorker 只读最后一步。张量回到客户端时已经在 CPU 上，EnvWorker 本来也要把它们送到 CPU。

最后一步的观测依然不小：40 个 ManiSkill 环境每个 chunk 要回传 35 MiB 的 640x480 相机图像。因此超过 1 MiB 的回复会按数据连接数切成连续的几段并行发送，这在限制单条 TCP 连接带宽的网络上很关键。在我们测过的两台机器之间，单条连接约 0.3 Gbit/s，一个 40 环境的 chunk 走 16 条连接约 23 ms 到达（含两端编解码），而仿真本身要 2.4 s。

代理会为每次 step 调用计时，EnvWorker 把累计值和它的其他计时一起记录：总往返 ``time/env/env/remote_round_trip``，其中服务端的部分 ``time/env/env/remote_server``，以及两者之差 ``time/env/env/remote_transfer``，也就是网络加上两端的编解码。只要 ``remote_transfer`` 远小于 ``remote_server``，限制训练速度的就不是网络。评测 rollout 记录同样的计时，键名带 ``eval_`` 前缀，例如 ``time/env/env/eval_remote_round_trip``。

服务端抛出的异常会以 ``RemoteError`` 的形式回到客户端，消息里写明原始异常，并附带服务端的 traceback；会话本身继续服务。

保证训练可复现
--------------

仿真器在构造和 reset 环境时，会从所在进程的随机数生成器取数。本地环境与它的 EnvWorker 共用这些生成器，而 EnvWorker 在构造任何环境之前会用 ``actor.seed`` 给它们播种。远程环境运行在单独的进程里，所以服务端用代理从 EnvWorker 的 torch 生成器里取出的一个数给这个进程播种；如果 EnvWorker 因 ``runner.deterministic`` 打开了确定性算法，服务端也会同样打开。

因此，设了种子的训练可以重复，每个会话也各自取一条独立的随机数流。这些随机数流和本地环境会取到的并不相同，所以远程训练与本地训练在统计上一致，而不是逐步一致。在同一张 GPU 上按同样的方式播种时，一个 ManiSkill 环境经服务端返回的观测、奖励和回合统计与在本进程里运行时逐位一致，跨过回合结束和随后的自动 reset 也是如此。

GPU 机器该怎么配
----------------

一张 GPU 能提供多少仿真，取决于一个进程里一起步进的环境有多少，而不是有多少个进程共用这张 GPU。在 RTX 4090 上，一个 ManiSkill 进程在 40、80、120 个环境时分别达到 60、101、147 env-step/s，RTX 4090 D 和 L20 与它相差不超过几个百分点。两个各有 40 个环境的进程共用一张 GPU 时会轮流执行：每个进程的速度只有单独运行时的一半，加起来也不比一个进程快。24 GB 的卡在单进程 160 个环境时显存不够，而 46 GB 的 L20 跑 160 个环境达到 166 env-step/s。

由此得出两条规则。每个会话独占一张 GPU，因为第二个会话挤进来，拖慢第一个会话的程度和它自己得到的一样多。GPU 还有余量时，增加每个 env worker 的环境数，而不是增加会话数。每个 env worker 在每个远程运行的 split 上，每个流水线阶段（``rollout.pipeline_stage_num``）开一个会话；每个会话分到该 split 的 ``total_num_envs`` 除以该 split 的会话数。

NVIDIA MPS 可以解除第一条限制。在 MPS 控制守护进程下，多个进程的 kernel 同时在 GPU 上执行，而不是轮流执行：在 L20 上，两个各有 40 个环境的会话合计达到 102 env-step/s，三个达到 132，和一个进程跑同样多环境时差不多；在 RTX 4090 上，两个会话合计 112。这时限制变成显存。一个 40 环境的 ManiSkill 会话无论开不开 MPS 都占约 8 GB，所以 24 GB 的卡能放两个。要这样服务会话，在 GPU 机器上启动一个控制守护进程，并让 env 服务使用同一个 pipe 目录：

.. code-block:: bash

   export CUDA_MPS_PIPE_DIRECTORY=/tmp/rlinf-mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/rlinf-mps/log
   mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
   CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d
   python -m rlinf.envs.remote.env_server --port 18991 --gpus 0

只有设置了同一个 ``CUDA_MPS_PIPE_DIRECTORY`` 的进程才会接入这个守护进程，所以同一台机器上的其他用户不受影响。env 服务退出后，用 ``echo quit | nvidia-cuda-mps-control`` 停掉守护进程。

.. warning::

   MPS 守护进程会给它的客户端重新编号 GPU。用 ``CUDA_VISIBLE_DEVICES=2,3`` 启动时，客户端看到的是 0 和 1，要用 GPU 2 的会话会找不到 CUDA 设备。启动守护进程时不要设置 ``CUDA_VISIBLE_DEVICES``，或者设置成不改变编号的值，例如 ``--gpus 0,1`` 对应 ``CUDA_VISIBLE_DEVICES=0,1``。在 GeForce 卡上，守护进程还必须加载宿主驱动的 ``libcuda``：如果容器镜像把 CUDA forward compatibility 的 ``libcuda`` 放在库路径最前面，MPS server 会以错误 804 启动失败。

以上速度是在没有 policy、用随机动作测得的；换成 π₀.₅ 来出动作时，同样 40 个环境在服务端的步进速度约快 1.4 倍。
