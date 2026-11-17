远程渲染
========

本页介绍如何把基于 MuJoCo 的仿真器的渲染放到一台 GPU 机器上，同时让物理仿真继续在 policy 旁边的 env worker 里运行。它适用于训练机器没有图形栈的集群，例如昇腾 NPU 主机；目前支持的仿真器是 LIBERO。

没有 GPU 渲染器时，每一帧相机画面都要在 CPU 上经 OSMesa 绘制。在 LIBERO 任务上，一对 256x256 的相机画面大约要 100 ms，每个仿真进程还要占将近两个 CPU 核，是一个动作 chunk 物理计算开销的好几倍。改用远程渲染后，仿真进程依旧负责物理、观测和 reset，只把「画」这一步交给渲染服务：在任意一张 NVIDIA GPU 上每帧只要几毫秒，画出的图像和在那张 GPU 上本地渲染逐位一致。下面我们先启动服务，再让一次训练连上它，最后看看网络上传的是什么、渲染机器该怎么配。

启动渲染服务
------------

远程渲染通过 HeteroPulse 实现，这是 RLinf 做远程仿真用的一个独立包。训练机器上把它装进 RLinf 的环境：

.. code-block:: bash

   pip install heteropulse

渲染服务是 GPU 机器上的一个独立进程。它不需要安装 RLinf，只要装上 HeteroPulse、和训练机器相同版本的 MuJoCo，以及带 EGL 的 NVIDIA 驱动；LIBERO 资产、robosuite 场景文件这类仿真器相关的内容都不需要放到这台机器上。

.. code-block:: bash

   pip install "heteropulse[render]" mujoco==3.3.7   # 与训练机器的版本一致
   heteropulse-render --port 18990 --gpus 0,1 --procs-per-gpu 4

这条命令会：

1. 在 CUDA 设备 0 和 1 上各启动四个 worker 进程。每个 worker 在自己的 GPU 上打开一个 EGL context，使用与该 CUDA 编号对应的 EGL 设备。
2. 监听 18990 端口，把每个新连接交给当前连接数最少的 worker。
3. 把收到的编译后模型保存在 ``--cache-dir`` 下（默认是 ``$TMPDIR/heteropulse_render_cache``）。某个 worker 遇到其他 worker 已经收到过的模型时，直接从磁盘加载，不再向客户端索要。

每个 worker 每分钟打印一次日志，记录这段时间画了多少帧、每帧在 GL 线程上用了多久；为渲染机器定规格时就看这个数。

.. warning::

   渲染服务没有任何认证。请把它绑定在只有训练机器能访问的网络上，例如用 ``--host`` 指定一个内网地址。

让训练连上渲染服务
------------------

是否使用远程渲染按 env 的 split 单独设置。在需要远程渲染的 split 下设置 ``remote_render.address``：

.. code-block:: yaml

   env:
     train:
       env_type: libero
       remote_render:
         address: "192.168.3.2:18990"
         connect_timeout_s: 30     # 可选，等待服务接受连接的时长
         request_timeout_s: 300    # 可选，单个请求的最长时长
         quickack: false           # 可选，设为 true 时要求立即回 ACK（仅 Linux）

在命令行上，对应的写法是 ``++env.train.remote_render.address=192.168.3.2:18990``；评测用 ``++env.eval.remote_render.address=...``。不设置 ``address`` 或设为 ``null`` 时仍然本地渲染；在 LIBERO 以外的仿真器上设置这一项，``validate_cfg`` 会直接报错。

配置里的其他内容都不用改。训练机器上的 ``MUJOCO_GL`` 可以保持 ``osmesa``：远程渲染时仿真进程从不创建 GL context，后端只是被 import，并不会被使用。控制「哪些帧需要画」的两个开关 ``skip_intermediate_renders`` 和 ``skip_reset_renders`` 含义不变，而且应该最先考虑，因为不画的帧在哪里都不花时间。

网络上传的是什么
----------------

每个 LIBERO 仿真进程在构建第一个环境时建立一条 TCP 连接，并在整个生命周期内复用。连接断开时（例如渲染服务重启了），进程会在下一帧重新连接，并重新上传模型。连接上只有两类消息。

渲染服务第一次见到某个场景时，需要这个场景的网格和贴图。客户端不让服务端去解析资产文件，而是直接上传自己编译好的模型，一个 LIBERO-Spatial 场景大约 260 MB。模型按「可视签名」索引，签名是对 MuJoCo 渲染 context 所依赖的全部内容求的哈希：网格、贴图、材质、高度场、plane geom 和可视化选项。物体位姿不在其中，所以 LIBERO-Spatial 的十个任务、以及每次 reset 时的重建，都共用一个签名：只上传一次，每个 worker 上也只有一个 context。

此后的每一帧都是一个请求，携带 MuJoCo 的 ``mjv_updateScene`` 在物理仿真旁边构建好的场景：所有 geom 的位姿和颜色、灯光与相机。一个 LIBERO 相机视角大约 20 KB。服务端用 ``mjr_render`` 画出这一帧，回复像素，一帧 256x256 RGB 图像为 196 KB。场景是在物理所在的地方构建的，所以服务端从不推进、甚至不持有仿真状态；robosuite 切换 site 可见性这类在两步之间改变画面的操作，也会随这一帧一起送到服务端。

.. note::

   在 GPU 型号与驱动相同的前提下，远程渲染的画面与本地渲染逐位一致，单测会与本地 context 的结果逐位对照。它与 CPU 上 OSMesa 渲染的画面并不一致，两者的光栅化方式不同；和 CPU 渲染的训练对比时，请比较学习曲线，不要比较像素。

远程渲染受限于延迟而不是带宽：LIBERO 仿真器每一帧都要等回复，而且每次物理步进（几十毫秒）之后才要一两帧。这样停顿之后，Linux 会推迟客户端的 ACK，196 KB 的回复就卡在等 ACK 上。在我们的训练机与渲染机之间，请求连续发出时一次往返 1.3 到 1.4 ms，停顿 20 到 40 ms 之后变成 2.4 到 4.6 ms。设 ``remote_render.quickack: true`` 后，客户端每次读之前都会设置 ``TCP_QUICKACK``，无论停顿多久，回复都稳定在 1.3 到 1.5 ms。16 张昇腾 950PR 共用一张 L20 渲染时，π₀.₅ LIBERO 每步从 38.7 s 降到 38.0 s。默认关闭。

有两种情况超出了场景能携带的内容。带蒙皮网格或柔性体的模型会在构建环境时被拒绝；在运行时修改贴图的仿真器会在第一次上传贴图时报错，因为服务端画的是模型上传时的贴图。

为渲染机器定规格
----------------

在 RTX 4090 上画一帧 LIBERO 画面只用 0.15 到 0.3 ms 的 GPU 时间。限制 worker 的是 CPU：把场景拷进 worker 的 MuJoCo 结构体并处理各个连接，轻载时每帧约占一个核 1.3 ms，同一个 worker 上有 30 个以上连接时约 2.4 ms。因此一个 worker 每秒大约能画 400 帧 LIBERO 画面；在 GPU 或机器的核用完之前，吞吐随 worker 数增长。在不经过网络的基准测试里，一张 RTX 4090、RTX 4090 D 或 L20 在 GPU 饱和前每秒能画 6,500 到 7,100 帧 LIBERO 画面；12 核主机上的 L4 在约 4,300 帧时先用完了 CPU。

所需的帧数由训练本身决定。带两个相机的 LIBERO 每走一步要画两帧，一个以 100 env-steps/s 运行的仿真器需要 200 帧/s；打开 ``skip_intermediate_renders`` 并使用五个动作一个 chunk 时，这个数会降到五分之一。用训练所需的帧数除以 400 左右得到 worker 数，并为每个 worker 留一个核。
