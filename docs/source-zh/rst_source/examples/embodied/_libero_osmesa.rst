在 AMD、华为昇腾和摩尔线程上运行 LIBERO 时，使用 CPU 渲染。在已激活的模型环境中设置以下变量，再启动训练：

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa
   export ROBOT_PLATFORM=LIBERO

``run_embodiment.sh`` 会保留这些变量。安装脚本通过 ``requirements/sys_deps.sh`` 安装所需的 ``libosmesa6`` 系统库。

.. warning::

   直接启动 Python 时也需要设置这两个渲染变量。软件渲染会占用 CPU 资源，增加 rollout 并行度前，应按主机能力调整环境数量。

有两个环境开关可以减少软件渲染实际出的帧数，而不改变训练看到的 observation。``skip_intermediate_renders`` 只渲染每个 action chunk 真正消费的那一帧，不再渲染 chunk 内部的每个仿真步。``skip_reset_renders`` 对 reset 做同样的事：LIBERO 在每次 reset 之后会用 ``num_steps_wait`` 步让场景静置，而 policy 只收到最后一帧，因此相机在那一步之前保持关闭。这两个 key 都没有写在 LIBERO 环境的默认配置里，需要用追加语法：

.. code-block:: bash

   +env.train.skip_intermediate_renders=true +env.train.skip_reset_renders=true

``skip_reset_renders`` 能省多少，取决于一次训练的 reset 发生在哪里。``env.train.auto_reset: False`` 是 LIBERO 的默认值，此时每个 rollout epoch 开头都会把所有环境整体 reset 一次，这次 reset 本身就是迭代里一段可测量的耗时。``auto_reset: True`` 时，reset 则分散在 episode 结束的时刻，发生在步进过程中。
