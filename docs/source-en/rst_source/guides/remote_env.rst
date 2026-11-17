Remote Simulators
=================

This guide shows how to run a simulator on a GPU machine while the env workers,
the policy and training stay on the training machines. It is for training
accelerators that cannot run the simulator themselves, such as Ascend NPU
hosts, and ManiSkill and LIBERO are the simulators it currently supports.

ManiSkill's GPU physics needs CUDA. Without it, ManiSkill falls back to CPU
physics, which runs one environment per process instead of hundreds per GPU. With a remote
simulator, EnvWorker builds a proxy where it would build the simulator. The
proxy opens a session on an env server, which builds the real environment on one
of its GPUs, and from then on every chunk step sends the actions across and
brings back what the environment returns at the end of the chunk. We'll start a
server, point a run at it, and then look at what crosses the network, how runs
stay reproducible, and how to size the GPU machine.

LIBERO has a lighter option. Its physics is cheap on a CPU and only its
rendering needs a GPU, so :doc:`remote_render` keeps the physics next to the
policy and sends only the drawing to a GPU machine. That mode moves less data
and keeps video recording and data collection available; run the whole LIBERO
environment remotely only when the training machines' CPUs cannot carry its
physics.

Start an Env Server
-------------------

The env server is a standalone process on the GPU machine. It does not join the
Ray cluster and needs no training accelerator. It needs the same RLinf code as the training
machines, HeteroPulse on both ends (``pip install heteropulse``), which carries
the sessions, and the simulator installed with its assets, the way a local run would:
ManiSkill with SAPIEN and ``MS_ASSET_DIR``, or LIBERO with its scene files. Its
NVIDIA driver must also run the CUDA version PyTorch was built for on that
machine. If you set the machine up without ``requirements/install.sh``, add the
system packages it installs as well. ManiSkill on NVIDIA needs ``libvulkan1``
and ``libegl1`` in particular: NVIDIA's Vulkan driver loads ``libEGL.so.1`` when
it starts, and without it SAPIEN cannot create its Vulkan instance.

.. code-block:: bash

   python -m rlinf.envs.remote.env_server --port 18991 --gpus 0,1

What this does:

1. Listens on port 18991. For every proxy that connects, it starts one process
   on the GPU in ``--gpus`` that holds the fewest sessions of the proxy's split,
   then the fewest sessions overall, so the training sessions spread over the
   GPUs before two share one. The process sets ``CUDA_VISIBLE_DEVICES`` to that
   GPU and, for MuJoCo-based simulators, ``MUJOCO_GL=egl`` with the EGL device
   that matches it. ``--max-sessions-per-gpu`` refuses new sessions once every
   GPU holds that many.
2. Builds the proxy's environment in that process with RLinf's own env classes,
   from the split config and arguments EnvWorker passed to the proxy, and serves
   it until the proxy disconnects. Then the process exits and frees the GPU. A
   training machine that vanishes without closing its connections is noticed
   within about three minutes.
3. Logs, at startup, a hash of the Python code under ``rlinf/envs``. A proxy
   whose hash differs is refused when it connects, unless its split sets
   ``remote.require_same_code: false``, because the two ends would otherwise
   disagree about what the environment does.

Each session logs which GPU it landed on and how long its environment took to
build.

.. warning::

   The env server has no authentication, and a session builds whatever
   environment its proxy asks for. Bind it to a network that only your training
   machines can reach, for example with ``--host`` set to a private address.

Point a Run at the Server
-------------------------

A run opts in per environment split. ``env_type`` keeps naming the real
simulator, so everything that depends on it, such as action formatting, stays as
it is; ``remote.address`` names the server:

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
         streams: 16               # optional, data connections per environment
         connect_timeout_s: 60     # optional, how long to wait for each connection
         request_timeout_s: 900    # optional, longest single call, building included
         require_same_code: true   # optional, refuse a server running other code

From the command line, the same switch is
``++env.train.remote.address=192.168.3.2:18991``, and ``env.eval.remote`` does the
same for evaluation. The placement puts the env workers on a node group that
ignores hardware: with the simulator on the server, an env worker only relays
actions and observations, and needs no accelerator.

A remote environment sits behind a proxy and returns only the last step of each
chunk, so ``validate_cfg`` refuses the features that need more of it:
``video_cfg.save_video`` and ``data_collection.enabled``, which read every step,
``enable_offload``, which calls into the simulator object, an external reward
model (``reward.use_reward_model``), and online LeRobot collection on the train
split. It also refuses a split that sets both ``remote`` and
``remote_render``.

What Crosses the Network
------------------------

Each proxy opens its connections when its environment is built and keeps them
for its lifetime: one control connection for requests and ``streams`` data
connections, so a run fails on its next call if the server restarts under it.

Requests carry the actions, a few kilobytes for a chunk. A reply to a chunk step
carries the observation and info at the end of the chunk, and the rewards,
terminations and truncations of every step in it. The observations of the
earlier steps stay on the server, because EnvWorker reads only the last one
whenever the features above are off. Tensors come back on the CPU, where
EnvWorker sends them anyway.

The last observation is still large: 40 ManiSkill environments return 35 MiB of
640x480 camera images per chunk. Replies above 1 MiB are therefore split into
one contiguous piece per data connection and sent in parallel, which matters on
networks that cap the bandwidth of each TCP connection. Between the hosts we
measured, where one connection carried about 0.3 Gbit/s, a 40-environment chunk
took about 23 ms to arrive over 16 connections, encoding included, against
2.4 s of simulation.

The proxy times its step calls and EnvWorker logs the totals with its other timers:
``time/env/env/remote_round_trip``, the server's share
``time/env/env/remote_server``, and their difference
``time/env/env/remote_transfer``, which is the network and the encoding at both
ends. When ``remote_transfer`` is small against ``remote_server``, the network is
not what limits the run. Evaluation rollouts log the same timers with an
``eval_`` prefix, such as ``time/env/env/eval_remote_round_trip``.

An exception raised on the server comes back as a ``RemoteError`` whose message
names the original exception and which carries the server's traceback, and the
session keeps serving.

Keep Runs Reproducible
----------------------

Simulators draw from their process's random number generators when they build
and reset environments. A local environment shares those of its EnvWorker,
which seeds them from ``actor.seed`` before building any environment. A served
environment runs in a process of its own, so the server seeds that process
with a number the proxy draws from the EnvWorker's torch generator, and turns
on deterministic algorithms when ``runner.deterministic`` turned them on in the
EnvWorker.

A seeded run therefore repeats, and every session draws a stream of its own.
The streams are not the ones a local environment would draw, so a remote run
matches a local run statistically, not step for step. Seeded the same way on
the same GPU, a ManiSkill environment returned the same observations,
rewards and episode statistics through the server as in-process, bit for bit,
across an episode boundary and its automatic reset.

Size the Simulator Machine
--------------------------

How much simulation a GPU delivers depends on how many environments one
process steps together, not on how many processes share the GPU. On an RTX 4090,
one ManiSkill process stepped 60, 101 and 147 env-steps/s with 40, 80 and 120
environments, and an RTX 4090 D and an L20 matched it within a few percent.
Two processes of 40 environments on one GPU took turns: each stepped half as
fast as alone, and together no faster than one. A 24 GB card ran out of
memory at 160 environments in one process, which the 46 GB L20 stepped at 166
env-steps/s.

Two rules follow. Give each session its own GPU, since a second session on a GPU
slows the first by as much as it gains. And when a GPU has headroom, raise the
number of environments per env worker rather than adding sessions. Each env
worker opens one session per pipeline stage (``rollout.pipeline_stage_num``) on
every split that runs remotely, and each session holds
``total_num_envs`` divided by the number of sessions on that split.

NVIDIA MPS lifts the first rule. Under an MPS control daemon, the kernels of
several processes run on the GPU at the same time instead of in turns: on an L20,
two sessions of 40 environments stepped 102 env-steps/s together and three
stepped 132, about what one process with as many environments steps, and on an
RTX 4090 two stepped 112. Memory then sets the limit. A 40-environment ManiSkill
session holds about 8 GB with or without MPS, so a 24 GB card fits two of them.
To serve sessions this way, start a control daemon on the GPU machine and give
the env server the same pipe directory:

.. code-block:: bash

   export CUDA_MPS_PIPE_DIRECTORY=/tmp/rlinf-mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/rlinf-mps/log
   mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
   CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d
   python -m rlinf.envs.remote.env_server --port 18991 --gpus 0

Only processes that set the same ``CUDA_MPS_PIPE_DIRECTORY`` join this daemon, so
other users of the machine are unaffected. Stop it with
``echo quit | nvidia-cuda-mps-control`` after the env server exits.

.. warning::

   An MPS daemon renumbers the GPUs its clients see. Started with
   ``CUDA_VISIBLE_DEVICES=2,3``, it shows its clients those GPUs as 0 and 1, and
   a session that asks for GPU 2 finds no CUDA device. Start the daemon with no
   ``CUDA_VISIBLE_DEVICES``, or with one that keeps the numbering, such as
   ``CUDA_VISIBLE_DEVICES=0,1`` for ``--gpus 0,1``.
   On GeForce cards, the daemon also needs the host driver's ``libcuda``: a CUDA
   forward-compatibility ``libcuda`` that a container image puts first on the
   library path makes its server fail with error 804.

These rates were measured with random actions, without a policy; with π₀.₅
acting, the same 40 environments stepped about 1.4 times faster on the server.
