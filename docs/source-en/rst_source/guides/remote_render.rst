Remote Rendering
================

This guide shows how to render a MuJoCo-based simulator on a GPU machine while
its physics keeps running in the env workers next to the policy. It is for
clusters where the training machines have no graphics stack, such as Ascend
NPU hosts, and LIBERO is the simulator it currently supports.

Without a GPU renderer, every camera frame goes through OSMesa on the CPU. On a
LIBERO task, a pair of 256x256 camera frames costs about 100 ms and close to two
CPU cores per simulator process, which is several times what the physics of an
action chunk costs. With remote rendering, the same simulator process keeps its
physics, its observations and its resets, and hands only the drawing to a render
server: a few milliseconds per frame on any NVIDIA GPU, with images that are
identical to rendering locally on that GPU. We'll start the server, point a run
at it, and then look at what crosses the network and how to size the render
machine.

Start a Render Server
---------------------

Remote rendering runs through HeteroPulse, a package RLinf uses for remote
simulation. Install it on the training machines, in RLinf's environment:

.. code-block:: bash

   pip install heteropulse

The render server is a standalone process on the GPU machine. It needs no RLinf
installation: only HeteroPulse with the MuJoCo version the training machines
run, and an NVIDIA driver with EGL. Nothing simulator-specific goes on the
machine: no LIBERO assets, no robosuite scene files.

.. code-block:: bash

   pip install "heteropulse[render]" mujoco==3.3.7   # the training machines' version
   heteropulse-render --port 18990 --gpus 0,1 --procs-per-gpu 4

What this does:

1. Starts four worker processes on each of CUDA devices 0 and 1. Each worker
   opens one EGL context on its GPU, on the EGL device that matches the CUDA
   id.
2. Listens on port 18990 and hands every incoming connection to the worker that
   currently holds the fewest.
3. Keeps the compiled models it receives under ``--cache-dir`` (by default
   ``$TMPDIR/heteropulse_render_cache``), so a worker that sees a model another
   worker already received loads it from disk instead of asking for it again.

Each worker logs once a minute how many frames it drew and how long each took on
its GL thread, which is the number to watch when sizing the server.

.. warning::

   The render server has no authentication. Bind it to a network that only your
   training machines can reach, for example with ``--host`` set to a private
   address.

Point a Run at the Server
-------------------------

A run opts in per environment split. Set ``remote_render.address`` on the split
whose simulator should render remotely:

.. code-block:: yaml

   env:
     train:
       env_type: libero
       remote_render:
         address: "192.168.3.2:18990"
         connect_timeout_s: 30     # optional, how long to wait for the server
         request_timeout_s: 300    # optional, longest single request
         quickack: false           # optional, true asks for immediate ACKs (Linux)

From the command line, the same switch is
``++env.train.remote_render.address=192.168.3.2:18990``, and
``++env.eval.remote_render.address=...`` does the same for evaluation. Leaving
``address`` unset, or ``null``, keeps local rendering, and
``validate_cfg`` rejects the block on simulators other than LIBERO.

Nothing else in the config changes. ``MUJOCO_GL`` may stay ``osmesa`` on the
training machines: with remote rendering, the simulator processes never create
a GL context, so the backend is only imported, never used. The switches that
control which frames are drawn, ``skip_intermediate_renders`` and
``skip_reset_renders``, keep their meaning and are the first lever to pull,
because a frame that is not drawn costs nothing anywhere.

What Crosses the Network
------------------------

Each LIBERO simulator process opens one TCP connection when it builds its first
environment and keeps it for its lifetime. If the connection breaks, for example
because the render server restarted, the process reconnects on its next frame
and uploads its model again. Two kinds of messages travel on the connection.

The first time the server sees a scene, it needs the scene's meshes and
textures. Rather than resolving asset files on the server, the client uploads
its compiled model, about 260 MB for a LIBERO-Spatial scene. Models are keyed by
a *visual signature*, a hash of everything a MuJoCo render context is built
from: meshes, textures, materials, height fields, plane geoms and the visual
options. Body poses are not part of it, so all ten LIBERO-Spatial tasks, and
every rebuild of a task at reset, share one signature, one upload and one
context per worker.

Every frame after that is a request carrying the scene MuJoCo's
``mjv_updateScene`` built next to the physics: the list of geoms with their
poses and colors, the lights and the camera, about 20 KB for a LIBERO camera
view. The server draws it with ``mjr_render`` and replies with the pixels, 196 KB
for a 256x256 RGB frame. Because the scene is built where the physics runs, the
server never steps or even holds the simulation state, and anything that
changes a frame between steps, such as robosuite toggling site visibility,
arrives with the frame.

.. note::

   Frames match local rendering bit for bit when both run on the same kind of
   GPU and driver, which the unit tests check against a local context. They do
   not match OSMesa frames from a CPU renderer, which rasterizes differently, so
   compare learning curves, not pixels, against a CPU-rendered run.

Rendering is bound by latency rather than bandwidth: a LIBERO simulator waits
for every frame, and asks for one or two after each physics step of a few tens
of milliseconds. After such a pause Linux delays the client's ACKs, and a 196 KB
reply then stalls waiting for them. Between our training and render machines it
took 1.3 to 1.4 ms when requests followed each other closely and 2.4 to 4.6 ms
after a pause of 20 to 40 ms. With ``remote_render.quickack: true`` the client
sets ``TCP_QUICKACK`` before every read, which kept the reply at 1.3 to 1.5 ms
after any pause. On 16 Ascend 950PR cards rendering on one L20, it cut a π₀.₅
LIBERO step from 38.7 s to 38.0 s. It is off by default.

Two things are outside what the scene carries. Models with skinned meshes or
flexible bodies are rejected when their environment is built, and a simulator
that edits textures at runtime fails on its first texture upload: the server
draws the textures the model had when it was uploaded.

Size the Render Machine
-----------------------

Drawing a LIBERO frame takes 0.15 to 0.3 ms of GPU time on an RTX 4090. What
limits a worker is CPU: copying a scene into the worker's MuJoCo structures and
answering its connections costs about 1.3 ms per frame on one core when lightly
loaded, and about 2.4 ms per frame once 30 or more connections share the
worker. One worker therefore draws about 400 LIBERO frames a second, and
throughput grows with workers until the GPU or the machine's cores run out. In
a benchmark without the network, one RTX 4090, RTX 4090 D or L20 drew 6,500 to
7,100 LIBERO frames a second before its GPU saturated, and an L4 on a 12-core
host ran out of cores at about 4,300.

The frames you need follow from the run. A LIBERO step with two cameras draws
two frames, so one simulator stepping at 100 env-steps/s needs 200 frames/s, and
``skip_intermediate_renders`` with five-action chunks cuts that by five. Divide
the frames a run needs by about 400 for the number of workers, and give each
worker a core.
