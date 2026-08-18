RL with the SO101 Arm (Real2Sim)
================================

The `SO101 <https://github.com/TheRobotStudio/SO-ARM100>`__ (SO-ARM101) is a low-cost
6-DoF open-source arm. This example rebuilds a **real** SO101 pick-and-place workcell
inside ManiSkill from measured geometry and a recorded `LeRobot
<https://github.com/huggingface/lerobot>`__ dataset, then RL-fine-tunes an OpenPI
π₀.₅ policy on it.

Unlike the other ManiSkill recipes, this one is **joint-space**: the arm is driven with
absolute joint positions (``pd_joint_pos``), and the 6-dim policy output maps directly
onto ``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]``.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ (OpenPI)

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      Pick red cube → place in tray → return home

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 4–8 GPUs

| **You'll do:** install deps → download the ManiSkill assets and an SO101 π₀.₅ SFT checkpoint → probe → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` with ``--model openpi --env maniskill_libero``.

Task
~~~~

``SO101GrabRedCube-v1`` reproduces the real workcell: a 2.9 cm red cube on a
light-brown board, a blue cube as a visual distractor, and an open tray next to the
board. Success requires **both** conditions, matching the real demonstrations:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Condition
     - Definition
   * - Cube in the tray
     - Red cube released inside the tray interior (not merely held above it)
   * - Arm returned home
     - Mean deviation of the 5 arm joints from the measured home pose < 0.08 rad

The home pose and its tolerance come from the recorded real episodes rather than being
chosen by hand.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Shape / meaning
   * - ``main_images``
     - ``[B, 480, 640, 3]`` front camera, intrinsics measured from the real frames
   * - ``wrist_images``
     - ``[B, 480, 640, 3]`` wrist camera mounted on ``Fixed_Jaw``
   * - ``states``
     - ``[B, 6]`` joint positions, in LeRobot normalized motor units
   * - actions
     - ``[B, chunk, 6]`` absolute joint targets, in LeRobot normalized motor units

Units: the checkpoint speaks LeRobot **normalized motor units** (arm ``[-100, 100]``,
gripper ``[0, 100]``) while ManiSkill speaks **radians**. The conversion lives in
``rlinf/envs/maniskill/so101_calib.py`` and is derived from the follower servo
calibration (Feetech, 4096 ticks/revolution). It is applied at the env boundary:
actions in ``action_utils.prepare_actions_for_maniskill``, observations in
``ManiskillEnv._wrap_obs`` (enabled by ``so101_state_norm: True``).

.. note::

   The gripper deliberately does **not** use the arm's tick→radian conversion. The real
   parallel gripper and the simulated revolute jaw have different linkage geometry;
   reusing the arm formula leaves the jaws at roughly an 8 cm gap against a 2.9 cm
   cube, so the policy can never close on it. The gripper instead uses a separately
   measured map, giving a 1.3 cm closed / 11.2 cm open span.

Robot model
~~~~~~~~~~~

The task uses a widened-limit variant of ManiSkill's built-in SO100. The stock URDF
under-models the real hardware — it clamps ``shoulder_lift`` and ``elbow_flex`` to
±1.5708 rad, while the real follower reaches +2.48 and −2.38 — and the simulated arm
cannot reach the tray at all until the limits are widened.

The widened URDF is **derived from the installed ManiSkill one at import time**
(``rlinf/envs/maniskill/so101_agent.py``) rather than shipped as an asset. Limits come
from ``so101_calib.JOINT_LIMITS_{LOW,HIGH}`` — the same table the action conversion
clips against — so the two cannot drift apart, and a fresh checkout needs no
out-of-band download.

Installation
------------

The SO101 task needs no dependencies beyond the standard ManiSkill + OpenPI
environment:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env maniskill_libero

Download the ManiSkill assets as described in
:doc:`ManiSkill <maniskill>`, then point the configs at an SO101 π₀.₅ SFT
checkpoint. The RL starting point must be a policy that already has non-zero success in
simulation — PPO amplifies success, it does not discover it.

Run It
------

.. code-block:: bash

   cd examples/embodiment
   bash run_embodiment.sh so101_ppo_openpi_pi05

Before committing to a long run, **probe under the rollout distribution** with a
frozen policy — deterministic evaluation and noisy rollout can differ by more than
50 points, and it is the noisy number that PPO learns from:

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-path $PWD/examples/embodiment/config/ \
     --config-name so101_ppo_openpi_pi05 \
     runner.max_epochs=1 runner.val_check_interval=1 \
     actor.optim.lr=1e-9 actor.optim.value_lr=1e-9

Require ``env/success_once`` ≥ 5%. Historically, the runs that amplified started in the
5–15% band and the runs that never took off started at 0.5–1.0%. It is a necessary but
not sufficient condition.

.. warning::

   Do not use ``runner.only_eval=True`` as the probe. It switches where the model spec
   comes from and skips training-environment creation, so it exercises a different code
   path than training does.

Key configuration
~~~~~~~~~~~~~~~~~

Three settings depart from the default π-RL recipe. Each was A/B'd on this task, and
the default combination is known to collapse here (deterministic eval fell to ~0 by
step 9), so treat them as task requirements rather than tuning knobs:

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Setting
     - Value
     - Why
   * - ``actor.model.num_action_chunks``
     - ``10``
     - Equals the checkpoint's ``action_horizon``. Executing only 5 discards half of
       every prediction and re-plans twice as often; using 10 is worth about 11 points
       of deterministic eval on its own, so evaluation and real-robot deployment want
       it too.
   * - ``actor.model.openpi.noise_logvar_range``
     - ``[0.02, 0.04]``
     - The field ``flow_noise`` actually reads. ``noise_params`` only feeds
       ``flow_sde`` and has no effect under ``noise_method: flow_noise``.
   * - ``actor.global_batch_size``
     - ``4096``
     - Equal to the samples produced per epoch
       (``num_envs × max_episode_steps / num_action_chunks × rollout_epoch``), giving
       exactly **one update per epoch**. The default 12 updates collapses this task:
       from the same start, same noise and same chunk length, 12 updates went
       61.7% → 7.0% while 1 update climbed to 73.4%. This task takes 64 noisy
       decisions per episode against the reference benchmark's 16.

Spawn coverage is set by ``env.*.init_params.spawn_mode`` and ``spawn_frac``. The
default is the full board. A curriculum region can be selected with
``spawn_frac: [0.4294, 0.9115, 0.5142, 0.9817]``; scores from different spawn regions
are not comparable, so always report which one a number came from.

Evaluation
----------

.. code-block:: bash

   cd examples/embodiment
   bash run_embodiment.sh so101_eval_openpi_pi05

Watch ``eval/success_once`` (deterministic) and ``env/success_once`` (noisy rollout).
Report the two separately: a peak picked from a fixed evaluation set is optimistic
relative to fresh, never-used seeds — on this task the same checkpoint scored 73.4%
under the former and 57.8% under the latter.

Verifying the setup
-------------------

A simulator-level smoke test checks the parts that need no policy checkpoint — the
derived URDF, both cameras, the joint-space observation, unit-conversion agreement
between the NumPy and Torch paths, and the jaw geometry:

.. code-block:: bash

   python -m toolkits.so101_smoke

Unit tests for the conversion contract run without a simulator:

.. code-block:: bash

   pytest tests/unit_tests/test_so101_maniskill.py
