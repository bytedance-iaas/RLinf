Use CPU rendering for LIBERO on AMD, Huawei Ascend, and Moore Threads. Set both
variables in the active model environment before launching a run:

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa
   export ROBOT_PLATFORM=LIBERO

``run_embodiment.sh`` preserves these values. The installer includes the
``libosmesa6`` system library through ``requirements/sys_deps.sh``.

.. warning::

   Set both rendering variables when launching Python directly as well.
   Software rendering uses CPU resources; adjust environment counts to the
   host's capacity before scaling up rollouts.

Two environment switches cut how many frames software rendering has to produce,
without changing the observations a run sees. ``skip_intermediate_renders``
renders only the frame that each action chunk consumes, instead of every
simulator step inside the chunk. ``skip_reset_renders`` does the same for
resets: LIBERO settles the scene for ``num_steps_wait`` steps after every reset
and the policy only receives the last of those frames, so the camera stays off
until that step. Neither key is declared in the LIBERO environment defaults, so
append them:

.. code-block:: bash

   +env.train.skip_intermediate_renders=true +env.train.skip_reset_renders=true

What ``skip_reset_renders`` is worth depends on where a run's resets happen.
With ``env.train.auto_reset: False`` -- the LIBERO default -- every rollout
epoch begins with a full reset of all environments, and that reset is a
measurable block of the iteration on its own. With ``auto_reset: True`` the
resets instead happen as episodes end during stepping.
