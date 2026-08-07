Run State Contract
==================

Read a run's status, liveness, phase, progress, and latest checkpoint from disk —
without attaching to the driver or parsing stdout.

RLinf's existing loggers (TensorBoard, wandb, SwanLab) are **data plane**: time
series of scalars. This page specifies the **control plane**: the one-per-run
facts you need to answer "is my job still alive, and how far along is it?"

.. note::

   The control plane is written by the training driver as of ``schema_version: 2``.
   Nothing here replaces the metric backends — see :doc:`Logging <../logger>`
   for those.

Where it lives
--------------

.. code-block:: text

   <runner.logger.log_path>/_rlinf/
   ├── runs/<run_id>/
   │   ├── manifest.json        # invariants: run_id, task_type, git sha, cmdline, placement
   │   ├── run.json             # current snapshot, replaced atomically
   │   ├── events.jsonl         # append-only lifecycle and phase events
   │   ├── heartbeat            # tiny file; mtime is the fallback liveness signal
   │   ├── checkpoints.jsonl    # one line appended after each completed save
   │   └── media.rank<k>.jsonl  # video index, sharded per writer
   └── latest -> runs/<run_id>  # symlink to the most recent launch

Files, not a database: zero dependencies, readable across virtualenvs, and still
readable after a crash. SQLite is deliberately avoided because its locking is
unreliable on NFS.

Reading ``run.json``
--------------------

``docs/schemas/run.v2.schema.json`` is the authoritative definition. The fields
that matter most:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - What it tells you
   * - ``state``
     - ``pending`` · ``running`` · ``finished`` · ``failed`` · ``stopped``.
       A write-side fact only.
   * - ``phase``
     - The driver's innermost active stage (``rollout``, ``train``, ``eval``, …),
       derived from the ``ScopedTimer`` scope.
   * - ``components``
     - For async runners, which of ``env`` / ``rollout`` / ``actor`` are active.
       A single ``phase`` cannot express three concurrent components.
   * - ``heartbeat_at``
     - Last tick of the background heartbeat thread. Proves the **process** is
       alive — nothing more.
   * - ``last_progress_at``
     - Last time ``step`` advanced. Proves the **training loop** is alive.
   * - ``last_metric_at``
     - Last time a metric reached a backend.
   * - ``progress``
     - ``step`` / ``max_steps`` / ``epoch``, plus ``step_semantics``.
   * - ``timing``
     - ``elapsed_s``, ``step_time_p50``, ``eta_s``, and ``eta_confidence``.
   * - ``latest_checkpoint``
     - Mirrors the last line of ``checkpoints.jsonl``.
   * - ``exit``
     - Non-null only for ``failed`` / ``stopped``: reason plus traceback tail.

Deriving liveness
-----------------

``state`` has no ``stalled`` value, and that is deliberate: **a dead writer
cannot record its own death.** If the driver is ``kill -9``'d, ``run.json`` is
frozen mid-``running`` forever. Liveness is therefore a *reader-side* judgement
over the three timestamps:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Verdict
     - Condition
   * - ``unreachable``
     - ``heartbeat_at`` is older than ``k × heartbeat_interval_s``. The writer
       is no longer reporting.
   * - ``degraded``
     - Heartbeat is fresh but ``last_progress_at`` is stale. The process lives
       while the training loop does not — an NCCL hang or a stuck environment.
   * - ``degraded``
     - Heartbeat and progress are fresh but ``last_metric_at`` is stale. The
       metric path broke.
   * - ``healthy``
     - Everything fresh, or the run reached a terminal state.

Three timestamps rather than one heartbeat is what separates "the process died"
from "the process is fine and training is wedged". The second failure is the
common one in distributed RL, and a single heartbeat reports it as healthy.

.. tip::

   Terminal runs are silent by design. Treat ``finished`` / ``failed`` /
   ``stopped`` as healthy instead of letting the silence age into
   ``unreachable``.

Comparing steps across runs
---------------------------

``progress.step_semantics`` declares what one step means, because runners do not
agree:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Value
     - Meaning
   * - ``rl_iteration``
     - One full RL iteration (embodied, SFT).
   * - ``minibatch``
     - One minibatch. Reasoning logs at minibatch granularity, so its x-axis is
       denser than an embodied run's by a factor of ``n_minibatches``.
   * - ``optimizer_step``
     - One optimizer step.

This field **labels** existing behaviour; it does not change any runner's step
accounting. Use it to annotate axes and to avoid comparing runs that are not
comparable.

Checkpoint visibility
---------------------

A line is appended to ``checkpoints.jsonl`` only **after** a save completes.
Readers that trust the file therefore never observe a half-written checkpoint,
so no explicit ``WRITING`` / ``READY`` protocol is needed.

Each line carries ``step``, ``path``, ``saved_at``, ``size_bytes``,
``duration_s``, ``is_best``, the metrics at that step, and the structured fields
``resume_dir`` / ``entry_script`` / ``config_name``. Build a resume command from
those fields rather than storing a pre-baked shell string, which goes stale.

Manual verification
-------------------

Check the contract itself — the fixtures, the schema, and the liveness
derivation — without launching a run:

.. code-block:: bash

   pytest tests/unit_tests/test_run_state_contract.py -v

Then, against a live run:

.. code-block:: bash

   # 1. Watch a short run advance.
   watch -n1 cat <log_path>/_rlinf/latest/run.json

   # 2. Kill the driver, then confirm a reader reports 'unreachable':
   #    heartbeat_at stops advancing while state stays "running".
   kill -9 <driver_pid>

   # 3. Confirm a failure records its reason.
   #    state == "failed" and exit.reason is non-empty.
   python -c "import json;print(json.load(open('<log_path>/_rlinf/latest/run.json'))['exit'])"

Validate any snapshot against the schema:

.. code-block:: bash

   python -c "
   import json, jsonschema
   schema = json.load(open('docs/schemas/run.v2.schema.json'))
   jsonschema.validate(json.load(open('<path>/run.json')), schema)
   print('valid')"

Schema versioning
-----------------

``schema_version`` is an integer that increments on any breaking change to the
layout or to ``run.json``. Version 2 added the three-timestamp liveness model,
``components``, structured resume fields, and ``step_semantics``.

The training side validates against the committed schema and fixtures in
``tests/fixtures/run_state/``. External readers should vendor the frozen schema
and enforce byte-for-byte parity in their integration tests.
