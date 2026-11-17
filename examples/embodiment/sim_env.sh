# Shared simulator environment for embodied training and evaluation.
# Source this file and call setup_sim_env after REPO_PATH is set.
# Derived paths are exported only when the corresponding base path is set.
# ROBOTWIN_PATH and DREAMZERO_PATH are added to PYTHONPATH only when set.

setup_sim_env() {
    # Hosts without a display GPU have no EGL, and MuJoCo then fails at import
    # rather than falling back. Ascend nodes are the case in the tree today, so
    # default them to software rendering; an explicit MUJOCO_GL still wins.
    local default_gl="egl"
    if compgen -G "/dev/davinci[0-9]*" > /dev/null \
        && [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
        default_gl="osmesa"
    fi
    export MUJOCO_GL="${MUJOCO_GL:-${default_gl}}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${default_gl}}"

    export OMNIGIBSON_NO_OMNI_LOGS="${OMNIGIBSON_NO_OMNI_LOGS:-1}"
    export OMNIGIBSON_DEBUG="${OMNIGIBSON_DEBUG:-0}"
    if [ -n "${OMNIGIBSON_DATA_PATH:-}" ]; then
        export OMNIGIBSON_DATASET_PATH="${OMNIGIBSON_DATASET_PATH:-${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/}"
        export OMNIGIBSON_KEY_PATH="${OMNIGIBSON_KEY_PATH:-${OMNIGIBSON_DATA_PATH}/omnigibson.key}"
        export OMNIGIBSON_ASSET_PATH="${OMNIGIBSON_ASSET_PATH:-${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/}"
    fi
    export OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}"

    if [ -n "${ISAAC_PATH:-}" ]; then
        export EXP_PATH="${EXP_PATH:-${ISAAC_PATH}/apps}"
        export CARB_APP_PATH="${CARB_APP_PATH:-${ISAAC_PATH}/kit}"
    fi

    if [ -n "${ROBOTWIN_PATH:-}" ]; then
        export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
    if [ -n "${DREAMZERO_PATH:-}" ]; then
        export PYTHONPATH="${DREAMZERO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
    fi

    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}
