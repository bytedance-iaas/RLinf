import traceback

def stage(name, fn):
    try:
        r = fn()
        print(f"[OK]   {name}: {r}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()

# 1) task registration
def t1():
    from rlinf.envs.maniskill import import_all_tasks
    import_all_tasks()
    from mani_skill.utils.registration import REGISTERED_ENVS
    assert "SO101PickCube-v1" in REGISTERED_ENVS, "SO101PickCube-v1 not registered"
    return "SO101PickCube-v1 registered"
stage("task registration", t1)

# 2) openpi pi05_so101 config entry present
def t2():
    from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
    assert "pi05_so101" in _CONFIGS_DICT, "pi05_so101 missing"
    c = _CONFIGS_DICT["pi05_so101"]
    return f"pi05_so101 present, data={type(c.data).__name__}, pi05={getattr(c.model,'pi05',None)}, horizon={getattr(c.model,'action_horizon',None)}"
stage("openpi pi05_so101 registry", t2)

# 3) policy transforms import + shapes
def t3():
    from rlinf.models.embodiment.openpi.policies import so101_policy as sp
    ex = sp.make_so101_example()
    return f"SO101Inputs/Outputs import ok, action_dim={sp.SO101_ACTION_DIM}, example_state={ex['observation/state'].shape}"
stage("so101 policy transforms", t3)

# 4) action passthrough branch
def t4():
    import numpy as np, torch
    from rlinf.envs.action_utils import prepare_actions_for_maniskill
    a = torch.zeros(4, 5, 6)
    out = prepare_actions_for_maniskill(a, num_action_chunks=5, action_dim=6, action_scale=1.0, policy="so100")
    assert out.shape == (4, 5, 6), out.shape
    return f"so100 passthrough ok, shape={tuple(out.shape)}"
stage("action passthrough (so100)", t4)

# 5) GPU env instantiation + obs contract (needs a GPU + SAPIEN rendering)
def t5():
    import gymnasium as gym
    from rlinf.envs.maniskill import import_all_tasks
    import_all_tasks()
    env = gym.make("SO101PickCube-v1", num_envs=2, obs_mode="rgb", control_mode="pd_joint_pos", sim_backend="gpu")
    obs, _ = env.reset()
    sd = obs["sensor_data"]
    qpos = env.unwrapped.agent.robot.get_qpos()
    third = sd["3rd_view_camera"]["rgb"].shape if "3rd_view_camera" in sd else "MISSING"
    wrist = sd["wrist_camera"]["rgb"].shape if "wrist_camera" in sd else "MISSING"
    instr = env.unwrapped.get_language_instruction()
    env.close()
    return f"qpos={tuple(qpos.shape)} 3rd_view={third} wrist={wrist} instr0={instr[0]!r}"
stage("GPU env reset + obs contract", t5)

print("SMOKE DONE")
