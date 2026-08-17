# STATUS: TOOL — 通用工具，与具体阶段无关。 渲染当前场景，看资产/位姿对不对
import os
"""Wrist + front view AFTER the raise-arm prefix -- i.e. what the policy sees
during the task, not at the folded home pose."""
import numpy as np, imageio.v2 as imageio, gymnasium as gym
from rlinf.envs.maniskill import import_all_tasks
OUT=os.environ.get("SCRATCH", "/tmp/so101_runs")
tn=lambda x: x.cpu().numpy() if hasattr(x,"cpu") else np.asarray(x)
import_all_tasks()
env=gym.make("SO101GrabRedCube-v1",num_envs=1,obs_mode="rgb",control_mode="pd_joint_pos",sim_backend="gpu")
obs,_=env.reset(seed=44)
q0=env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
READY=np.array([0.0,0.0,0.0,np.pi/2,np.pi/2,0.5],dtype=np.float32)
for i in range(1,25):
    obs,*_=env.step((q0+(READY-q0)*i/24).astype(np.float32))
s=obs["sensor_data"]
front=tn(s["3rd_view_camera"]["rgb"])[0]; wrist=tn(s["wrist_camera"]["rgb"])[0]
imageio.imwrite(f"{OUT}/so101_ready_pair.png", np.concatenate([front,wrist],axis=1))
print("red=",env.unwrapped.red_cube.pose.sp.p.tolist()," wrist rgb mean=",float(wrist.mean()))
env.close()
