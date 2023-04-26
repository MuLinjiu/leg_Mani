from envs.pmtg_task import QuadrupedGymEnv
import pybullet as bc
import numpy as np
import time

# _pybullet_client = bc.BulletClient()
env = QuadrupedGymEnv(time_step=0.001, action_repeat=20, obs_hist_len=5, render=True)
low =  env.action_space.low
high = env.action_space.high
obs = env.reset()

# print((low > obs).any(), (high < obs).any())
# print(obs - low)

pDes = np.array([0, -0.0838, -0.3, 0, 0.0838, -0.3, 0, -0.0838, -0.3, 0, 0.0838, -0.3])
for _ in range(100000):
    qDes = np.zeros(12)
    q = env._robot.GetMotorAngles()
    actualfoot_pos = np.zeros(12)
    for i in range(4):
        qDes[i*3:i*3+3] = env._robot.ComputeLegIK(pDes[i*3:i*3+3], i)
        actualfoot_pos[i*3:i*3+3] = env._robot.ComputeFootPosHipFrame(q[i*3:i*3+3], i)
    env._robot.ApplyAction(np.array([60]*12), np.array([2]*12),qDes, np.zeros(12), np.zeros(12))
    print(q)
    print(actualfoot_pos)
    env._pybullet_client.stepSimulation()
    time.sleep(0.001)
    # obs,r,d,i = env.step(np.zeros(16))
    # print(np.isnan(obs).any())
    # print(obs - low)
    # print(obs)
    