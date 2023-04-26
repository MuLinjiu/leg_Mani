from envs.loco_manip_task import QuadrupedManipEnv
import pybullet as bc
import numpy as np
import time

env = QuadrupedManipEnv(time_step=0.001, action_repeat=100, obs_hist_len=1, render=True)
low =  env.action_space.low
high = env.action_space.high
env.reset()
# print(env._robot.get_body_position())
# print(env._robot.get_body_yaw())
counter = 0
for _ in range(1000):
    a = np.array([1, 0, 0])
    # if counter >= 5:
    #     a[0] = 0
    env.step(a)
    
    counter += 1

