from envs.MPC_Nav_env import QuadrupedGymEnv
import pybullet as bc
import numpy as np
import time
env = QuadrupedGymEnv(time_step=0.001, action_repeat= 300, obs_hist_len=1, render=True)
env.reset()

# _pybullet_client = bc.BulletClient()
# _robot = A1.A1(pybullet_client = _pybullet_client)
# controller = MPClocomotion.MPCLocomotion(0.001, 30)
# controller.initialize()
# controller.setupCmd(0.0, 0.0, 0.0, 0.3)

for _ in range(20000):
    # controller.run(_robot)
    # tau = _robot.ComputeForceControl(controller.f_ff.reshape((12)))
    
    env.step(action = np.zeros(3))