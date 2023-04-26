from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import inspect
import math
import os
import random
import time
from collections import deque
import random

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import scipy.interpolate
from absl import app
from absl import flags
from gym import spaces
from new_mpc_implementation import MPClocomotion

import envs.A1 as a1
import envs.assets as assets

def get_action_balance(p_des_leg):
    robot = a1.A1(pybullet_client = self._pybullet_client)
    controller = MPClocomotion.MPCLocomotion(0.001, 30)
    controller.initialize()
    controller.setupCmd(0.0, 0.0, 0.0, 0.3)
    controller.run(self._robot) # 计算force f_ff 计算p_des_leg(contact)
    tau = robot.ComputeForceControl(self.controller.f_ff.reshape((12)))
    v_des_leg = np.zeros((4,3), dtype=np.float16)
    # p_des_leg = np.zeros((4,3), dtype=np.float16)
    # p_des_leg[0] = [0.1478, 0.11459, -0.3]
    # p_des_leg[1] = [0.1478, -0.11459, -0.3]
    # p_des_leg[2] = [-0.2895, 0.11459, -0.3]
    # p_des_leg[3] = [-0.2895, -0.11459, -0.3]
    for i in range(4):
        if self.controller.contactState[i] == 0:
            self._robot.SetCartesianPD(np.diag([450, 450, 250]), np.diag([10, 10, 10]))
            tau[i*3:i*3+3] += self._robot.ComputeLegImpedanceControl(p_des_leg[i], v_des_leg[i], i)
            # qDes[i*3:i*3+3] = self._robot.ComputeLegIK(self.controller.p_des_leg[i], i)
            # Jointkp[i*3:i*3+3] = np.array([60, 60, 60])
            # Jointkd[i*3:i*3+3] = np.array([2,2,2])
        else:
            # Jointkp[i*3:i*3+3] = np.array([0, 0, 0])
            # Jointkd[i*3:i*3+3] = np.array([2, 2, 2])
            self._robot.SetCartesianPD(np.diag([0, 0, 0]), np.diag([10, 10, 10]))
            tau[i*3:i*3+3] += self._robot.ComputeLegImpedanceControl(p_des_leg[i], v_des_leg[i], i)

    return tau
