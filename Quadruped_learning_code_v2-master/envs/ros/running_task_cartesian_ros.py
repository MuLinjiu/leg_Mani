import os, inspect
import envs.assets as assets

import math
import time, datetime
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
from collections import deque

from envs.ros.A1_ROS import A1ROS

OBSERVATION_EPS = 0.05

INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * 4)

UPPER_JOINT_LIMIT = np.array([0.802851455917, 4.18879020479, -0.916297857297] * 4)
LOWER_JOINT_LIMIT = np.array([-0.802851455917, -1.0471975512, -2.69653369433] * 4)

DES_VEL_LOW = 1.0
DES_VEL_HIGH = 2.0


class QuadrupedGymEnv(gym.Env):
    def __init__(
            self,
            time_step,
            action_repeat,
            obs_hist_len,
            obs_hist_space=1,
            render=False,
            **kwargs):

        self._action_repeat = action_repeat
        self._render = render
        self._action_bound = 1.0
        self._time_step = time_step
        self._num_bullet_solver_iterations = 60
        self._obs_hist_len = obs_hist_len
        self._obs_hist_space = obs_hist_space

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._last_frame_time = 0.0

        self._last_pDes = np.zeros(12)

        self._robot = A1ROS()
        _kpCartesian = np.diag([500, 500, 500])
        _kdCartesian = np.diag([10, 10, 10])
        self._robot.SetCartesianPD(_kpCartesian, _kdCartesian)

        self._des_velocity_x = 0
        self._des_velocity_y = 0
        self._des_yaw_rate = 0

        self.setupActionSpace()
        self.setupObservationSpace()
        self.seed()

    def setupActionSpace(self):
        action_dim = 12
        self._action_dim = action_dim
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self._last_action_rl = np.zeros(self._action_dim)

    def setupObservationSpace(self):
        obs_high = self.ObservationUpperBound() + OBSERVATION_EPS
        obs_low = self.ObservationLowerBound() - OBSERVATION_EPS

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def reset(self):
        self._last_pDes = np.zeros(12)

        # self._des_velocity_x = DES_VEL_LOW + (DES_VEL_HIGH - DES_VEL_LOW) * np.random.random()
        self._des_velocity_x = 2.0
        self._des_yaw_rate = 0.0

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]

        self._last_action_rl = np.zeros(self._action_dim)
        # reset commnads 

        self._last_feet_contact_time = [0] * 4
        self._feet_in_air = [False] * 4

        self._obs_buffer = deque()
        self._initial_obs = self._getCurrentObservation()

        return self.getObservation()

    def ObservationUpperBound(self):
        upper_bound_joint = np.array([0.0] * 24)
        upper_bound_joint[0:12] = UPPER_JOINT_LIMIT
        upper_bound_joint[12:24] = np.array([21.0] * 12)
        base_upper_bound = np.concatenate((upper_bound_joint,
                                           np.array([1.0] * 4),  # quaternion
                                           np.array([5.0] * 3),  # linear vel
                                           np.array([10.0] * 3),  # angular vel
                                           np.array([1.0, 1.0, 1.0] * 4),  # last foot cmd
                                           np.array([1.] * 4),  # contact
                                           np.array([2, 0.5, 2])  # cmd
                                           ))
        upper_bound = np.concatenate([base_upper_bound] * self._obs_hist_len)
        return upper_bound

    def ObservationLowerBound(self):
        lower_bound_joint = np.array([0.0] * 24)
        lower_bound_joint[0:12] = LOWER_JOINT_LIMIT
        lower_bound_joint[12:24] = np.array([-21.0] * 12)
        base_lower_bound = np.concatenate((lower_bound_joint,
                                           np.array([-1.0] * 4),  # quaternion
                                           np.array([-5.0] * 3),  # linear vel
                                           np.array([-10.0] * 3),  # angular vel
                                           np.array([-1.0, -1.0, -1.0] * 4),  # last foot cmd
                                           np.array([0.0] * 4),  # contact
                                           np.array([-2, -0.5, -2])  # cmd
                                           ))
        lower_bound = np.concatenate([base_lower_bound] * self._obs_hist_len)
        return lower_bound

    def _getCurrentObservation(self):
        observation = []
        observation.extend(list(self._robot.GetMotorAngles()))
        observation.extend(list(self._robot.GetMotorVelocities()))
        observation.extend(list(self._robot.GetBaseOrientation()))
        observation.extend(list(self._robot.GetBaseLinearVelocity()))
        observation.extend(list(self._robot.GetBaseAngularVelocity()))
        observation.extend(self._last_pDes)
        observation.extend(list(self._robot.GetFootContacts()))
        base_cmd = np.array([self._des_velocity_x, self._des_velocity_y, self._des_yaw_rate])
        observation.extend(base_cmd)
        observation = np.array(observation)

        return observation

    def getObservation(self):
        observation = self._getCurrentObservation()
        self._obs_buffer.appendleft(observation)

        obs = []
        for i in range(self._obs_hist_len):
            obs_idx = i * self._obs_hist_space
            if obs_idx < len(self._obs_buffer):
                obs.extend(self._obs_buffer[i])
            else:
                obs.extend(self._initial_obs)

        return np.array(obs)

    def step(self, action):
        action = np.clip(action, -self._action_bound, self._action_bound)
        pDes = self._get_foot_pos_cmd(action)
        for _ in range(self._action_repeat):
            # send cmd with IK
            tau = self._robot.ComputeImpedanceControl(pDes, np.zeros(12))

            kpJoint = np.array([0] * 12)
            kdJoint = np.array([2] * 12)
            qDes = INIT_MOTOR_ANGLES

            self._robot.ApplyAction(kpJoint, kdJoint, qDes, np.zeros(12), tau)
            self._sim_step_counter += 1

        self._last_action_rl = action
        self._last_pDes = pDes
        self._env_step_counter += 1
        done = False
        timeout = False

        if self.termination():
            done = True

        return np.array(self.getObservation()), 0, done, {'base_pos': self._robot.GetBasePosition(),
                                                          "TimeLimit.truncated": timeout}

    def _get_foot_pos_cmd(self, action):
        upp_xyz = np.array([0.2, -0.05, -0.15, 0.2, 0.05, -0.15] * 2)
        lb_xyz = np.array([-0.2, 0.05, -0.33, -0.2, -0.05, -0.33] * 2)

        xyz = self._scale_helper(action, lb_xyz, upp_xyz)

        pDes = np.zeros(12)
        for i in range(4):
            if i == 0 or i == 2:
                FR_hip = np.array([0, -0.0838, 0])
            else:
                FR_hip = np.array([0, 0.0838, 0])
            pDes[3 * i:3 * i + 3] = FR_hip + xyz[3 * i:3 * i + 3]

        return pDes

    def _scale_helper(self, action, lower_lim, upper_lim):
        a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        a = np.clip(a, lower_lim, upper_lim)
        return a

    def get_sim_time(self):
        return self._sim_step_counter * self._time_step

    def termination(self):
        orientation = self._robot.GetBaseOrientation()
        rpy = self._robot.GetBaseRPY()
        rot_mat = self._robot.GetBaseOrientationMatrix()
        local_up = rot_mat[6:]
        pos = self._robot.GetBasePosition()

        # return self.is_fallen() or distance > self._distance_limit #or numInvalidContacts > 0
        return (abs(rpy[0]) > 0.5 or abs(rpy[1]) > 1 or pos[2] < 0.15 or self._robot.GetInvalidContacts())

    def scale_rand(self, num_rand, low, high):
        """ scale number of rand numbers between low and high """
        return low + np.random.random(num_rand) * (high - low)

    # ========================================= render ======================================#

    def close(self):
        pass

    def set_env_randomizer(self, env_randomizer):
        self._env_randomizer = env_randomizer

    def configure(self, args):
        self._args = args

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        return np.array([])
