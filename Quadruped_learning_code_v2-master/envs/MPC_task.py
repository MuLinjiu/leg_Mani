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

ACTION_EPS = 0.01
OBSERVATION_EPS = 0.05

INIT_MOTOR_ANGLES = np.array([0, 0.5, -1.4] * 4)
EPISODE_LENGTH = 20

_STANCE_DURATION_SECONDS = [0.3] * 4 
_DUTY_FACTOR = [0.5] * 4
_INIT_PHASE_FULL_CYCLE = [0.0, 0, 0, 0.0]
_BODY_INERTIA = np.array([0.0168, 0, 0, 0, 0.0565, 0, 0, 0, 0.064])

class QuadrupedGymEnv(gym.Env):
    def __init__(
        self,
        time_step,
        action_repeat,
        obs_hist_len,
        render = False,
        **kwargs):

        self._action_repeat = action_repeat
        self._render = render
        self._action_bound = 1.0
        self._time_step = time_step
        self._num_bullet_solver_iterations = 60
        self._obs_hist_len = obs_hist_len
        self._MAX_EP_LEN = EPISODE_LENGTH
        self._urdf_root = assets.getDataPath()
        self._last_cmd = np.zeros(8)

        self.num_obs = 40

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._last_frame_time = 0.0
        self._terminate = False

        if self._render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()
        self._configure_visualizer()

        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
        self._pybullet_client.setTimeStep(self._time_step)
        self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root,
                                                  basePosition=[80,0,0],
                                                  )
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
        self._pybullet_client.setGravity(0, 0, -9.8)
        self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=1)

        self._robot = a1.A1(pybullet_client = self._pybullet_client)

        self.setupActionSpace()
        self.setupObservationSpace()

        self.controller = MPClocomotion.MPCLocomotion(0.001, 30)


    def setupActionSpace(self):
        action_dim = 8
        self._action_dim = action_dim
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high - ACTION_EPS, action_high + ACTION_EPS, dtype=np.float32)
        self._last_action_rl = np.zeros(self._action_dim)
    
    def setupObservationSpace(self):
        upper_bound = np.array([1000] * self.num_obs * self._obs_hist_len)
        lower_bound = np.array([-1000] * self.num_obs * self._obs_hist_len)

        self.observation_space = spaces.Box(lower_bound, upper_bound, dtype=np.float32)
    
    def reset(self):
        self._robot.Reset()
        self.lin_speed, self.ang_speed = (0.2, 0.0, 0.), 0.
        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._terminate = False
        self.controller.initialize()
        self.controller.setupCmd(0.0, 0.0, 0.0, 0.3)
        self._settle_robot()

        self._obs_buffer = deque([np.zeros(self.observation_space.shape[0])] * self._obs_hist_len)
        for _ in range(self._obs_hist_len):
            self.getObservation()

        self._last_action_rl = np.zeros(8)

        if self._render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                       self._cam_pitch, [0, 0, 0])
        
        return self.getObservation()

    def _settle_robot(self):
        kp_joint = np.array([60]*12)
        kd_joint = np.array([5]*12)
        pDes = np.array([0, -0.0838, -0.3, 0, 0.0838, -0.3, 0, -0.0838, -0.3, 0, 0.0838, -0.3])
        for _ in range(2000):
            qDes = np.zeros(12)
            for i in range(4):
                qDes[i*3:i*3+3] = self._robot.ComputeLegIK(pDes[i*3:i*3+3], i)
            self._robot.ApplyAction(kp_joint, kd_joint, qDes, np.zeros(12), np.zeros(12))
            if self._render:
                time.sleep(0.001)
            self._pybullet_client.stepSimulation()
    
    def getObservation(self): # dummy function to test MPC
        observation = []
        body_states = np.array([self._robot.GetBasePosition()[2], self._robot.GetBaseRPY()[0], self._robot.GetBaseRPY()[1]])
        observation.extend(body_states)
        observation.extend(list(self._robot.GetBaseLinearVelocity())) # use world frame for now
        observation.extend(list(self._robot.GetBaseAngularVelocityLocalFrame()))
        observation.extend(self._robot.GetFootPositionsInBaseFrame()[0].tolist())
        observation.extend(self._robot.GetFootPositionsInBaseFrame()[1].tolist())
        observation.extend(self._robot.GetFootPositionsInBaseFrame()[2].tolist())
        observation.extend(self._robot.GetFootPositionsInBaseFrame()[3].tolist())
        observation.extend(list(self._robot.GetFootContacts()))
        observation.extend(list(self.controller.gait.getLegPhases()))
        observation.extend(self._last_cmd)
        vx = self.lin_speed[0]
        vy = self.lin_speed[1]
        yaw_rate = self.ang_speed
        observation.extend(np.array([vx,vy,yaw_rate]))
        self._obs_buffer.appendleft(observation)
        obs = []
        for i in range(self._obs_hist_len):
            obs.extend(self._obs_buffer[i])
        return obs
    
    def _scale_helper(self, action, lower_lim, upper_lim):
        a = lower_lim + 0.5 * (action + 1)*(upper_lim - lower_lim)
        a = np.clip(a, lower_lim, upper_lim)
        return a

    def step(self, action):
        action = np.clip(action, -self._action_bound, self._action_bound)
        self._dt_motor_torques = []
        self._dt_motor_velocities = []
        vxCommand = self._env_step_counter * 0.01
        if vxCommand > 1.5:
            vxCommand = 1.5
        self.controller.setupCmd(0.0, 0.0, 0.5, 0.28)
        foot_pos_offset = self._get_foot_pos_offsets(action)

        for _ in range(self._action_repeat):
            self.controller.run(self._robot)
            tau = self._robot.ComputeForceControl(self.controller.f_ff.reshape((12)))
                
            Jointkp = np.zeros(12)
            Jointkd = np.zeros(12)
            qDes = np.zeros(12)
                
            for i in range(4):
                if self.controller.contactState[i] == 0:
                    self._robot.SetCartesianPD(np.diag([450, 450, 250]), np.diag([10, 10, 10]))
                    tau[i*3:i*3+3] += self._robot.ComputeLegImpedanceControl(self.controller.p_des_leg[i], self.controller.v_des_leg[i], i)
                    # qDes[i*3:i*3+3] = self._robot.ComputeLegIK(self.controller.p_des_leg[i], i)
                    # Jointkp[i*3:i*3+3] = np.array([60, 60, 60])
                    # Jointkd[i*3:i*3+3] = np.array([2,2,2])
                else:
                    # Jointkp[i*3:i*3+3] = np.array([0, 0, 0])
                    # Jointkd[i*3:i*3+3] = np.array([2, 2, 2])
                    self._robot.SetCartesianPD(np.diag([0, 0, 0]), np.diag([10, 10, 10]))
                    tau[i*3:i*3+3] += self._robot.ComputeLegImpedanceControl(self.controller.p_des_leg[i], self.controller.v_des_leg[i], i)

            self._robot.ApplyAction(Jointkp, Jointkd, qDes, np.zeros(12), tau)
            self._pybullet_client.stepSimulation()
            self._sim_step_counter += 1

        if self._render:
            self._render_step_helper()
            time.sleep(0.001)

        self._last_action_rl = action
        self._last_cmd = foot_pos_offset
        self._env_step_counter += 1
        done = False
        reward = self.get_reward()

        if self.termination():
            done = True
            reward -= 10
        if self.get_sim_time() > self._MAX_EP_LEN:
            done = True

        return np.array(self.getObservation()), reward, done, {'base_pos': self._robot.GetBasePosition()}
    
    def _get_foot_pos_offsets(self, action):
        ub_foot_pos_offset = np.array([0.1, -0.05, 0.1, 0.05]*2)
        lb_foot_pos_offest = np.array([-0.1, 0.05, -0.1, -0.05]*2)
        foot_pos_offset = self._scale_helper(action, lb_foot_pos_offest, ub_foot_pos_offset)

        return foot_pos_offset
    
    def get_reward(self):
        survival_reward = 0.03
        com_vel = self._robot.GetBaseLinearVelocity()
        yaw_dot = self._robot.GetBaseAngularVelocityLocalFrame()[2]
        vel_reward = .2*(1 - abs(com_vel[0] - self.lin_speed[0])) + .2*(1 - abs(com_vel[1] - self.lin_speed[1])) + .2 * (1 - abs(yaw_dot - self.ang_speed))

        orn_reward = - 0.01 * (abs(self._robot.GetBaseRPY()[0]) + abs(self._robot.GetBaseRPY()[1]))

        return vel_reward + orn_reward + survival_reward

    def get_sim_time(self):
        return self._sim_step_counter * self._time_step

    def termination(self):
        rpy = self._robot.GetBaseRPY()
        pos = self._robot.GetBasePosition()

        # return self.is_fallen() or distance > self._distance_limit #or numInvalidContacts > 0
        return (abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1 or pos[2] < 0.15 or self._robot.GetInvalidContacts())

#========================================= render ======================================#
    def close(self):
        self._pybullet_client.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render_step_helper(self):
        """ Helper to configure the visualizer camera during step(). """
        # Sleep, otherwise the computation takes less time than real time,
        # which will make the visualization like a fast-forward video.
        time_spent = time.time() - self._last_frame_time
        # print('time_spent ', time_spent)
        self._last_frame_time = time.time()
        # time_to_sleep = self._action_repeat * self._time_step - time_spent
        time_to_sleep = self._time_step - time_spent
        if time_to_sleep > 0 and (time_to_sleep < self._time_step):
            time.sleep(time_to_sleep)
        base_pos = self._robot.GetBasePosition()
        camInfo = self._pybullet_client.getDebugVisualizerCamera()
        curTargetPos = camInfo[11]
        distance = camInfo[10]
        yaw = camInfo[8]
        pitch = camInfo[9]
        targetPos = [
            0.95 * curTargetPos[0] + 0.05 * base_pos[0], 0.95 * curTargetPos[1] + 0.05 * base_pos[1],
            curTargetPos[2]
        ]
        self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)
    
    def _configure_visualizer(self):

        self._render_width = 960
        self._render_height = 720
        self._cam_dist = 1.5 # .75 #for paper
        self._cam_yaw = 20
        self._cam_pitch = -20 # -10 # for paper
        # get rid of random visualizer things
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = np.array([self._robot.GetBasePosition()[0],self._robot.GetBasePosition()[1], 0.3])
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                    aspect=float(self._render_width)/self._render_height,
                                                                    nearVal=0.1,
                                                                    farVal=100.0)
        (_, _, px, _,
        _) = self._pybullet_client.getCameraImage(width=self._render_width,
                                                height=self._render_height,
                                                viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix,
                                                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
