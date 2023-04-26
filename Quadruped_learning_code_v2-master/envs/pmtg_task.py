from email.mime import base
import os, inspect
from turtle import forward
import envs.assets as assets
import math
import time, datetime
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import random
from collections import deque

import envs.A1 as a1
from utils.foot_trajectory_generator import FootTrajectoryGenerator
from pkg_resources import parse_version

ACTION_EPS = 0.01
OBSERVATION_EPS = 0.05

INIT_MOTOR_ANGLES = np.array([0, 0.5, -1.4] * 4)

UPPER_JOINT_LIMIT = np.array([ 0.802851455917,  4.18879020479, -0.916297857297 ] * 4)
LOWER_JOINT_LIMIT = np.array([-0.802851455917, -1.0471975512 , -2.69653369433  ] * 4)

DES_VEL_LOW = 0.5
DES_VEL_HIGH = 1.5

class QuadrupedGymEnv(gym.Env):
    def __init__(
        self,
        time_step,
        action_repeat,
        obs_hist_len,
        render = False,
        **kwargs):

        self._is_record_video = False
        self._action_repeat = action_repeat
        self._render = render
        self._action_bound = 1.0
        self._time_step = time_step
        self._num_bullet_solver_iterations = 60
        self._obs_hist_len = obs_hist_len
        self._MAX_EP_LEN = 20
        self._urdf_root = assets.getDataPath()

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
                                                  basePosition=[0,0,0],
                                                  )
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
        self._pybullet_client.setGravity(0, 0, -9.8)
        self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=1)



        self._last_fi = np.zeros(4)
        self._last_pDes = np.zeros(12)
        # self.videoLogID = None

        self._robot = a1.A1(pybullet_client=self._pybullet_client)
        _kpCartesian = np.diag([500, 500, 500])
        _kdCartesian = np.diag([10, 10, 10])
        self._robot.SetCartesianPD(_kpCartesian, _kdCartesian)

        self.setupActionSpace()
        self.setupObservationSpace()
        self.seed(20)
        self.setupFTG()


    def setupActionSpace(self):
        action_dim = 16
        self._action_dim = action_dim
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high - ACTION_EPS, action_high + ACTION_EPS, dtype=np.float32)
        self._last_action_rl = np.zeros(self._action_dim)
    
    def setupObservationSpace(self):
        obs_high = self.ObservationUpperBound() + OBSERVATION_EPS
        obs_low = self.ObservationLowerBound() - OBSERVATION_EPS

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def setupFTG(self):
        self.FTG = FootTrajectoryGenerator(T = 0.4, max_foot_height=0.1, dt=self._action_repeat/1000.0)

    def reset(self):

        mu_min = 0.5
        self._ground_mu_k = mu_min + 0.5 * np.random.random()
        self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=self._ground_mu_k)
        self._robot.Reset()

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._terminate = False
        if self._render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                       self._cam_pitch, [0, 0, 0])

        self._last_action_rl = np.zeros(self._action_dim)
        # reset commnads 
        self._des_velocity_x = DES_VEL_LOW + (DES_VEL_HIGH - DES_VEL_LOW) * np.random.random()
        self._des_velocity_y = -1 + 2 * np.random.random()
        self._des_yaw_rate = 0
        # -1 + 2 * np.random.random()
 
        self._obs_buffer = deque([np.zeros(self.observation_space.shape[0])] * self._obs_hist_len)
        for _ in range(self._obs_hist_len):
            self.getObservation()

        self._settle_robot()
        self.setupFTG()

        if self.render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                       self._cam_pitch, [0, 0, 0])
            # self.recordVideoHelper()

        return self.getObservation()

    def _settle_robot(self):
        kp_joint = np.array([60]*12)
        kd_joint = np.array([5]*12)
        for _ in range(200):
            self._robot.ApplyAction(kp_joint, kd_joint, INIT_MOTOR_ANGLES, np.zeros(12), np.zeros(12))
            if self._render:
                time.sleep(0.001)
            self._pybullet_client.stepSimulation()

    def ObservationUpperBound(self):
        upper_bound_joint = np.array([0.0] * 24)
        upper_bound_joint[0:12] = UPPER_JOINT_LIMIT
        upper_bound_joint[12:24] = np.array([21.0] * 12)
        base_upper_bound = np.concatenate((upper_bound_joint,
                                       np.array([1.0]*4), # quaternion
                                       np.array([5.0]*3), # linear vel
                                       np.array([10.0]*3), # angular vel
                                       np.array([1.0, 1.0, 1.0, 1.0]), # last freq offset
                                       np.array([1.0,1.0,1.0]*4), # last foot cmd
                                       np.array([1.]*4), # contact
                                       np.array([2, 0.5, 2]) #cmd
                                       ))
        upper_bound = np.concatenate([base_upper_bound] * self._obs_hist_len)
        return upper_bound

    def ObservationLowerBound(self):
        lower_bound_joint = np.array([0.0] * 24)
        lower_bound_joint[0:12] = LOWER_JOINT_LIMIT
        lower_bound_joint[12:24] = np.array([-21.0] * 12)
        base_lower_bound = np.concatenate((lower_bound_joint,
                                       np.array([-1.0]*4), # quaternion
                                       np.array([-5.0]*3), # linear vel
                                       np.array([-10.0]*3), # angular vel
                                       np.array([-1., -1., -1., -1.]), # last freq offset
                                       np.array([-1.0,-1.0,-1.0]*4), # last foot cmd
                                       np.array([0.0]*4), # contact
                                       np.array([-2, -0.5, -2]) #cmd
                                       ))
        lower_bound = np.concatenate([base_lower_bound] * self._obs_hist_len)
        return lower_bound

    def getObservation(self):
        observation = []
        observation.extend(list(self._robot.GetMotorAngles()))
        observation.extend(list(self._robot.GetMotorVelocities()))
        observation.extend(list(self._robot.GetBaseOrientation()))
        observation.extend(list(self._robot.GetBaseLinearVelocity()))
        observation.extend(list(self._robot.GetBaseAngularVelocity()))
        observation.extend(self._last_fi)
        observation.extend(self._last_pDes)
        observation.extend(list(self._robot.GetFootContacts()))
        base_cmd = np.array([self._des_velocity_x, self._des_velocity_y, self._des_yaw_rate])
        observation.extend(base_cmd)
        self._obs_buffer.appendleft(observation)
        obs = []
        for i in range(self._obs_hist_len):
            obs.extend(self._obs_buffer[i])
        return obs

    def step(self, action):
        action = np.clip(action, -self._action_bound - ACTION_EPS, self._action_bound + ACTION_EPS)
        self._dt_motor_torques = []
        self._dt_motor_velocities = []
        pDes = self._get_PMTG_foot_pos_cmd(action)
        qDes = np.zeros(12)
        for i in range(4):
            qDes[i*3:i*3+3] = self._robot.ComputeLegIK(pDes[i*3:i*3+3], i)

        for _ in range(self._action_repeat):
            #send cmd with IK
            # tau = self._robot.ComputeImpedanceControl(pDes, np.zeros(12))
            kpJoint = np.array([60]*12)
            kdJoint = np.array([5]*12)

            self._robot.ApplyAction(kpJoint, kdJoint, qDes, np.zeros(12), np.zeros(12))
            self._pybullet_client.stepSimulation()
            self._sim_step_counter += 1
            self._dt_motor_torques.append(self._robot.GetMotorTorqueCmds())
            self._dt_motor_velocities.append(self._robot.GetMotorVelocities())

            if self._render:
                self._render_step_helper()
                # time.sleep(0.001)
            
        self._last_action_rl = action
        self._last_pDes = pDes
        self._env_step_counter += 1
        done = False
        reward = self.get_PMTG_reward()

        if self.termination():
            done = True
            reward -= 10
        if self.get_sim_time() > self._MAX_EP_LEN:
            done = True
        
        return np.array(self.getObservation()), reward, done, {'base_pos': self._robot.GetBasePosition()}
    
    def _get_PMTG_foot_pos_cmd(self,action):
        a = action[4:]
        upp_fi = np.array([0.3, 0.3, 0.3, 0.3])#1#/0.2 (0-0.5 works)
        low_fi = np.array([-0.3, -0.3, -0.3, -0.3]) #-1#/0.2

        fi = self._scale_helper(action[:4], low_fi, upp_fi)
        self._last_fi = fi
        foot_dh = self.FTG.setPhasesAndGetDeltaHeights(self.get_sim_time(), fi=fi)
        upp_xyz = np.array([0.2, -0.1, 0.03, 0.2, 0.1, 0.03]*2)
        lb_xyz = np.array([-0.2, 0.1, -0.02, -0.2, -0.1, -0.02]*2)

        xyz = self._scale_helper(a, lb_xyz, upp_xyz)

        pDes = np.zeros(12)
        for i in range(4):
            if i==0 or i==2:
                FR_hip = np.array([0, -0.0838, -0.2])
            else:
                FR_hip = np.array([0, 0.0838, -0.3])
            pDes[3*i:3*i+3] = FR_hip + xyz[3*i:3*i+3]
            pDes[i*3+2] += foot_dh[i] 
        
        return pDes
    
    def get_PMTG_reward(self):
        base_vel = self._robot.GetBaseLinearVelocity()

        energy_reward = 0
        for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
            energy_reward -= 0.0015 * np.abs(np.dot(tau,vel)) * self._time_step 

        des_vel_x = self._des_velocity_x
        des_vel_y = self._des_velocity_y
        des_yaw_rate = self._des_yaw_rate

        vel_reward = -0.02 * (abs(base_vel[0] - des_vel_x)+ abs(base_vel[1] - des_vel_y))

        orn_weight = 0.01#1e-2 #0.01
        orn_reward = - orn_weight * (abs(self._robot.GetBaseRPY()[0]) + abs(self._robot.GetBaseRPY()[1])) 
        
        yaw_rate_act = self._robot.GetBaseAngularVelocity()[2]
        yaw_rate_reward = -0.015 * (abs(yaw_rate_act - des_yaw_rate)) #-0.01


        height_reward = -0.01 * abs(self._robot.GetBasePosition()[2] - 0.3)

        # print('vel_reward ', vel_reward)
        # print('energy_reward ', energy_reward)
        # print('orn_reward ', orn_reward)
        # print('yaw_rate_reward ', yaw_rate_reward)
        # print('height reward ', height_reward)

        return vel_reward + energy_reward \
            + height_reward + yaw_rate_reward  + orn_reward      

    def _scale_helper(self, action, lower_lim, upper_lim):
        a = lower_lim + 0.5 * (action + 1)*(upper_lim - lower_lim)
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

#========================================= render ======================================#

    def close(self):
        self._pybullet_client.disconnect()

    def configure(self, args):
        self._args = args

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
        # self._camera_dist = 1
        # self._camera_pitch = -30
        # self._camera_yaw = 0
        self._cam_dist = 1.5 # .75 #for paper
        self._cam_yaw = 30
        self._cam_pitch = -30 # -10 # for paper
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