import os, inspect
import envs.assets as assets

URDF_ROOT = assets.getDataPath()

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

OBSERVATION_EPS = 0.05

INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * 4)

UPPER_JOINT_LIMIT = np.array([0.802851455917, 4.18879020479, -0.916297857297] * 4)
LOWER_JOINT_LIMIT = np.array([-0.802851455917, -1.0471975512, -2.69653369433] * 4)

DES_VEL_LOW = 0.5
DES_VEL_HIGH = 2.0


class QuadrupedGymEnv(gym.Env):
    def __init__(
            self,
            time_step,
            action_repeat,
            obs_hist_len,
            render=False,
            **kwargs):

        self._is_record_video = False
        self._domain_randomization = True

        self._action_repeat = action_repeat
        self._render = render
        self._action_bound = 1.0
        self._time_step = time_step
        self._num_bullet_solver_iterations = 60
        self._obs_hist_len = obs_hist_len
        self._MAX_EP_LEN = 10  # in seconds

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
        self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % URDF_ROOT,
                                                    basePosition=[80, 0, 0],
                                                    # to extend available running space (shift)
                                                    )
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
        self._pybullet_client.setGravity(0, 0, -9.8)

        self._gait_duration = 0.4
        self._phase = 0

        self._last_qDes = np.zeros(12)

        self._robot = a1.A1(pybullet_client=self._pybullet_client)
        self.base_block_ID = -1
        _kpCartesian = np.diag([500, 500, 500])
        _kdCartesian = np.diag([10, 10, 10])
        self._robot.SetCartesianPD(_kpCartesian, _kdCartesian)

        self.box_ids = []

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
        mu_min = 0.5
        self._ground_mu_k = mu_min + 0.3 * np.random.random()
        self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=self._ground_mu_k)
        self._robot.Reset()
        self._settle_robot()

        self._phase = 0
        self._last_qDes = np.zeros(12)

        if self._domain_randomization:
            self.add_random_boxes()
            # if self.base_block_ID != -1:
            #     self._pybullet_client.removeBody(self.base_block_ID)
            #     self.base_block_ID = -1
            # if np.random.random() < 0.8:
            #     self.add_base_mass_offset()
            self._robot.RandomizePhysicalParams()

        # self._des_velocity_x = 0.5 + 1.5 * np.random.random()
        self._des_velocity_x = 1.0
        self._des_velocity_y = 0.0
        self._des_yaw_rate = 0.0 #+ np.random.random()
        self._des_yaw = 0.0

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._terminate = False
        if self._render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                             self._cam_pitch, [0, 0, 0])
            print("Ground friction: ", self._ground_mu_k)
            print("Des vel x: ", self._des_velocity_x)
            print("Des vel y: ", self._des_velocity_y)
            print("Des turn rate: ", self._des_yaw_rate)

        self._last_action_rl = np.zeros(self._action_dim)
        # reset commnads 

        self._last_feet_contact_time = [0] * 4
        self._feet_in_air = [False] * 4
        self._foot_max_height = [0] * 4

        self._obs_buffer = deque([np.zeros(self.observation_space.shape[0])] * self._obs_hist_len)
        self._settle_robot()
        for _ in range(self._obs_hist_len):
            self.getObservation()

        if self.render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                             self._cam_pitch, [0, 0, 0])

        return self.getObservation()

    def _settle_robot(self):
        if self._render:
            time.sleep(0.2)

        kp_joint = np.array([60] * 12)
        kd_joint = np.array([5] * 12)
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
                                           np.array([1.0] * 4),  # quaternion
                                           np.array([5.0] * 3),  # linear vel
                                           np.array([10.0] * 3),  # angular vel
                                           np.array([1.0, 1.0, 1.0] * 4),  # last foot cmd
                                           np.array([1.] * 4),  # contact
                                           np.array([2, 2, 2])  # cmd
                                           ))
        upper_bound = np.concatenate([base_upper_bound] * self._obs_hist_len)
        upper_bound = np.append(upper_bound, np.array([1, 1]))
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
                                           np.array([-2, -2 , -2])  # cmd
                                           ))
        lower_bound = np.concatenate([base_lower_bound] * self._obs_hist_len)
        lower_bound = np.append(lower_bound, np.array([-1, -1]))
        return lower_bound

    def getObservation(self):
        observation = []
        observation.extend(list(self._robot.GetMotorAngles()))
        observation.extend(list(self._robot.GetMotorVelocities()))
        observation.extend(list(self._robot.GetBaseOrientation()))
        observation.extend(list(self._robot.GetBaseLinearVelocity()))
        observation.extend(list(self._robot.GetBaseAngularVelocity()))
        observation.extend(self._last_qDes)
        observation.extend(list(self._robot.GetFootContacts()))
        base_cmd = np.array([self._des_velocity_x, self._des_velocity_y, self._des_yaw_rate])
        observation.extend(base_cmd)
        observation = np.array(observation)

        # obs_noise = np.random.normal(size=observation.shape) * self.getObservationNoiseScale()
        # observation += obs_noise
        self._obs_buffer.appendleft(observation)

        obs = []
        for i in range(self._obs_hist_len):
            obs.extend(self._obs_buffer[i])
        signal_1 = np.sin(2 * np.pi * self._phase)
        signal_2 = np.cos(2 * np.pi * self._phase)
        obs.extend([signal_1, signal_2])

        return np.array(obs)

    def getObservationNoiseScale(self):
        obs_scale = []
        obs_scale.extend([0.1, 0.1, 0.1] * 4)  # Motor angles
        obs_scale.extend([0.2] * 12)  # Motor vel
        obs_scale.extend([0.1] * 4)  # quaternion
        obs_scale.extend([0.5] * 3)  # linear vel
        obs_scale.extend([0.2] * 3)  # angular vel

        obs_scale.extend([0.0, 0.0, 0.0] * 4)  # last foot cmd
        obs_scale.extend([0.0] * 4)  # contact
        obs_scale.extend([0.0, 0.0, 0.0])  # cmd

        return np.array(obs_scale)

    def step(self, action):
        action = np.clip(action, -self._action_bound, self._action_bound)
        self._dt_motor_torques = []
        self._dt_motor_velocities = []
        qDes = self._get_motor_pos_cmd(action)
        for _ in range(self._action_repeat):
            # send cmd with IK
            kpJoint = np.array([60] * 12)
            kdJoint = np.array([5] * 12)

            self._robot.ApplyAction(kpJoint, kdJoint, qDes, np.zeros(12), np.zeros(12),
                                    add_noise=self._domain_randomization)
            self._pybullet_client.stepSimulation()
            self._sim_step_counter += 1
            self._dt_motor_torques.append(self._robot.GetMotorTorqueCmds())
            self._dt_motor_velocities.append(self._robot.GetMotorVelocities())

            if self._render:
                self._render_step_helper()
                # time.sleep(0.001)
                

        self._last_action_rl = action
        self._last_qDes = qDes
        self._env_step_counter += 1
        self._phase = (self.get_sim_time() % self._gait_duration) / self._gait_duration
        done = False
        timeout = False
        reward = self.get_reward()

        if self.termination():
            done = True
            reward -= 10
        if self.get_sim_time() > self._MAX_EP_LEN:
            done = True
            timeout = True

        return np.array(self.getObservation()), reward, done, {'base_pos': self._robot.GetBasePosition(),
                                                               "TimeLimit.truncated": timeout}

    def _get_motor_pos_cmd(self, action):

        # upp_q_offset = np.array([0.8] * 12)
        # lb_q_offset = np.array([-0.8] * 12)
        upp_q_offset = np.array([0.5] * 12)
        lb_q_offset = np.array([-0.5] * 12)
        # upp_q_offset = np.array([0.4, 0.8, 0.8] * 4)
        # lb_q_offset = np.array([-0.4, -0.8, -0.8] * 4)

        qDes_offset = self._scale_helper(action, lb_q_offset, upp_q_offset)

        qDes_temp = INIT_MOTOR_ANGLES + qDes_offset
        qDes = np.clip(qDes_temp, LOWER_JOINT_LIMIT, UPPER_JOINT_LIMIT)

        return qDes

    def get_reward(self):
        survival_reward = 0.06
        base_vel = self._robot.GetBaseLinearVelocity()
        robot_yaw = self._robot.GetBaseRPY()[2]
        des_vel_x_world_frame = np.cos(robot_yaw) * self._des_velocity_x - np.sin(robot_yaw) * self._des_velocity_y
        des_vel_y_world_frame = np.sin(robot_yaw) * self._des_velocity_x + np.cos(robot_yaw) * self._des_velocity_y
        self._des_yaw += self._action_repeat * self._time_step * self._des_yaw_rate
        energy_reward = 0
        for tau, vel in zip(self._dt_motor_torques, self._dt_motor_velocities):
            energy_reward -= 0.008 * np.abs(np.dot(tau, vel)) * self._time_step

        vel_reward = .2 * (1 - (abs(base_vel[0] - des_vel_x_world_frame) + abs(base_vel[1] - des_vel_y_world_frame)))
        # orn_weight = 1e-3
        orn_weight = 0.2 # 1e-2
        orn_reward = - orn_weight * (abs(self._robot.GetBaseRPY()[0]) + abs(self._robot.GetBaseRPY()[1]))
        yaw_rate_act = self._robot.GetBaseAngularVelocity()[2]
        yaw_rate_reward = 0.05 * (0.1 - abs(yaw_rate_act - self._des_yaw_rate))
        y_reward = -0.1 * abs(self._robot.GetBasePosition()[1])

        height_reward = -0.2 * abs(self._robot.GetBasePosition()[2] - 0.3)

        feetInContactBool = self._robot.GetFootContacts()
        # * Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;

        # feet_in_air_weight = 0.2
        # feet_in_air_reward = 0
        # cur_time = self.get_sim_time()
        # for i in range(len(feetInContactBool)):
        #     if feetInContactBool[i] == 1 and self._feet_in_air[i]:
        #         in_air_time = cur_time - self._last_feet_contact_time[i]
        #         feet_in_air_reward += (in_air_time - 0.4) * feet_in_air_weight
        #         self._last_feet_contact_time[i] = cur_time
        #         self._feet_in_air[i] = False
        #     elif feetInContactBool[i] == 0 and not self._feet_in_air[i]:
        #         self._feet_in_air[i] = True

        foot_clearance_weight = -0.1
        foot_clearance_reward = 0
        foot_clearance_target = 0.05

        foot_pos = self._robot.GetFootPositionsInBaseFrame()
        rBody = self._robot.GetBaseOrientationMatrix()
        foot_pos = np.matmul(rBody.T, foot_pos.T).T + self._robot.GetBasePosition()

        for i in range(4):
            if feetInContactBool[i] == 1 and self._feet_in_air[i]:
                foot_clearance_reward += foot_clearance_weight * (self._foot_max_height[i] < foot_clearance_target)
                self._feet_in_air[i] = False
                self._foot_max_height[i] = 0
            elif feetInContactBool[i] == 0 and not self._feet_in_air[i]:
                self._feet_in_air[i] = True
                self._foot_max_height[i] = max(self._foot_max_height[i], foot_pos[i][2])

        gait_reward_weight = 0.02
        gait_reward = 0
        grp_1_signal = np.sin(2 * np.pi * self._phase)  # pos: should stance; neg: should swing
        grp_2_signal = -grp_1_signal
        grp_1_leg_idx = [0, 3]
        # grp_2_leg_idx = [1, 2]

        for i in range(len(feetInContactBool)):
            if i in grp_1_leg_idx:
                gait_reward -= feetInContactBool[i] ^ (grp_1_signal > 0)
            else:
                gait_reward -= feetInContactBool[i] ^ (grp_2_signal > 0)
        gait_reward *= gait_reward_weight

        # print("======================================")
        # print("Vel: ", vel_reward)
        # print("Energy: ", energy_reward)
        # print("Y: ", y_reward)
        # print("Gait: ", gait_reward)
        # print("Height: ", height_reward)
        # print("Orn: ", orn_reward)
        # print("Survival: ", survival_reward)
        # print("Foot clearance: ", foot_clearance_reward)
        #
        # total = vel_reward + energy_reward + y_reward + gait_reward \
        #        + height_reward + orn_reward + survival_reward + foot_clearance_reward
        # print("### Total: ###", total)

        return vel_reward + energy_reward + yaw_rate_reward + gait_reward + y_reward \
               + height_reward + orn_reward + survival_reward + foot_clearance_reward

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

    def add_random_boxes(self, num_rand=50):
        """Add random boxes in front of the robot, should be in x [0.5, 50] and y [-5,5]
    how many?
    how large?
    """
        # print('-'*80,'\nadding boxes\n','-'*80)
        # x location
        x_low = 2
        x_upp = 20
        # y location
        y_low = -1.5
        y_upp = 1.5
        # z, orig [0.01, 0.03]
        z_low = 0.005  #
        z_upp = 0.015  # max height, was 0.025
        # z_upp = 0.04 # max height, was 0.025
        # block dimensions
        block_x_max = 1
        block_x_min = 0.1
        block_y_max = 1
        block_y_min = 0.1
        # block orientations
        # roll_low, roll_upp = -0.1, 0.1
        # pitch_low, pitch_upp = -0.1, 0.1
        roll_low, roll_upp = -0.01, 0.01
        pitch_low, pitch_upp = -0.01, 0.01  # was 0.001,
        yaw_low, yaw_upp = -np.pi, np.pi

        x = self.scale_rand(num_rand, x_low, x_upp)
        y = self.scale_rand(num_rand, y_low, y_upp)
        z = self.scale_rand(num_rand, z_low, z_upp)
        block_x = self.scale_rand(num_rand, block_x_min, block_x_max)
        block_y = self.scale_rand(num_rand, block_y_min, block_y_max)
        roll = self.scale_rand(num_rand, roll_low, roll_upp)
        pitch = self.scale_rand(num_rand, pitch_low, pitch_upp)
        yaw = self.scale_rand(num_rand, yaw_low, yaw_upp)
        # loop through
        if not self.box_ids:
            for i in range(num_rand):
                sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
                                                                       halfExtents=[block_x[i] / 2, block_y[i] / 2,
                                                                                    z[i] / 2])
                orn = self._pybullet_client.getQuaternionFromEuler([roll[i], pitch[i], yaw[i]])
                block2 = self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                               basePosition=[x[i], y[i], z[i] / 2], baseOrientation=orn)
                # set friction coeff to 1
                self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)
                self.box_ids.append(block2)

            # add walls
            orn = self._pybullet_client.getQuaternionFromEuler([0, 0, 0])
            sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
                                                                   halfExtents=[x_upp / 2, 0.5, 0.5])
            block2 = self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                           basePosition=[x_upp / 2, y_low, 0.5], baseOrientation=orn)
            block2 = self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                                                           basePosition=[x_upp / 2, -y_low, 0.5], baseOrientation=orn)
        else:
            for i in range(num_rand):
                orn = self._pybullet_client.getQuaternionFromEuler([roll[i], pitch[i], yaw[i]])
                self._pybullet_client.resetBasePositionAndOrientation(self.box_ids[i],
                                                                      posObj=[x[i], y[i], z[i] / 2], ornObj=orn)

    def add_base_mass_offset(self, spec_mass=None, spec_location=None):
        quad_base = np.array(self._robot.GetBasePosition())
        quad_ID = self._robot.A1

        # offset_low = np.array([-0.15, -0.02, -0.05])
        # offset_upp = np.array([ 0.15,  0.02,  0.05])
        offset_low = np.array([-0.15, -0.05, -0.05])
        offset_upp = np.array([0.15, 0.05, 0.05])
        #   block_pos_delta_base_frame = -1*np.array([-0.2, 0.1, -0.])
        if spec_location is None:
            block_pos_delta_base_frame = self.scale_rand(3, offset_low, offset_upp)
        else:
            block_pos_delta_base_frame = np.array(spec_location)
        if spec_mass is None:
            # base_mass = 8*np.random.random()
            # base_mass = 15*np.random.random()
            # base_mass = 12*np.random.random()
            base_mass = 2 * np.random.random()
        else:
            base_mass = spec_mass
        if self._render:
            print('=========================== Random Mass:')
            print('Mass:', base_mass, 'location:', block_pos_delta_base_frame)

            # if rendering, also want to set the halfExtents accordingly
            # 1 kg water is 0.001 cubic meters
            boxSizeHalf = [(base_mass * 0.001) ** (1 / 3) / 2] * 3
            # boxSizeHalf = [0.05]*3
            translationalOffset = [0, 0, 0.1]
        else:
            boxSizeHalf = [0.05] * 3
            translationalOffset = [0] * 3

        # sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX, halfExtents=[0.05]*3)
        sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX, halfExtents=boxSizeHalf,
                                                               collisionFramePosition=translationalOffset)
        # orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
        self.base_block_ID = self._pybullet_client.createMultiBody(baseMass=base_mass,
                                                                   baseCollisionShapeIndex=sh_colBox,
                                                                   basePosition=quad_base + block_pos_delta_base_frame,
                                                                   baseOrientation=[0, 0, 0, 1])

        cid = self._pybullet_client.createConstraint(quad_ID, -1, self.base_block_ID, -1,
                                                     self._pybullet_client.JOINT_FIXED,
                                                     [0, 0, 0], [0, 0, 0], -block_pos_delta_base_frame)
        # disable self collision between box and each link
        for i in range(-1, self._pybullet_client.getNumJoints(quad_ID)):
            self._pybullet_client.setCollisionFilterPair(quad_ID, self.base_block_ID, i, -1, 0)

    # ========================================= render ======================================#

    def close(self):
        self._pybullet_client.disconnect()

    def set_env_randomizer(self, env_randomizer):
        self._env_randomizer = env_randomizer

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
        self._cam_dist = 1.0  # .75 #for paper
        self._cam_yaw = 0
        self._cam_pitch = -30  # -10 # for paper
        # get rid of random visualizer things
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI, 0)

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = self._robot.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                       aspect=float(
                                                                           self._render_width) / self._render_height,
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
