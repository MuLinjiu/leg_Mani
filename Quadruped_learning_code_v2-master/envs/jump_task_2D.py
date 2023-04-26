import os, inspect, io
import math, time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import random
from utils.traj_motion_data import TrajTaskMotionData
from collections import deque

import envs.A1 as a1

FILENAME_OPT = '../envs/data/data9_forward/jumpingFull_A1_1ms_h00_d60.csv'

PRE_LANDING_CONFIG = np.array([0, 0.7, -1.8] * 4)

UPPER_JOINT_LIMIT = np.array([ 0.802851455917,  4.18879020479, -0.916297857297 ] * 4)
LOWER_JOINT_LIMIT = np.array([-0.802851455917, -1.0471975512 , -2.69653369433  ] * 4)

class QuadrupedGymEnv(gym.Env):
    def __init__(
        self,
        time_step,
        action_repeat,
        obs_hist_len,
        obs_hist_space=1,
        render = False,
        **kwargs):

        self._time_step = time_step
        self._action_repeat = action_repeat
        self._obs_hist_len = obs_hist_len
        self._render = render
        self._action_bound = 1.0
        self._obs_hist_space = obs_hist_space
        self._num_bullet_solver_iterations = 60

        self.num_obs = 41 + 36*3 + 12 + 2

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._last_frame_time = 0.0
        self._terminate = False

        self._des_pos_x = 0.0
        self._des_pos_y = 0.0

        if self._render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
            # self.q_file = io.open('qDes.txt', 'w')
        else:
            self._pybullet_client = bc.BulletClient()
        self._configure_visualizer()

        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
        self._pybullet_client.setTimeStep(self._time_step)
        self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath(),
                                                  basePosition=[0,0,0],
                                                  )
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
        self._pybullet_client.setGravity(0, 0, -9.8)
        self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=1.0)


        # self.add_box_ff(0.02)
        # self.add_box_rr(0.02)

        self._robot = a1.A1(pybullet_client = self._pybullet_client)

        self._optimization_traj = TrajTaskMotionData(filename=FILENAME_OPT, dt=0.001, useCartesianData=True)
        self._traj_duration = self._optimization_traj.trajLen
        self._last_qDes_cmd = np.zeros(12)

        self.setupActionSpace()
        self.setupObservationSpace()

    def setupActionSpace(self):
        action_dim = 4
        self._action_dim = action_dim
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self._last_action_rl = np.zeros(self._action_dim)

    def setupObservationSpace(self):
        upper_bound = np.array([100.0] * self.num_obs * self._obs_hist_len)
        lower_bound = np.array([-100.0] * self.num_obs * self._obs_hist_len)

        self.observation_space = spaces.Box(lower_bound, upper_bound, dtype=np.float32)

    def reset(self):
        self._robot.Reset()
        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._terminate = False
        self._last_qDes_cmd = np.zeros(12)
        self._max_height = 0

        self._dt_motor_torques = []
        self._dt_motor_velocities = []

        self._settle_robot()

        self.kpJoint = np.array([300]*12)
        self.kdJoint = np.array([3]*12)

        # self._des_pos_x = 0.4
        self._des_pos_x = 0.3

        self._obs_buffer = deque()
        self._initial_obs = self._getCurrentObservation()
        
        self._last_action_rl = np.zeros(4)
        if self._render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                       self._cam_pitch, [0, 0, 0])
        
        return self.getObservation()       


    def _settle_robot(self):
        kp_joint = np.array([60]*12)
        kd_joint = np.array([5]*12)
        # des_state = self._optimization_traj.get_state_at_index(0)
        # qDes = self._optimization_traj.get_joint_pos_from_state(des_state)
        qDes = self._optimization_traj.q[:,0]

        for _ in range(1000):
            self._robot.ApplyAction(kp_joint, kd_joint, qDes, np.zeros(12), np.zeros(12))
            if self._render:
                time.sleep(0.001)
            self._pybullet_client.stepSimulation()
        
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

    def _getCurrentObservation(self):
        observation = []
        observation.extend(list(self._robot.GetMotorAngles()))
        observation.extend(list(self._robot.GetMotorVelocities()))
        observation.extend(list(self._robot.GetBaseOrientation()))
        observation.extend(list(self._robot.GetBasePosition()))
        observation.extend(list(self._robot.GetBaseLinearVelocity()))
        observation.extend(list(self._robot.GetBaseAngularVelocity()))       
        observation.extend(list(self._robot.GetFootContacts()))
        
        traj_index = np.array([1, 2, 3])
        
        for idx in traj_index:
            observation.extend(self._optimization_traj.get_qDes_at_index(self._sim_step_counter + idx * self._action_repeat))
            observation.extend(self._optimization_traj.get_qdDes_at_index(self._sim_step_counter + idx * self._action_repeat))
            observation.extend(self._optimization_traj.get_torques_at_index(self._sim_step_counter + idx * self._action_repeat))
        observation.extend(self._last_qDes_cmd)
        observation.append(self._des_pos_x)
        observation.append(self._des_pos_y)
        observation = np.array(observation)

        return observation

    def step(self, action):
        action = np.clip(action, -self._action_bound, self._action_bound)
        
        delta_q = self._get_delta_q(action)

        on_ground_flag = 1.0
        if self._sim_step_counter >= 800:
            on_ground_flag = 0.0

        if self._sim_step_counter == 0:
            start_qDes = self._robot.GetMotorAngles()
        else:
            start_qDes = self._last_qDes_cmd

        end_qDes = self._optimization_traj.get_qDes_at_index(self._sim_step_counter + self._action_repeat) + on_ground_flag * delta_q
        if self._sim_step_counter > 1000: # prepare for landing
            end_qDes = PRE_LANDING_CONFIG
            self.kpJoint = np.array([20.0] * 12)
            self.kdJoint = np.array([5.0] * 12)

        for i in range(self._action_repeat):
            self._max_height = max(self._max_height, self._robot.GetBasePosition()[2])
            qdDes_opt = self._optimization_traj.get_qdDes_at_index(self._sim_step_counter)
            tauDes_opt = self._optimization_traj.get_torques_at_index(self._sim_step_counter)

            qDes = start_qDes + (i/self._action_repeat) * (end_qDes - start_qDes)
            qDes = np.clip(qDes, LOWER_JOINT_LIMIT, UPPER_JOINT_LIMIT)

            self._robot.ApplyAction(self.kpJoint, self.kdJoint, qDes, qdDes_opt, tauDes_opt)
            self._pybullet_client.stepSimulation()
            self._sim_step_counter += 1
            self._dt_motor_torques.append(self._robot.GetMotorTorqueCmds())
            self._dt_motor_velocities.append(self._robot.GetMotorVelocities())
            
            if self._render:
                # for i in range(12):
                #     self.q_file.write( "%.5f " %qDes[i])
                # for i in range(12):
                #     self.q_file.write( "%.5f " %self._robot.GetMotorAngles()[i])
                # for i in range(12):
                #     self.q_file.write( "%.5f " %self._robot.GetMotorVelocities()[i])
                # for i in range(12):
                #     self.q_file.write( "%.5f " %self._robot.GetMotorTorqueCmds()[i])
                # for i in range(3):
                #     self.q_file.write( "%.5f " %self._robot.GetBasePosition()[i])
                # for i in range(3):
                #     self.q_file.write( "%.5f " %self._robot.GetBaseRPY()[i])
                # self.q_file.write('\n')
                self._render_step_helper()
                # time.sleep(0.002)
    
        self._last_action_rl = action
        self._last_qDes_cmd = end_qDes
        self._env_step_counter += 1
        done = False
        reward, done = self.get_reward_and_done()
        # if self._sim_step_counter == 1200:
        #     print(self._robot.GetBasePosition()[0])

        return np.array(self.getObservation()), reward, done, {'base_pos': self._robot.GetBasePosition(), 'base_rpy': self._robot.GetBaseRPY()}
    
    def get_reward_and_done(self):
        done = False
        fail = False
        reward = 0
        # print('energy rewrad %f', energy_weight * energy_reward)
        energy_weight = 0.25
        if self._robot.GetInvalidContacts():
            fail = True

        if (self._sim_step_counter >= self._optimization_traj.trajLen - 1 and np.sum(self._robot.GetFootContacts()) >= 2) or fail:
            done = True
            robot_position_xy = np.array([self._robot.GetBasePosition()[0], self._robot.GetBasePosition()[1]])
            w1 = 80
            w2 = -25.0
            w3 = -0.2

            energy_reward = 0
            for tau, vel in zip(self._dt_motor_torques, self._dt_motor_velocities):
                energy_reward -= np.abs(np.dot(tau, vel)) * self._time_step
            
            reward = energy_weight * energy_reward

            reward += w1 * (0.05 - np.linalg.norm(np.array([self._des_pos_x, self._des_pos_y]) - robot_position_xy)) \
                      + w2 * np.linalg.norm(self._robot.GetBaseOrientation() - np.array([0,0,0,1])) + w3 * abs(self._robot.GetBaseLinearVelocity()[2]) # need to change the des_orientation for 3D implementation

            if self._max_height < 0.45:
                fail = True
            
            if abs(robot_position_xy[0] - self._des_pos_x) < 0.03 and fail == False:
                reward += 30   
            
            if fail:
                done = True
                reward -= 30
            
            # print('max height ', self._max_height)

            # print('energy rewrad %f', energy_weight * energy_reward)
            # print('final dist reward %f', w1 * (0.05 - np.linalg.norm(np.array([self._des_pos_x, self._des_pos_y]) - robot_position_xy)) )
            # print('final ori reward %f',  w2 * np.linalg.norm(self._robot.GetBaseOrientation() - np.array([0,0,0,1])))
            # print('final z-vel reward %f', w3 * abs(self._robot.GetBaseLinearVelocity()[2]))

        return reward, done
        

    def _get_delta_q(self, action):
        # lb = np.array([-0.15]*4)
        # ub = np.array([0.15]*4)
        lb = np.array([-0.2]*4)
        ub = np.array([0.2]*4)
        delta_q_2D = self._scale_helper(action, lb, ub)
        delta_q = np.zeros(12)
        delta_q[1] = delta_q_2D[0]
        delta_q[4] = delta_q_2D[0]
        delta_q[2] = delta_q_2D[1]
        delta_q[5] = delta_q_2D[1]
        delta_q[7] = delta_q_2D[2]
        delta_q[10] = delta_q_2D[2]
        delta_q[8] = delta_q_2D[3]
        delta_q[11] = delta_q_2D[3]

        return delta_q

    def _scale_helper(self, action, lower_lim, upper_lim):
        a = lower_lim + 0.5 * (action + 1)*(upper_lim - lower_lim)
        a = np.clip(a, lower_lim, upper_lim)
        return a
    
    def get_sim_time(self):
        return self._sim_step_counter * self._time_step
        
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
        self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, targetPos)

    def _configure_visualizer(self):

        self._render_width = 960
        self._render_height = 720

        self._cam_dist = 1.0  # .75 #for paper
        self._cam_yaw = 0.0
        self._cam_pitch = -30.0  # -10 # for paper
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

    def add_box_ff(self, z_height):
        """ add box under front feet"""
        #box_x = 0.2
        box_x = 0.25
        box_y = 0.5
        box_z = 2*z_height #0.05
        sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
                halfExtents=[box_x/2,box_y/2,box_z/2])

        self.box_ff=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                                basePosition = [box_x/2+0.08,0,0],baseOrientation=[0,0,0,1])
        self._pybullet_client.changeDynamics(self.box_ff, -1, lateralFriction=1.0)

    def add_box_rr(self, z_height=0.025):
        """ add box under rear feet, temp test """
        #box_x = 0.2
        box_x = 0.25
        box_y = 0.5
        box_z = 2*z_height #0.05
        sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
                halfExtents=[box_x/2,box_y/2,box_z/2])

        self.box_rr=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                                basePosition = [-box_x/2 - 0.05,0,0],baseOrientation=[0,0,0,1])
        self._pybullet_client.changeDynamics(self.box_rr, -1, lateralFriction=1.0)