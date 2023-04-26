import os, inspect
import envs.assets as assets

URDF_ROOT = assets.getDataPath()

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data

import numpy as np
from cmath import pi
from collections import deque
import random
import os
import time

from envs.manip_setup import BoxRobot, EnvObject

INIT_ROBOT_ORIENTATION = [0, 0, 0, 1]

class QuadrupedManipEnv(gym.Env):
    def __init__(self,
                 time_step,
                 action_repeat,
                 obs_hist_len,
                 render=False,
                 **kwargs):
        
        self._time_step = time_step
        self._action_repeat = action_repeat
        self._obs_hist_len = obs_hist_len
        self._render = render
        self._max_EP_len = 10
        self._num_bullet_solver_iterations = 60

        self.num_obs = 19

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._action_bound = 1.0
        self._terminate = False
        self._last_base_position = [0, 0, 0]
        self._last_frame_time = 0.0
        self._last_cmd = np.zeros(3)

        if self._render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()
        self._configure_visualizer()
        
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
        self._pybullet_client.setTimeStep(self._time_step)
        self.plane = self._pybullet_client.loadURDF("%s/plane.urdf" % URDF_ROOT,
                                                    basePosition=[0, 0, 0],
                                                    # to extend available running space (shift)
                                                    )
        self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
        self._pybullet_client.configureDebugVisualizer(
            self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
        self._pybullet_client.setGravity(0, 0, -9.8) 
        
        self.yaw_init = 0
        self.init_orientation = self._pybullet_client.getQuaternionFromEuler([0,0,self.yaw_init])

        self.yaw_goal = 0 # np.pi/2.0
        self.goal_orientation = self._pybullet_client.getQuaternionFromEuler([0,0,self.yaw_goal]) # in degrees
        self.goal_position = np.array([0, 0])

        self.init_pos = np.array([-5.0, 0.0, 0.1])

        self.manipulation_target = np.array([self.goal_position[0] - self.init_pos[0],
                                        self.goal_position[1] - self.init_pos[1],
                                        self.yaw_goal - self.yaw_init])

        self._robot = BoxRobot(self._pybullet_client, 
                               self.init_pos - np.array([0.3, 0, 0]),
                               INIT_ROBOT_ORIENTATION)

        self._box = EnvObject(self._pybullet_client,
                              self.init_pos,
                              self.init_orientation)

        self.setupActionSpace()
        self.setupObservationSpace()

    def setupActionSpace(self):
        action_dim = 3
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self._last_action_rl = np.zeros(action_dim)
    
    def setupObservationSpace(self):
        upper_bound = np.array([100] * self.num_obs * self._obs_hist_len)
        lower_bound = np.array([-100] * self.num_obs * self._obs_hist_len)

        self.observation_space = spaces.Box(lower_bound, upper_bound, dtype=np.float32)

    def reset(self):
        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._terminate = False

        self.contact_time = 0
        self.contact_start = 0

        self._robot.Reset(self.init_pos - np.array([0.8, 0, 0]), INIT_ROBOT_ORIENTATION)
        self._box.Reset(self.init_pos, self.init_orientation)

        self._last_action_rl = np.zeros(3)
        
        # settle robot
        for _ in range(200):
            self._pybullet_client.stepSimulation()

        self._obs_buffer = deque([np.zeros(self.observation_space.shape[0])] * self._obs_hist_len)
        for _ in range(self._obs_hist_len):
            self.getObservation()

        if self._render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                       self._cam_pitch, [0, 0, 0])
            self._pybullet_client.changeVisualShape(self._robot.uid, -1, rgbaColor=[0, 0, 0, 1])
            self._pybullet_client.changeVisualShape(self._box.uid, -1, rgbaColor=[255, 166, 1, 1])

        return self.getObservation()

    def getObservation(self):
        observation = []

        robot_states = np.array([self._robot.get_body_position()[0], 
                                 self._robot.get_body_position()[1], 
                                 self._robot.get_body_yaw(),
                                 self._robot.get_body_linear_velocity()[0],
                                 self._robot.get_body_linear_velocity()[1],
                                 self._robot.get_body_angular_velocity()[2]])
        observation.extend(list(robot_states))
        box_states = np.array([self._box.get_body_position()[0], 
                               self._box.get_body_position()[1], 
                               self._box.get_body_yaw(),
                               self._box.get_body_linear_velocity()[0],
                               self._box.get_body_linear_velocity()[1],
                               self._box.get_body_angular_velocity()[2]])
        observation.extend(list(box_states))
        self.manipulation_target = np.array([self.goal_position[0] - self.init_pos[0],
                                        self.goal_position[1] - self.init_pos[1],
                                        self.yaw_goal - self.yaw_init])
        observation.extend(self.manipulation_target)
        observation.extend(self._last_cmd)

        if len(self._pybullet_client.getContactPoints()) <= 8: # contact between robot and box
            observation.extend([0])
        else:
            observation.extend([1])

        self._obs_buffer.appendleft(observation)
        obs = []
        for i in range(self._obs_hist_len):
            obs.extend(self._obs_buffer[i])
        return obs

    def _scale_helper(self, action, lower_lim, upper_lim):
        a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        a = np.clip(a, lower_lim, upper_lim)
        return a

    def step(self, action):
        action = np.clip(action, -self._action_bound, self._action_bound)
        cmd = self.get_manipulation_cmd(action)

        for _ in range(self._action_repeat):
            self._robot.apply_forces((cmd[0], cmd[1], 0))
            self._robot.apply_torques((0, 0, cmd[2]))

            # self._robot.set_velcotiy(np.array([cmd[0], cmd[1], 0]),np.array([0, 0, cmd[2]]) )

            self._pybullet_client.stepSimulation()
            self._sim_step_counter += 1

            if self._render:
                self._render_step_helper()
                time.sleep(0.001)

        self._last_action_rl = action
        self._last_cmd = cmd
        self._env_step_counter += 1
        
        done, reward = self.get_done_and_reward()

        return np.array(self.getObservation()), reward, done, {'box_pos': self._box.get_body_position()}

    def get_manipulation_cmd(self, action):
        lowerbound = np.array([-60.0, -40.0, -23.0]) # range for force/torque control
        upperbound = np.array([60.0, 40.0, 23.0])
        # lowerbound = np.array([-1.0, -1.0, -1.0]) # range for velocity control
        # upperbound = np.array([1.0, 1.0, 1.0])
        cmd = self._scale_helper(action, lowerbound, upperbound)

        return cmd

    def get_done_and_reward(self):

        done = False
        robot_position = self._robot.get_body_position()
        robot_vel = self._robot.get_body_linear_velocity()
        robot_ang_vel = self._robot.get_body_angular_velocity()
        box_position = self._box.get_body_position()
        box_yaw = self._box.get_body_yaw()
        box_vel = self._box.get_body_linear_velocity()

        dist_box_goal = np.sqrt((self.goal_position[0] - box_position[0])**2 \
                                + (self.goal_position[1] - box_position[1])**2)
        
        dist_box_moved = np.array([box_position[0] - self.init_pos[0],
                                   box_position[1] - self.init_pos[1]])

        dist_robot_box = np.sqrt((robot_position[0] - box_position[0])**2 \
                                + (robot_position[1] - box_position[1])**2)
        
        factor = 0
        if dist_box_goal <= 3.5:
            factor = 1

        reward = 0.2 * (0.1 - abs(dist_box_moved[0] - self.manipulation_target[0])) \
                + 0.2 * (0.1 - abs(dist_box_moved[1] - self.manipulation_target[1])) \
                + 0.2 * (0.1 - abs(box_yaw - self.manipulation_target[2])) \
                - 0.1 * dist_robot_box ** 2 + 0.3 * abs(box_vel[0]) + 0.3*abs(box_vel[1])
        
        if dist_box_goal < 0.1 and abs(box_yaw - self.yaw_goal) < 0.1 and (abs(box_vel[0]) + abs(box_vel[1]) + robot_vel[0] + robot_vel[1]) < 0.1:
            done = True
            reward += 10
            return done, reward
        
        if dist_robot_box > 1.5:
            done = True
            # reward -= 10
            if robot_position[0] < self.init_pos[0]:
                reward -= 20
            if dist_box_moved[0] < 0.1:
                reward -= 10
            reward -= 10 + 5 * dist_box_goal + 2.5 * (abs(robot_vel[0]) + abs(robot_vel[1])+ abs(robot_ang_vel[2])) + 5 * abs(box_yaw - self.yaw_goal) + 20 * dist_robot_box
            return done, reward
        
        # if abs(robot_vel[0]) > 5 or abs(robot_vel[1]) > 5 or abs(robot_ang_vel[2]) > 5:
        #     done = True
        #     reward -= 20 + 20 * dist_box_goal + 10 * (abs(box_vel[0]) + abs(box_vel[1])) + 20 * abs(box_yaw - self.yaw_goal)
        #     return done, reward

        if self.get_sim_time() > self._max_EP_len:
            done = True
            if robot_position[0] < self.init_pos[0]:
                reward -= 20
            if dist_box_moved[0] < 0.1:
                reward -= 10
            # reward -= 10
            reward -= 10 + 5 * dist_box_goal + 2.5 * (abs(box_vel[0]) + abs(box_vel[1]) + abs(robot_vel[0]) + abs(robot_vel[1])+ abs(robot_ang_vel[2])) + 5 * abs(box_yaw - self.yaw_goal)
            # at this point, more penalty if the box is further away from the desired state
        return done, reward

    def get_sim_time(self):
        return self._sim_step_counter * self._time_step 

    

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
        # time_to_sleep = self._action_rep(0, 0, cmd[2])
        base_pos = self._robot.get_body_position()
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
        self._cam_dist = 8 # .75 #for paper
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
        base_pos = np.array([self._robot.get_body_position()[0],self._robot.get_body_position()[1], 0.3])
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
