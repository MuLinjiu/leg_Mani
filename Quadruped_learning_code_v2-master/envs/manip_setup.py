import numpy as np
import envs as envs
from cmath import pi
import os, inspect
import re
import pybullet as p


env_base_path = os.path.dirname(inspect.getfile(envs))
URDF_ROOT = os.path.join(env_base_path, 'assets/')
ROBOT_URDF_FILENAME = "box_robot.urdf"
BOX_URDF_FILENAME = "cube.urdf"

class BoxRobot(object):
    def __init__(self, pybullet_client, init_position, init_orientation):
        self.pybullet_client = pybullet_client

        self.init_position = init_position
        self.init_orientation = init_orientation
        self.load_urdf()

    def load_urdf(self):
        l = 1
        urdf_file = os.path.join(URDF_ROOT, ROBOT_URDF_FILENAME)
        self.uid = self.pybullet_client.loadURDF(\
            urdf_file, \
            useFixedBase=False, \
            basePosition=self.init_position) # baseOrientation=self.init_orientation
        p.addUserDebugLine([0, 0, 0], [l, 0, 0], [1, 0, 0], parentObjectUniqueId=self.uid)
        p.addUserDebugLine([0, 0, 0], [0, l, 0], [0, 1, 0], parentObjectUniqueId=self.uid)
        p.addUserDebugLine([0, 0, 0], [0, 0, l], [0, 0, 1], parentObjectUniqueId=self.uid)

        return self.uid
    
    def get_body_position(self):
        body_position, _ = (self.pybullet_client.getBasePositionAndOrientation(self.uid))
        return body_position

    def get_body_orientation(self):
        _, body_orientation = (self.pybullet_client.getBasePositionAndOrientation(self.uid))
        return body_orientation

    def get_body_linear_velocity(self):
        body_linear_velocity, _ = self.pybullet_client.getBaseVelocity(self.uid)
        return np.asarray(body_linear_velocity)

    def get_body_angular_velocity(self):
        _, body_angular_velocity = self.pybullet_client.getBaseVelocity(self.uid)
        return np.asarray(body_angular_velocity)
    
    def get_body_yaw(self):
        ori = self.get_body_orientation()
        rpy = self.pybullet_client.getEulerFromQuaternion(ori)
        return np.asarray(rpy)[2]

    def apply_forces(self, command):
        self.pybullet_client.applyExternalForce(objectUniqueId=self.uid,
                                                linkIndex=-1,
                                                forceObj=command,
                                                posObj=(0,0,0),
                                                flags=1) #LINK_FRAME
    
    def apply_torques(self, command):
        self.pybullet_client.applyExternalTorque(objectUniqueId=self.uid,
                                                linkIndex=-1,
                                                torqueObj=command,
                                                # posObj=(0,0,0),\
                                                flags=1) #LINK_FRAME
                                        
    def set_velcotiy(self, linear_vel, ang_vel):
        self.pybullet_client.resetBaseVelocity(objectUniqueId = self.uid,
                                              linearVelocity = linear_vel,
                                              angularVelocity = ang_vel)

    def Reset(self, position, orientation):
        self.pybullet_client.resetBasePositionAndOrientation(self.uid,
                                                             position,
                                                             orientation,)
        self.pybullet_client.resetBaseVelocity(self.uid, [0, 0, 0], [0, 0, 0])


class EnvObject(object):
    def __init__(self, pybullet_client, init_position, init_orientation):
        self.pybullet_client = pybullet_client

        self.init_position = init_position
        self.init_orientation = init_orientation
        self.load_urdf()

    def load_urdf(self, fixed=False):
        l = 1
        urdf_file = os.path.join(URDF_ROOT, BOX_URDF_FILENAME)
        self.uid = self.pybullet_client.loadURDF(urdf_file, useFixedBase=fixed, basePosition=self.init_position,
        baseOrientation=self.init_orientation)
        p.addUserDebugLine([0, 0, 0], [l, 0, 0], [1, 0, 0], parentObjectUniqueId=self.uid)
        p.addUserDebugLine([0, 0, 0], [0, l, 0], [0, 1, 0], parentObjectUniqueId=self.uid)
        p.addUserDebugLine([0, 0, 0], [0, 0, l], [0, 0, 1], parentObjectUniqueId=self.uid)

        return self.uid
    
    def get_body_position(self):
        body_position, _ = (self.pybullet_client.getBasePositionAndOrientation(self.uid))
        return body_position

    def get_body_orientation(self):
        _, body_orientation = (self.pybullet_client.getBasePositionAndOrientation(self.uid))
        return body_orientation
    
    def get_body_yaw(self):
        ori = self.get_body_orientation()
        rpy = self.pybullet_client.getEulerFromQuaternion(ori)
        return np.asarray(rpy)[2]

    def get_body_linear_velocity(self):
        body_linear_velocity, _ = self.pybullet_client.getBaseVelocity(self.uid)
        return np.asarray(body_linear_velocity)
    
    def get_body_angular_velocity(self):
        _, body_ang_velocity = self.pybullet_client.getBaseVelocity(self.uid)
        return np.asarray(body_ang_velocity)

    def Reset(self, position, orientation):
        self.pybullet_client.resetBasePositionAndOrientation(self.uid,
                                                             position,
                                                             orientation,)
        self.pybullet_client.resetBaseVelocity(self.uid, [0, 0, 0], [0, 0, 0])