"""This file implements the functionalities of a quadruped using pybullet.
getstate
send control
"""

from cmath import pi
import os
import re
import math
import numpy as np
import os, inspect
import envs_ as envs

# modify robot model urdf dir here
env_base_path = os.path.dirname(inspect.getfile(envs))
URDF_ROOT = os.path.join(env_base_path, 'assets/')
URDF_FILENAME = "a1_description/urdf/a1_rm_fixhips_stl_v2.urdf"

_CHASSIS_NAME_PATTERN = re.compile(r"\w*floating_base\w*")
_HIP_NAME_PATTERN = re.compile(r"\w+_hip_j\w+")
_THIGH_NAME_PATTERN = re.compile(r"\w+_thigh_j\w+")
_CALF_NAME_PATTERN = re.compile(r"\w+_calf_j\w+")
_FOOT_NAME_PATTERN = re.compile(r"\w+_foot_\w+")

ROBOT_MASS = 12
NUM_LEGS = 4
INIT_POSITION = [0, 0, 0.3]
INIT_ORIENTATION = [0, 0, 0, 1]
INIT_MOTOR_ANGLES = np.array([0, 0.4, -1.5] * NUM_LEGS)
TORQUE_LIMITS = np.asarray( [33.5] * 12 )

COM_OFFSET = -np.array([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = np.array([[0.183, -0.047, 0.], [0.183, 0.047, 0.],
                        [-0.183, -0.047, 0.], [-0.183, 0.047, 0.]
                        ])

LEG_OFFSETS = np.array([[0, -0.0838, 0], [0, 0.0838, 0], [0, -0.0838, 0], [0, 0.0838, 0]])

class A1(object):
    def __init__(self,
                 pybullet_client,
                 init_pos = [0, 0, 0.32],
                 init_ori = [0, 0, 0, 1],
                 time_step = 0.001):
        # self._config = robot_config
        self._num_motors = 12
        self._num_legs = 4
        self._pybullet_client = pybullet_client
        self._init_pos = init_pos
        self._init_ori = init_ori
        self.Kp_Joint = np.zeros(12)
        self.Kd_Joint = np.zeros(12)
        self.Kp_Cartesian = np.diag([0,0,0])
        self.Kd_Cartesian = np.diag([0,0,0])
        self.torque_cmds = np.zeros(12)

        self._LoadRobotURDF()
        self._BuildJointNameToIdDict()
        self._BuildUrdfIds()
        self._BuildUrdfMasses()
        self._RemoveDefaultJointDamping()
        self._BuildMotorIdList()
        self._SetMaxJointVelocities()

        self.Reset()

#================================= load urdf and setup ====================================#
    def _LoadRobotURDF(self):
        urdf_file = os.path.join(URDF_ROOT, URDF_FILENAME)
        self.A1 = self._pybullet_client.loadURDF(
            urdf_file,
            self._GetDefaultInitPosition(),
            self._GetDefaultInitOrientation(),
            flags = self._pybullet_client.URDF_USE_SELF_COLLISION
        )

        # self._pybullet_client.createConstraint(self.A1,  -1, -1, -1,
        #                                        self._pybullet_client.JOINT_FIXED, [0, 0, 0],
        #                                        [0, 0, 0], [0, 0, 1], childFrameOrientation=self._GetDefaultInitOrientation())

        return self.A1

    def _BuildJointNameToIdDict(self):
        """_BuildJointNameToIdDict """
        num_joints = self._pybullet_client.getNumJoints(self.A1)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.A1, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildUrdfIds(self):
        """Build the link Ids from its name in the URDF file.

        Raises:
        ValueError: Unknown category of the joint name.
        """
        num_joints = self._pybullet_client.getNumJoints(self.A1)
        self._chassis_link_ids = [-1] # just base link
        self._leg_link_ids = []   # all leg links (hip, thigh, calf)
        self._motor_link_ids = [] # all leg links (hip, thigh, calf)

        self._joint_ids=[]      # all motor joints
        self._hip_ids = []      # hip joint indices only
        self._thigh_ids = []    # thigh joint indices only
        self._calf_ids = []     # calf joint indices only
        self._foot_link_ids = [] # foot joint indices

        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.A1, i)
            # print(joint_info)
            joint_name = joint_info[1].decode("UTF-8")
            joint_id = self._joint_name_to_id[joint_name]
            if _CHASSIS_NAME_PATTERN.match(joint_name):
                self._chassis_link_ids = [joint_id]
            elif _HIP_NAME_PATTERN.match(joint_name):
                self._hip_ids.append(joint_id)
            elif _THIGH_NAME_PATTERN.match(joint_name):
                self._thigh_ids.append(joint_id)
            elif _CALF_NAME_PATTERN.match(joint_name):
                self._calf_ids.append(joint_id)
            elif _FOOT_NAME_PATTERN.match(joint_name):
                self._foot_link_ids.append(joint_id)
            else:
                continue
                raise ValueError("Unknown category of joint %s" % joint_name)

        # everything associated with the leg links
        self._joint_ids.extend(self._hip_ids)
        self._joint_ids.extend(self._thigh_ids)
        self._joint_ids.extend(self._calf_ids)
        # sort in case any weird order
        self._joint_ids.sort()
        self._hip_ids.sort()
        self._thigh_ids.sort()
        self._calf_ids.sort()
        self._foot_link_ids.sort()

        # print('joint ids', self._joint_ids)
        # sys.exit()

    def _BuildUrdfMasses(self):
        self._base_mass_urdf = []
        self._leg_masses_urdf = []
        self._foot_masses_urdf = []
        for chassis_id in self._chassis_link_ids:
            self._base_mass_urdf.append(self._pybullet_client.getDynamicsInfo(self.A1, chassis_id)[0])
        for leg_id in self._joint_ids:
            self._leg_masses_urdf.append(self._pybullet_client.getDynamicsInfo(self.A1, leg_id)[0])
        for foot_id in self._foot_link_ids:
            self._foot_masses_urdf.append(self._pybullet_client.getDynamicsInfo(self.A1, foot_id)[0])

    def _RemoveDefaultJointDamping(self):
        """Pybullet convention/necessity  """
        num_joints = self._pybullet_client.getNumJoints(self.A1)
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.A1, i)
            self._pybullet_client.changeDynamics(
                joint_info[0], -1, linearDamping=0, angularDamping=0)
        # for i in range(num_joints):
        #     self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.5)
    # set link friction (consistnet with Gazebo setup)
        for i in self._foot_link_ids:
            self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.6)
        for i in self._chassis_link_ids:
            self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.2)
        for i in self._hip_ids:
            self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.2)
        for i in self._thigh_ids:
            self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.2)
        for i in self._calf_ids:
            self._pybullet_client.changeDynamics(self.A1, i, lateralFriction=0.2)

    def _BuildMotorIdList(self):
    # self._motor_id_list = [self._joint_name_to_id[motor_name]
    #                         for motor_name in self._robot_config.MOTOR_NAMES]
        self._motor_id_list = self._joint_ids

    def _SetMaxJointVelocities(self):
        """Set maximum joint velocities from robot_config, the pybullet default is 100 rad/s """
        for i, link_id in enumerate(self._joint_ids):
            self._pybullet_client.changeDynamics(self.A1, link_id, maxJointVelocity=21.0)

    def _rand_helper_1d(self, orig_vec, percent_change):
        """Scale appropriately random in low/upp range, 1d vector """
        vec = np.zeros(len(orig_vec))
        for i, elem in enumerate(orig_vec):
            delta = percent_change * np.random.random() * orig_vec[i]
            vec[i] = orig_vec[i] + delta
        return vec

    def RandomizePhysicalParams(self):
        """Randomize physical robot parameters: masses. """
        base_mass = np.array(self._base_mass_urdf)
        leg_masses = np.array(self._leg_masses_urdf)
        foot_masses = np.array(self._foot_masses_urdf)

        new_base_mass = self._rand_helper_1d(base_mass, 0.8)
        new_leg_masses = self._rand_helper_1d(leg_masses, 0.5)
        new_foot_masses = self._rand_helper_1d(foot_masses, 0.5)

        self._pybullet_client.changeDynamics(self.A1, self._chassis_link_ids[0], mass=new_base_mass)
        for i, link_id in enumerate(self._joint_ids):
            self._pybullet_client.changeDynamics(self.A1, link_id, mass=new_leg_masses[i])
        for i, link_id in enumerate(self._foot_link_ids):
            self._pybullet_client.changeDynamics(self.A1, link_id, mass=new_foot_masses[i])

#================================= State feedback ====================================#
    def _GetDefaultInitPosition(self):
        return INIT_POSITION

    def _GetDefaultInitOrientation(self):
        return INIT_ORIENTATION

    def GetBasePosition(self):
        # in world frame
        position, _ = (self._pybullet_client.getBasePositionAndOrientation(self.A1))
        return position

    def GetBaseOrientation(self):
        _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.A1))
        return orientation

    def GetBaseRPY(self):
        ori = self.GetBaseOrientation()
        rpy = self._pybullet_client.getEulerFromQuaternion(ori)
        return np.asarray(rpy)

    def GetBaseOrientationMatrix(self):
        """ Get the base orientation matrix, as numpy array """
        baseOrn = self.GetBaseOrientation()
        return np.asarray(self._pybullet_client.getMatrixFromQuaternion(baseOrn)).reshape((3,3))

    def GetBaseLinearVelocity(self):
        """ Get base linear velocities (dx, dy, dz) in world frame"""
        linVel,_ = self._pybullet_client.getBaseVelocity(self.A1)
        return np.asarray(linVel)

    def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
        # Treat angular velocity as a position vector, then transform based on the
        # orientation given by dividing (or multiplying with inverse).
        # Get inverse quaternion assuming the vector is at 0,0,0 origin.
        _, orientation_inversed = self._pybullet_client.invertTransform([0, 0, 0],
                                                                       orientation)
        # Transform the angular_velocity at neutral orientation using a neutral
        # translation and reverse of the given orientation.
        relative_velocity, _ = self._pybullet_client.multiplyTransforms(
            [0, 0, 0], orientation_inversed, angular_velocity,
            self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
        return np.asarray(relative_velocity)

    def GetBaseAngularVelocity(self):
        """ Get base angular velocities (droll, dpitch, dyaw) in world frame"""
        _, angVel = self._pybullet_client.getBaseVelocity(self.A1)
        return np.asarray(angVel)

    def GetBaseAngularVelocityLocalFrame(self):
        _, angVel = self._pybullet_client.getBaseVelocity(self.A1)
        orientation = self.GetBaseOrientation()
        return self.TransformAngularVelocityToLocalFrame(angVel, orientation)

    def GetMotorAngles(self):
        """Get all motor angles """
        motor_angles = [
            self._pybullet_client.getJointState(self.A1, motor_id)[0]
            for motor_id in self._motor_id_list
        ]
        return motor_angles

    def GetMotorVelocities(self):
        """Get the velocity of all motors."""
        motor_velocities = [
            self._pybullet_client.getJointState(self.A1, motor_id)[1]
            for motor_id in self._motor_id_list
        ]
        return motor_velocities

    def GetMotorTorqueCmds(self):
        return self.torque_cmds

    def GetHipPositionsInBaseFrame(self):
        return HIP_OFFSETS

    def GetHipOffsetsInBaseFrame(self):
        return (HIP_OFFSETS + LEG_OFFSETS)

    def GetFootContacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.A1)

        contacts = [False, False, False, False]
        for contact in all_contacts:
            if contact[2] == self.A1:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(contact[3])
                contacts[toe_link_index] = True
            except ValueError:
                continue

        return contacts

    def GetInvalidContacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.A1)
        for c in all_contacts:
            if c[3] in self._thigh_ids or c[4] in self._thigh_ids:
                return True
            if c[3] in self._calf_ids or c[4] in self._calf_ids:
                return True
            if c[3] in self._hip_ids or c[4] in self._hip_ids:
                return True
            if c[3] in self._chassis_link_ids or c[4] in self._chassis_link_ids:
                return True

        return False

    def get_cam_view(self):
        original_cam_look_direction = np.array([1, 0, 0])  # Same as original robot orientation

        pos = self.GetBasePosition()
        orientation = self.GetBaseOrientation()
        axis, ori = self._pybullet_client.getAxisAngleFromQuaternion(orientation)
        axis = np.array(axis)

        original_cam_up_vector = np.array([0, 0, 1])  # Original camera up vector

        new_cam_up_vector = np.cos(ori) * original_cam_up_vector + np.sin(ori) * np.cross(axis, original_cam_up_vector) + (
                1 - np.cos(ori)) * np.dot(axis, original_cam_up_vector) * axis  # New camera up vector

        new_cam_look_direction = np.cos(ori) * original_cam_look_direction + np.sin(ori) * np.cross(axis, original_cam_look_direction) + (
                1 - np.cos(ori)) * np.dot(axis, original_cam_look_direction) * axis  # New camera look direction

        new_target_pos = pos + new_cam_look_direction  # New target position for camera to look at

        new_cam_pos = pos + 0.3 * new_cam_look_direction

        viewMatrix = self._pybullet_client.computeViewMatrix(
        cameraEyePosition=new_cam_pos,
        cameraTargetPosition=new_target_pos,
        cameraUpVector=new_cam_up_vector)

        projectionMatrix = self._pybullet_client.computeProjectionMatrixFOV(
        fov=87.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=3)

        _, _, _, depth_buffer, _ = self._pybullet_client.getCameraImage(
        width=32,
        height=32,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix)

        near = 0.3
        far = 3

        depthImg =  far * near / (far - (far - near) * depth_buffer)

        return depthImg

#================================== Send Cmd =======================================#
    def _setMotorTorqueById(self, motor_id, torque):
        self._pybullet_client.setJointMotorControl2(bodyIndex=self.A1,
                                                    jointIndex=motor_id,
                                                    controlMode=self._pybullet_client.TORQUE_CONTROL,
                                                    force=torque)

    def ApplyAction(self, kpJoint, kdJoint, qDes, qdotDes, tauDes):
        q = self.GetMotorAngles()
        qdot = self.GetMotorVelocities()

        motor_torque = kpJoint * (qDes - q) + kdJoint * (qdotDes - qdot) + tauDes
        self.torque_cmds = self.ApplyMotorDynamicsConstraint(np.array(qdot), motor_torque)
        
        for motor_id, torque in zip(self._motor_id_list, self.torque_cmds):
            self._setMotorTorqueById(motor_id, torque)

    def SetCartesianPD(self, kpCartesian, kdCartesian):
        self.Kp_Cartesian = kpCartesian
        self.Kd_Cartesian = kdCartesian
    
    def ComputeLegImpedanceControl(self, pDes, vDes, legID):
        q = self.GetMotorAngles()
        qdot = self.GetMotorVelocities()
        torque = np.zeros(3)

        pFoot = self.ComputeFootPosHipFrame(q[legID*3: legID*3+3], legID)
        J = self.ComputeLegJacobian(q[legID*3: legID*3+3], legID)
        vFoot = self.ComputeFootVelHipFrame(q[legID*3: legID*3+3], qdot[legID*3: legID*3+3], legID)

        torque = J.T @ (self.Kp_Cartesian@(pDes - pFoot) \
                          + self.Kd_Cartesian@(vDes - vFoot))

        return torque


    def ComputeImpedanceControl(self, pDes, vDes):
        q = self.GetMotorAngles()
        qdot = self.GetMotorVelocities()
        torque = np.zeros(12)
        for i in range(4):
            pFoot = self.ComputeFootPosHipFrame(q[i*3: i*3+3], i)
            J = self.ComputeLegJacobian(q[i*3: i*3+3], i)
            vFoot = self.ComputeFootVelHipFrame(q[i*3: i*3+3], qdot[i*3: i*3+3], i)

            torque[i*3:i*3+3] = J.T @ (self.Kp_Cartesian@(pDes[i*3:i*3+3] - pFoot) \
                          + self.Kd_Cartesian@(vDes[i*3:i*3+3] - vFoot))

        return torque

    def ComputeForceControl(self, force_cmd):
        q = self.GetMotorAngles()
        torque = np.zeros(12)
        for i in range(4):
            J = self.ComputeLegJacobian(q[i*3:i*3+3], i)
            torque[i*3:i*3+3] = J.T @ (force_cmd[i*3:i*3+3])

        return torque

    def ApplyMotorDynamicsConstraint(self, qdot, tau):
        Kt = 4 / 34  # from Unitree
        torque_motor_max = 4
        speed_motor_max = 1700 * 2 * 3.14 / 60
        max_js = speed_motor_max
        min_js = 940 * 2 * 3.14 / 60
        _alpha_motor = (min_js - max_js) / (torque_motor_max - 0.2)
        _voltage_max = 21.6
        _gear_ratio = 9.1
        _joint_torque_max = 33.5

        voltage = np.zeros(12)
        voltage = (tau * _alpha_motor / _gear_ratio + qdot * _gear_ratio) * Kt
        voltage = np.clip(voltage, -_voltage_max, _voltage_max)
        tau = ((1/Kt) * voltage - qdot*_gear_ratio) * _gear_ratio / _alpha_motor

        return np.clip(tau, -_joint_torque_max, _joint_torque_max)
        


#=================================== Kinematics & motor dynamics constraint ========================================#
    def ComputeLegIK(self, foot_position, leg_id):
        side_sign = 1

        if leg_id == 0 or leg_id == 2:
            side_sign = -1

        l_up = 0.2
        l_low = 0.2
        l_hip = 0.0838 * side_sign
        x, y, z = foot_position[0], foot_position[1], foot_position[2]
        theta_knee = -np.arccos(
                (x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) / (2 * l_low * l_up))
        l = np.sqrt(l_up**2 + l_low**2 + 2 * l_up * l_low * np.cos(theta_knee))
        theta_hip = np.arcsin(-x / l) - theta_knee / 2
        c1 = l_hip * y - l * np.cos(theta_hip + theta_knee / 2) * z
        s1 = l * np.cos(theta_hip + theta_knee / 2) * y + l_hip * z
        theta_ab = np.arctan2(s1, c1)
        qDes = np.array([theta_ab, theta_hip, theta_knee])
        # if np.isnan(qDes).any():
        # 	qDes = np.array([0, 0.5, -1.4])
        return qDes

    def ComputeFootPosHipFrame(self, q ,leg_id):

        side_sign = 1
        if leg_id == 0 or leg_id == 2:
            side_sign = -1

        pos = np.zeros(3)
        l1 = 0.0838
        l2 = 0.2
        l3 = 0.2

        s1 = np.sin(q[0])
        s2 = np.sin(q[1])
        s3 = np.sin(q[2])

        c1 = np.cos(q[0])
        c2 = np.cos(q[1])
        c3 = np.cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        pos[0] = -l3 * s23 - l2 * s2
        pos[1] = l1 * side_sign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
        pos[2] = l1 * side_sign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

        return pos

    def ComputeLegJacobian(self, q, leg_id):
        side_sign = 1
        if leg_id == 0 or leg_id == 2:
            side_sign = -1

        l1 = 0.0838
        l2 = 0.2
        l3 = 0.2

        s1 = np.sin(q[0])
        s2 = np.sin(q[1])
        s3 = np.sin(q[2])

        c1 = np.cos(q[0])
        c2 = np.cos(q[1])
        c3 = np.cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        J = np.zeros((3,3))
        J[1, 0] = -side_sign * l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
        J[2, 0] =  side_sign * l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
        J[0, 1] = -l3 * c23 - l2 * c2
        J[1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
        J[2, 1] = l2 * s2 * c1 + l3 * s23 * c1
        J[0, 2] = -l3 * c23
        J[1, 2] = -l3 * s23 *s1
        J[2, 2] = l3 * s23 * c1

        return J

    def ComputeFootVelHipFrame(self, q, qdot, leg_id):
        J = self.ComputeLegJacobian(q, leg_id)
        foot_vel = J @ qdot
        return foot_vel

    def GetFootPositionsInBaseFrame(self):
        q = np.asarray(self.GetMotorAngles())
        joint_angles = q.reshape((4,3))
        foot_positions = np.zeros((4,3))
        for i in range(4):
            foot_positions[i] = self.ComputeFootPosHipFrame(joint_angles[i], i)
        return foot_positions + HIP_OFFSETS
    


    #==================================== Reset =======================================#
    def Reset(self):
        self._pybullet_client.resetBasePositionAndOrientation(self.A1,
                                                              self._GetDefaultInitPosition(),
                                                              self._GetDefaultInitOrientation())
        self._pybullet_client.resetBaseVelocity(self.A1, [0, 0, 0], [0, 0, 0])
        for name in self._joint_name_to_id:
            joint_id = self._joint_name_to_id[name]
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.A1,
                jointIndex=(joint_id),
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0)
        for i, jointId in enumerate(self._joint_ids):
            angle = INIT_MOTOR_ANGLES[i]
            self._pybullet_client.resetJointState(
                self.A1, jointId, angle, targetVelocity=0
            )
